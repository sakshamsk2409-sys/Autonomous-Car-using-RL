import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import IntEnum

class CILCommand(IntEnum):
    """
    High-level navigational commands conditioned in the CIL policy (Codevilla et al., 2017).
    """
    FOLLOW_LANE = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2
    GO_STRAIGHT = 3

    @classmethod
    def from_carla_road_option(cls, road_option_val):
        """
        Maps CARLA GlobalRoutePlanner RoadOption enum values to CIL command branches.
        CARLA RoadOption mapping:
          1: LEFT -> TURN_LEFT
          2: RIGHT -> TURN_RIGHT
          3: STRAIGHT -> GO_STRAIGHT
          4: LANEFOLLOW -> FOLLOW_LANE
        """
        if road_option_val == 1:
            return cls.TURN_LEFT
        elif road_option_val == 2:
            return cls.TURN_RIGHT
        elif road_option_val == 3:
            return cls.GO_STRAIGHT
        else:
            return cls.FOLLOW_LANE


class CILBranchedNetwork(nn.Module):
    """
    Branched Conditional Imitation Learning Network Architecture (Codevilla et al., CoRL 2017 / ICRA 2018).
    
    Instead of feeding high-level navigational commands as a flat feature input, 
    the representation trunk splits into specialized, command-gated branches.
    This prevents catastrophic path averaging at intersections.
    """
    def __init__(self, state_dim=24, action_dim=2, hidden_dim=256, dropout_prob=0.2):
        super(CILBranchedNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_branches = 4

        # 1. Shared Perception / State Trunk
        self.shared_trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob)
        )

        # 2. Specialized Command Branches:
        # Branch 0: Follow Lane
        # Branch 1: Turn Left
        # Branch 2: Turn Right
        # Branch 3: Go Straight
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, action_dim),
                nn.Tanh() # Action range: [-1.0, 1.0] for steer and throttle/brake
            )
            for _ in range(self.num_branches)
        ])

    def forward(self, state, commands=None):
        """
        Forward pass.
        Args:
            state (Tensor): [batch_size, state_dim]
            commands (Tensor, optional): [batch_size] long tensor of command indices (0..3).
        Returns:
            If commands is None:
                Tensor of shape [batch_size, num_branches, action_dim] containing outputs of all branches.
            If commands is provided:
                Tensor of shape [batch_size, action_dim] selected by the respective command for each sample.
        """
        features = self.shared_trunk(state)
        branch_outputs = [branch(features) for branch in self.branches]
        all_actions = torch.stack(branch_outputs, dim=1) # [batch_size, 4, action_dim]

        if commands is None:
            return all_actions

        batch_size = state.size(0)
        # Ensure commands are in range [0, 3]
        clamped_commands = torch.clamp(commands.long(), 0, self.num_branches - 1)
        # Gather the specific branch output for each sample in the batch
        # clamped_commands shape: [batch_size, 1, 1] -> expand to [batch_size, 1, action_dim]
        cmd_indices = clamped_commands.view(batch_size, 1, 1).expand(-1, -1, self.action_dim)
        selected_actions = all_actions.gather(1, cmd_indices).squeeze(1)
        return selected_actions


class CILSingleHeadBaseline(nn.Module):
    """
    Command-Input Baseline Architecture (Section 4 baseline in Codevilla et al.).
    
    Here, the high-level command is concatenated as a one-hot auxiliary vector directly
    into the state vector. Used as an ablation baseline to demonstrate the superiority
    of the branched architecture over naive multi-modal behavioral cloning.
    """
    def __init__(self, state_dim=24, action_dim=2, num_commands=4, hidden_dim=256, dropout_prob=0.2):
        super(CILSingleHeadBaseline, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_commands = num_commands

        self.net = nn.Sequential(
            nn.Linear(state_dim + num_commands, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh()
        )

    def forward(self, state, commands):
        """
        Args:
            state (Tensor): [batch_size, state_dim]
            commands (Tensor): [batch_size] long tensor of command indices (0..3)
        """
        batch_size = state.size(0)
        clamped_commands = torch.clamp(commands.long(), 0, self.num_commands - 1)
        one_hot = F.one_hot(clamped_commands, num_classes=self.num_commands).float()
        x = torch.cat([state, one_hot], dim=1)
        return self.net(x)


class CILVisionBranchedNetwork(nn.Module):
    """
    Vision-based Branched CIL Network (Codevilla et al. 2017).
    8 Convolutional layers + 2 Fully Connected layers + 4 Command Branches.
    """
    def __init__(self, in_channels=3, action_dim=2, hidden_dim=512, dropout_prob=0.5):
        super(CILVisionBranchedNetwork, self).__init__()
        self.num_branches = 4
        self.action_dim = action_dim

        # 8 Conv layers with alternating strides
        self.conv_trunk = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        # Image feature compression
        self.fc_image = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob)
        )

        # Measurement stream (forward speed input)
        self.speed_trunk = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        # Joint fusion
        self.joint_fc = nn.Sequential(
            nn.Linear(hidden_dim + 128, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob)
        )

        # 4 Specialized command branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, action_dim),
                nn.Tanh()
            )
            for _ in range(self.num_branches)
        ])

    def forward(self, img, speed, commands=None):
        img_feats = self.fc_image(self.conv_trunk(img))
        speed_feats = self.speed_trunk(speed.view(-1, 1))
        joint_feats = self.joint_fc(torch.cat([img_feats, speed_feats], dim=1))

        branch_outputs = [branch(joint_feats) for branch in self.branches]
        all_actions = torch.stack(branch_outputs, dim=1)

        if commands is None:
            return all_actions

        batch_size = img.size(0)
        clamped_commands = torch.clamp(commands.long(), 0, self.num_branches - 1)
        cmd_indices = clamped_commands.view(batch_size, 1, 1).expand(-1, -1, self.action_dim)
        return all_actions.gather(1, cmd_indices).squeeze(1)
