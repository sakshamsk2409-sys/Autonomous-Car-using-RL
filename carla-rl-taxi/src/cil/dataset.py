import os
import numpy as np
import torch
from torch.utils.data import Dataset
from .models import CILCommand

class CILDataset(Dataset):
    """
    Dataset loader for Conditional Imitation Learning (Codevilla et al. 2017).
    Supports loading states, actions, and conditioned navigational commands.
    """
    def __init__(self, data_path, augment_perturbations=True, noise_std=0.02):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Expert dataset not found at '{data_path}'.")

        data = np.load(data_path)
        self.states = data['states'].astype(np.float32)
        self.actions = data['actions'].astype(np.float32)
        
        # Check if explicit 'commands' array exists in the archive
        if 'commands' in data:
            self.commands = data['commands'].astype(np.int64)
        else:
            # Infer commands from trajectory kinematics (curvature, heading, steering)
            print("Inferring navigation commands from kinematic trajectory features...")
            self.commands = self._infer_commands_from_trajectory(self.states, self.actions)

        self.augment_perturbations = augment_perturbations
        self.noise_std = noise_std

        # Verify command distributions
        unique_cmds, counts = np.unique(self.commands, return_counts=True)
        print("CIL Dataset loaded successfully. Command distribution:")
        for cmd_idx, count in zip(unique_cmds, counts):
            cmd_name = CILCommand(cmd_idx).name if cmd_idx in CILCommand._value2member_map_ else f"CMD_{cmd_idx}"
            print(f"  Branch {cmd_idx} ({cmd_name}): {count} samples ({count / len(self.commands) * 100:.1f}%)")

    def _infer_commands_from_trajectory(self, states, actions):
        """
        Infers high-level navigational commands when explicit planner flags are absent:
        Column 8: norm_curvature
        Column 3: norm_heading
        Action 0: expert steer
        """
        num_samples = len(states)
        commands = np.zeros(num_samples, dtype=np.int64) # Default 0: FOLLOW_LANE

        curvatures = states[:, 8] if states.shape[1] > 8 else np.zeros(num_samples)
        headings = states[:, 3] if states.shape[1] > 3 else np.zeros(num_samples)
        steers = actions[:, 0]

        for i in range(num_samples):
            curv = curvatures[i]
            steer = steers[i]
            hd = headings[i]

            # Right Turn Detection
            if curv > 0.04 or (curv > 0.015 and steer > 0.06):
                commands[i] = CILCommand.TURN_RIGHT
            # Left Turn Detection
            elif curv < -0.04 or (curv < -0.015 and steer < -0.06):
                commands[i] = CILCommand.TURN_LEFT
            # Straight Crossing Detection (low curvature, non-zero speed, straight trajectory)
            elif abs(curv) < 0.01 and abs(hd) < 0.02 and abs(steer) < 0.03:
                # Modulo partitioning for straight intersections vs standard lane following
                if i % 5 == 0:
                    commands[i] = CILCommand.GO_STRAIGHT
                else:
                    commands[i] = CILCommand.FOLLOW_LANE
            else:
                commands[i] = CILCommand.FOLLOW_LANE

        return commands

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state = self.states[idx].copy()
        action = self.actions[idx].copy()
        cmd = self.commands[idx]

        # Synthetic Perturbation Augmentation (Covariate Shift Defense)
        # Injects slight drift noise on lane offset and heading with corrective steering adjustment
        if self.augment_perturbations and np.random.rand() < 0.25:
            # Feature 1 is norm_lane_offset, Feature 3 is norm_heading
            drift_noise = np.random.normal(0, self.noise_std)
            state[1] += drift_noise
            # Counter-steer proportionally to recover back to center
            action[0] = np.clip(action[0] - 0.5 * drift_noise, -1.0, 1.0)

        return (
            torch.from_numpy(state),
            torch.tensor(cmd, dtype=torch.long),
            torch.from_numpy(action)
        )
