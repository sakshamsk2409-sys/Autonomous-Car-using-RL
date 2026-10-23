import torch
import torch.nn as nn
import torch.nn.functional as F

class CILLoss(nn.Module):
    """
    Branch-conditional loss function from Codevilla et al. (2017).
    
    Backpropagates error strictly through the specialist branch indicated by command c:
        L = weight_steer * |steer_pred - steer_gt| + weight_tb * |tb_pred - tb_gt|
    """
    def __init__(self, loss_type="l1", weight_steer=1.2, weight_throttle_brake=1.0):
        super(CILLoss, self).__init__()
        self.loss_type = loss_type
        self.weight_steer = weight_steer
        self.weight_tb = weight_throttle_brake

    def forward(self, pred_actions, target_actions):
        """
        Args:
            pred_actions (Tensor): [batch_size, 2] actions selected by the command branch
            target_actions (Tensor): [batch_size, 2] ground truth expert actions [steer, throttle_brake]
        Returns:
            loss (Tensor): scalar loss
            metrics (dict): detailed error metrics per action component
        """
        steer_diff = pred_actions[:, 0] - target_actions[:, 0]
        tb_diff = pred_actions[:, 1] - target_actions[:, 1]

        if self.loss_type == "l1":
            steer_loss = torch.abs(steer_diff).mean()
            tb_loss = torch.abs(tb_diff).mean()
        else: # mse
            steer_loss = torch.mean(steer_diff ** 2)
            tb_loss = torch.mean(tb_diff ** 2)

        total_loss = self.weight_steer * steer_loss + self.weight_tb * tb_loss

        metrics = {
            "total_loss": total_loss.item(),
            "steer_loss": steer_loss.item(),
            "tb_loss": tb_loss.item(),
            "steer_mae": torch.abs(steer_diff).mean().item(),
            "tb_mae": torch.abs(tb_diff).mean().item()
        }
        return total_loss, metrics


def compute_per_branch_metrics(all_branch_preds, target_actions, commands, num_branches=4):
    """
    Computes validation diagnostics for each individual command branch.
    Args:
        all_branch_preds (Tensor): [batch_size, num_branches, action_dim]
        target_actions (Tensor): [batch_size, action_dim]
        commands (Tensor): [batch_size] command indices
    Returns:
        dict: branch_name -> {"count": int, "steer_mae": float, "tb_mae": float}
    """
    branch_names = ["FOLLOW_LANE", "TURN_LEFT", "TURN_RIGHT", "GO_STRAIGHT"]
    results = {}

    for b_idx in range(num_branches):
        mask = (commands == b_idx)
        count = mask.sum().item()
        b_name = branch_names[b_idx] if b_idx < len(branch_names) else f"BRANCH_{b_idx}"

        if count == 0:
            results[b_name] = {"count": 0, "steer_mae": 0.0, "tb_mae": 0.0}
            continue

        branch_pred = all_branch_preds[mask, b_idx, :]
        branch_tgt = target_actions[mask, :]

        steer_mae = torch.abs(branch_pred[:, 0] - branch_tgt[:, 0]).mean().item()
        tb_mae = torch.abs(branch_pred[:, 1] - branch_tgt[:, 1]).mean().item()

        results[b_name] = {
            "count": count,
            "steer_mae": steer_mae,
            "tb_mae": tb_mae
        }

    return results
