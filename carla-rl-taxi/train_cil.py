import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# Add carla-rl-taxi to path if executed from root
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from src.cil.models import CILBranchedNetwork, CILSingleHeadBaseline, CILCommand
from src.cil.dataset import CILDataset
from src.cil.loss import CILLoss, compute_per_branch_metrics


def train_epoch(model, dataloader, optimizer, criterion, device, is_branched=True):
    model.train()
    total_loss = 0.0
    total_steer_mae = 0.0
    total_tb_mae = 0.0
    num_batches = 0

    for states, commands, actions in dataloader:
        states = states.to(device)
        commands = commands.to(device)
        actions = actions.to(device)

        optimizer.zero_grad()
        if is_branched:
            pred_actions = model(states, commands)
        else:
            pred_actions = model(states, commands)

        loss, metrics = criterion(pred_actions, actions)
        loss.backward()
        optimizer.step()

        total_loss += metrics["total_loss"]
        total_steer_mae += metrics["steer_mae"]
        total_tb_mae += metrics["tb_mae"]
        num_batches += 1

    return {
        "loss": total_loss / max(1, num_batches),
        "steer_mae": total_steer_mae / max(1, num_batches),
        "tb_mae": total_tb_mae / max(1, num_batches)
    }


def validate_epoch(model, dataloader, criterion, device, is_branched=True):
    model.eval()
    total_loss = 0.0
    total_steer_mae = 0.0
    total_tb_mae = 0.0
    num_batches = 0

    all_preds = []
    all_targets = []
    all_commands = []
    all_branch_tensors = []

    with torch.no_grad():
        for states, commands, actions in dataloader:
            states = states.to(device)
            commands = commands.to(device)
            actions = actions.to(device)

            if is_branched:
                # Get both all-branches tensor and selected actions
                all_branch_out = model(states, commands=None) # [batch, 4, 2]
                all_branch_tensors.append(all_branch_out.cpu())
                pred_actions = model(states, commands=commands)
            else:
                pred_actions = model(states, commands)

            loss, metrics = criterion(pred_actions, actions)

            total_loss += metrics["total_loss"]
            total_steer_mae += metrics["steer_mae"]
            total_tb_mae += metrics["tb_mae"]
            num_batches += 1

            all_preds.append(pred_actions.cpu())
            all_targets.append(actions.cpu())
            all_commands.append(commands.cpu())

    summary = {
        "loss": total_loss / max(1, num_batches),
        "steer_mae": total_steer_mae / max(1, num_batches),
        "tb_mae": total_tb_mae / max(1, num_batches)
    }

    if is_branched and len(all_branch_tensors) > 0:
        stacked_branches = torch.cat(all_branch_tensors, dim=0)
        stacked_targets = torch.cat(all_targets, dim=0)
        stacked_commands = torch.cat(all_commands, dim=0)
        branch_metrics = compute_per_branch_metrics(stacked_branches, stacked_targets, stacked_commands)
        summary["branch_diagnostics"] = branch_metrics

    return summary


def run_experiment(data_path="data/expert_data.npz", epochs=25, batch_size=64, lr=5e-4, val_split=0.2):
    # Resolve relative path
    if not os.path.exists(data_path):
        alt_path = os.path.join(current_dir, data_path)
        if os.path.exists(alt_path):
            data_path = alt_path
        else:
            raise FileNotFoundError(f"Expert data not found at {data_path} or {alt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting CIL Reproduction Training Pipeline on Device: {device} ===")
    print(f"Loading dataset from: {data_path}")

    full_dataset = CILDataset(data_path, augment_perturbations=True)
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    obs_dim = full_dataset.states.shape[1]
    print(f"Training set: {train_size} samples | Validation set: {val_size} samples | State dimension: {obs_dim}")

    criterion = CILLoss(loss_type="l1", weight_steer=1.2, weight_throttle_brake=1.0)

    # -------------------------------------------------------------
    # 1. Train Proposed CIL Branched Network (Codevilla et al. 2017)
    # -------------------------------------------------------------
    print("\n[1/2] Training CIL Branched Network (Specialized Action Heads)...")
    branched_net = CILBranchedNetwork(state_dim=obs_dim, action_dim=2, hidden_dim=256).to(device)
    optimizer_branched = optim.Adam(branched_net.parameters(), lr=lr, weight_decay=1e-5)
    scheduler_branched = optim.lr_scheduler.ReduceLROnPlateau(optimizer_branched, mode='min', factor=0.5, patience=3)

    best_branched_loss = float("inf")
    branched_history = []

    models_dir = os.path.join(current_dir, "models", "cil")
    logs_dir = os.path.join(current_dir, "logs", "cil")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    best_branched_path = os.path.join(models_dir, "best_cil_branched.pth")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_res = train_epoch(branched_net, train_loader, optimizer_branched, criterion, device, is_branched=True)
        val_res = validate_epoch(branched_net, val_loader, criterion, device, is_branched=True)
        scheduler_branched.step(val_res["loss"])
        elapsed = time.time() - t0

        is_best = val_res["loss"] < best_branched_loss
        if is_best:
            best_branched_loss = val_res["loss"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": branched_net.state_dict(),
                "optimizer_state_dict": optimizer_branched.state_dict(),
                "val_loss": val_res["loss"],
                "val_steer_mae": val_res["steer_mae"],
                "val_tb_mae": val_res["tb_mae"],
                "obs_dim": obs_dim
            }, best_branched_path)

        branched_history.append({"epoch": epoch, "train": train_res, "val": val_res, "time": elapsed})
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | Train Loss: {train_res['loss']:.4f} | "
                  f"Val Loss: {val_res['loss']:.4f} | Steer MAE: {val_res['steer_mae']:.4f} | TB MAE: {val_res['tb_mae']:.4f}"
                  f"{' *BEST*' if is_best else ''}")

    print(f"CIL Branched Model trained successfully! Best Val Loss: {best_branched_loss:.4f} saved to {best_branched_path}")

    # -------------------------------------------------------------
    # 2. Train Ablation Baseline (Single-Head Command-Input Model)
    # -------------------------------------------------------------
    print("\n[2/2] Training Ablation Baseline (Single-Head Command Input)...")
    baseline_net = CILSingleHeadBaseline(state_dim=obs_dim, action_dim=2, hidden_dim=256).to(device)
    optimizer_baseline = optim.Adam(baseline_net.parameters(), lr=lr, weight_decay=1e-5)
    scheduler_baseline = optim.lr_scheduler.ReduceLROnPlateau(optimizer_baseline, mode='min', factor=0.5, patience=3)

    best_baseline_loss = float("inf")
    baseline_history = []
    best_baseline_path = os.path.join(models_dir, "best_cil_baseline.pth")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_res = train_epoch(baseline_net, train_loader, optimizer_baseline, criterion, device, is_branched=False)
        val_res = validate_epoch(baseline_net, val_loader, criterion, device, is_branched=False)
        scheduler_baseline.step(val_res["loss"])
        elapsed = time.time() - t0

        is_best = val_res["loss"] < best_baseline_loss
        if is_best:
            best_baseline_loss = val_res["loss"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": baseline_net.state_dict(),
                "optimizer_state_dict": optimizer_baseline.state_dict(),
                "val_loss": val_res["loss"],
                "val_steer_mae": val_res["steer_mae"],
                "val_tb_mae": val_res["tb_mae"],
                "obs_dim": obs_dim
            }, best_baseline_path)

        baseline_history.append({"epoch": epoch, "train": train_res, "val": val_res, "time": elapsed})
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | Train Loss: {train_res['loss']:.4f} | "
                  f"Val Loss: {val_res['loss']:.4f} | Steer MAE: {val_res['steer_mae']:.4f} | TB MAE: {val_res['tb_mae']:.4f}"
                  f"{' *BEST*' if is_best else ''}")

    print(f"Ablation Baseline trained successfully! Best Val Loss: {best_baseline_loss:.4f} saved to {best_baseline_path}")

    # -------------------------------------------------------------
    # 3. Comparative Summary & Export
    # -------------------------------------------------------------
    branched_best_val = min(h["val"]["loss"] for h in branched_history)
    baseline_best_val = min(h["val"]["loss"] for h in baseline_history)
    branched_best_steer = min(h["val"]["steer_mae"] for h in branched_history)
    baseline_best_steer = min(h["val"]["steer_mae"] for h in baseline_history)

    steer_improvement = ((baseline_best_steer - branched_best_steer) / max(1e-6, baseline_best_steer)) * 100.0

    print("\n=================================================================")
    print("           REPRODUCTION BENCHMARK SUMMARY (CIL vs Baseline)      ")
    print("=================================================================")
    print(f"  Metric                     | Baseline (Single-Head) | CIL Branched (Ours) | Improvement")
    print(f"  ---------------------------+------------------------+---------------------+------------")
    print(f"  Best Validation Loss (L1)  | {baseline_best_val:22.4f} | {branched_best_val:19.4f} | {((baseline_best_val - branched_best_val)/baseline_best_val)*100:+.1f}%")
    print(f"  Steering MAE               | {baseline_best_steer:22.4f} | {branched_best_steer:19.4f} | {steer_improvement:+.1f}%")
    print("=================================================================")

    # Per-Branch Diagnostic breakdown of final branched model
    final_diag = branched_history[-1]["val"].get("branch_diagnostics", {})
    if final_diag:
        print("\nPer-Branch Validation Diagnostics (Branched Network):")
        for b_name, b_data in final_diag.items():
            print(f"  • {b_name:12s} | Count: {b_data['count']:4d} | Steer MAE: {b_data['steer_mae']:.4f} | TB MAE: {b_data['tb_mae']:.4f}")

    results_export = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "epochs": epochs,
        "batch_size": batch_size,
        "obs_dim": obs_dim,
        "branched_best_val_loss": branched_best_val,
        "branched_best_steer_mae": branched_best_steer,
        "baseline_best_val_loss": baseline_best_val,
        "baseline_best_steer_mae": baseline_best_steer,
        "steer_improvement_pct": steer_improvement,
        "branched_branch_diagnostics": final_diag
    }

    results_json_path = os.path.join(logs_dir, "reproduction_benchmark.json")
    with open(results_json_path, "w") as f:
        json.dump(results_export, f, indent=4)
    print(f"\nDetailed benchmark metrics written to: {results_json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CIL Branched and Baseline Networks")
    parser.add_argument("--data", type=str, default="data/expert_data.npz", help="Path to expert dataset")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    args = parser.parse_args()

    run_experiment(data_path=args.data, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
