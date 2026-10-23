import os
import sys
import csv
import time
import math
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from src.utils import connect_to_carla
from src.environment import CarlaTaxiEnv
from src.cil.models import CILBranchedNetwork, CILCommand

BASE_OBS_HEADERS = [
    "norm_speed", "norm_lane_offset", "norm_z_diff", "norm_heading",
    "norm_obstacle_dist", "norm_yaw_rate", "norm_last_steer", "norm_last_tb",
    "norm_curvature", "norm_dist_next", "norm_traffic_light"
]

ENHANCED_OBS_HEADERS = [
    "norm_animal_dist", "norm_animal_lat", "lane_confidence",
    "norm_pothole_dist", "norm_pothole_offset", "pothole_severity",
    "norm_bike_dist", "norm_bike_lat_vel", "threat_intent", "intent_conf",
    "siren_sound_level", "siren_bearing", "emergency_yield_req"
]

def load_cil_model(checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        alt_path = os.path.join(SCRIPT_DIR, checkpoint_path)
        if os.path.exists(alt_path):
            checkpoint_path = alt_path
        else:
            raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'")

    print(f"Loading CIL Branched Model from: {checkpoint_path}")
    data = torch.load(checkpoint_path, map_location=device)
    obs_dim = data.get("obs_dim", 24)

    model = CILBranchedNetwork(state_dim=obs_dim, action_dim=2, hidden_dim=256)
    model.load_state_dict(data["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"CIL Model initialized successfully (Input Dim: {obs_dim}, Hidden: 256, Branches: 4).")
    return model, obs_dim

def run_cil_evaluation(model_path="models/cil/best_cil_branched.pth", episodes=3, max_steps=1200):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, expected_obs_dim = load_cil_model(model_path, device)

    print("Connecting to CARLA simulator...")
    client, world = connect_to_carla(port=2000)
    
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    env = CarlaTaxiEnv(client, world)

    logs_dir = os.path.join(SCRIPT_DIR, "logs", "cil")
    os.makedirs(logs_dir, exist_ok=True)
    csv_path = os.path.join(logs_dir, "eval_cil_observations.csv")

    obs_headers = BASE_OBS_HEADERS + (ENHANCED_OBS_HEADERS if expected_obs_dim >= 24 else [])
    csv_headers = ["episode", "step", "command"] + obs_headers + ["action_steer", "action_throttle_brake", "reward"]

    csv_file = open(csv_path, mode='w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(csv_headers)

    print(f"\n=== Starting Closed-Loop CIL Evaluation ({episodes} episodes, max {max_steps} steps/ep) ===")
    
    total_jerks = []
    episode_rewards = []
    
    try:
        for ep in range(1, episodes + 1):
            obs, info = env.reset()
            ep_reward = 0.0
            last_steer = 0.0
            jerks = []

            for step in range(max_steps):
                # 1. Determine active high-level command from route
                current_command = CILCommand.FOLLOW_LANE
                if len(env.gps_route) > 0 and env.route_index < len(env.gps_route):
                    lookahead_idx = min(env.route_index + 3, len(env.gps_route) - 1)
                    _, road_opt = env.gps_route[lookahead_idx]
                    if road_opt is not None and hasattr(road_opt, "value"):
                        current_command = CILCommand.from_carla_road_option(road_opt.value)

                # 2. Format observation tensor
                state_tensor = torch.tensor(obs[:expected_obs_dim], dtype=torch.float32).unsqueeze(0).to(device)
                cmd_tensor = torch.tensor([current_command], dtype=torch.long).to(device)

                # 3. Predict action using the active specialist head
                with torch.no_grad():
                    action_tensor = model(state_tensor, cmd_tensor)
                    action = action_tensor.squeeze(0).cpu().numpy()

                steer = float(action[0])
                throttle_brake = float(action[1])

                # Measure steering jerk
                jerk = abs(steer - last_steer)
                jerks.append(jerk)
                last_steer = steer

                # 4. Step simulation environment
                next_obs, reward, terminated, truncated, _ = env.step(action)
                ep_reward += reward

                # 5. Log telemetry row
                row = [ep, step, int(current_command)] + list(obs[:expected_obs_dim]) + [steer, throttle_brake, reward]
                writer.writerow(row)

                obs = next_obs
                if terminated or truncated:
                    break

            mean_jerk = np.mean(jerks) if len(jerks) > 0 else 0.0
            total_jerks.append(mean_jerk)
            episode_rewards.append(ep_reward)

            print(f"Episode {ep:02d} Complete | Steps: {step+1:4d} | Reward: {ep_reward:7.2f} | Mean Steer Jerk: {mean_jerk:.4f}")

    finally:
        csv_file.close()
        print(f"\nEvaluation telemetry saved to: {csv_path}")
        print(f"Mean Steer Jerk across episodes: {np.mean(total_jerks):.4f}")
        print(f"Mean Episode Reward: {np.mean(episode_rewards):.2f}")


if __name__ == "__main__":
    run_cil_evaluation()
