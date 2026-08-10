import os
import csv
import time
import numpy as np
from stable_baselines3 import PPO
from src.utils import connect_to_carla
from src.environment import CarlaTaxiEnv

CSV_HEADERS = [
    "step",
    "norm_speed",
    "norm_lane_offset",
    "norm_z_diff",
    "norm_heading",
    "norm_obstacle_dist",
    "norm_yaw_rate",
    "norm_last_steer",
    "norm_last_tb",
    "norm_curvature",
    "norm_dist_next",
    "norm_traffic_light",
    "pred_steer",
    "pred_throttle_brake",
    "step_reward"
]

def enjoy_and_log(model_path="models/cloned_taxi_model.zip", max_steps=2000, csv_path="logs/eval_observations.csv"):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    print("Connecting to CARLA for evaluation...")
    client, world = connect_to_carla(port=2000)
    env = CarlaTaxiEnv(client, world)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Cannot find model at {model_path}")
        
    print(f"Loading policy model from {model_path}...")
    model = PPO.load(model_path, env=env)
    
    csv_file = open(csv_path, mode='w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(CSV_HEADERS)
    
    obs, _ = env.reset()
    print(f"Starting evaluation run. Logging telemetry to '{csv_path}'...")
    
    step_count = 0
    try:
        while step_count < max_steps:
            # Predict action using trained policy
            action, _ = model.predict(obs, deterministic=True)
            
            # Step environment forward
            obs, reward, terminated, truncated, _ = env.step(action)
            
            # Log observation vector + predicted actions + step reward
            row = [step_count] + obs.tolist() + [float(action[0]), float(action[1]), float(reward)]
            writer.writerow(row)
            
            step_count += 1
            if step_count % 500 == 0:
                print(f"Evaluated {step_count} / {max_steps} steps...")
                csv_file.flush()
                
            if terminated or truncated:
                print("Episode finished (crashed or off-lane). Resetting...")
                obs, _ = env.reset()
                
    finally:
        csv_file.close()
        env.close()
        print(f"Evaluation finished. Logs saved to {csv_path}")

if __name__ == "__main__":
    enjoy_and_log()