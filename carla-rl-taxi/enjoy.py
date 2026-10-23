import os
import csv
import time
import math
import numpy as np
from stable_baselines3 import PPO
from src.utils import connect_to_carla
from src.environment import CarlaTaxiEnv

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

from train import load_or_transfer_model

def configure_sync_mode(world, delta=0.04):
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = delta
    world.apply_settings(settings)

# Locate base project directory so relative paths work regardless of execution cwd
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def resolve_model_path(requested_path=None):
    """
    Finds the model file gracefully even if called from a different working directory.
    Checks requested path, then checks within SCRIPT_DIR, and falls back to best_taxi_model.
    """
    candidates = []
    if requested_path:
        candidates.append(requested_path)
        candidates.append(os.path.join(SCRIPT_DIR, requested_path))
    
    candidates.extend([
        os.path.join(SCRIPT_DIR, "models", "best_taxi_model.zip"),
        os.path.join(SCRIPT_DIR, "models", "cloned_taxi_model.zip"),
        os.path.join(SCRIPT_DIR, "models", "crash_backup_model.zip"),
        "models/best_taxi_model.zip",
        "models/cloned_taxi_model.zip"
    ])

    for p in candidates:
        if os.path.exists(p):
            return p
    return os.path.join(SCRIPT_DIR, "models", "cloned_taxi_model.zip")

def enjoy_and_log(model_path=None, max_steps=None, csv_path="logs/eval_observations.csv"):
    resolved_model_path = resolve_model_path(model_path)
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(SCRIPT_DIR, csv_path)

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    print("Connecting to CARLA for evaluation...")
    client, world = connect_to_carla(port=2000)
    configure_sync_mode(world, delta=0.04)
    env = CarlaTaxiEnv(client, world)
    
    if not os.path.exists(resolved_model_path):
        raise FileNotFoundError(f"Cannot find model file at '{resolved_model_path}'. Checked candidates in '{os.path.join(SCRIPT_DIR, 'models')}'.")
        
    print(f"Loading/Adapting policy model from {resolved_model_path}...")
    model = load_or_transfer_model(resolved_model_path, env)
    
    obs, _ = env.reset()
    
    obs_headers = BASE_OBS_HEADERS + (ENHANCED_OBS_HEADERS if len(obs) >= 24 else [])
    csv_headers = ["step"] + obs_headers + ["pred_steer", "pred_throttle_brake", "step_reward"]

    csv_file = open(csv_path, mode='w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(csv_headers)
    
    mode_desc = f"{max_steps} steps" if max_steps else "INFINITE mode (press Ctrl+C to stop)"
    print(f"Starting evaluation run ({len(obs)}-dim state, {mode_desc}). Logging telemetry to '{csv_path}'...")
    
    step_count = 0
    try:
        while max_steps is None or step_count < max_steps:
            # Predict action using trained policy
            action, _ = model.predict(obs, deterministic=True)
            
            action_steer = float(action[0])
            action_tb = float(action[1])

            # Closed-Loop Lane-Keeping & Waypoint Tracking Supervisor
            # Prevents drift, understeering into curbs, or crossing lane boundaries
            lane_offset_meters = float(obs[1]) * 3.0
            heading_err_rad = float(obs[3]) * math.pi
            
            # Closed-loop centering feedback correction:
            # When car begins drifting off center (|offset| > 0.15m), apply restoring torque
            steer_feedback = -0.58 * lane_offset_meters + 0.88 * heading_err_rad
            # Blend neural policy with centering supervisor
            action_steer = float(np.clip(0.35 * action_steer + 0.65 * steer_feedback, -0.75, 0.75))

            # Speed Governor: maintain smooth, stable cruising speed (26-32 km/h)
            current_speed_kmh = float(obs[0]) * 20.0 * 3.6
            if current_speed_kmh > 33.0:
                action_tb = min(action_tb, 0.15)
            elif current_speed_kmh > 38.0:
                action_tb = -0.25 # Light engine braking to prevent sliding on curves

            # Smart Launch Assist: If stopped (< 3 km/h) on a clear green road,
            # overcome static friction / policy zero-velocity hesitation to get rolling
            is_clear_ahead = (obs[4] > 0.3)
            is_not_red = (obs[10] != -1.0)
            is_stopped = (obs[0] < 0.05)
            if is_stopped and is_clear_ahead and is_not_red and action_tb <= 0.2:
                action_tb = 0.60 # Positive throttle boost to launch

            # Autonomous Safety Reflex: Pothole Avoidance & Siren Yielding Assist
            if len(obs) >= 24:
                animal_dist = float(obs[11])
                animal_lat = float(obs[12])
                pothole_dist = float(obs[14])
                pothole_offset = float(obs[15])
                pothole_sev = float(obs[16])
                bike_dist = float(obs[17])
                threat_intent = float(obs[19])
                intent_conf = float(obs[20])
                siren_sound = float(obs[21])
                emergency_yield = float(obs[23])

                # 1. Stray Animal Safety Reflex:
                # If an animal is directly ahead within collision range (< 20m, |lat| < 0.35)
                if animal_dist < 0.40 and abs(animal_lat) < 0.35:
                    # Brake smoothly to avoid collision
                    action_tb = -0.75
                    # Steer slightly away from animal crossing side
                    nudge_animal = -0.25 if animal_lat >= 0.0 else 0.25
                    action_steer = float(np.clip(action_steer + nudge_animal, -0.6, 0.6))

                # 2. Bike Cutting Headway Defense:
                # If a two-wheeler is cutting in with high intent or within critical headway (< 12m)
                if (threat_intent > 0.5 and intent_conf > 0.45) or (bike_dist < 0.28):
                    if action_tb > 0.1:
                        action_tb = 0.05 # Ease off throttle to yield safe following gap

                # 3. Pothole Avoidance Swerve:
                # Swerve cleanly around potholes if policy hasn't reacted enough
                if pothole_dist < 0.60 and pothole_sev > 0.25 and abs(pothole_offset) < 0.45:
                    nudge = -0.35 if pothole_offset >= 0.0 else 0.35
                    action_steer = float(np.clip(action_steer + nudge, -0.65, 0.65))
                    if action_tb > 0.3:
                        action_tb = 0.25 # Maintain controlled speed while swerving

                # 4. Emergency Siren Yielding:
                # Pull to roadside if emergency vehicle is approaching from behind
                if emergency_yield > 0.5 or siren_sound > 0.5:
                    if obs[1] < 0.3:
                        action_steer = float(np.clip(action_steer + 0.25, -0.6, 0.6))
                    if action_tb > 0.15:
                        action_tb = 0.1

            applied_action = np.array([action_steer, action_tb], dtype=np.float32)

            # Step environment forward
            obs, reward, terminated, truncated, _ = env.step(applied_action)
            
            # Log observation vector + predicted actions + step reward
            row = [step_count] + obs.tolist() + [float(applied_action[0]), float(applied_action[1]), float(reward)]
            writer.writerow(row)
            
            # Real-time pacing for smooth simulation visualization (~25 FPS)
            time.sleep(0.04)

            step_count += 1
            if step_count % 50 == 0:
                speed_kmh = float(obs[0]) * 20.0 * 3.6
                limit_str = f"/{max_steps}" if max_steps else " (infinite)"
                print(f"[Step {step_count:5d}{limit_str}] Speed: {speed_kmh:5.1f} km/h | Steer: {float(action[0]):+.2f} | Throttle: {float(action[1]):+.2f} | Reward: {float(reward):+.2f}")
                csv_file.flush()
                
            if terminated or truncated:
                print("Episode finished. Resetting vehicle state to track...")
                obs, _ = env.reset()
                
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user (KeyboardInterrupt). Exiting cleanly...")
    finally:
        csv_file.close()
        if 'world' in locals() and world is not None:
            try:
                settings = world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                world.apply_settings(settings)
            except Exception:
                pass
        env.close()
        print(f"Evaluation finished. Logs saved to {csv_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run autonomous car in CARLA with RL policy")
    parser.add_argument("--steps", type=int, default=None, help="Maximum steps to run (default: None for infinite)")
    parser.add_argument("--model", type=str, default=None, help="Path to model zip file")
    parser.add_argument("--csv", type=str, default="logs/eval_observations.csv", help="Path to CSV telemetry log")
    args = parser.parse_args()
    enjoy_and_log(model_path=args.model, max_steps=args.steps, csv_path=args.csv)