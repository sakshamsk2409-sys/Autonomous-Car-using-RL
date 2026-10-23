import os
import sys
import csv
import time
import numpy as np

carla_pythonapi = r"C:\Users\sahak\OneDrive\Documents\AIDS\ML\UE4\PythonAPI"
if os.path.exists(carla_pythonapi):
    if carla_pythonapi not in sys.path:
        sys.path.append(carla_pythonapi)
    p_carla = os.path.join(carla_pythonapi, "carla")
    if p_carla not in sys.path:
        sys.path.append(p_carla)

import carla
from agents.navigation.controller import VehiclePIDController
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

def build_pid_controller(vehicle):
    """
    Constructs a robust, tuned PID controller that drives strictly centered
    along the lane and follows route waypoints without swerving.
    """
    args_lateral_dict = {
        'K_P': 1.95,
        'K_D': 0.2,
        'K_I': 0.07,
        'dt': 0.05
    }
    args_long_dict = {
        'K_P': 1.0,
        'K_D': 0.0,
        'K_I': 0.75,
        'dt': 0.05
    }
    return VehiclePIDController(
        vehicle,
        args_lateral=args_lateral_dict,
        args_longitudinal=args_long_dict,
        max_throttle=0.75,
        max_brake=0.8,
        max_steering=0.7
    )

def collect_expert_data(total_steps=12000, save_path="data/expert_data.npz", csv_path="logs/expert_observations.csv"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    print("Initiating handshake with CARLA server...")
    client, world = connect_to_carla(port=2000)
    env = CarlaTaxiEnv(client, world)
    
    observations = []
    actions = []
    
    obs, _ = env.reset()
    pid = build_pid_controller(env.vehicle)
    
    obs_headers = BASE_OBS_HEADERS + (ENHANCED_OBS_HEADERS if len(obs) >= 24 else [])
    csv_headers = ["step"] + obs_headers + ["action_steer", "action_throttle_brake"]

    csv_file = open(csv_path, mode='w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(csv_headers)
    
    target_speed_kmh = 28.0  # Smooth cruising speed
    print(f"Starting route-locked PID expert demonstration collection ({total_steps} frames, {len(obs)}-dim state)...")
    
    step_count = 0
    try:
        while step_count < total_steps:
            if not env.vehicle or not env.vehicle.is_alive:
                obs, _ = env.reset()
                pid = build_pid_controller(env.vehicle)
                continue

            # Query forward lookahead waypoint from gps_route
            if len(env.gps_route) > 0:
                lookahead_idx = min(env.route_index + 3, len(env.gps_route) - 1)
                target_wp, _ = env.gps_route[lookahead_idx]
            else:
                target_wp = env.map.get_waypoint(env.vehicle.get_location(), project_to_road=True)

            # Compute expert control via PID
            control = pid.run_step(target_speed_kmh, target_wp)
            steer = float(control.steer)
            throttle_brake = float(-control.brake if control.brake > 0.0 else control.throttle)

            # =========================================================
            # Reactive Expert Guidance for Potholes & Emergency Sirens
            # =========================================================
            if len(obs) >= 24:
                pothole_dist = float(obs[14])      # [0.0 close, 1.0 none]
                pothole_offset = float(obs[15])    # [-1.0 left, +1.0 right]
                pothole_sev = float(obs[16])       # severity
                siren_sound = float(obs[21])
                emergency_yield = float(obs[23])

                # 1. Pothole Avoidance Nudge:
                # If a significant pothole is detected ahead (< 20m, dist < 0.65) and aligned with ego
                if pothole_dist < 0.65 and pothole_sev > 0.25:
                    # If pothole is within car width envelope (|offset| < 0.45)
                    if abs(pothole_offset) < 0.45:
                        # Steer in the opposite direction of the pothole to swerve cleanly around it
                        steer_nudge = -0.32 if pothole_offset >= 0.0 else 0.32
                        steer = float(np.clip(steer + steer_nudge, -0.65, 0.65))
                        # Ease throttle to safely navigate around pothole
                        if throttle_brake > 0.2:
                            throttle_brake = 0.25

                # 2. Emergency Siren Yielding:
                # If siren approaching from behind, pull over to shoulder (offset > 0.25) and slow down
                if emergency_yield > 0.5 or siren_sound > 0.5:
                    current_lane_off = float(obs[1])
                    if current_lane_off < 0.25:
                        steer = float(np.clip(steer + 0.25, -0.6, 0.6)) # Nudge rightwards to yield lane
                    if throttle_brake > 0.15:
                        throttle_brake = 0.1 # Decelerate smoothly to let emergency vehicle pass
            
            # Step the environment with this expert action
            action = np.array([steer, throttle_brake], dtype=np.float32)
            obs, reward, terminated, truncated, _ = env.step(action)
            
            observations.append(obs)
            actions.append(action)
            writer.writerow([step_count] + obs.tolist() + [steer, throttle_brake])
            
            step_count += 1
            if step_count % 1000 == 0:
                print(f"Collected {step_count} / {total_steps} clean route-centered frames...")
                csv_file.flush()
                
            if terminated or truncated:
                reason = "Collision/Crash" if terminated else "Route Finished or Border Exceeded"
                print(f"Resetting episode ({reason}). Respawning vehicle...")
                obs, _ = env.reset()
                pid = build_pid_controller(env.vehicle)
                
    finally:
        csv_file.close()
        
    print(f"Saving dataset to {save_path}...")
    np.savez(save_path, states=np.array(observations), actions=np.array(actions))
    print(f"Data collection complete! Saved {len(observations)} clean demonstrations to {save_path}.")

if __name__ == "__main__":
    try:
        collect_expert_data()
    except KeyboardInterrupt:
        print("\nData collection stopped by user command.")