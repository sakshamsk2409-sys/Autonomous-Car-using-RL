import os
import csv
import time
import numpy as np
import carla
from src.utils import connect_to_carla
from src.environment import CarlaTaxiEnv

CSV_HEADERS = [
    "step", "norm_speed", "norm_lane_offset", "norm_z_diff", "norm_heading",
    "norm_obstacle_dist", "norm_yaw_rate", "norm_last_steer", "norm_last_tb",
    "norm_curvature", "norm_dist_next", "norm_traffic_light",
    "action_steer", "action_throttle_brake"
]

def collect_expert_data(total_steps=15000, save_path="data/expert_data.npz", csv_path="logs/expert_observations.csv"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    print("Initiating handshake with CARLA server...")
    client, world = connect_to_carla(port=2000)
    env = CarlaTaxiEnv(client, world)
    
    tm = client.get_trafficmanager(8000)
    tm.global_percentage_speed_difference(30.0)
    
    observations = []
    actions = []
    
    obs, _ = env.reset()
    env.vehicle.set_autopilot(True, tm.get_port())
    
    csv_file = open(csv_path, mode='w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(CSV_HEADERS)
    
    print(f"Starting data collection for {total_steps} frames...")
    
    step_count = 0
    try:
        while step_count < total_steps:
            world.tick()
            
            # Read Autopilot control output
            control = env.vehicle.get_control()
            steer = control.steer
            throttle_brake = -control.brake if control.brake > 0.0 else control.throttle
            
            # Sync action memory inside environment so _get_obs() gets real action values
            env.last_steer = steer
            env.last_throttle_brake = throttle_brake
            
            # Extract observation (now with correct action memory)
            obs = env._get_obs()
            action = np.array([steer, throttle_brake], dtype=np.float32)
            
            observations.append(obs)
            actions.append(action)
            
            writer.writerow([step_count] + obs.tolist() + [steer, throttle_brake])
            
            step_count += 1
            if step_count % 1000 == 0:
                print(f"Collected {step_count} / {total_steps} frames...")
                csv_file.flush()
                
            # Reset if crashed, off-lane, OR if reached the end of the GPS route!
            route_finished = (env.route_index >= len(env.gps_route) - 1)
            if env.is_crashed or abs(env.current_lane_offset) > 4.5 or route_finished:
                reason = "Route Completed!" if route_finished else "Boundary/Collision Hit"
                print(f"Resetting env ({reason}). Respawning vehicle...")
                obs, _ = env.reset()
                env.vehicle.set_autopilot(True, tm.get_port())
                
    finally:
        csv_file.close()
        
    print(f"Saving dataset to {save_path}...")
    np.savez(save_path, states=np.array(observations), actions=np.array(actions))
    print("Data collection complete!")
    env.close()

if __name__ == "__main__":
    collect_expert_data(total_steps=15000)