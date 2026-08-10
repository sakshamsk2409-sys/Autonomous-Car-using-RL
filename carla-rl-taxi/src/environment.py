import os
import sys
import glob
import random
import time
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    sys.path.append(glob.glob(os.path.abspath('../autonomous/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64')))[0])
except IndexError:
    pass

sys.path.append(os.path.abspath('../autonomous/PythonAPI'))
sys.path.append(os.path.abspath('../autonomous/PythonAPI/carla'))

import carla
from src.rewards import calculate_taxi_reward
from agents.navigation.global_route_planner import GlobalRoutePlanner

class CarlaTaxiEnv(gym.Env):
    def __init__(self, client, world):
        super(CarlaTaxiEnv, self).__init__()
        self.client = client
        self.world = world
        self.map = self.world.get_map()
        
        # Action space: [Steering, Throttle/Brake]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # Updated to 11-element normalized observation space
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(11,), dtype=np.float32
        )
        
        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicle_bp = self.blueprint_library.filter('vehicle.tesla.model3')[0]
        
        self.vehicle = None
        self.collision_sensor = None
        self.is_crashed = False
        self.current_waypoint = None
        self.current_lane_offset = 0.0
        
        # Action memory variables for action-derivative feedback
        self.last_steer = 0.0
        self.last_throttle_brake = 0.0
        
        # GPS Routing Setup
        self.route_planner = GlobalRoutePlanner(self.map, 2.0)
        self.gps_route = []
        self.route_index = 0
        
        # Configurable preview window for future waypoint targeting
        self.lookahead_waypoints = 10
        
        # Single-car validation context
        self.traffic_vehicles = []
        print("Running in single-car environment with 11-feature state vector.")

    def _get_obs(self):
        if not self.vehicle:
            return np.zeros(11, dtype=np.float32)
            
        vehicle_transform = self.vehicle.get_transform()
        vehicle_loc = vehicle_transform.location
        veh_yaw = math.radians(vehicle_transform.rotation.yaw)
        
        # 1. Forward Speed (2D) - Normalized by 20 m/s (~72 km/h max reference)
        v = self.vehicle.get_velocity()
        speed_2d = math.sqrt(v.x**2 + v.y**2)
        norm_speed = np.clip(speed_2d / 20.0, 0.0, 1.0)
        
        # 2. Signed Lane Offset (Distance from road center line)
        local_wp = self.map.get_waypoint(vehicle_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        dx = vehicle_loc.x - local_wp.transform.location.x
        dy = vehicle_loc.y - local_wp.transform.location.y
        forward = local_wp.transform.get_forward_vector()
        right_vector = np.array([-forward.y, forward.x])
        lane_offset = np.dot(np.array([dx, dy]), right_vector)
        self.current_lane_offset = lane_offset
        norm_lane_offset = np.clip(lane_offset / 3.0, -1.0, 1.0) # Normalized over +/- 3.0 meters
        
        # 3. Z-Difference (Vertical offset for slopes and road surface bumps)
        z_diff = vehicle_loc.z - local_wp.transform.location.z
        norm_z_diff = np.clip(z_diff / 1.0, -1.0, 1.0)
        
        # Update GPS Target Waypoint Tracking
        if self.route_index < len(self.gps_route):
            target_wp, _ = self.gps_route[self.route_index]
            if vehicle_loc.distance(target_wp.transform.location) < 2.0:
                self.route_index = min(self.route_index + 1, len(self.gps_route) - 1)
                target_wp, _ = self.gps_route[self.route_index]
        else:
            target_wp = local_wp

        # 4. Current Heading Error
        wp_yaw = math.radians(target_wp.transform.rotation.yaw)
        heading_error = math.atan2(math.sin(wp_yaw - veh_yaw), math.cos(wp_yaw - veh_yaw))
        norm_heading = np.clip(heading_error / math.pi, -1.0, 1.0)
        
        # 5. Obstacle Distance (Frontal Raycast check against surrounding actors)
        obstacle_dist = 50.0  # Max detection range in meters
        actors = self.world.get_actors().filter('vehicle.*')
        for actor in actors:
            if actor.id != self.vehicle.id:
                loc = actor.get_location()
                dist = vehicle_loc.distance(loc)
                if dist < obstacle_dist:
                    fwd = vehicle_transform.get_forward_vector()
                    vec_to_actor = carla.Vector3D(loc.x - vehicle_loc.x, loc.y - vehicle_loc.y, loc.z - vehicle_loc.z)
                    dot = fwd.x * vec_to_actor.x + fwd.y * vec_to_actor.y
                    if dot > 0: # Actor is in front of ego vehicle
                        obstacle_dist = min(obstacle_dist, dist)
        norm_obstacle_dist = np.clip(obstacle_dist / 50.0, 0.0, 1.0)

        # 6. Yaw Rate (Angular velocity around Z-axis in rad/s)
        ang_vel = self.vehicle.get_angular_velocity()
        norm_yaw_rate = np.clip(math.radians(ang_vel.z) / math.pi, -1.0, 1.0)
        
        # 7 & 8. Last Applied Controls (Steering angle & Throttle/Brake)
        norm_last_steer = np.clip(self.last_steer, -1.0, 1.0)
        norm_last_tb = np.clip(self.last_throttle_brake, -1.0, 1.0)

        # 9. Waypoint Curvature (Rate of turn over look-ahead horizon)
        future_index = min(self.route_index + self.lookahead_waypoints, len(self.gps_route) - 1)
        future_wp, _ = self.gps_route[future_index] if len(self.gps_route) > 0 else (target_wp, None)
        future_yaw = math.radians(future_wp.transform.rotation.yaw)
        curvature = math.atan2(math.sin(future_yaw - wp_yaw), math.cos(future_yaw - wp_yaw))
        norm_curvature = np.clip(curvature / math.pi, -1.0, 1.0)

        # 10. Distance to Next Waypoint (Euclidean distance to target)
        dist_to_next = vehicle_loc.distance(target_wp.transform.location)
        norm_dist_next = np.clip(dist_to_next / 10.0, 0.0, 1.0)

        # 11. Traffic Light State (Red=-1.0, Yellow=0.0, Green/None=1.0)
        traffic_light_val = 1.0
        if self.vehicle.is_at_traffic_light():
            traffic_light = self.vehicle.get_traffic_light()
            if traffic_light:
                state = traffic_light.get_state()
                if state == carla.TrafficLightState.Red:
                    traffic_light_val = -1.0
                elif state == carla.TrafficLightState.Yellow:
                    traffic_light_val = 0.0
                elif state == carla.TrafficLightState.Green:
                    traffic_light_val = 1.0
        norm_traffic_light = traffic_light_val

        return np.array([
            norm_speed,             # 0
            norm_lane_offset,       # 1
            norm_z_diff,            # 2
            norm_heading,           # 3
            norm_obstacle_dist,     # 4
            norm_yaw_rate,          # 5
            norm_last_steer,        # 6
            norm_last_tb,           # 7
            norm_curvature,         # 8
            norm_dist_next,         # 9
            norm_traffic_light      # 10
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.is_crashed = False
        self.last_steer = 0.0
        self.last_throttle_brake = 0.0
        
        # Clean up active sensors
        if self.collision_sensor is not None:
            try:
                self.collision_sensor.stop()
                self.collision_sensor.destroy()
            except Exception:
                pass
            self.collision_sensor = None

        if self.vehicle is not None:
            try:
                self.vehicle.destroy()
            except Exception:
                pass
            self.vehicle = None

        # Safe Actor Spawning Blocks
        spawn_points = self.map.get_spawn_points()
        start_wp = spawn_points[20]
        end_wp = spawn_points[149]
        
        self.vehicle = self.world.try_spawn_actor(self.vehicle_bp, start_wp)
        while self.vehicle is None:
            time.sleep(0.1)
            self.vehicle = self.world.try_spawn_actor(self.vehicle_bp, start_wp)
            
        # Initialize route path trajectories
        self.gps_route = self.route_planner.trace_route(start_wp.location, end_wp.location)
        self.route_index = 0

        # Attach Collision Sensor
        collision_bp = self.blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: self._on_collision(event))

        self.world.tick()
        return self._get_obs(), {}

    def _on_collision(self, event):
        self.is_crashed = True

    def step(self, action):
        steer = float(action[0])
        throttle_brake = float(action[1])
        
        # Store actions in memory for action-derivative feedback
        self.last_steer = steer
        self.last_throttle_brake = throttle_brake
        
        control = carla.VehicleControl()
        control.steer = np.clip(steer, -1.0, 1.0)
        
        if throttle_brake >= 0.0:
            control.throttle = np.clip(throttle_brake, 0.0, 1.0)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = np.clip(abs(throttle_brake), 0.0, 1.0)
            
        if self.vehicle:
            if random.random() < 0.01:
                print(
                    f"Action={action}, "
                    f"Throttle={control.throttle:.2f}, "
                    f"Brake={control.brake:.2f}, "
                    f"Steer={control.steer:.2f}"
                )
            self.vehicle.apply_control(control)
            
        self.world.tick()
        
        obs = self._get_obs()
        reward = calculate_taxi_reward(obs, self.is_crashed)
        
        terminated = self.is_crashed
        truncated = False
        
        if abs(self.current_lane_offset) > 4.5:
            truncated = True
            
        return obs, reward, terminated, truncated, {}