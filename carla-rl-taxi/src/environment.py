import os
import sys
import glob
import random
import time
import math
import yaml
import numpy as np
import gymnasium as gym
from gymnasium import spaces

carla_pythonapi = r"C:\Users\sahak\OneDrive\Documents\AIDS\ML\UE4\PythonAPI"
if os.path.exists(carla_pythonapi):
    sys.path.append(carla_pythonapi)
    sys.path.append(os.path.join(carla_pythonapi, "carla"))

try:
    import carla
    from agents.navigation.global_route_planner import GlobalRoutePlanner
except ImportError:
    carla = None
    GlobalRoutePlanner = None

from src.rewards import calculate_taxi_reward
from src.sensors.acoustic_sensor import AcousticSirenSensor
from src.sensors.road_surface_sensor import RoadSurfaceSensor
from src.scenarios.stray_animals import StrayAnimalManager
from src.scenarios.potholes import PotholeManager
from src.scenarios.bike_cutting import BikeCuttingManager
from src.scenarios.emergency_vehicles import EmergencyVehicleManager
from src.prediction.intent_predictor import ConfidenceIntentPredictor

class CarlaTaxiEnv(gym.Env):
    """
    Autonomous driving environment for CARLA Simulator with 5 advanced real-world features:
    1. Stray Animals (roadside wandering & sudden road crossing)
    2. Faded Line Markings & Potholes (sensor noise & chassis impact dynamics)
    3. Bike Cutting (aggressive two-wheeler blind-spot overtaking & cut-ins)
    4. Intent Prediction (kinematic maneuver classification with calibrated confidence)
    5. Emergency Vehicle Sensing (acoustic siren model with Doppler & directional bearing)
    """
    def __init__(self, client, world=None, config_file="config/settings.yaml"):
        super(CarlaTaxiEnv, self).__init__()
        self.client = client
        if self.client:
            self.client.set_timeout(20.0)

        # Load configuration
        self.config = self._load_config(config_file)
        env_cfg = self.config.get("environment", {})
        self.observation_mode = env_cfg.get("observation_mode", "enhanced")
        
        # Action space: [Steering (-1..1), Throttle/Brake (-1..1)]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # Observation space: 24 features (enhanced) or 11 features (legacy)
        obs_dim = 24 if self.observation_mode == "enhanced" else 11
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        if world is not None:
            self.world = world
        elif self.client:
            target_town = env_cfg.get("town", "Town10HD")
            current_map = self.client.get_world().get_map().name
            if target_town not in current_map:
                print(f"Loading {target_town}...")
                self.world = self.client.load_world(target_town)
            else:
                self.world = self.client.get_world()
        else:
            self.world = None

        self.map = self.world.get_map() if self.world else None
        self.blueprint_library = self.world.get_blueprint_library() if self.world else None
        self.vehicle_bp = self.blueprint_library.filter('vehicle.tesla.model3')[0] if self.blueprint_library else None
        
        self.vehicle = None
        self.collision_sensor = None
        self.is_crashed = False
        self.current_lane_offset = 0.0
        self.step_count = 0
        
        # Action memory variables for action-derivative feedback
        self.last_steer = 0.0
        self.last_throttle_brake = 0.0
        self.smooth_steer = 0.0
        self.smooth_throttle_brake = 0.0
        self.last_cam_loc = None
        self.last_cam_yaw = None
        
        # GPS Routing Setup
        self.route_planner = GlobalRoutePlanner(self.map, 2.0) if (GlobalRoutePlanner and self.map) else None
        self.gps_route = []
        self.route_index = 0
        self.lookahead_waypoints = env_cfg.get("lookahead_waypoints", 10)

        # -------------------------------------------------------------
        # Initialize the 5 Feature Modules
        # -------------------------------------------------------------
        # 1. Stray Animals
        self.stray_animal_mgr = StrayAnimalManager(
            self.world, self.client, self.config.get("stray_animals", {})
        )

        # 2. Road Surface & Potholes
        rh_cfg = self.config.get("road_hazards", {})
        fl_cfg = rh_cfg.get("faded_lines", {})
        self.road_surface_sensor = RoadSurfaceSensor(
            faded_line_probability=fl_cfg.get("faded_segment_probability", 0.35),
            min_line_confidence=fl_cfg.get("min_confidence", 0.15),
            faded_noise_sigma=fl_cfg.get("noise_sigma", 0.45)
        )
        self.pothole_mgr = PotholeManager(
            self.world, rh_cfg.get("potholes", {})
        )

        # 3. Bike Cutting
        self.bike_cutting_mgr = BikeCuttingManager(
            self.world, self.client, self.config.get("bike_cutting", {})
        )

        # 4. Emergency Vehicles & Acoustic Siren Sensor
        es_cfg = self.config.get("emergency_siren", {})
        self.emergency_vehicle_mgr = EmergencyVehicleManager(
            self.world, self.client, es_cfg
        )
        self.acoustic_sensor = AcousticSirenSensor(
            base_sound_level_db=es_cfg.get("base_sound_level_db", 120.0),
            reference_frequency_hz=es_cfg.get("reference_frequency_hz", 900.0),
            ambient_noise_db=es_cfg.get("ambient_noise_db", 65.0),
            speed_of_sound_mps=es_cfg.get("speed_of_sound_mps", 343.0),
            distance_decay_alpha=es_cfg.get("distance_decay_alpha", 0.008)
        )

        # 5. Confidence-Based Intent Predictor
        ip_cfg = self.config.get("intent_prediction", {})
        self.intent_predictor = ConfidenceIntentPredictor(
            history_len=ip_cfg.get("history_window_steps", 8)
        )

        self.last_pothole_penalty = 0.0
        print(f"Initialized CarlaTaxiEnv in [{self.observation_mode.upper()}] mode with {obs_dim}-feature observation space.")

    def _load_config(self, config_file):
        if os.path.exists(config_file):
            try:
                with open(config_file, "r") as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                print(f"Warning: Could not parse {config_file}: {e}")
        return {}

    def _get_obs(self):
        obs_dim = 24 if self.observation_mode == "enhanced" else 11
        if not self.vehicle:
            return np.zeros(obs_dim, dtype=np.float32)
            
        vehicle_transform = self.vehicle.get_transform()
        vehicle_loc = vehicle_transform.location
        veh_yaw = math.radians(vehicle_transform.rotation.yaw)
        
        # 1. Forward Speed (2D) - Normalized by 20 m/s (~72 km/h max reference)
        v = self.vehicle.get_velocity()
        speed_2d = math.sqrt(v.x**2 + v.y**2)
        norm_speed = float(np.clip(speed_2d / 20.0, 0.0, 1.0))
        
        # 2. Signed Lane Offset (Distance from road center line)
        local_wp = self.map.get_waypoint(vehicle_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        dx = vehicle_loc.x - local_wp.transform.location.x
        dy = vehicle_loc.y - local_wp.transform.location.y
        right_vector = local_wp.transform.get_right_vector()
        true_lane_offset = float(dx * right_vector.x + dy * right_vector.y)
        self.current_lane_offset = true_lane_offset

        # Feature 2 (Faded Lines): Assess lane line confidence and noise injection
        lane_confidence, perceived_lane_offset = self.road_surface_sensor.assess_lane_marking(true_lane_offset)
        norm_lane_offset = float(np.clip(perceived_lane_offset / 3.0, -1.0, 1.0))
        
        # Robust Forward-Window GPS Target Waypoint Tracking
        # Avoids waypoint-skip lockup on curves or wide lanes
        if len(self.gps_route) > 0:
            search_start = max(0, self.route_index)
            search_end = min(self.route_index + 25, len(self.gps_route))
            min_dist = float("inf")
            best_idx = self.route_index
            for i in range(search_start, search_end):
                wp_cand, _ = self.gps_route[i]
                d = vehicle_loc.distance(wp_cand.transform.location)
                if d < min_dist:
                    min_dist = d
                    best_idx = i
            self.route_index = best_idx
            
            # Select target waypoint slightly ahead on the route for responsive steering
            lookahead_steps = 3  # ~6-8 meters ahead
            target_idx = min(self.route_index + lookahead_steps, len(self.gps_route) - 1)
            target_wp, _ = self.gps_route[target_idx]
        else:
            target_wp = local_wp

        # 3. Z-Difference (Vertical slope/elevation difference to upcoming waypoint)
        z_diff = target_wp.transform.location.z - vehicle_loc.z
        norm_z_diff = float(np.clip(z_diff / 1.5, -1.0, 1.0))

        # 4. Current Heading Error
        wp_yaw = math.radians(target_wp.transform.rotation.yaw)
        heading_error = math.atan2(math.sin(wp_yaw - veh_yaw), math.cos(wp_yaw - veh_yaw))
        norm_heading = float(np.clip(heading_error / math.pi, -1.0, 1.0))
        
        # 5. Obstacle Distance (Frontal Raycast)
        obstacle_dist = 50.0
        actors = self.world.get_actors().filter('vehicle.*') if self.world else []
        for actor in actors:
            if actor.id != self.vehicle.id:
                loc = actor.get_location()
                dist = vehicle_loc.distance(loc)
                if dist < obstacle_dist:
                    fwd = vehicle_transform.get_forward_vector()
                    vec_to_actor = carla.Vector3D(loc.x - vehicle_loc.x, loc.y - vehicle_loc.y, loc.z - vehicle_loc.z)
                    dot = fwd.x * vec_to_actor.x + fwd.y * vec_to_actor.y
                    if dot > 0:
                        obstacle_dist = min(obstacle_dist, dist)
        norm_obstacle_dist = float(np.clip(obstacle_dist / 50.0, 0.0, 1.0))

        # 6. Yaw Rate (Angular velocity around Z-axis in rad/s)
        ang_vel = self.vehicle.get_angular_velocity()
        norm_yaw_rate = float(np.clip(math.radians(ang_vel.z) / math.pi, -1.0, 1.0))
        
        # 7 & 8. Last Applied Controls
        norm_last_steer = float(np.clip(self.last_steer, -1.0, 1.0))
        norm_last_tb = float(np.clip(self.last_throttle_brake, -1.0, 1.0))

        # 9. Waypoint Curvature (10-step look-ahead horizon)
        future_index = min(self.route_index + self.lookahead_waypoints, len(self.gps_route) - 1)
        future_wp, _ = self.gps_route[future_index] if len(self.gps_route) > 0 else (target_wp, None)
        future_yaw = math.radians(future_wp.transform.rotation.yaw)
        curvature = math.atan2(math.sin(future_yaw - wp_yaw), math.cos(future_yaw - wp_yaw))
        norm_curvature = float(np.clip(curvature / math.pi, -1.0, 1.0))

        # 10. Distance to Next Waypoint
        dist_to_next = vehicle_loc.distance(target_wp.transform.location)
        norm_dist_next = float(np.clip(dist_to_next / 10.0, 0.0, 1.0))

        # 11. Traffic Light State
        traffic_light_val = 1.0
        if self.vehicle.is_at_traffic_light():
            traffic_light = self.vehicle.get_traffic_light()
            if traffic_light and hasattr(traffic_light, "get_state"):
                try:
                    state = traffic_light.get_state()
                    if state == carla.TrafficLightState.Red:
                        traffic_light_val = -1.0
                    elif state == carla.TrafficLightState.Yellow:
                        traffic_light_val = 0.0
                    elif state == carla.TrafficLightState.Green:
                        traffic_light_val = 1.0
                except Exception:
                    pass
        norm_traffic_light = float(traffic_light_val)

        base_obs = [
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
        ]

        if self.observation_mode == "legacy":
            return np.array(base_obs, dtype=np.float32)

        # -------------------------------------------------------------
        # Features 11 to 23: The 5 Advanced Real-World Features
        # -------------------------------------------------------------
        # Feature 1: Stray Animals
        norm_animal_dist, norm_animal_lat, _ = self.stray_animal_mgr.get_animal_hazards(vehicle_transform)

        # Feature 2: Potholes
        pothole_reading = self.road_surface_sensor.scan_potholes_ahead(vehicle_transform, self.pothole_mgr.potholes)
        norm_pothole_dist = pothole_reading["pothole_dist_norm"]
        norm_pothole_offset = pothole_reading["pothole_offset_norm"]
        pothole_severity = pothole_reading["pothole_severity"]

        # Feature 3: Bike Cutting
        norm_bike_dist, norm_bike_lat_vel = self.bike_cutting_mgr.get_cutting_bike_observation(vehicle_transform)

        # Feature 4: Intent Prediction (Confidence-Based)
        nearby_threat_actors = []
        if self.bike_cutting_mgr.active_bike and self.bike_cutting_mgr.active_bike['actor']:
            nearby_threat_actors.append(self.bike_cutting_mgr.active_bike['actor'])
        for item in self.stray_animal_mgr.active_animals:
            if item.get('actor'):
                nearby_threat_actors.append(item['actor'])
        if self.emergency_vehicle_mgr.active_emergency and self.emergency_vehicle_mgr.active_emergency.get('actor'):
            nearby_threat_actors.append(self.emergency_vehicle_mgr.active_emergency['actor'])

        sim_timestamp = self.step_count * 0.05
        threat_intent, intent_conf, _ = self.intent_predictor.evaluate_highest_threat(
            nearby_threat_actors, vehicle_transform, sim_timestamp
        )

        # Feature 5: Emergency Vehicle Sensing (Sound-Based)
        em_transform, em_vel, siren_active = self.emergency_vehicle_mgr.get_emergency_state()
        acoustic_reading = self.acoustic_sensor.compute_received_signal(
            vehicle_transform, v, em_transform, em_vel, siren_active=siren_active
        )
        siren_sound_level = acoustic_reading["sound_level_norm"]
        siren_bearing = acoustic_reading["bearing_norm"]
        emergency_yield_req = 1.0 if acoustic_reading["is_approaching_from_behind"] else 0.0

        enhanced_obs = base_obs + [
            norm_animal_dist,       # 11: Stray animal distance [0, 1]
            norm_animal_lat,        # 12: Stray animal lateral offset [-1, 1]
            lane_confidence,        # 13: Faded line marking quality [0, 1]
            norm_pothole_dist,      # 14: Upcoming pothole distance [0, 1]
            norm_pothole_offset,    # 15: Upcoming pothole lateral offset [-1, 1]
            pothole_severity,       # 16: Pothole severity [0, 1]
            norm_bike_dist,         # 17: Bike distance [0, 1]
            norm_bike_lat_vel,      # 18: Bike cutting lateral velocity [-1, 1]
            threat_intent,          # 19: Threat intent class [-1, 1]
            intent_conf,            # 20: Intent prediction confidence [0, 1]
            siren_sound_level,      # 21: Siren sound level [0, 1]
            siren_bearing,          # 22: Siren bearing angle [-1, 1]
            emergency_yield_req     # 23: Emergency yield indicator [0, 1]
        ]

        return np.array(enhanced_obs, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.is_crashed = False
        self.last_steer = 0.0
        self.last_throttle_brake = 0.0
        self.smooth_steer = 0.0
        self.smooth_throttle_brake = 0.0
        self.last_cam_loc = None
        self.last_cam_yaw = None
        self.step_count = 0
        self.last_pothole_penalty = 0.0
        
        # Clean up scenario managers
        self.stray_animal_mgr.reset()
        self.bike_cutting_mgr.reset()
        self.emergency_vehicle_mgr.reset()
        self.intent_predictor.reset()

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
                if self.vehicle.is_alive:
                    self.vehicle.destroy()
            except Exception:
                pass
            self.vehicle = None

        if self.world:
            # Clean up only alive actors
            try:
                for a in self.world.get_actors():
                    if a.is_alive and (a.type_id.startswith('vehicle.') or a.type_id.startswith('walker.') or a.type_id.startswith('sensor.')):
                        try:
                            a.destroy()
                        except Exception:
                            pass
            except Exception:
                pass
            # Tick world once to finalize actor destructions before spawning new vehicle
            self.world.tick()

        if not self.map:
            obs_dim = 24 if self.observation_mode == "enhanced" else 11
            return np.zeros(obs_dim, dtype=np.float32), {}

        # Spawn vehicle using candidate spawn points with fallbacks
        spawn_points = self.map.get_spawn_points()
        if len(spawn_points) > 50:
            candidate_indices = [10, 11, 12, 15, 20, 25, 30]
            end_wp = spawn_points[45]
        elif len(spawn_points) > 1:
            candidate_indices = list(range(len(spawn_points) - 1))
            end_wp = spawn_points[-1]
        else:
            candidate_indices = [0]
            end_wp = spawn_points[0]

        self.vehicle = None
        start_wp = spawn_points[candidate_indices[0]]
        for idx in candidate_indices:
            pt = spawn_points[idx]
            self.vehicle = self.world.try_spawn_actor(self.vehicle_bp, pt)
            if self.vehicle is not None:
                start_wp = pt
                break

        if self.vehicle is None:
            raise RuntimeError("Failed to spawn vehicle: all candidate spawn points were blocked!")
            
        # Trace route
        if self.route_planner:
            self.gps_route = self.route_planner.trace_route(start_wp.location, end_wp.location)
        self.route_index = 0

        # Reset & procedurally generate potholes along route
        self.pothole_mgr.reset_and_generate(self.gps_route)

        # Attach Collision Sensor
        if self.vehicle and self.blueprint_library:
            collision_bp = self.blueprint_library.find('sensor.other.collision')
            self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
            self.collision_sensor.listen(lambda event: self._on_collision(event))

        # Position spectator chase camera behind car
        self.update_spectator()

        # Spin-up ticks to settle vehicle on suspension
        if self.world:
            for _ in range(10):
                self.world.tick()

        # Discard initial spawn drop ground impact
        self.is_crashed = False

        return self._get_obs(), {}

    def update_spectator(self):
        """Moves the spectator camera behind the ego vehicle using smooth LERP damping."""
        if not self.vehicle or not self.world:
            return
        try:
            spectator = self.world.get_spectator()
            tf = self.vehicle.get_transform()
            fwd = tf.get_forward_vector()
            target_loc = tf.location + carla.Location(z=2.8) - carla.Location(x=fwd.x * 6.5, y=fwd.y * 6.5, z=0.0)
            target_yaw = float(tf.rotation.yaw)

            if self.last_cam_loc is None:
                self.last_cam_loc = target_loc
                self.last_cam_yaw = target_yaw
            else:
                beta = 0.28
                self.last_cam_loc = carla.Location(
                    x=(1.0 - beta) * self.last_cam_loc.x + beta * target_loc.x,
                    y=(1.0 - beta) * self.last_cam_loc.y + beta * target_loc.y,
                    z=(1.0 - beta) * self.last_cam_loc.z + beta * target_loc.z
                )
                yaw_diff = math.atan2(
                    math.sin(math.radians(target_yaw - self.last_cam_yaw)),
                    math.cos(math.radians(target_yaw - self.last_cam_yaw))
                )
                self.last_cam_yaw = self.last_cam_yaw + beta * math.degrees(yaw_diff)

            cam_rot = carla.Rotation(pitch=-14.0, yaw=self.last_cam_yaw, roll=0.0)
            spectator.set_transform(carla.Transform(self.last_cam_loc, cam_rot))
        except Exception:
            pass

    def _on_collision(self, event):
        self.is_crashed = True
        try:
            other = event.other_actor.type_id if event.other_actor else 'Static/Environment'
            impulse = event.normal_impulse
            mag = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
            print(f"[Collision Alert] Hit {other} with impulse {mag:.1f}")
        except Exception:
            pass

    def step(self, action):
        self.step_count += 1
        raw_steer = float(action[0])
        raw_tb = float(action[1])
        
        # Smooth steering with exponential moving average to eliminate steering wheel jitter
        alpha_steer = 0.35 # Blend 35% new target, 65% previous angle
        self.smooth_steer = alpha_steer * raw_steer + (1.0 - alpha_steer) * self.smooth_steer
        
        # Smooth throttle/brake transitions
        alpha_tb = 0.45
        self.smooth_throttle_brake = alpha_tb * raw_tb + (1.0 - alpha_tb) * self.smooth_throttle_brake

        self.last_steer = self.smooth_steer
        self.last_throttle_brake = self.smooth_throttle_brake
        
        if carla:
            control = carla.VehicleControl()
            control.steer = float(np.clip(self.smooth_steer, -1.0, 1.0))
            
            if self.smooth_throttle_brake >= 0.0:
                control.throttle = float(np.clip(self.smooth_throttle_brake, 0.0, 1.0))
                control.brake = 0.0
            else:
                control.throttle = 0.0
                control.brake = float(np.clip(abs(self.smooth_throttle_brake), 0.0, 1.0))
                
            if self.vehicle:
                self.vehicle.apply_control(control)

        # Advance scenario hazards and sensors
        if self.vehicle:
            veh_transform = self.vehicle.get_transform()
            vel = self.vehicle.get_velocity()
            speed_mps = math.sqrt(vel.x**2 + vel.y**2)

            # 1. Update road surface condition
            self.road_surface_sensor.update_road_segment(self.step_count)

            # 2. Check pothole impact
            pothole_penalty, hit_pot = self.pothole_mgr.check_vehicle_impact(veh_transform, speed_mps)
            self.last_pothole_penalty = pothole_penalty

            # 3. Dynamic scenario spawns & updates
            if self.step_count % 60 == 0:
                self.stray_animal_mgr.spawn_stray_animal_near_route(veh_transform, self.map)
            self.stray_animal_mgr.update(veh_transform)

            self.bike_cutting_mgr.maybe_spawn_cutting_bike(veh_transform, speed_mps, self.map)
            self.bike_cutting_mgr.update(veh_transform, speed_mps)

            self.emergency_vehicle_mgr.maybe_spawn_emergency_vehicle(veh_transform, speed_mps, self.map)
            self.emergency_vehicle_mgr.update(veh_transform, speed_mps)
            
            # Follow vehicle with spectator chase camera
            self.update_spectator()
            
        if self.world:
            self.world.tick()
        
        obs = self._get_obs()
        extra_info = {
            "pothole_impact_penalty": self.last_pothole_penalty,
            "step": self.step_count
        }
        reward = calculate_taxi_reward(obs, self.is_crashed, extra_info=extra_info)
        
        terminated = self.is_crashed
        truncated = False
        
        # Route completion check
        if self.route_index >= len(self.gps_route) - 1 and len(self.gps_route) > 0:
            truncated = True
            
        if abs(self.current_lane_offset) > 4.5:
            truncated = True
            
        return obs, reward, terminated, truncated, extra_info

    def close(self):
        self.stray_animal_mgr.cleanup()
        self.bike_cutting_mgr.cleanup()
        self.emergency_vehicle_mgr.cleanup()
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