import math
import random
import numpy as np

try:
    import carla
except ImportError:
    carla = None

class BikeCuttingManager:
    """
    Simulates aggressive two-wheelers (motorcycles, scooters, bicycles) overtaking
    from blind spots/adjacent lanes and sharply cutting across the ego vehicle's path.
    """
    def __init__(self, world, client=None, config=None):
        self.world = world
        self.client = client
        self.config = config or {}

        self.enabled = self.config.get("enabled", True)
        self.spawn_interval_steps = self.config.get("spawn_interval_steps", 120)
        self.speed_ratio_to_ego = self.config.get("speed_ratio_to_ego", 1.35)
        self.overtake_start_distance = self.config.get("overtake_start_distance", -15.0)
        self.cut_in_distance_ahead = self.config.get("cut_in_distance_ahead", 7.0)
        self.cut_in_lateral_speed = self.config.get("cut_in_lateral_speed", 2.2)

        self.active_bike = None # dict: {'actor': carla.Actor, 'state': str, 'side': float, 'lateral_offset': float}
        self.steps_since_last_spawn = 0
        self.blueprint_library = self.world.get_blueprint_library() if self.world else None

    def reset(self):
        """Clean up active bikes."""
        self.cleanup()
        self.steps_since_last_spawn = 0

    def cleanup(self):
        if self.active_bike and self.active_bike.get('actor'):
            try:
                self.active_bike['actor'].destroy()
            except Exception:
                pass
        self.active_bike = None

    def _get_bike_blueprint(self):
        if not self.blueprint_library:
            return None
        # Two wheeler blueprints
        bps = (
            list(self.blueprint_library.filter('vehicle.yamaha.*')) +
            list(self.blueprint_library.filter('vehicle.kawasaki.*')) +
            list(self.blueprint_library.filter('vehicle.harley-davidson.*')) +
            list(self.blueprint_library.filter('vehicle.bh.*')) +
            list(self.blueprint_library.filter('vehicle.gazelle.*')) +
            list(self.blueprint_library.filter('vehicle.diamondback.*'))
        )
        if not bps:
            # Fallback to general vehicle if no specific 2-wheeler is loaded
            bps = list(self.blueprint_library.filter('vehicle.*'))
        return random.choice(bps) if bps else None

    def maybe_spawn_cutting_bike(self, ego_transform, ego_velocity_mps, carla_map):
        """
        Periodically spawns a bike behind or adjacent to ego in an overtaking position.
        """
        if not self.enabled or self.world is None:
            return

        if self.active_bike is not None:
            return

        self.steps_since_last_spawn += 1
        if self.steps_since_last_spawn < self.spawn_interval_steps:
            return

        bp = self._get_bike_blueprint()
        if not bp:
            return

        ego_loc = ego_transform.location
        ego_yaw_rad = math.radians(ego_transform.rotation.yaw)
        
        # Choose side: left (+1) or right (-1)
        side = 1.0 if random.random() > 0.5 else -1.0
        lateral_offset = side * 2.8 # Adjacent lane/shoulder offset
        behind_dist = self.overtake_start_distance

        spawn_x = ego_loc.x + behind_dist * math.cos(ego_yaw_rad) - lateral_offset * math.sin(ego_yaw_rad)
        spawn_y = ego_loc.y + behind_dist * math.sin(ego_yaw_rad) + lateral_offset * math.cos(ego_yaw_rad)
        spawn_z = ego_loc.z + 0.3

        spawn_transform = carla.Transform(
            carla.Location(x=spawn_x, y=spawn_y, z=spawn_z),
            carla.Rotation(yaw=math.degrees(ego_yaw_rad))
        )

        actor = self.world.try_spawn_actor(bp, spawn_transform)
        if actor is not None:
            self.steps_since_last_spawn = 0
            target_speed = max(ego_velocity_mps * self.speed_ratio_to_ego, 7.0)
            self.active_bike = {
                'actor': actor,
                'state': 'OVERTAKING', # 'OVERTAKING' -> 'CUTTING_IN' -> 'AHEAD'
                'side': side,
                'lateral_offset': lateral_offset,
                'target_speed': target_speed,
                'cut_in_progress': 0.0
            }

    def update(self, ego_transform, ego_velocity_mps):
        """
        Updates bike movement each environment step: overtakes, then cuts in directly ahead of ego.
        """
        if not self.enabled or self.active_bike is None:
            return

        actor = self.active_bike['actor']
        if not actor.is_alive:
            self.active_bike = None
            return

        ego_loc = ego_transform.location
        ego_yaw_rad = math.radians(ego_transform.rotation.yaw)
        cos_yaw = math.cos(ego_yaw_rad)
        sin_yaw = math.sin(ego_yaw_rad)

        bike_loc = actor.get_location()
        dx = bike_loc.x - ego_loc.x
        dy = bike_loc.y - ego_loc.y

        longitudinal = dx * cos_yaw + dy * sin_yaw
        lateral = -dx * sin_yaw + dy * cos_yaw

        # State transition logic
        state = self.active_bike['state']
        side = self.active_bike['side']
        target_speed = self.active_bike['target_speed']

        if state == 'OVERTAKING':
            # Check if bike reached the forward cut-in threshold (e.g. 5-8m ahead of ego)
            if longitudinal >= self.cut_in_distance_ahead:
                self.active_bike['state'] = 'CUTTING_IN'
        elif state == 'CUTTING_IN':
            # Transition laterally toward center line (lateral = 0)
            self.active_bike['cut_in_progress'] += 0.05
            if abs(lateral) < 0.4 or self.active_bike['cut_in_progress'] >= 1.0:
                self.active_bike['state'] = 'AHEAD'

        # Apply controls to the bike
        if carla:
            control = carla.VehicleControl()
            # Longitudinal throttle
            control.throttle = 0.8
            control.brake = 0.0
            
            # Steering toward cut-in target
            if state == 'CUTTING_IN':
                # Steer towards ego's lane center
                steer_val = -0.35 * side # Steer opposite to side to cut across
                control.steer = float(np.clip(steer_val, -0.6, 0.6))
            elif state == 'AHEAD':
                # Align straight with road
                control.steer = 0.0
            else:
                # Keep parallel
                control.steer = 0.0
                
            try:
                actor.apply_control(control)
            except Exception:
                pass

        # Cleanup if bike is far ahead (> 40m) or far behind (< -30m)
        if longitudinal > 40.0 or longitudinal < -35.0:
            try:
                actor.destroy()
            except Exception:
                pass
            self.active_bike = None

    def get_cutting_bike_observation(self, ego_transform):
        """
        Extracts observations regarding the cutting bike:
        - norm_bike_dist: float in [0.0, 1.0] (1.0 = none or safe distance)
        - norm_bike_lat_vel: float in [-1.0, 1.0] (lateral velocity towards ego lane center)
        """
        if not self.active_bike or not self.active_bike['actor'].is_alive:
            return 1.0, 0.0

        actor = self.active_bike['actor']
        ego_loc = ego_transform.location
        ego_yaw_rad = math.radians(ego_transform.rotation.yaw)
        cos_yaw = math.cos(ego_yaw_rad)
        sin_yaw = math.sin(ego_yaw_rad)

        bike_loc = actor.get_location()
        dx = bike_loc.x - ego_loc.x
        dy = bike_loc.y - ego_loc.y
        dist = math.sqrt(dx**2 + dy**2)

        norm_dist = float(np.clip(dist / 40.0, 0.0, 1.0))

        # Lateral velocity projected onto ego's lateral axis
        vel = actor.get_velocity()
        # lateral axis is [-sin(yaw), cos(yaw)]
        lat_vel = -vel.x * sin_yaw + vel.y * cos_yaw
        norm_lat_vel = float(np.clip(lat_vel / 5.0, -1.0, 1.0))

        return norm_dist, norm_lat_vel
