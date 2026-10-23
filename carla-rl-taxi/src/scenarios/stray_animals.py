import math
import random
import numpy as np

try:
    import carla
except ImportError:
    carla = None

class StrayAnimalManager:
    """
    Manages stray animal hazards in the CARLA environment.
    Simulates erratic behaviors such as roadside wander and sudden road-crossing/darting.
    """
    def __init__(self, world, client=None, config=None):
        self.world = world
        self.client = client
        self.config = config or {}
        
        self.enabled = self.config.get("enabled", True)
        self.max_animals = self.config.get("max_animals", 2)
        self.spawn_distance_min = self.config.get("spawn_distance_min", 25.0)
        self.spawn_distance_max = self.config.get("spawn_distance_max", 45.0)
        self.crossing_probability = self.config.get("crossing_probability", 0.6)
        self.wander_speed = self.config.get("wander_speed", 1.8)
        
        self.active_animals = [] # list of dicts: {'actor': carla.Actor, 'state': str, 'crossing_dir': float}
        self.blueprint_library = self.world.get_blueprint_library() if self.world else None

    def reset(self):
        """Clean up active animal actors."""
        self.cleanup()
        self.active_animals = []

    def cleanup(self):
        """Destroy spawned actors safely."""
        for item in self.active_animals:
            actor = item.get('actor')
            if actor is not None:
                try:
                    actor.destroy()
                except Exception:
                    pass
        self.active_animals = []

    def spawn_stray_animal_near_route(self, ego_transform, carla_map):
        """
        Spawns a stray animal ahead of the ego vehicle near the road shoulder.
        """
        if not self.enabled or self.world is None or not self.blueprint_library:
            return

        if len(self.active_animals) >= self.max_animals:
            return

        # Find walker blueprints
        walker_bps = list(self.blueprint_library.filter('walker.pedestrian.*'))
        if not walker_bps:
            return
        
        # Pick blueprint
        animal_bp = random.choice(walker_bps)

        ego_loc = ego_transform.location
        ego_yaw_rad = math.radians(ego_transform.rotation.yaw)
        
        # Spawn distance ahead along heading
        spawn_dist = random.uniform(self.spawn_distance_min, self.spawn_distance_max)
        
        # Spawn offset: roadside (3.0 to 5.0 meters left or right of lane center)
        side = 1.0 if random.random() > 0.5 else -1.0
        lateral_offset = side * random.uniform(3.0, 5.0)
        
        spawn_x = ego_loc.x + spawn_dist * math.cos(ego_yaw_rad) - lateral_offset * math.sin(ego_yaw_rad)
        spawn_y = ego_loc.y + spawn_dist * math.sin(ego_yaw_rad) + lateral_offset * math.cos(ego_yaw_rad)
        spawn_z = ego_loc.z + 0.5

        spawn_point = carla.Transform(
            carla.Location(x=spawn_x, y=spawn_y, z=spawn_z),
            carla.Rotation(yaw=math.degrees(ego_yaw_rad) - 90.0 * side)
        )

        actor = self.world.try_spawn_actor(animal_bp, spawn_point)
        if actor is not None:
            will_cross = (random.random() < self.crossing_probability)
            self.active_animals.append({
                'actor': actor,
                'state': 'ready_to_cross' if will_cross else 'wander',
                'crossing_dir': -side, # Towards road center
                'speed': self.wander_speed,
                'has_triggered': False
            })

    def update(self, ego_transform):
        """
        Updates animal movement each environment step.
        Triggers sudden crossing if ego gets within triggering range.
        """
        if not self.enabled or not self.active_animals:
            return

        ego_loc = ego_transform.location
        surviving = []

        for item in self.active_animals:
            actor = item['actor']
            if not actor.is_alive:
                continue

            actor_loc = actor.get_location()
            dx = actor_loc.x - ego_loc.x
            dy = actor_loc.y - ego_loc.y
            dist = math.sqrt(dx**2 + dy**2)

            # Cull animals that are far behind ego (> 20m behind)
            ego_yaw_rad = math.radians(ego_transform.rotation.yaw)
            fwd_proj = dx * math.cos(ego_yaw_rad) + dy * math.sin(ego_yaw_rad)
            if fwd_proj < -15.0 or dist > 80.0:
                try:
                    actor.destroy()
                except Exception:
                    pass
                continue

            # Behavior logic
            if item['state'] == 'ready_to_cross':
                # Trigger sudden dart across road when ego is 15-25m away
                if 8.0 < dist < 25.0 and fwd_proj > 0.0:
                    item['state'] = 'crossing'
                    item['has_triggered'] = True
                    speed = self.wander_speed * 1.6 # Dart speed
                else:
                    speed = 0.5 # Lingering roadside
            elif item['state'] == 'crossing':
                speed = self.wander_speed * 1.6
            else:
                # Stochastic wander
                speed = self.wander_speed * 0.8

            # Apply movement direction
            if carla:
                rot = actor.get_transform().rotation
                rad_yaw = math.radians(rot.yaw)
                control = carla.WalkerControl()
                control.direction = carla.Vector3D(
                    x=math.cos(rad_yaw),
                    y=math.sin(rad_yaw),
                    z=0.0
                )
                control.speed = float(speed)
                try:
                    actor.apply_control(control)
                except Exception:
                    pass

            surviving.append(item)

        self.active_animals = surviving

    def get_animal_hazards(self, ego_transform):
        """
        Perception query for the nearest stray animal ahead.
        Returns:
        - norm_animal_dist: float in [0.0, 1.0] (1.0 = none in forward envelope)
        - norm_animal_lateral: float in [-1.0, 1.0] (lateral offset relative to ego lane)
        - nearest_dist_m: float
        """
        if not self.active_animals:
            return 1.0, 0.0, 999.0

        ego_loc = ego_transform.location
        ego_yaw_rad = math.radians(ego_transform.rotation.yaw)
        cos_yaw = math.cos(ego_yaw_rad)
        sin_yaw = math.sin(ego_yaw_rad)

        min_dist = float('inf')
        lat_offset = 0.0

        for item in self.active_animals:
            actor = item['actor']
            if not actor.is_alive:
                continue
            loc = actor.get_location()
            dx = loc.x - ego_loc.x
            dy = loc.y - ego_loc.y
            
            longitudinal = dx * cos_yaw + dy * sin_yaw
            lateral = -dx * sin_yaw + dy * cos_yaw

            # Only consider actors in front of ego within 50m and within roadside zone (+/- 8m)
            if 0.5 < longitudinal < 50.0 and abs(lateral) < 8.0:
                dist = math.sqrt(longitudinal**2 + lateral**2)
                if dist < min_dist:
                    min_dist = dist
                    lat_offset = lateral

        if min_dist < 50.0:
            norm_dist = float(np.clip(min_dist / 50.0, 0.0, 1.0))
            norm_lat = float(np.clip(lat_offset / 4.0, -1.0, 1.0))
            return norm_dist, norm_lat, min_dist

        return 1.0, 0.0, 999.0
