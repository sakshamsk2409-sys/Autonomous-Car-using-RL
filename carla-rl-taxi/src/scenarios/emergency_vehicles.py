import math
import random
import numpy as np

try:
    import carla
except ImportError:
    carla = None

class EmergencyVehicleManager:
    """
    Manages emergency vehicles (police, ambulance) approaching with active sirens,
    allowing the acoustic sensor to model directional sound and evaluate yielding.
    """
    def __init__(self, world, client=None, config=None):
        self.world = world
        self.client = client
        self.config = config or {}

        self.enabled = self.config.get("enabled", True)
        self.spawn_interval_steps = self.config.get("spawn_interval_steps", 250)
        self.speed_ratio_to_ego = self.config.get("speed_ratio_to_ego", 1.5)
        self.spawn_distance_behind = self.config.get("spawn_distance_behind", -55.0)

        self.active_emergency = None # dict: {'actor', 'siren_active', 'state', 'target_speed'}
        self.steps_since_last_spawn = 0
        self.blueprint_library = self.world.get_blueprint_library() if self.world else None

    def reset(self):
        """Clean up active emergency vehicles."""
        self.cleanup()
        self.steps_since_last_spawn = 0

    def cleanup(self):
        if self.active_emergency and self.active_emergency.get('actor'):
            try:
                self.active_emergency['actor'].destroy()
            except Exception:
                pass
        self.active_emergency = None

    def _get_emergency_blueprint(self):
        if not self.blueprint_library:
            return None
        # Police / Ambulance blueprints
        bps = (
            list(self.blueprint_library.filter('vehicle.dodge.charger_police*')) +
            list(self.blueprint_library.filter('vehicle.*police*')) +
            list(self.blueprint_library.filter('vehicle.*ambulance*')) +
            list(self.blueprint_library.filter('vehicle.*firetruck*'))
        )
        if not bps:
            bps = list(self.blueprint_library.filter('vehicle.tesla.*')) or list(self.blueprint_library.filter('vehicle.*'))
        return random.choice(bps) if bps else None

    def maybe_spawn_emergency_vehicle(self, ego_transform, ego_velocity_mps, carla_map):
        """
        Spawns an emergency vehicle behind the ego vehicle that approaches with active sirens.
        """
        if not self.enabled or self.world is None:
            return

        if self.active_emergency is not None:
            return

        self.steps_since_last_spawn += 1
        if self.steps_since_last_spawn < self.spawn_interval_steps:
            return

        bp = self._get_emergency_blueprint()
        if not bp:
            return

        ego_loc = ego_transform.location
        ego_yaw_rad = math.radians(ego_transform.rotation.yaw)
        
        # Position behind ego in adjacent overtaking lane/corridor (1.8m to 2.2m offset)
        behind_dist = self.spawn_distance_behind
        lateral_side = random.choice([-1.0, 1.0])
        lat_spawn_offset = lateral_side * 2.0
        
        spawn_x = ego_loc.x + behind_dist * math.cos(ego_yaw_rad) - lat_spawn_offset * math.sin(ego_yaw_rad)
        spawn_y = ego_loc.y + behind_dist * math.sin(ego_yaw_rad) + lat_spawn_offset * math.cos(ego_yaw_rad)
        spawn_z = ego_loc.z + 0.3

        spawn_transform = carla.Transform(
            carla.Location(x=spawn_x, y=spawn_y, z=spawn_z),
            carla.Rotation(yaw=math.degrees(ego_yaw_rad))
        )

        actor = self.world.try_spawn_actor(bp, spawn_transform)
        if actor is not None:
            self.steps_since_last_spawn = 0
            
            # Enable emergency siren lights if supported
            if carla and hasattr(carla, 'VehicleLightState'):
                try:
                    actor.set_light_state(carla.VehicleLightState(
                        carla.VehicleLightState.Special1 | carla.VehicleLightState.Special2
                    ))
                except Exception:
                    pass

            target_speed = max(ego_velocity_mps * self.speed_ratio_to_ego, 14.0)
            self.active_emergency = {
                'actor': actor,
                'siren_active': True,
                'target_speed': target_speed,
                'lateral_offset': lat_spawn_offset,
                'state': 'APPROACHING'
            }

    def update(self, ego_transform, ego_velocity_mps):
        """
        Advances the emergency vehicle forward to overtake ego.
        """
        if not self.enabled or self.active_emergency is None:
            return

        actor = self.active_emergency['actor']
        if not actor.is_alive:
            self.active_emergency = None
            return

        ego_loc = ego_transform.location
        ego_yaw_rad = math.radians(ego_transform.rotation.yaw)
        cos_yaw = math.cos(ego_yaw_rad)
        sin_yaw = math.sin(ego_yaw_rad)

        em_loc = actor.get_location()
        dx = em_loc.x - ego_loc.x
        dy = em_loc.y - ego_loc.y
        longitudinal = dx * cos_yaw + dy * sin_yaw
        lateral = -dx * sin_yaw + dy * cos_yaw

        if carla:
            # Calculate lateral and longitudinal offset relative to ego orientation
            lat_offset = self.active_emergency.get('lateral_offset', 1.8)
            current_target_speed = self.active_emergency.get('target_speed', 16.0)

            # Check if emergency vehicle is close behind (between -35m and -3m)
            # If so, it actively steers into the adjacent overtaking corridor and brakes if obstructed
            steer_cmd = 0.0
            throttle_cmd = 0.85
            brake_cmd = 0.0

            if longitudinal < 0.0: # Behind ego
                # Steering adjustment: steer slightly right/left towards overtaking corridor
                # cross-track error to overtaking lateral offset
                lateral_error = lat_offset - lateral
                steer_cmd = float(np.clip(0.35 * lateral_error, -0.4, 0.4))
                
                # Proximity braking: if closing within 8.5 meters and directly behind ego (|lateral| < 1.4m),
                # brake hard to avoid rear-ending the ego vehicle!
                if longitudinal > -8.5 and abs(lateral) < 1.4:
                    throttle_cmd = 0.0
                    brake_cmd = 0.85
                elif longitudinal > -18.0 and abs(lateral) < 1.0:
                    throttle_cmd = 0.3
                    brake_cmd = 0.2
            else: # Passed ego
                # Merge smoothly back ahead or continue straight
                throttle_cmd = 0.95
                steer_cmd = float(np.clip(-0.15 * lateral, -0.2, 0.2))

            control = carla.VehicleControl()
            control.throttle = throttle_cmd
            control.brake = brake_cmd
            control.steer = steer_cmd
            try:
                actor.apply_control(control)
            except Exception:
                pass

        # Cleanup if emergency vehicle is well past ego (> 60m ahead)
        if longitudinal > 60.0 or longitudinal < -80.0:
            try:
                actor.destroy()
            except Exception:
                pass
            self.active_emergency = None

    def get_emergency_state(self):
        """
        Returns actor, velocity, and siren active state for acoustic sensor evaluation.
        """
        if not self.active_emergency or not self.active_emergency['actor'].is_alive:
            return None, None, False
        actor = self.active_emergency['actor']
        return actor.get_transform(), actor.get_velocity(), self.active_emergency['siren_active']
