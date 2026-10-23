import math
import random
import numpy as np

try:
    import carla
except ImportError:
    carla = None

class PotholeManager:
    """
    Procedurally places potholes along the planned route and calculates
    vehicle collision / suspension impact dynamics when driven over.
    """
    def __init__(self, world=None, config=None):
        self.world = world
        self.config = config or {}
        
        self.enabled = self.config.get("enabled", True)
        self.count_per_km = self.config.get("count_per_km", 8)
        self.radius_min = self.config.get("radius_min", 0.4)
        self.radius_max = self.config.get("radius_max", 1.1)
        self.severity_min = self.config.get("severity_min", 0.3)
        self.severity_max = self.config.get("severity_max", 1.0)
        self.impact_penalty_scale = self.config.get("impact_penalty_scale", 15.0)

        self.potholes = [] # list of dicts: {'x', 'y', 'z', 'radius', 'severity', 'hit': bool}

    def reset_and_generate(self, route_waypoints):
        """
        Generates potholes along the given GPS route waypoints.
        """
        self.potholes = []
        if not self.enabled or not route_waypoints or len(route_waypoints) < 10:
            return

        total_route_len = len(route_waypoints) * 2.0 # Approx 2m step size
        num_potholes = max(1, int((total_route_len / 1000.0) * self.count_per_km))
        
        # Don't place potholes within the first 15 waypoints (spawn area) or last 5
        valid_indices = list(range(15, len(route_waypoints) - 5))
        if len(valid_indices) < num_potholes:
            chosen_indices = valid_indices
        else:
            chosen_indices = sorted(random.sample(valid_indices, num_potholes))

        for idx in chosen_indices:
            wp, _ = route_waypoints[idx]
            loc = wp.transform.location
            forward = wp.transform.get_forward_vector()
            # Perpendicular right vector in road plane
            right_x = -forward.y
            right_y = forward.x

            # Place within the drivable lane width (+/- 1.3 meters from lane center)
            lat_jitter = random.uniform(-1.3, 1.3)
            pot_x = loc.x + right_x * lat_jitter
            pot_y = loc.y + right_y * lat_jitter
            pot_z = loc.z

            radius = random.uniform(self.radius_min, self.radius_max)
            severity = random.uniform(self.severity_min, self.severity_max)

            self.potholes.append({
                'x': pot_x,
                'y': pot_y,
                'z': pot_z,
                'radius': radius,
                'severity': severity,
                'hit': False
            })

        self.visualize_potholes()

    def visualize_potholes(self):
        """Draws visual markers in the CARLA simulator so potholes can be seen."""
        if not self.world or not carla or not hasattr(self.world, 'debug'):
            return

        for pot in self.potholes:
            loc = carla.Location(x=pot['x'], y=pot['y'], z=pot['z'] + 0.1)
            # Red/orange crater marker
            self.world.debug.draw_point(
                loc,
                size=0.18,
                color=carla.Color(r=200, g=60, b=10),
                life_time=600.0
            )
            # Ring or text indicator
            self.world.debug.draw_string(
                loc + carla.Location(z=0.3),
                f"POTHOLE ({pot['severity']:.1f})",
                draw_shadow=False,
                color=carla.Color(r=255, g=100, b=0),
                life_time=600.0
            )

    def check_vehicle_impact(self, vehicle_transform, speed_mps):
        """
        Checks if the vehicle has driven over any pothole in this step.
        Returns:
        - impact_penalty: float >= 0.0 (penalty to apply in reward)
        - pothole_hit: dict or None
        """
        if not self.enabled or not self.potholes:
            return 0.0, None

        veh_loc = vehicle_transform.location
        impact_penalty = 0.0
        hit_pot = None

        for pot in self.potholes:
            if pot['hit']:
                continue

            dx = veh_loc.x - pot['x']
            dy = veh_loc.y - pot['y']
            dist = math.sqrt(dx**2 + dy**2)

            # Check overlap with vehicle footprint (approx 1.2m wheel radius + pothole radius)
            if dist < (pot['radius'] + 1.0):
                pot['hit'] = True
                hit_pot = pot
                # Higher speed and deeper pothole = higher chassis impact
                impact_energy = (speed_mps / 20.0) * pot['severity']
                impact_penalty = self.impact_penalty_scale * impact_energy
                break

        return float(impact_penalty), hit_pot
