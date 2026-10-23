import math
import numpy as np

class RoadSurfaceSensor:
    """
    Sensor for detecting road surface conditions:
    1. Upcoming potholes (distance, lateral offset within lane, and severity).
    2. Faded lane line marking detection confidence & simulated vision noise.
    """
    def __init__(
        self,
        lookahead_distance_max=30.0,
        faded_line_probability=0.35,
        min_line_confidence=0.15,
        faded_noise_sigma=0.45
    ):
        self.lookahead_distance_max = lookahead_distance_max
        self.faded_line_probability = faded_line_probability
        self.min_line_confidence = min_line_confidence
        self.faded_noise_sigma = faded_noise_sigma
        
        # State tracking for continuous faded road patches
        self.current_patch_is_faded = False
        self.patch_remaining_steps = 0
        self.smoothed_noise = 0.0
        self.current_confidence = 1.0

    def update_road_segment(self, step_count):
        """
        Simulates entering/exiting road segments with worn or faded paint.
        Maintains continuity for 50-150 steps per segment.
        """
        if self.patch_remaining_steps <= 0:
            self.current_patch_is_faded = (np.random.rand() < self.faded_line_probability)
            self.patch_remaining_steps = np.random.randint(50, 150)
            self.current_confidence = float(np.random.uniform(self.min_line_confidence, 0.45)) if self.current_patch_is_faded else 1.0
        else:
            self.patch_remaining_steps -= 1

    def assess_lane_marking(self, true_lane_offset):
        """
        Assesses the visibility of lane markings with temporally coherent sensor uncertainty.
        
        Returns:
        - confidence: float in [0.0, 1.0] (1.0 = crisp lines, 0.15 = faded/eroded)
        - perceived_lane_offset: float (true offset corrupted by smooth sensor drift when faded)
        """
        if self.current_patch_is_faded:
            if self.current_confidence >= 0.5:
                self.current_confidence = float(np.random.uniform(self.min_line_confidence, 0.45))
            # Smooth low-pass filtered drift rather than high-frequency white noise chatter
            raw_noise = float(np.random.normal(0.0, self.faded_noise_sigma * (1.0 - self.current_confidence)))
            self.smoothed_noise = 0.85 * self.smoothed_noise + 0.15 * raw_noise
            confidence = self.current_confidence
            perceived_lane_offset = true_lane_offset + self.smoothed_noise
        else:
            self.current_confidence = float(np.random.uniform(0.92, 1.0))
            self.smoothed_noise *= 0.8
            confidence = self.current_confidence
            perceived_lane_offset = true_lane_offset + self.smoothed_noise
            
        return confidence, perceived_lane_offset

    def scan_potholes_ahead(self, ego_transform, potholes_list):
        """
        Scans upcoming potholes in the forward path envelope.
        
        Parameters:
        - ego_transform: carla.Transform of ego vehicle
        - potholes_list: list of dicts [{'x': float, 'y': float, 'radius': float, 'severity': float}]
        
        Returns dict:
        - pothole_dist_norm: float in [0.0, 1.0] (1.0 = none within lookahead)
        - pothole_offset_norm: float in [-1.0, 1.0] (lateral offset relative to ego heading)
        - pothole_severity: float in [0.0, 1.0] (depth/roughness)
        - closest_pothole: dict or None
        """
        if not potholes_list:
            return {
                "pothole_dist_norm": 1.0,
                "pothole_offset_norm": 0.0,
                "pothole_severity": 0.0,
                "closest_pothole": None
            }

        ego_loc = ego_transform.location
        ego_yaw_rad = math.radians(ego_transform.rotation.yaw)
        cos_yaw = math.cos(ego_yaw_rad)
        sin_yaw = math.sin(ego_yaw_rad)

        closest_dist = float('inf')
        closest_pothole = None
        closest_lat_offset = 0.0

        for pot in potholes_list:
            dx = pot['x'] - ego_loc.x
            dy = pot['y'] - ego_loc.y
            
            # Project into ego coordinate frame:
            # forward: longitudinal distance along heading
            # lateral: cross-track offset (positive = right, negative = left)
            longitudinal = dx * cos_yaw + dy * sin_yaw
            lateral = -dx * sin_yaw + dy * cos_yaw

            # Only look ahead within lookahead horizon and within roadway width envelope (+/- 4.0m)
            if 0.5 < longitudinal < self.lookahead_distance_max and abs(lateral) < 4.0:
                dist = math.sqrt(longitudinal**2 + lateral**2)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_pothole = pot
                    closest_lat_offset = lateral

        if closest_pothole is not None:
            dist_norm = float(np.clip(closest_dist / self.lookahead_distance_max, 0.0, 1.0))
            offset_norm = float(np.clip(closest_lat_offset / 3.0, -1.0, 1.0))
            severity = float(np.clip(closest_pothole['severity'], 0.0, 1.0))
            return {
                "pothole_dist_norm": dist_norm,
                "pothole_offset_norm": offset_norm,
                "pothole_severity": severity,
                "closest_pothole": closest_pothole
            }

        return {
            "pothole_dist_norm": 1.0,
            "pothole_offset_norm": 0.0,
            "pothole_severity": 0.0,
            "closest_pothole": None
        }
