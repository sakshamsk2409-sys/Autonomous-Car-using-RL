import math
import numpy as np

class AcousticSirenSensor:
    """
    Simulated directional acoustic sensor / microphone array for detecting 
    emergency vehicle sirens (sound-based sensing).
    
    Models:
    - Sound pressure level decay with distance (Inverse-Square Law with atmospheric damping)
    - Directional Angle-of-Arrival (bearing estimation via simulated beamforming)
    - Doppler frequency shift based on relative velocity
    - Signal-to-Noise Ratio (SNR) against ambient urban road noise
    """
    def __init__(
        self,
        base_sound_level_db=120.0,
        reference_frequency_hz=900.0,
        ambient_noise_db=65.0,
        speed_of_sound_mps=343.0,
        distance_decay_alpha=0.008,
        detection_threshold_snr=5.0
    ):
        self.base_sound_level_db = base_sound_level_db
        self.ref_freq_hz = reference_frequency_hz
        self.ambient_noise_db = ambient_noise_db
        self.c = speed_of_sound_mps
        self.alpha = distance_decay_alpha
        self.detection_threshold_snr = detection_threshold_snr
        
        self.last_sound_level = 0.0
        self.sound_derivative = 0.0

    def compute_received_signal(self, ego_transform, ego_velocity, emergency_transform, emergency_velocity, siren_active=True):
        """
        Calculates the acoustic reading at the ego vehicle given an emergency vehicle source.
        
        Parameters:
        - ego_transform: carla.Transform or object with .location (x, y, z) and .rotation (yaw in degrees)
        - ego_velocity: carla.Vector3D or object with (x, y, z)
        - emergency_transform: carla.Transform of emergency vehicle
        - emergency_velocity: carla.Vector3D of emergency vehicle
        - siren_active: bool, whether the siren is emitting sound
        
        Returns dict:
        - sound_level_norm: float in [0.0, 1.0] (normalized intensity)
        - bearing_norm: float in [-1.0, 1.0] (relative angle from -pi to +pi)
        - doppler_shift_hz: float, frequency shift in Hz
        - snr_db: float, signal-to-noise ratio in dB
        - is_siren_detected: bool, whether SNR exceeds detection threshold
        - is_approaching_from_behind: bool, whether siren is behind ego and closing in
        """
        if not siren_active or emergency_transform is None:
            self.sound_derivative = -self.last_sound_level
            self.last_sound_level = 0.0
            return {
                "sound_level_norm": 0.0,
                "bearing_norm": 0.0,
                "doppler_shift_hz": 0.0,
                "snr_db": -20.0,
                "is_siren_detected": False,
                "is_approaching_from_behind": False
            }

        ego_loc = ego_transform.location
        em_loc = emergency_transform.location
        
        dx = em_loc.x - ego_loc.x
        dy = em_loc.y - ego_loc.y
        dz = getattr(em_loc, 'z', 0.0) - getattr(ego_loc, 'z', 0.0)
        distance = math.sqrt(dx**2 + dy**2 + dz**2)
        distance = max(distance, 1.0) # Avoid singularity at d=0

        # 1. Received Sound Pressure Level (dB)
        # Free-field spherical spreading: L = L0 - 20*log10(d) - alpha*d
        received_db = self.base_sound_level_db - 20.0 * math.log10(distance) - self.alpha * distance
        snr_db = received_db - self.ambient_noise_db
        is_detected = snr_db >= self.detection_threshold_snr

        # Normalized sound intensity between 0.0 (ambient) and 1.0 (ear-splitting proximity)
        sound_level_norm = float(np.clip((received_db - self.ambient_noise_db) / (self.base_sound_level_db - self.ambient_noise_db), 0.0, 1.0))

        # Track rate of change (approach rate)
        self.sound_derivative = sound_level_norm - self.last_sound_level
        self.last_sound_level = sound_level_norm

        # 2. Relative Bearing (Angle of Arrival)
        # Vector from ego to emergency vehicle
        ego_yaw_rad = math.radians(ego_transform.rotation.yaw)
        source_angle_world = math.atan2(dy, dx)
        relative_bearing = math.atan2(math.sin(source_angle_world - ego_yaw_rad), math.cos(source_angle_world - ego_yaw_rad))
        bearing_norm = float(np.clip(relative_bearing / math.pi, -1.0, 1.0))

        # 3. Doppler Frequency Shift
        # Relative velocity projected along line of sight
        unit_los_x = dx / distance
        unit_los_y = dy / distance
        unit_los_z = dz / distance

        rel_vx = getattr(emergency_velocity, 'x', 0.0) - getattr(ego_velocity, 'x', 0.0)
        rel_vy = getattr(emergency_velocity, 'y', 0.0) - getattr(ego_velocity, 'y', 0.0)
        rel_vz = getattr(emergency_velocity, 'z', 0.0) - getattr(ego_velocity, 'z', 0.0)

        # Velocity towards ego (closing speed > 0 means getting closer)
        v_closing = -(rel_vx * unit_los_x + rel_vy * unit_los_y + rel_vz * unit_los_z)
        
        # Doppler formula: f_obs = f0 * (c / (c - v_source_rel))
        denom = max(self.c - v_closing, 10.0)
        observed_freq = self.ref_freq_hz * (self.c / denom)
        doppler_shift = observed_freq - self.ref_freq_hz

        # 4. Check if approaching from rear flank (|bearing| > 90 deg / 0.5 norm)
        is_approaching_from_behind = (
            is_detected and 
            abs(bearing_norm) > 0.45 and # In rear hemisphere
            (v_closing > 0.5 or self.sound_derivative >= 0.0) # Gaining ground
        )

        return {
            "sound_level_norm": sound_level_norm if is_detected else 0.0,
            "bearing_norm": bearing_norm if is_detected else 0.0,
            "doppler_shift_hz": float(doppler_shift),
            "snr_db": float(snr_db),
            "is_siren_detected": is_detected,
            "is_approaching_from_behind": is_approaching_from_behind
        }
