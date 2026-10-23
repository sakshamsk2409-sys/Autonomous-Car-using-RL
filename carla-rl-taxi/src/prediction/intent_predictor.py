import math
import collections
import numpy as np

class ConfidenceIntentPredictor:
    """
    Confidence-Based Intent Predictor for surrounding dynamic obstacles
    (vehicles, two-wheelers, stray animals, pedestrians).
    
    Tracks kinematic history and outputs predicted maneuver class alongside 
    a calibrated Bayesian/softmax confidence score in [0.0, 1.0].
    """
    INTENT_MAP = {
        "NONE": 0.0,
        "LANE_FOLLOW": 0.0,
        "CUT_IN": 1.0,
        "CROSSING": 0.6,
        "DECELERATING": -0.6
    }

    def __init__(self, history_len=8, confidence_temperature=1.0):
        self.history_len = history_len
        self.temperature = confidence_temperature
        
        # actor_id -> deque of dicts: {'loc': (x,y,z), 'vel': (vx,vy,vz), 'yaw': rad, 'time': t}
        self.actor_tracks = collections.defaultdict(lambda: collections.deque(maxlen=self.history_len))

    def reset(self):
        """Clear tracking history."""
        self.actor_tracks.clear()

    def update_track(self, actor_id, location, velocity, yaw_deg, timestamp):
        """
        Updates the kinematic history for a given actor.
        """
        track = self.actor_tracks[actor_id]
        track.append({
            'x': location.x,
            'y': location.y,
            'z': getattr(location, 'z', 0.0),
            'vx': velocity.x,
            'vy': velocity.y,
            'vz': getattr(velocity, 'z', 0.0),
            'yaw': math.radians(yaw_deg),
            't': timestamp
        })

    def predict_intent(self, actor_id, ego_transform):
        """
        Predicts the intent and confidence score for a specific tracked actor.
        
        Returns:
        - intent_class: str ("CUT_IN", "CROSSING", "DECELERATING", "LANE_FOLLOW", "NONE")
        - confidence: float in [0.0, 1.0]
        - intent_score: float in [-1.0, 1.0]
        """
        track = self.actor_tracks.get(actor_id)
        if not track or len(track) < 3:
            return "NONE", 0.0, 0.0

        latest = track[-1]
        earliest = track[0]
        dt = max(latest['t'] - earliest['t'], 0.05)

        ego_loc = ego_transform.location
        ego_yaw = math.radians(ego_transform.rotation.yaw)
        cos_eyaw = math.cos(ego_yaw)
        sin_eyaw = math.sin(ego_yaw)

        # Vector from ego to actor
        dx = latest['x'] - ego_loc.x
        dy = latest['y'] - ego_loc.y
        dist = math.sqrt(dx**2 + dy**2)
        if dist > 50.0:
            return "NONE", 0.0, 0.0

        # Project into ego frame
        longitudinal = dx * cos_eyaw + dy * sin_eyaw
        lateral = -dx * sin_eyaw + dy * cos_eyaw

        # Relative velocity in ego frame
        rel_vx = latest['vx']
        rel_vy = latest['vy']
        v_lon = rel_vx * cos_eyaw + rel_vy * sin_eyaw
        v_lat = -rel_vx * sin_eyaw + rel_vy * cos_eyaw

        # Acceleration estimate
        a_lon = (track[-1]['vx'] - track[0]['vx']) / dt

        # Heading error relative to ego heading
        heading_diff = math.atan2(math.sin(latest['yaw'] - ego_yaw), math.cos(latest['yaw'] - ego_yaw))

        # -------------------------------------------------------------
        # Unnormalized Logit Evidence Functions
        # -------------------------------------------------------------
        # 1. CUT_IN evidence: Moving towards ego lateral centerline while ahead/alongside
        # Lateral velocity towards center: if lateral > 0 (right), v_lat < 0 (moving left towards 0)
        moving_toward_center = (lateral > 0.3 and v_lat < -0.3) or (lateral < -0.3 and v_lat > 0.3)
        cut_in_evidence = 0.0
        if moving_toward_center and -5.0 < longitudinal < 20.0:
            cut_in_evidence = 2.5 * abs(v_lat) + 1.5 * abs(math.sin(heading_diff))

        # 2. CROSSING evidence: High lateral speed across roadway with large heading disparity
        crossing_evidence = 0.0
        if abs(math.sin(heading_diff)) > 0.6 and abs(v_lat) > 0.8:
            crossing_evidence = 2.0 * abs(v_lat) + 2.0 * abs(math.sin(heading_diff))

        # 3. DECELERATING evidence: Negative longitudinal acceleration in forward path
        decel_evidence = 0.0
        if longitudinal > 2.0 and abs(lateral) < 2.0 and a_lon < -1.0:
            decel_evidence = 2.0 * abs(a_lon)

        # 4. LANE_FOLLOW evidence: Low lateral speed, parallel heading
        lane_follow_evidence = 1.0
        if abs(v_lat) < 0.4 and abs(heading_diff) < 0.25:
            lane_follow_evidence = 3.0 / (1.0 + abs(v_lat) + abs(heading_diff))

        logits = np.array([lane_follow_evidence, cut_in_evidence, crossing_evidence, decel_evidence], dtype=np.float32)
        classes = ["LANE_FOLLOW", "CUT_IN", "CROSSING", "DECELERATING"]

        # Softmax computation
        scaled_logits = logits / max(self.temperature, 0.1)
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        probs = exp_logits / np.sum(exp_logits)

        best_idx = int(np.argmax(probs))
        predicted_class = classes[best_idx]
        raw_confidence = float(probs[best_idx])

        # Distance decay on confidence (further actors have higher sensor uncertainty)
        dist_factor = float(np.exp(-dist / 35.0))
        calibrated_confidence = float(np.clip(raw_confidence * dist_factor, 0.0, 1.0))

        intent_score = self.INTENT_MAP.get(predicted_class, 0.0)

        return predicted_class, calibrated_confidence, intent_score

    def evaluate_highest_threat(self, actors_list, ego_transform, timestamp):
        """
        Evaluates all dynamic obstacles and identifies the one posing the highest threat.
        
        Returns:
        - threat_intent_norm: float in [-1.0, 1.0]
        - intent_confidence: float in [0.0, 1.0]
        - highest_threat_info: dict
        """
        if not actors_list:
            return 0.0, 0.0, {"class": "NONE", "confidence": 0.0, "actor_id": None}

        highest_threat_score = 0.0
        best_intent_norm = 0.0
        best_confidence = 0.0
        best_info = {"class": "NONE", "confidence": 0.0, "actor_id": None}

        # Threat weights for prioritization: Cut-in and Crossing pose immediate collision hazards
        threat_priority = {
            "CUT_IN": 3.0,
            "CROSSING": 2.5,
            "DECELERATING": 2.0,
            "LANE_FOLLOW": 0.5,
            "NONE": 0.0
        }

        for actor in actors_list:
            if not actor.is_alive:
                continue
            
            loc = actor.get_location()
            vel = actor.get_velocity()
            yaw = actor.get_transform().rotation.yaw
            self.update_track(actor.id, loc, vel, yaw, timestamp)

            p_class, conf, score = self.predict_intent(actor.id, ego_transform)
            priority = threat_priority.get(p_class, 0.0)
            threat_level = priority * conf

            if threat_level > highest_threat_score:
                highest_threat_score = threat_level
                best_intent_norm = score
                best_confidence = conf
                best_info = {
                    "class": p_class,
                    "confidence": conf,
                    "actor_id": actor.id
                }

        return float(best_intent_norm), float(best_confidence), best_info
