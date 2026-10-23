import numpy as np
import math

def calculate_taxi_reward(obs, is_crashed, extra_info=None):
    """
    Multi-objective reward function supporting both legacy 11-feature observations
    and enhanced 24-feature observations with:
    1. Stray animal collision prevention & proactive braking
    2. Pothole chassis impact penalty & proactive avoidance
    3. Faded line marking adaptive lane keeping stability
    4. Bike cutting headway defense
    5. Emergency vehicle siren detection & yielding logic
    """
    if is_crashed:
        return -100.0

    extra_info = extra_info or {}

    speed = float(obs[0])
    lane_offset = float(obs[1])
    z_diff = float(obs[2])
    heading_error = float(obs[3])
    obstacle_dist = float(obs[4])
    traffic_light = float(obs[10])

    # 1. Progression along route (Forward speed projected onto route alignment)
    # Rewards moving forward when pointing correctly; penalizes speed when facing backwards/perpendicular
    alignment = math.cos(heading_error * math.pi)
    route_speed_reward = speed * max(-0.2, alignment)
    heading_bonus = 1.0 * max(-1.0, alignment)

    # 2. Gaussian Lane Centering Reward
    # If yielding to an emergency vehicle or avoiding an active pothole, relax centering to allow lateral swerving
    is_swerving_or_yielding = False
    if len(obs) >= 24:
        pothole_dist = float(obs[14])
        siren_sound = float(obs[21])
        emergency_yield_req = float(obs[23])
        if (pothole_dist < 0.65 and float(obs[16]) > 0.2) or emergency_yield_req > 0.5 or siren_sound > 0.4:
            is_swerving_or_yielding = True

    if is_swerving_or_yielding:
        centering_reward = 0.5 * math.exp(-1.0 * (lane_offset ** 2))
        lane_penalty = -1.0 * abs(lane_offset)
    else:
        centering_reward = 2.0 * math.exp(-4.0 * (lane_offset ** 2))
        lane_penalty = -3.0 * abs(lane_offset)

    if abs(lane_offset) > 1.4:
        lane_penalty -= 8.0 * (abs(lane_offset) - 1.4)  # Steep dropoff near road borders

    bump_penalty = -1.0 * abs(z_diff)

    # 3. Traffic Light Violation Penalty
    traffic_penalty = 0.0
    if traffic_light == -1.0 and speed > 0.05:
        traffic_penalty = -10.0 * speed

    # 4. Proximity Frontal Obstacle Penalty
    obstacle_penalty = 0.0
    if obstacle_dist < 0.2:  # Distance under 10 meters
        obstacle_penalty = -5.0 * (0.2 - obstacle_dist)

    # -------------------------------------------------------------
    # Enhanced Features (if present in observation vector >= 24)
    # -------------------------------------------------------------
    animal_penalty = 0.0
    pothole_penalty = float(extra_info.get("pothole_impact_penalty", 0.0))
    bike_defense_penalty = 0.0
    emergency_reward_penalty = 0.0

    if len(obs) >= 24:
        animal_dist = float(obs[11])
        animal_lateral = float(obs[12])
        lane_confidence = float(obs[13])
        pothole_dist = float(obs[14])
        pothole_offset = float(obs[15])
        pothole_sev = float(obs[16])
        bike_dist = float(obs[17])
        bike_lat_vel = float(obs[18])
        threat_intent = float(obs[19])
        intent_conf = float(obs[20])
        siren_sound = float(obs[21])
        siren_bearing = float(obs[22])
        emergency_yield_req = float(obs[23])

        # Feature 1: Stray Animal Preemptive Braking
        # If animal is in frontal collision cone (< 20m, lateral offset < 1.5m)
        if animal_dist < 0.4 and abs(animal_lateral) < 0.4:
            # Penalize forward speed proportional to proximity
            animal_penalty = 6.0 * speed * (0.4 - animal_dist)

        # Feature 2: Faded Lane Lines & Potholes
        # When lane markings are faded, relax strict lane penalty to reduce jitter
        if lane_confidence < 0.5:
            lane_penalty *= (0.5 + 0.5 * lane_confidence)

        # Approaching pothole preview:
        # If a pothole is ahead (< 20m, pothole_dist < 0.65) and aligned with the vehicle path (|pothole_offset| < 0.4)
        if pothole_dist < 0.65 and pothole_sev > 0.2:
            lat_pothole_gap = abs(pothole_offset)
            if lat_pothole_gap < 0.4:
                # Penalize steering straight into the pothole crater
                pothole_penalty += 8.0 * (0.4 - lat_pothole_gap) * (0.65 - pothole_dist) * (0.5 + speed)
            else:
                # Reward successful lateral avoidance swerve
                pothole_penalty -= 1.5 * (lat_pothole_gap - 0.4)

        # Feature 3 & 4: Bike Cutting & Intent Prediction
        # If dynamic threat is CUT_IN with high confidence or bike is close with cutting lateral velocity
        is_cutting_threat = (threat_intent > 0.5 and intent_conf > 0.5) or (bike_dist < 0.35 and abs(bike_lat_vel) > 0.2)
        if is_cutting_threat and speed > 0.2:
            # Penalize failing to create safe following gap for cutting bike
            bike_defense_penalty = -4.0 * (0.35 - min(bike_dist, 0.35)) * speed

        # Feature 5: Emergency Vehicle Yielding
        if emergency_yield_req > 0.5 or (siren_sound > 0.4 and abs(siren_bearing) > 0.4):
            # Emergency siren approaching from behind!
            # Reward ego for pulling to roadside (lane_offset > 0.2) and slowing down
            if lane_offset > 0.2 and speed < 0.3:
                emergency_reward_penalty = +2.5 # Positive yield reward!
            elif speed > 0.3 and abs(lane_offset) < 0.2:
                # Obstructing emergency vehicle path
                emergency_reward_penalty = -8.0 * speed

    total_reward = (
        route_speed_reward
        + centering_reward
        + heading_bonus
        + lane_penalty
        + bump_penalty
        + traffic_penalty
        + obstacle_penalty
        - animal_penalty
        - pothole_penalty
        + bike_defense_penalty
        + emergency_reward_penalty
    )

    if speed < 0.05:
        total_reward -= 0.5  # Modest penalty for remaining motionless on road

    return float(total_reward)