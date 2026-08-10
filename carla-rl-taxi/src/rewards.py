import numpy as np

def calculate_taxi_reward(obs, is_crashed):
    if is_crashed:
        return -100.0

    speed = obs[0]
    lane_offset = obs[1]
    z_diff = obs[2]
    heading_error = obs[3]
    obstacle_dist = obs[4]
    traffic_light = obs[10]

    # Progression Reward
    speed_reward = speed

    # Penalties for misalignment and surface instability
    lane_penalty = -2.0 * abs(lane_offset)
    heading_penalty = -2.0 * abs(heading_error)
    bump_penalty = -1.0 * abs(z_diff)

    # Traffic Light Violation Penalty (Penalize forward speed on red light)
    traffic_penalty = 0.0
    if traffic_light == -1.0 and speed > 0.05:
        traffic_penalty = -10.0 * speed

    # Proximity Emergency Penalty
    obstacle_penalty = 0.0
    if obstacle_dist < 0.2: # Distance under 10 meters
        obstacle_penalty = -5.0 * (0.2 - obstacle_dist)

    total_reward = (
        speed_reward
        + lane_penalty
        + heading_penalty
        + bump_penalty
        + traffic_penalty
        + obstacle_penalty
    )

    if speed < 0.05:
        total_reward -= 0.02

    return float(total_reward)