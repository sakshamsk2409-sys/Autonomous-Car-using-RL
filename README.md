# Autonomous Car using RL

An end-to-end autonomous driving pipeline developed in the CARLA Simulator that solves horizon blindness, steering jitter, and covariate shift by combining Supervised Behavioral Cloning (BC) warm-starting with Proximal Policy Optimization (PPO) Reinforcement Learning.

---

## 🌟 Advanced Real-World Driving Features

The environment is equipped with 5 safety-critical features tailored for complex, unstructured driving environments:

### 1. 🐕 Stray Animals (`src/scenarios/stray_animals.py`)
- **Stochastic Verge Wandering & Road Crossing**: Spawns animal/pedestrian actors along roadside shoulders and road edges.
- **Sudden Dart Behavior**: Animals graze roadside until ego approaches within a critical time-to-collision window ($15-25$m), triggering sudden perpendicular road crossing.
- **Perception & Rewards**: Frontal perception extracts distance to animal and lateral offset relative to the lane center. Collisions incur heavy penalties ($-100$), while proactive deceleration when animals enter the corridor is rewarded.

### 2. 🕳️ Faded Line Markings & Potholes (`src/sensors/road_surface_sensor.py`, `src/scenarios/potholes.py`)
- **Faded Lane Line Detection**: Simulates vision degradation on worn or unpainted road segments. On faded patches, line confidence drops ($\gamma < 0.5$) and stochastic Gaussian observation noise is added to the perceived lane offset, incentivizing the agent to rely on path curvature and inertial stability rather than erratic line-chasing.
- **Pothole Ground Profiling & Suspension Impact**: Potholes are procedurally seeded along route waypoints. The ground-surface profiler scans for upcoming potholes (distance, lateral offset within lane, and depth severity). Rolling over a pothole applies a speed-dependent chassis impact penalty:
  $$\text{Penalty} = -k_{pot} \cdot \left(\frac{v_{ego}}{20}\right) \cdot \text{severity}$$
  encouraging the car to slow down or navigate around the crater.

### 3. 🏍️ Bike Cutting (`src/scenarios/bike_cutting.py`)
- **Aggressive Two-Wheeler Overtaking**: High-agility two-wheelers (motorcycles, scooters, bicycles) spawn in adjoining lanes or rear blind spots.
- **Sharp Cut-In Maneuvers**: The bike overtakes at higher longitudinal speed ($v_{bike} \approx 1.35 \cdot v_{ego}$) and cuts sharply in front of the ego vehicle's bumper ($5-8$m ahead).
- **Headway Defense**: Ego perceives the bike's distance and lateral cut-in velocity, incentivizing defensive gap creation and braking.

### 4. 🧠 Intent Prediction (Confidence-Based) (`src/prediction/intent_predictor.py`)
- **Multi-Actor Kinematic History**: Tracks surrounding dynamic obstacles across a sliding history window.
- **Maneuver Classification**: Evaluates kinematic evidence for 4 maneuver classes:
  - `CUT_IN` (high lateral approach velocity towards centerline)
  - `CROSSING` (perpendicular speed across road)
  - `DECELERATING` (rapid longitudinal braking)
  - `LANE_FOLLOW` (parallel heading and low lateral drift)
- **Bayesian Confidence Calibration**: Softmax probabilities modulated by distance-decay reliability factors produce calibrated confidence scores $C \in [0.0, 1.0]$. The highest-threat maneuver and confidence are directly fed into the RL state.

### 5. 🚑 Emergency Vehicle Sensing (Sound-Based) (`src/sensors/acoustic_sensor.py`, `src/scenarios/emergency_vehicles.py`)
- **Directional Siren Acoustic Array**: Simulates an on-board acoustic microphone array detecting emergency vehicle sirens ($f_0 \approx 900\text{ Hz}$, $L_0 \approx 120\text{ dB}$).
- **Physical Wave Decay & Doppler Shift**:
  - Inverse-Square sound decay with atmospheric attenuation:
    $$L_{rec} = L_0 - 20\log_{10}(d) - \alpha d$$
  - Doppler frequency shift based on relative approach velocity:
    $$f_{obs} = f_0 \left(\frac{c}{c - v_{rel}}\right)$$
  - Direction of arrival (bearing $\theta_{siren} \in [-\pi, \pi]$ relative to ego heading).
- **Yielding Behavior**: When a siren approaches from behind, the agent receives a $+2.5$ yielding bonus for pulling towards the shoulder and decelerating, and a severe penalty ($-8.0 \cdot v_{ego}$) for obstructing the emergency vehicle.

---

## 📊 24-Dimensional State Observation Vector

| Index | Feature | Range | Description |
|:---:|:---|:---:|:---|
| 0 | `norm_speed` | $[0, 1]$ | Forward vehicle speed normalized by 20 m/s |
| 1 | `norm_lane_offset` | $[-1, 1]$ | Perceived lateral distance from lane center (incorporates faded noise) |
| 2 | `norm_z_diff` | $[-1, 1]$ | Elevation difference to target waypoint |
| 3 | `norm_heading` | $[-1, 1]$ | Heading error relative to target waypoint |
| 4 | `norm_obstacle_dist` | $[0, 1]$ | Frontal vehicle raycast distance |
| 5 | `norm_yaw_rate` | $[-1, 1]$ | Yaw angular rate in rad/s |
| 6 | `norm_last_steer` | $[-1, 1]$ | Previous steering command |
| 7 | `norm_last_tb` | $[-1, 1]$ | Previous throttle/brake command |
| 8 | `norm_curvature` | $[-1, 1]$ | Lookahead waypoint path curvature |
| 9 | `norm_dist_next` | $[0, 1]$ | Distance to current GPS waypoint |
| 10 | `norm_traffic_light` | $[-1, 1]$ | Signal state (-1 Red, 0 Yellow, 1 Green) |
| **11** | `norm_animal_dist` | $[0, 1]$ | **Distance to nearest stray animal ahead** |
| **12** | `norm_animal_lat` | $[-1, 1]$ | **Lateral offset of stray animal relative to lane center** |
| **13** | `lane_confidence` | $[0, 1]$ | **Lane marking visibility confidence (faded vs crisp)** |
| **14** | `norm_pothole_dist` | $[0, 1]$ | **Distance to upcoming pothole ahead** |
| **15** | `norm_pothole_offset` | $[-1, 1]$ | **Lateral offset of pothole within lane** |
| **16** | `pothole_severity` | $[0, 1]$ | **Depth and roughness severity of pothole** |
| **17** | `norm_bike_dist` | $[0, 1]$ | **Distance to nearest aggressive two-wheeler** |
| **18** | `norm_bike_lat_vel` | $[-1, 1]$ | **Lateral cut-in velocity of the bike** |
| **19** | `threat_intent` | $[-1, 1]$ | **Predicted maneuver class of highest-threat obstacle** |
| **20** | `intent_conf` | $[0, 1]$ | **Calibrated confidence score of intent prediction** |
| **21** | `siren_sound_level` | $[0, 1]$ | **Perceived acoustic siren sound level** |
| **22** | `siren_bearing` | $[-1, 1]$ | **Angle of arrival of emergency siren sound** |
| **23** | `emergency_yield_req`| $[0, 1]$ | **Indicator flag that emergency yielding is required** |

> **Backward Compatibility**: To run in legacy 11-dimension mode, set `environment.observation_mode: "legacy"` in `config/settings.yaml`. The training pipeline also includes warm-start weight transfer to adapt pre-trained 11-D models directly to 24-D networks without loss of prior skills.

---

## 🛠️ Usage & Workflow

### 1. Run Unit Tests
Validate all sensors, scenario managers, intent predictions, and reward terms:
```powershell
python -m unittest tests/test_new_features.py
```

### 2. Collect Expert Data
Collect autopilot driving telemetry across Town10HD with all 5 active hazard scenarios:
```powershell
python collect_data.py
```

### 3. Supervised Behavioral Cloning (Warm-Start)
Clone policy weights into the PPO architecture:
```powershell
python clone_behavior.py
```

### 4. Reinforcement Learning Fine-Tuning
Train the policy network using PPO with multi-objective rewards:
```powershell
python train.py
```

### 5. Evaluation & Telemetry Logging
Evaluate trained agents and log telemetry to CSV:
```powershell
python enjoy.py
```

---

## 📄 Research Paper Reproduction: Conditional Imitation Learning (CIL)

This repository includes a reproducible implementation of the landmark autonomous driving paper:
> **"End-to-end Driving via Conditional Imitation Learning"**  
> *Felipe Codevilla, Matthias Müller, Antonio López, Vladlen Koltun, Alexey Dosovitskiy* (CoRL 2017 / ICRA 2018)  
> [arXiv:1710.02410](https://arxiv.org/abs/1710.02410)

### 🧩 Core Contribution & Architecture
Standard behavioral cloning collapses at intersections due to multi-modal path averaging (e.g. human demonstrations turning left and right average to driving straight into obstacles). CIL solves this by conditioning on discrete high-level navigational commands ($c \in \{\text{FOLLOW\_LANE}, \text{TURN\_LEFT}, \text{TURN\_RIGHT}, \text{GO\_STRAIGHT}\}$):
- **Shared Representation Trunk**: Computes shared geometric features from vehicle kinematics and sensors.
- **Specialized Command Branches**: 4 discrete action heads dynamically switched by the active navigational command.
- **Covariate Shift Perturbation Augmentation**: Injected state perturbations with proportional counter-steering to defend against trajectory drift.

### 📊 Benchmark Reproduction Results (`logs/cil/reproduction_benchmark.json`)

| Model Architecture | Command Conditioning | Best Val Loss ($L_1$) | Steering MAE | Primary Failure Mode Prevented |
| :--- | :--- | :---: | :---: | :--- |
| **Ablation Baseline** | Concatenated One-Hot Vector | $0.0523$ | $0.0050$ | Branch interference, steering hesitation |
| **CIL Branched (Reproduced)** | Dynamic Specialized Heads | $\mathbf{0.0527}$ | $\mathbf{0.0049}$ | Multi-modal path averaging at intersections |

#### Per-Branch Diagnostics:
- `FOLLOW_LANE` (Branch 0): Steer MAE $0.0036$ | Throttle/Brake MAE $0.0456$
- `TURN_RIGHT` (Branch 2): Steer MAE $0.0130$ | Throttle/Brake MAE $0.0542$
- `GO_STRAIGHT` (Branch 3): Steer MAE $0.0035$ | Throttle/Brake MAE $0.0448$

### 🚀 Running CIL Training & Evaluation

1. **Train CIL Branched & Baseline Models**:
   ```powershell
   python carla-rl-taxi/train_cil.py --data carla-rl-taxi/data/expert_data.npz --epochs 20
   ```
2. **Evaluate in Closed-Loop CARLA**:
   ```powershell
   python carla-rl-taxi/enjoy_cil.py
   ```

---

## 📚 Academic References & Citations

1. **End-to-End Driving via Conditional Imitation Learning**  
   *Codevilla, F., Müller, M., López, A., Koltun, V., & Dosovitskiy, A.* (CoRL 2017 / ICRA 2018). [arXiv:1710.02410](https://arxiv.org/abs/1710.02410)
2. **Exploring the Limitations of Behavior Cloning for Autonomous Driving**  
   *Codevilla, F., Santana, E., López, A. M., & Gaidon, A.* (ICCV 2019). [arXiv:1904.08980](https://arxiv.org/abs/1904.08980)
3. **ChauffeurNet: Learning to Drive by Imitating the Best and Synthesizing the Worst**  
   *Bansal, M., Krizhevsky, A., & Ogale, A.* (RSS 2019 / Waymo). [Paper Link](https://waymo.com/research/chauffeurnet-learning-to-drive-by-imitating-the-best-and-synthesizing-the-worst/)
4. **Learning to Drive in a Day**  
   *Kendall, A., Hawke, J., Janz, D., et al.* (ICRA 2019 / Wayve AI). [arXiv:1807.00412](https://arxiv.org/abs/1807.00412)
5. **CARLA: An Open Urban Driving Simulator**  
   *Dosovitskiy, A., Ros, G., Codevilla, F., López, A., & Koltun, V.* (CoRL 2017). [arXiv:1711.03938](https://arxiv.org/abs/1711.03938)

