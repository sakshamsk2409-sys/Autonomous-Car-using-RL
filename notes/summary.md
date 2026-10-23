# Research Paper Summary & Reproduction Plan

## Paper Reference
- **Title**: End-to-end Driving via Conditional Imitation Learning (CIL)
- **Authors**: Felipe Codevilla, Matthias Müller, Antonio López, Vladlen Koltun, Alexey Dosovitskiy
- **Conference / Venue**: ICRA 2018 / CoRL 2017
- **arXiv Link**: [arXiv:1710.02410](https://arxiv.org/abs/1710.02410)

---

## 1. Core Understanding (The 4 Questions)

### Q1: What problem is the paper solving?
Standard end-to-end imitation learning (behavioral cloning) maps sensory input directly to driving controls $f(I) \to a$. However, at decision points (such as road intersections), human demonstrations branch multi-modally: some drivers turn left, others turn right or proceed straight. Standard imitation learning networks suffer from **catastrophic path averaging** (e.g., averaging a left turn and a right turn results in driving straight into a barrier or divider) and cannot be instructed where to navigate. 

The paper solves this by **conditioning the policy on a high-level navigational command** ($c \in \{\text{Follow Lane}, \text{Turn Left}, \text{Turn Right}, \text{Go Straight}\}$).

### Q2: What dataset(s) does it use?
- Recorded inside the open-source **CARLA Simulator** across dynamic urban layouts.
- **Training Environment**: Town01 (under training weather conditions, e.g., Clear Noon, Heavy Rain).
- **Testing Environment**: Town01 (same town, unseen weather) and Town02 (unseen town, unseen test conditions) to assess generalization and domain transfer.
- Inputs captured: RGB camera image, current vehicle measurements (forward speed), and the active navigation command. Ground-truth controls: steering angle, throttle, and brake.

### Q3: What model architecture does it propose?
The **Branched Conditional Network (CIL)**:
1. **Perception Backbone**:
   - 8 Convolutional layers with kernel sizes $5\times 5$ (1st layer) and $3\times 3$ (subsequent 7 layers), strides of 2 on alternating layers for spatial downsampling, Batch Normalization, and ReLU activations.
   - Followed by 2 Fully Connected layers (512 units each) with Dropout ($p=0.5$).
2. **Measurement Stream**:
   - Vehicle forward speed processed via FC layers ($128 \to 128 \to 512$).
3. **Joint State Fusion**:
   - Concatenation of image latent features and speed latent features ($512 + 512 = 1024$ or joint embedding).
4. **Specialist Action Branches (The Core Innovation)**:
   - Instead of feeding the discrete command as a flat one-hot input vector to a single controller head (which suffers from branch interference), the joint embedding branches into **4 separate sub-networks** corresponding to:
     - Branch 0: `Follow Lane` / Lane Keeping
     - Branch 1: `Turn Left`
     - Branch 2: `Turn Right`
     - Branch 3: `Go Straight`
   - During the forward pass, the active command $c$ acts as a computational switch, routing gradients and predicting continuous control action vectors: $\mathbf{a} = [\text{steer}, \text{throttle}, \text{brake}]$.

### Q4: How is performance evaluated?
- **Offline / Validation Loss**: Mean Absolute Error ($L_1$) and Mean Squared Error ($L_2$) between predicted actions and expert human actions on hold-out sequences.
- **Online Closed-Loop CARLA Benchmark**:
  - Success Rate (% of missions completed without collision or leaving lane corridor).
  - Infraction rate (collisions with pedestrians, vehicles, static layout).
  - Distance traveled before critical intervention.
  - Generalization across 4 benchmark suites: *Straight*, *One Turn*, *Navigation (Town01)*, and *Navigation in Unseen Town (Town02)*.

---

## 2. Implementation Tasks Checklist (Ordered Easiest to Hardest)

- [ ] **Task 1: Command Enum & Observation Representation** (Easiest)
  - Define high-level command definitions (`LANE_FOLLOW=0`, `TURN_LEFT=1`, `TURN_RIGHT=2`, `GO_STRAIGHT=3`).
  - Extend the environment observation pipeline to emit command tokens.
- [ ] **Task 2: PyTorch Branched CIL Network Architecture**
  - Implement `ConditionalImitationModel` featuring shared representation + 4 specialized command branches.
  - Implement single-head command-input baseline for comparative ablation.
- [ ] **Task 3: Command-Gated Loss Function & Training Loop**
  - Implement selective branch loss computation: only backpropagate through branch $b == c$.
  - Implement L1 / L2 loss with steer weighting.
- [ ] **Task 4: Dataset Loader & Synthetic Recovery Augmentation**
  - Batch data loader with command index masking.
  - Data collection script in CARLA recording command flags and vehicle trajectory.
- [ ] **Task 5: Offline Model Validation & Metric Logging**
  - Measure per-branch validation loss, steering error, and throttle/brake accuracy.
- [ ] **Task 6: Online Closed-Loop CARLA Evaluation Runner**
  - Execute live closed-loop runs in CARLA Town01 & Town02.
  - Compute completion rate and collision metrics comparing CIL vs standard unconditioned imitation learning.
- [ ] **Task 7: Meaningful Extension (CIL + Reinforcement Learning Hybrid)** (Hardest)
  - Bridge the trained CIL branch weights into the existing PPO policy network in `Autonomous-Car-using-RL`, demonstrating policy improvement and obstacle avoidance (stray animals / cut-ins).
