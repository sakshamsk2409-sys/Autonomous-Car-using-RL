import os
import torch
import torch.nn.functional as F
import numpy as np
from stable_baselines3 import PPO
from torch.utils.data import DataLoader, TensorDataset
from src.environment import CarlaTaxiEnv
from src.utils import connect_to_carla

def clone_behavior(data_path="data/expert_data.npz", save_path="models/cloned_taxi_model", epochs=20, batch_size=64, lr=7e-4):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Expert dataset not found at '{data_path}'. Please run collect_data.py first!")

    print(f"Loading expert dataset from {data_path}...")
    dataset = np.load(data_path)
    raw_states = dataset['states']
    raw_actions = dataset['actions']
    
    # 1. Initialize CARLA to construct the exact policy network dimensions
    print("Initiating handshake with CARLA to build policy network architecture...")
    client, world = connect_to_carla(port=2000)
    env = CarlaTaxiEnv(client, world)
    
    env_dim = env.observation_space.shape[0]
    data_dim = raw_states.shape[1]
    
    # Zero-pad or slice if dimension mismatch exists
    if data_dim < env_dim:
        print(f"Adapting {data_dim}-dim dataset to {env_dim}-dim observation space (zero-padding auxiliary features)...")
        pad_width = ((0, 0), (0, env_dim - data_dim))
        states_adapted = np.pad(raw_states, pad_width, mode='constant', constant_values=0.0)
    elif data_dim > env_dim:
        print(f"Truncating {data_dim}-dim dataset to match {env_dim}-dim observation space...")
        states_adapted = raw_states[:, :env_dim]
    else:
        states_adapted = raw_states

    expert_states = torch.tensor(states_adapted, dtype=torch.float32)
    expert_actions = torch.tensor(raw_actions, dtype=torch.float32)

    # 2. Instantiate blank PPO model
    print(f"Instantiating PPO policy with {env_dim}-dim input space...")
    model = PPO("MlpPolicy", env, verbose=0)
    device = model.device
    
    expert_states = expert_states.to(device)
    expert_actions = expert_actions.to(device)
    
    # 3. Create PyTorch DataLoader
    torch_dataset = TensorDataset(expert_states, expert_actions)
    dataloader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=True)
    
    # 4. Adam optimizer on policy parameters
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=lr)
    
    print(f"Starting Behavioral Cloning (Direct Action Regression) for {epochs} epochs on {len(expert_states)} samples...")
    model.policy.train()
    
    # 5. Supervised MSE Regression Training Loop
    # Directly fits policy action output to demonstrated PID expert control actions
    for epoch in range(epochs):
        epoch_loss = 0.0
        steer_loss_total = 0.0
        tb_loss_total = 0.0
        
        for batch_states, batch_actions in dataloader:
            # Predict action distribution from policy
            distribution = model.policy.get_distribution(batch_states)
            pred_actions = distribution.mode()  # Deterministic action output
            
            # Weighted MSE loss: higher emphasis on steering accuracy to prevent lane departure
            loss_steer = F.mse_loss(pred_actions[:, 0], batch_actions[:, 0])
            loss_tb = F.mse_loss(pred_actions[:, 1], batch_actions[:, 1])
            loss = 2.0 * loss_steer + 1.0 * loss_tb
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            steer_loss_total += loss_steer.item()
            tb_loss_total += loss_tb.item()
            
        num_batches = len(dataloader)
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Total Loss: {epoch_loss/num_batches:.5f} | Steer MSE: {steer_loss_total/num_batches:.5f} | Throttle/Brake MSE: {tb_loss_total/num_batches:.5f}")
        
    print(f"Supervised cloning complete! Saving brain to {save_path}.zip...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    env.close()
    print("Policy successfully saved and ready for evaluation or RL fine-tuning!")

if __name__ == "__main__":
    try:
        clone_behavior()
    except KeyboardInterrupt:
        print("\nCloning stopped by user command.")