import os
import torch
import numpy as np
from stable_baselines3 import PPO
from torch.utils.data import DataLoader, TensorDataset
from src.environment import CarlaTaxiEnv
from src.utils import connect_to_carla

def clone_behavior(data_path="data/expert_data.npz", save_path="models/cloned_taxi_model", epochs=15, batch_size=64):
    print(f"Loading expert dataset from {data_path}...")
    dataset = np.load(data_path)
    expert_states = torch.tensor(dataset['states'], dtype=torch.float32)
    expert_actions = torch.tensor(dataset['actions'], dtype=torch.float32)
    
    # 1. Initialize CARLA to generate the required network architecture shapes
    print("Initiating handshake with CARLA to build network architecture...")
    client, world = connect_to_carla(port=2000)
    env = CarlaTaxiEnv(client, world)
    
    # 2. Initialize a blank PPO Model
    print("Initializing blank PPO Model...")
    model = PPO("MlpPolicy", env, verbose=1)
    
    # Move tensors to the correct device (GPU/CPU)
    device = model.device
    expert_states = expert_states.to(device)
    expert_actions = expert_actions.to(device)
    
    # 3. Create a PyTorch DataLoader for batching the data
    torch_dataset = TensorDataset(expert_states, expert_actions)
    dataloader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=True)
    
    # 4. Setup a PyTorch optimizer specifically targeting the policy network weights
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=1e-3)
    
    print(f"Starting Behavioral Cloning (Supervised Learning) for {epochs} epochs...")
    model.policy.train()
    
    # 5. The Training Loop
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_states, batch_actions in dataloader:
            # Evaluate actions to extract the log probabilities
            _, log_prob, _ = model.policy.evaluate_actions(batch_states, batch_actions)
            
            # The Behavioral Cloning objective: Maximize the likelihood of the expert's actions
            # In PyTorch, optimization minimizes the objective, so we use the negative log-likelihood
            loss = -log_prob.mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss (Negative Log-Likelihood): {epoch_loss/len(dataloader):.4f}")
        
    print(f"Cloning complete! Saving cloned brain to {save_path}.zip...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    env.close()

if __name__ == "__main__":
    try:
        clone_behavior()
    except KeyboardInterrupt:
        print("\nCloning stopped by user command.")