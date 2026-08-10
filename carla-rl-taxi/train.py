import os
from src.utils import connect_to_carla
from src.environment import CarlaTaxiEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback # <-- ADD THIS

def configure_sync_mode(world):
    settings = world.get_settings()
    settings.synchronous_mode = True       
    settings.fixed_delta_seconds = 0.05     
    settings.substepping = True             
    settings.max_substep_delta_seconds = 0.01
    settings.max_substeps = 10
    world.apply_settings(settings)
    print("CARLA engine forced into strict synchronous physics mode.")

def main():
    model = None # Initialize handle for exception block visibility
    try:
        print("Initiating handshake with manually opened CARLA server...")
        client, world = connect_to_carla(port=2000)
        configure_sync_mode(world)
        env = CarlaTaxiEnv(client, world)
        
        cloned_model_path = "models/cloned_taxi_model.zip"
        if not os.path.exists(cloned_model_path):
            raise FileNotFoundError(f"Could not find the cloned model at {cloned_model_path}.")
            
        print(f"Loading pre-trained cloned brain from {cloned_model_path}...")
        model = PPO.load(cloned_model_path, env=env)
        model.learning_rate = 1e-4
        
        # --- THE INSURANCE POLICY FOR A 2 MILLION RUN ---
        # Saves progress automatically every 20,000 steps
        checkpoint_callback = CheckpointCallback(
            save_freq=20000, 
            save_path='./models/checkpoints/',
            name_prefix='taxi_rl_step'
        )
        
        total_timesteps = 2000000 # <-- Changed to your chosen 2 Million mark!
        print(f"Beginning RL Fine-Tuning optimization loop ({total_timesteps} steps)...")
        
        model.learn(total_timesteps=total_timesteps, tb_log_name="PPO_fine_tuned", callback=checkpoint_callback)
        
        os.makedirs("models", exist_ok=True)
        model.save("models/best_taxi_model")
        print("Fine-tuning complete! Final model saved safely as models/best_taxi_model.zip")
        
    except KeyboardInterrupt:
        print("\nFine-tuning interrupted by user command. Saving intermediate progress...")
        if model: model.save("models/best_taxi_model")
    except Exception as e:
        print(f"An unexpected error occurred during execution: {e}")
        if model:
            model.save("models/crash_backup_model")
            print("Emergency save written to models/crash_backup_model.zip")
    finally:
        print("Closed python script handles.")

if __name__ == "__main__":
    main()