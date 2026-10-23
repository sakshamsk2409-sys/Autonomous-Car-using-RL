import os
import torch
from src.utils import connect_to_carla
from src.environment import CarlaTaxiEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

def configure_sync_mode(world):
    settings = world.get_settings()
    settings.synchronous_mode = True       
    settings.fixed_delta_seconds = 0.05     
    settings.substepping = True             
    settings.max_substep_delta_seconds = 0.01
    settings.max_substeps = 10
    world.apply_settings(settings)
    print("CARLA engine forced into strict synchronous physics mode.")

def load_or_transfer_model(model_path, env):
    """
    Loads pre-trained model into the environment.
    If the pre-trained model was trained on legacy 11-D observations and the current
    env uses enhanced 24-D observations, performs surgical weight transfer so the
    agent retains previous driving skills while learning new safety features.
    """
    try:
        model = PPO.load(model_path, env=env)
        print("Model loaded directly without dimension mismatch.")
        return model
    except ValueError as e:
        print(f"Detected observation space transition ({e}). Initiating warm-start weight transfer...")
        # Load old model without env binding
        old_model = PPO.load(model_path)
        new_model = PPO("MlpPolicy", env, verbose=1)

        old_state_dict = old_model.policy.state_dict()
        new_state_dict = new_model.policy.state_dict()

        for key in new_state_dict:
            if key in old_state_dict:
                if new_state_dict[key].shape == old_state_dict[key].shape:
                    new_state_dict[key] = old_state_dict[key].clone()
                elif len(new_state_dict[key].shape) == 2 and new_state_dict[key].shape[0] == old_state_dict[key].shape[0]:
                    # Input projection layer weight matrix
                    old_in = old_state_dict[key].shape[1]
                    new_state_dict[key][:, :old_in] = old_state_dict[key].clone()
                    new_state_dict[key][:, old_in:] = 0.0 # Zero-init new feature connections
                    print(f"Transferred and expanded weights for layer: {key}")

        new_model.policy.load_state_dict(new_state_dict)
        print("Warm-start weight transfer complete! Retained prior policy behavior.")
        return new_model

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    model = None
    try:
        print("Initiating handshake with manually opened CARLA server...")
        client, world = connect_to_carla(port=2000)
        configure_sync_mode(world)
        env = CarlaTaxiEnv(client, world)
        
        # Check for model path robustly relative to script dir
        candidate_models = [
            os.path.join(SCRIPT_DIR, "models", "cloned_taxi_model.zip"),
            os.path.join(SCRIPT_DIR, "models", "best_taxi_model.zip"),
            os.path.join(SCRIPT_DIR, "models", "crash_backup_model.zip"),
            "models/cloned_taxi_model.zip",
            "models/best_taxi_model.zip"
        ]
        cloned_model_path = next((p for p in candidate_models if os.path.exists(p)), None)
        if not cloned_model_path:
            raise FileNotFoundError(f"Could not find any base model in '{os.path.join(SCRIPT_DIR, 'models')}'. Run clone_behavior.py first.")
            
        print(f"Loading/Adapting pre-trained brain from {cloned_model_path}...")
        model = load_or_transfer_model(cloned_model_path, env)
        model.learning_rate = 1e-4
        
        # Saves progress automatically every 2,000 steps so intermediate checkpoints are immediately available
        checkpoint_callback = CheckpointCallback(
            save_freq=2000, 
            save_path='./models/checkpoints/',
            name_prefix='taxi_rl_step'
        )
        
        total_timesteps = 50000
        print(f"Beginning RL Fine-Tuning optimization loop ({total_timesteps} steps)...")
        
        model.learn(total_timesteps=total_timesteps, tb_log_name="PPO_fine_tuned", callback=checkpoint_callback)
        
        save_dir = os.path.join(SCRIPT_DIR, "models")
        os.makedirs(save_dir, exist_ok=True)
        final_save_path = os.path.join(save_dir, "best_taxi_model")
        model.save(final_save_path)
        print(f"Fine-tuning complete! Final model saved safely as {final_save_path}.zip")
        
    except KeyboardInterrupt:
        print("\nFine-tuning interrupted by user command. Saving intermediate progress...")
        if model:
            save_dir = os.path.join(SCRIPT_DIR, "models")
            os.makedirs(save_dir, exist_ok=True)
            model.save(os.path.join(save_dir, "best_taxi_model"))
    except Exception as e:
        print(f"An unexpected error occurred during execution: {e}")
        if model:
            save_dir = os.path.join(SCRIPT_DIR, "models")
            os.makedirs(save_dir, exist_ok=True)
            backup_path = os.path.join(save_dir, "crash_backup_model")
            model.save(backup_path)
            print(f"Emergency save written to {backup_path}.zip")
    finally:
        if 'world' in locals() and world is not None:
            try:
                settings = world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                world.apply_settings(settings)
                print("Restored asynchronous mode on CARLA world.")
            except Exception:
                pass
        print("Closed python script handles.")

if __name__ == "__main__":
    main()