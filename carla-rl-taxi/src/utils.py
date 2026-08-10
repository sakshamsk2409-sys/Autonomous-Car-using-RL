import os
import subprocess
import time
import carla

def launch_carla_server(carla_path, port=2000):
    """
    Launches the CARLA simulator executable in low-rendering mode 
    to preserve resources on your graphics card.
    """
    executable = os.path.join(carla_path, "CarlaUE4.exe")
    if not os.path.exists(executable):
        raise FileNotFoundError(f"Could not locate CARLA executable at {executable}")
    
    print("Launching CARLA Simulator Server...")
    
    # FIXED: Added -dx11 flag to prevent DXGI device reset crashes
    cmd = [executable, f"-carla-rpc-port={port}", "-quality-level=Low", "-dx11"]
    
    process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(20)
    return process
    
    
def connect_to_carla(port=2000, town_name="Town01"):
    """
    Connects our custom Python script to the running simulator server 
    and returns initialized handles to the client and world instances.
    """
    print(f"Connecting to CARLA server on port {port}...")
    client = carla.Client("localhost", port)
    client.set_timeout(60.0)  
    
    # FIXED: Completely removed client.load_world to prevent map switching freezes.
    # It will use whatever map is already up and running.
    world = client.get_world()
    
    # Configure environmental variables for clean baseline visibility
    weather = carla.WeatherParameters.ClearNoon
    world.set_weather(weather)
    
    print("Successfully connected and environment initialized.")
    return client, world