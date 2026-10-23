import os
import sys
import subprocess
import time

carla_pythonapi = r"C:\Users\sahak\OneDrive\Documents\AIDS\ML\UE4\PythonAPI"
if os.path.exists(carla_pythonapi):
    if carla_pythonapi not in sys.path:
        sys.path.append(carla_pythonapi)
    p_carla = os.path.join(carla_pythonapi, "carla")
    if p_carla not in sys.path:
        sys.path.append(p_carla)

import carla

def launch_carla_server(carla_path, port=2000, quality_level="Epic"):
    """
    Launches the CARLA simulator executable in full high-fidelity Epic mode.
    """
    executable = os.path.join(carla_path, "CarlaUE4.exe")
    if not os.path.exists(executable):
        raise FileNotFoundError(f"Could not locate CARLA executable at {executable}")
    
    print(f"Launching CARLA Simulator Server in [{quality_level}] graphics quality...")
    
    cmd = [executable, f"-carla-rpc-port={port}", f"-quality-level={quality_level}", "-dx11"]
    
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