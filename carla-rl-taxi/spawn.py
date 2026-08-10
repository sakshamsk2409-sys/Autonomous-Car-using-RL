import carla

def main():
    print("Connecting to CARLA...")
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    map = world.get_map()
    
    spawn_points = map.get_spawn_points()
    print(f"Found {len(spawn_points)} spawn points! Painting numbers in the sky...")

    for i, spawn_point in enumerate(spawn_points):
        # Draw the index number 2 meters above the road
        world.debug.draw_string(
            spawn_point.location + carla.Location(z=2.0),
            str(i),
            draw_shadow=False,
            color=carla.Color(r=255, g=0, b=0),
            life_time=120.0 # Numbers will stay visible for 120 seconds
        )

    print("Done! Look at the CARLA window. Fly around and find your perfect Start and End indices.")

if __name__ == '__main__':
    main()