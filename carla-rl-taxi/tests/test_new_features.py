import sys
import os
import math
import unittest
import numpy as np

# Ensure project root is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

# Helper mock classes for testing without requiring a live CARLA GPU server
class MockLocation:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def distance(self, other):
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)

class MockRotation:
    def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
        self.pitch = float(pitch)
        self.yaw = float(yaw)
        self.roll = float(roll)

class MockTransform:
    def __init__(self, location=None, rotation=None):
        self.location = location or MockLocation()
        self.rotation = rotation or MockRotation()

    def get_forward_vector(self):
        rad = math.radians(self.rotation.yaw)
        return MockLocation(x=math.cos(rad), y=math.sin(rad), z=0.0)

class MockVector3D:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

class MockActor:
    def __init__(self, actor_id=1, transform=None, velocity=None):
        self.id = actor_id
        self.transform = transform or MockTransform()
        self.velocity = velocity or MockVector3D()
        self.is_alive = True

    def get_location(self):
        return self.transform.location

    def get_transform(self):
        return self.transform

    def get_velocity(self):
        return self.velocity

    def destroy(self):
        self.is_alive = False

    def apply_control(self, control):
        pass


class TestAutonomousFeatures(unittest.TestCase):

    def test_acoustic_siren_sensor(self):
        from src.sensors.acoustic_sensor import AcousticSirenSensor
        sensor = AcousticSirenSensor(base_sound_level_db=120.0, ambient_noise_db=65.0)

        ego_tf = MockTransform(MockLocation(0, 0, 0), MockRotation(yaw=0))
        ego_vel = MockVector3D(10, 0, 0) # Moving forward at 10 m/s

        # Case 1: Inactive siren
        res = sensor.compute_received_signal(ego_tf, ego_vel, None, None, siren_active=False)
        self.assertFalse(res["is_siren_detected"])
        self.assertEqual(res["sound_level_norm"], 0.0)

        # Case 2: Siren approaching from behind (at -30m, moving at 20 m/s)
        em_tf_close = MockTransform(MockLocation(-20, 0, 0), MockRotation(yaw=0))
        em_vel = MockVector3D(20, 0, 0)
        res_close = sensor.compute_received_signal(ego_tf, ego_vel, em_tf_close, em_vel, siren_active=True)
        self.assertTrue(res_close["is_siren_detected"])
        self.assertGreater(res_close["sound_level_norm"], 0.3)
        self.assertGreater(res_close["doppler_shift_hz"], 0.0) # Frequency shift should be positive (approaching)
        self.assertTrue(res_close["is_approaching_from_behind"])

        # Case 3: Far siren (at 150m) should have significantly lower sound level
        em_tf_far = MockTransform(MockLocation(150, 0, 0), MockRotation(yaw=0))
        res_far = sensor.compute_received_signal(ego_tf, ego_vel, em_tf_far, em_vel, siren_active=True)
        self.assertLess(res_far["sound_level_norm"], res_close["sound_level_norm"])

    def test_road_surface_sensor_and_potholes(self):
        from src.sensors.road_surface_sensor import RoadSurfaceSensor
        from src.scenarios.potholes import PotholeManager

        surface_sensor = RoadSurfaceSensor()
        ego_tf = MockTransform(MockLocation(0, 0, 0), MockRotation(yaw=0))

        # Test Pothole Scanning
        potholes = [
            {'x': 15.0, 'y': 0.5, 'z': 0.0, 'radius': 0.8, 'severity': 0.85, 'hit': False},
            {'x': 80.0, 'y': 0.0, 'z': 0.0, 'radius': 0.5, 'severity': 0.5, 'hit': False} # Beyond lookahead
        ]
        scan = surface_sensor.scan_potholes_ahead(ego_tf, potholes)
        self.assertIsNotNone(scan["closest_pothole"])
        self.assertAlmostEqual(scan["closest_pothole"]['x'], 15.0)
        self.assertGreater(scan["pothole_severity"], 0.8)
        self.assertAlmostEqual(scan["pothole_dist_norm"], 15.0 / 30.0, places=2)

        # Test Faded Line Marking Assessment
        surface_sensor.current_patch_is_faded = True
        conf_faded, off_faded = surface_sensor.assess_lane_marking(0.2)
        self.assertLess(conf_faded, 0.5)

        surface_sensor.current_patch_is_faded = False
        conf_crisp, off_crisp = surface_sensor.assess_lane_marking(0.2)
        self.assertGreater(conf_crisp, 0.8)

        # Test Pothole Impact dynamics
        pot_mgr = PotholeManager(world=None, config={"enabled": True, "impact_penalty_scale": 20.0})
        pot_mgr.potholes = [{'x': 5.0, 'y': 0.0, 'z': 0.0, 'radius': 0.8, 'severity': 0.9, 'hit': False}]
        
        # Vehicle passing right over it at 15 m/s
        veh_tf_hit = MockTransform(MockLocation(5.2, 0.1, 0), MockRotation())
        penalty, hit = pot_mgr.check_vehicle_impact(veh_tf_hit, speed_mps=15.0)
        self.assertIsNotNone(hit)
        self.assertGreater(penalty, 10.0)
        self.assertTrue(hit['hit'])

    def test_confidence_intent_predictor(self):
        from src.prediction.intent_predictor import ConfidenceIntentPredictor
        predictor = ConfidenceIntentPredictor(history_len=6)
        ego_tf = MockTransform(MockLocation(0, 0, 0), MockRotation(yaw=0))

        # Simulate dynamic actor executing an aggressive CUT_IN maneuver:
        # Starting alongside at lateral offset 3.0m, moving laterally left into lane center at 2.0 m/s
        for step in range(5):
            t = step * 0.1
            loc = MockLocation(x=10.0 + step * 0.5, y=3.0 - step * 0.5, z=0.0)
            vel = MockVector3D(x=12.0, y=-2.0, z=0.0) # Sharp lateral velocity towards center
            yaw = -15.0 # Angled into ego lane
            predictor.update_track(actor_id=42, location=loc, velocity=vel, yaw_deg=yaw, timestamp=t)

        pred_class, conf, score = predictor.predict_intent(42, ego_tf)
        self.assertEqual(pred_class, "CUT_IN")
        self.assertGreater(conf, 0.5)
        self.assertEqual(score, 1.0)
        self.assertLessEqual(conf, 1.0)
        self.assertGreaterEqual(conf, 0.0)

    def test_stray_animal_and_bike_cutting(self):
        from src.scenarios.stray_animals import StrayAnimalManager
        from src.scenarios.bike_cutting import BikeCuttingManager

        # Stray Animal Hazard Query
        animal_mgr = StrayAnimalManager(world=None)
        mock_animal = MockActor(101, MockTransform(MockLocation(20.0, 1.5, 0), MockRotation()))
        animal_mgr.active_animals = [{'actor': mock_animal, 'state': 'crossing', 'speed': 2.0, 'has_triggered': True}]
        
        ego_tf = MockTransform(MockLocation(0, 0, 0), MockRotation(yaw=0))
        dist_norm, lat_norm, raw_dist = animal_mgr.get_animal_hazards(ego_tf)
        self.assertAlmostEqual(raw_dist, math.sqrt(20.0**2 + 1.5**2), places=1)
        self.assertLess(dist_norm, 0.5)

        # Bike Cutting Observation Query
        bike_mgr = BikeCuttingManager(world=None)
        mock_bike = MockActor(102, MockTransform(MockLocation(8.0, 1.0, 0), MockRotation()), MockVector3D(12.0, -1.8, 0.0))
        bike_mgr.active_bike = {'actor': mock_bike, 'state': 'CUTTING_IN'}
        bike_dist_norm, bike_lat_vel = bike_mgr.get_cutting_bike_observation(ego_tf)
        self.assertLess(bike_dist_norm, 0.3)
        self.assertLess(bike_lat_vel, 0.0) # Moving left toward center

    def test_reward_function(self):
        from src.rewards import calculate_taxi_reward

        # Test Crash Condition
        r_crash = calculate_taxi_reward(np.zeros(24), is_crashed=True)
        self.assertEqual(r_crash, -100.0)

        # Test Normal Cruising Reward (24 features)
        normal_obs = np.zeros(24, dtype=np.float32)
        normal_obs[0] = 0.6  # Good speed
        normal_obs[1] = 0.0  # Center of lane
        normal_obs[4] = 1.0  # Obstacle far
        normal_obs[10] = 1.0 # Green light
        normal_obs[11] = 1.0 # Animal far
        normal_obs[13] = 1.0 # Crisp lines
        normal_obs[14] = 1.0 # No pothole
        normal_obs[17] = 1.0 # No cutting bike
        normal_obs[21] = 0.0 # No siren
        r_cruise = calculate_taxi_reward(normal_obs, is_crashed=False)
        self.assertGreater(r_cruise, 0.4)

        # Test Stray Animal Proximity Penalty (Animal close ahead and car speeding)
        animal_threat_obs = normal_obs.copy()
        animal_threat_obs[11] = 0.15 # Very close (7.5m)
        animal_threat_obs[12] = 0.05 # Directly in path
        r_animal = calculate_taxi_reward(animal_threat_obs, is_crashed=False)
        self.assertLess(r_animal, r_cruise)

        # Test Emergency Siren Yielding Bonus vs Obstruction Penalty
        # Yielding: ego pulls to shoulder (lane_offset = 0.3) and slows down (speed = 0.15)
        yield_obs = normal_obs.copy()
        yield_obs[0] = 0.15
        yield_obs[1] = 0.3
        yield_obs[21] = 0.8
        yield_obs[22] = -0.8 # Siren directly behind
        yield_obs[23] = 1.0  # Yield required
        r_yield = calculate_taxi_reward(yield_obs, is_crashed=False)

        # Obstructing: ego speeds in center lane blocking siren
        block_obs = normal_obs.copy()
        block_obs[0] = 0.7
        block_obs[1] = 0.0
        block_obs[21] = 0.8
        block_obs[22] = -0.8
        block_obs[23] = 1.0
        r_block = calculate_taxi_reward(block_obs, is_crashed=False)
        self.assertGreater(r_yield, r_block)

        # Test Pothole Impact Penalty
        r_pothole_hit = calculate_taxi_reward(normal_obs, is_crashed=False, extra_info={"pothole_impact_penalty": 12.5})
        self.assertAlmostEqual(r_cruise - r_pothole_hit, 12.5, places=3)

    def test_environment_observation_mode_toggle(self):
        from src.environment import CarlaTaxiEnv
        
        # Test enhanced mode (24 features)
        env_enhanced = CarlaTaxiEnv(client=None, world=None, config_file="config/settings.yaml")
        self.assertEqual(env_enhanced.observation_space.shape[0], 24)
        
        obs_enhanced, _ = env_enhanced.reset()
        self.assertEqual(obs_enhanced.shape[0], 24)
        self.assertTrue(np.all(obs_enhanced >= -1.0) and np.all(obs_enhanced <= 1.0))

        # Test legacy mode (11 features)
        env_legacy = CarlaTaxiEnv(client=None, world=None, config_file="config/settings.yaml")
        env_legacy.observation_mode = "legacy"
        env_legacy.observation_space = env_enhanced.action_space # temporary
        obs_legacy = env_legacy._get_obs()
        self.assertEqual(obs_legacy.shape[0], 11)

    def test_intent_predictor_all_maneuvers(self):
        from src.prediction.intent_predictor import ConfidenceIntentPredictor
        predictor = ConfidenceIntentPredictor(history_len=6)
        ego_tf = MockTransform(MockLocation(0, 0, 0), MockRotation(yaw=0))

        # 1. Crossing maneuver (perpendicular velocity across road)
        for step in range(5):
            t = step * 0.1
            loc = MockLocation(x=15.0, y=-4.0 + step * 1.5, z=0.0)
            vel = MockVector3D(x=0.0, y=3.0, z=0.0) # Moving across road
            predictor.update_track(actor_id=1, location=loc, velocity=vel, yaw_deg=90.0, timestamp=t)
        p_class, conf, _ = predictor.predict_intent(1, ego_tf)
        self.assertEqual(p_class, "CROSSING")
        self.assertGreater(conf, 0.4)

        # 2. Decelerating maneuver (sudden braking in front)
        for step in range(5):
            t = step * 0.1
            # Rapid deceleration: vx dropping from 15 to 3 m/s
            speed = max(15.0 - step * 3.0, 1.0)
            loc = MockLocation(x=10.0 + step * 1.0, y=0.0, z=0.0)
            vel = MockVector3D(x=speed, y=0.0, z=0.0)
            predictor.update_track(actor_id=2, location=loc, velocity=vel, yaw_deg=0.0, timestamp=t)
        p_class, conf, _ = predictor.predict_intent(2, ego_tf)
        self.assertEqual(p_class, "DECELERATING")
        self.assertGreater(conf, 0.4)

        # 3. Lane Follow maneuver (steady forward motion in lane)
        for step in range(5):
            t = step * 0.1
            loc = MockLocation(x=20.0 + step * 1.0, y=0.1, z=0.0)
            vel = MockVector3D(x=10.0, y=0.0, z=0.0)
            predictor.update_track(actor_id=3, location=loc, velocity=vel, yaw_deg=0.0, timestamp=t)
        p_class, conf, _ = predictor.predict_intent(3, ego_tf)
        self.assertEqual(p_class, "LANE_FOLLOW")
        self.assertGreater(conf, 0.4)

    def test_acoustic_sensor_receding(self):
        from src.sensors.acoustic_sensor import AcousticSirenSensor
        sensor = AcousticSirenSensor()
        ego_tf = MockTransform(MockLocation(0, 0, 0), MockRotation(yaw=0))
        ego_vel = MockVector3D(15, 0, 0)

        # Emergency vehicle moving away from ego (receding: speed lower or heading opposite)
        em_tf_ahead = MockTransform(MockLocation(30, 0, 0), MockRotation(yaw=0))
        em_vel_ahead = MockVector3D(25, 0, 0) # Moving away faster ahead of ego
        res = sensor.compute_received_signal(ego_tf, ego_vel, em_tf_ahead, em_vel_ahead, siren_active=True)
        # Doppler shift should be negative when source is moving away
        self.assertLess(res["doppler_shift_hz"], 0.0)
        self.assertFalse(res["is_approaching_from_behind"])

    def test_road_surface_pothole_filtering(self):
        from src.sensors.road_surface_sensor import RoadSurfaceSensor
        sensor = RoadSurfaceSensor()
        ego_tf = MockTransform(MockLocation(0, 0, 0), MockRotation(yaw=0))

        # Potholes strictly behind ego (x < 0) or off-road laterally (|y| > 5m)
        potholes = [
            {'x': -10.0, 'y': 0.0, 'z': 0.0, 'radius': 0.5, 'severity': 0.5, 'hit': False}, # Behind
            {'x': 15.0, 'y': 8.0, 'z': 0.0, 'radius': 0.5, 'severity': 0.5, 'hit': False}   # Offroad
        ]
        res = sensor.scan_potholes_ahead(ego_tf, potholes)
        self.assertEqual(res["pothole_dist_norm"], 1.0)
        self.assertIsNone(res["closest_pothole"])

if __name__ == "__main__":
    unittest.main()

