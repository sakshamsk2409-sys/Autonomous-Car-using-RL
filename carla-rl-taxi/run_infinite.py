"""
Infinite Autonomous Taxi Driver Runner
Runs the fine-tuned RL autonomous vehicle infinitely in CARLA with all 5 active scenarios:
- Stray animal avoidance
- Pothole detection & proactive swerving
- Two-wheeler cutting-in mitigation
- Intent prediction
- Emergency siren detection & yielding

Press Ctrl+C at any time in the terminal to stop cleanly.
"""
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from enjoy import enjoy_and_log

if __name__ == "__main__":
    print("=" * 70)
    print("   AUTONOMOUS TAXI - INFINITE DRIVING MODE (PRESENTATION READY)   ")
    print("   Running indefinitely. Press Ctrl+C at any time to exit safely. ")
    print("=" * 70)
    enjoy_and_log(max_steps=None)
