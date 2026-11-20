"""pytest path setup: real_env.py imports env.py / ppo.py from the
parent (deep-rl-market-making) directory."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
