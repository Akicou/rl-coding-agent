"""Entry point for starting training."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_agent.config import RLConfig
from rl_agent.train import train

cfg = RLConfig.from_env()
# Users can override here before calling train()
train(cfg)
