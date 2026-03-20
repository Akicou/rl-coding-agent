"""Entry point for starting training."""

import os

from rl_agent.config import RLConfig
from rl_agent.train import train

cfg = RLConfig.from_env()
# Users can override here before calling train()
train(cfg)
