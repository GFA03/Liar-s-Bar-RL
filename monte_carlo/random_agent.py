import random

from monte_carlo.mc_env import LiarsBarEdiEnv


class RandomAgent:
    def __init__(self, env: LiarsBarEdiEnv):
        self.env = env

    def __act__(self):
        actions = self.env._get_available_actions()
        return random.choice(actions)