from qlearn.q_agent import QLearningAgent
from qlearn.q_env import QLearningEnv


class QLearningTrainer:
    def __init__(self, env: QLearningEnv, agent: QLearningAgent):
        self.env = env
        self.agent = agent

    def train(self, episodes=100):
        for episode_number in range(episodes):
            self.env.reset()

            done = False
            while not done:
                state = self.env.get_obs()
                action = self.agent.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)

                self.agent.learn(state, action, reward, next_state, done)

            # print(f"Finished episode {episode_number}")