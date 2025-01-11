from monte_carlo.mc_env import LiarsBarEdiEnv
from monte_carlo.mc_agent import MonteCarloAgent
from monte_carlo.random_agent import RandomAgent


class MonteCarloTrainer:
    def __init__(self, env: LiarsBarEdiEnv, agent: MonteCarloAgent):
        self.env = env
        self.agent = agent

    def train(self, episodes = 100):
        for episode_number in range(episodes):
            self.env.reset()

            done = False
            while not done:
                state = self.env.get_obs()
                action = self.agent.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)


            # After episode ends, update Q values based on rewards
            episode = self.env.get_player_reward_history()
            for i in range(4):
                self.agent.learn(episode[i])

            print(f"Finished episode {episode_number}")




if __name__ == "__main__":
    # Train the agent using Monte Carlo
    env = LiarsBarEdiEnv()
    agent = MonteCarloAgent(env, epsilon=0.1, gamma=0.9)
    random_agent = RandomAgent(env)
    trainer = MonteCarloTrainer(env, agent)

    trainer.train(episodes=100000)
    # for key, value in agent.Q.items():
    #     print(f"{key}: {value}")


    for tests in range(50):
        print("********************")
        env.reset()
        done = False
        while not done:
            state = env.get_obs()
            action = agent.act(state)
            _, reward, done, _ = env.step(action)
            print(f"State: {state}, Action: {action}, Reward: {reward}")