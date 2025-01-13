from monte_carlo.mc_env import LiarsBarEdiEnv
from dqn.dqn_agent import DQNAgent
import numpy as np

def train_n_dqn(env: LiarsBarEdiEnv, no_agents: int = 4, episodes=100):
    '''
        Trains n dqn agents for a number of episodes and returns the best rated agent
    '''
    agents = [DQNAgent(env, gamma=0.0-i, epsilon=0.2 + i/10, lr=1e-3 * i, batch_size=32, buffer_capacity=20000) for i in range(no_agents)]

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_rewards = [0] * no_agents
        steps = 0

        while not done:
           for i in range(no_agents):
                action = agents[i].choose_action(state)
                next_state, reward, done, _ = env.step(action)
                total_rewards[i] += reward
                steps += 1

                agents[i].remember(state, action, reward, next_state, done)
                agents[i].train_step()

                state = next_state
                # print(f'Agent{i} {state} {reward}')

                if done:
                    break
        # print(f"Episode: {episode + 1}/{episodes}, Steps: {steps}, Total Reward: {total_rewards}")

    print("Training finished!")
    return agents
