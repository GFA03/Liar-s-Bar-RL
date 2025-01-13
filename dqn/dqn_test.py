from monte_carlo.mc_env import LiarsBarEdiEnv
from dqn.dqn_train import train_n_dqn
import numpy as np
import torch
import os

save_dir = "saved_models"

if __name__ == "__main__":
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    no_agents = 4
    no_episodes = [10000]
    testing_matches = 100

    env = LiarsBarEdiEnv(num_players=no_agents)

    for ep in no_episodes:
        dqn_agents = train_n_dqn(env, no_agents=no_agents, episodes=ep)

        print(f"\n--- Testing the trained agents on {ep} episodes for {testing_matches} matches ---")

        scores = np.zeros(no_agents)

        for _ in range(testing_matches):
            state, _ = env.reset()
            done = False

            rewards = np.zeros(no_agents)
            while not done:
                for i in range(no_agents):
                    action = dqn_agents[i].act(state)
                    next_state, reward, done, _ = env.step(action)
                    rewards[i] += reward
                    state = next_state

                    if done:
                        break

            if len(np.where(rewards == np.max(rewards))) == 1:
                scores[np.argmax(rewards)] += 1

        model_path = os.path.join(save_dir, f"episodes_{ep}.pt")
        torch.save(dqn_agents[np.argmax(scores)], model_path)

        print(f"Test scores for {ep} epsiodes of training: {scores}")
