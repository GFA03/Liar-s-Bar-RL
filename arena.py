from random import shuffle

from HumanAgent import HumanAgent
from LiarsBarArena import LiarsBarGame
from dqn.dqn_train import train_n_dqn
from monte_carlo.mc_agent import MonteCarloAgent
from monte_carlo.mc_env import LiarsBarEdiEnv
from monte_carlo.mc_trainer import MonteCarloTrainer
from monte_carlo.random_agent import RandomAgent
from qlearn.q_agent import QLearningAgent
from qlearn.q_trainer import QLearningTrainer
from sarsa.sarsa_agent import SarsaAgent, SarsaTrainer

env = LiarsBarEdiEnv()
mc10000 = MonteCarloAgent(env)
mc1000 = MonteCarloAgent(env)
mc100 = MonteCarloAgent(env)
mc10 = MonteCarloAgent(env)

human = HumanAgent()

sarsa1000 = SarsaAgent(env)
sarsa100 = SarsaAgent(env)

qlearn1000 = QLearningAgent(env)
qlearn100 = QLearningAgent(env)

wins = [0, 0, 0, 0]
agents = []
agents_index = [0,1,2,3]

no_agents = 4
dqn_env = LiarsBarEdiEnv(num_players=4)
dqn_agents = train_n_dqn(dqn_env, no_agents=no_agents, episodes=100)

# trainer = MonteCarloTrainer(env, mc10000)
# trainer.train(episodes=10000)
trainer = MonteCarloTrainer(env, mc1000)
trainer.train(episodes=1000)
trainer = MonteCarloTrainer(env, mc100)
trainer.train(episodes=100)
# trainer = MonteCarloTrainer(env, mc10)
# trainer.train(episodes=10)

trainer = SarsaTrainer(env, sarsa1000)
trainer.train(episodes=1000)
trainer = SarsaTrainer(env, sarsa100)
trainer.train(episodes=100)

trainer = QLearningTrainer(env, qlearn1000)
trainer.train(episodes=1000)
trainer = QLearningTrainer(env, qlearn100)
trainer.train(episodes=100)

agents.append(qlearn1000)
agents.append(sarsa1000)
agents.append(dqn_agents[0])
agents.append(mc1000)
# arena.register_agent(dqn_agents[0])
# arena.register_agent(dqn_agents[1])

for i in range(1000):
    arena = LiarsBarGame()
    arena.register_agent(agents[agents_index[0]])
    arena.register_agent(agents[agents_index[1]])
    arena.register_agent(agents[agents_index[2]])
    arena.register_agent(agents[agents_index[3]])
    print(i)
    winner = arena.run_game()
    wins[agents_index[winner]] += 1
    shuffle(agents_index)


print(wins)

# mc10000: 21 - mc1000: 8 - dqn100: 61 - dqn100: 10
# mc10000: 17 - mc1000:  64 - dqn100: 13 - dqn100: 6
# mc10000: 185 - mc1000:  529 - dqn300: 243 - dqn300: 43
# sarsa1000: 603 - dqn100: 46 - dqn100: 59 - mc1000: 292 (with shuffled players)
# qlearn1000: 205 - dqn_agent[1]: 46 - dqn_agent[0]:374 - mc1000: 375
# qlearn1000: 135 - sarsa1000: 483 - dqn_agent[0]: 49 - mc1000: 333