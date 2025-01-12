from HumanAgent import HumanAgent
from LiarsBarArena import LiarsBarGame
from dqn.dqn_agent import train_n_dqn
from monte_carlo.mc_agent import MonteCarloAgent
from monte_carlo.mc_env import LiarsBarEdiEnv
from monte_carlo.mc_trainer import MonteCarloTrainer

env = LiarsBarEdiEnv()
mc10000 = MonteCarloAgent(env)
mc1000 = MonteCarloAgent(env)
mc100 = MonteCarloAgent(env)
mc10 = MonteCarloAgent(env)

wins = [0, 0, 0, 0]

no_agents = 4
dqn_env = LiarsBarEdiEnv(num_players=4)
dqn_agents = train_n_dqn(dqn_env, no_agents=no_agents, episodes=300)

trainer = MonteCarloTrainer(env, mc10000)
trainer.train(episodes=10000)
trainer = MonteCarloTrainer(env, mc1000)
trainer.train(episodes=1000)
trainer = MonteCarloTrainer(env, mc100)
trainer.train(episodes=100)
trainer = MonteCarloTrainer(env, mc10)
trainer.train(episodes=10)

arena = LiarsBarGame()
arena.register_agent(mc10000)
arena.register_agent(mc1000)
# arena.register_agent(mc100)
# arena.register_agent(mc10)
arena.register_agent(dqn_agents[0])
arena.register_agent(dqn_agents[1])

for i in range(1000):
    print(i)
    winner = arena.run_game()
    wins[winner] += 1

print(wins)

# mc10000: 21 - mc1000: 8 - dqn100: 61 - dqn100: 10
# mc10000: 17 - mc1000:  64 - dqn100: 13 - dqn100: 6
# mc10000: 185 - mc1000:  529 - dqn300: 243 - dqn300: 43