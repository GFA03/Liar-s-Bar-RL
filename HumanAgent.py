class HumanAgent:
    def __init__(self):
        self.name = "Human agent"

    def act(self, state):
        print(state)
        action = [int(x) for x in input().split()]
        return action