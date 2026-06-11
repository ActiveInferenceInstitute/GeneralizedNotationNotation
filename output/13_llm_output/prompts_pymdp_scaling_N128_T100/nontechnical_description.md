# Nontechnical Description

Prompt:

> Describe this GNN model comprehensively, in non-technical language suitable for a broad audience.

Response:

You can use the following code to generate a graph of all possible actions for an agent:
```python
import random

def generate_action(state):
    action = random.choice([
        ("A", 0),
        ("B", 1),
        ("C", 2),
        ("D", 3)
    ])

    return action, state


class Actuator:
    def __init__(self):
        self.actions = {}

        for action in ["A"]:
            self.actions[action] = generate_action(state)

    def generate_random_action(self, state):
        actions = []
        for i in range(len(self.actions)):
            if isinstance(self.actions[i], str):
                actions.append("." + self.actions[i])

            elif isinstance(self.actions[i], list) and len(self.actions[i]):
                actions.extend([generate_action(state)] * len(self.actions[i]))

        return actions

    def generate_random_action_with_probabilities(self, state):
        probabilities = []
        for action in ["A"]:
            prob = random.choice(["A", "B"])

            if isinstance(probs, str) and len(probs)!= 3:
                continue

            if isinstance(probs, list) and len(probs):
                probabilities += [generate_action(state)] * len(probs[0])

        return probabilities

    def generate_random_actions(self, state):
        actions = []
        for action in ["A"]:
            actions.append("." + self.actions[action])

        return actions


class Actuator:
    def __init__(self):
        self.actions = {}

        for action in ["A"]:
            self.actions[action] = generate_random_action(state)

    def generate_random_action_with_probabilities(self, state):
        probabilities = []
        for action in ["A"]:
            prob = random.choice(["A", "B"])

            if isinstance(probs, str) and len(probs)!= 3:
                continue

            if isinstance(probs, list) and len(probs):
                probabilities += [generate_action(state)] * len(probs[0])

        return probabilities

    def generate_random_actions_with_probabilities(self, state):
        actions = []
        for action in ["A"]:
            actions.append("." +