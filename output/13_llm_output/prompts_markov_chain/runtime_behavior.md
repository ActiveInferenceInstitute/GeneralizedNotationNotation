# Runtime Behavior

Prompt:

> Describe what happens when this GNN model runs and how it would behave in different settings or domains.

Response:

```python
import json
from typing import List

class SimpleMDP:
    def __init__(self):
        self._states = {}

    def add(self, state1, state2):
        """Add a new state to the system."""
        if not isinstance(state1, (list, tuple)):
            raise TypeError("State must be of type list or tuple")

        self.add_transition(state1)
        self.add_transition(state2)

    def add_transition(self, state):
        """Add a transition from one state to another."""
        if not isinstance(state, (list, tuple)):
            raise TypeError("State must be of type list or tuple")

        for i in range(len(state)):
            self.add_transition([
                [
                    [
                        [
                            [
                                [
                                    [
                                        [
                                            [
                                                [
                                                    # Add a transition from state1 to state2.
                                      # This is not allowed, as it would create an action-based transition
                      ],
                      ]],
                     []]
                  ]]
              ]
            ):

    def add_transition(self, state):
        """Add a transition from one state to another."""
        if not isinstance(state, (list, tuple)):
            raise TypeError("State must be of type list or tuple")

        for i in range(len(state)):
            self.add_transition([
                [
                    [
                      [
                        [
                            [
                              # Add an action-based transition from state1 to state2
                          ]],
                     []]
                  ]]

            )

    def add_action(self, state):
        """Add a transition from one state to another."""
        if not isinstance(state, (list, tuple)):
            raise TypeError("State must be of type list or tuple")

        for i in range(len(state)):
            self.add_transition([
                [
                    [
                      [
                        [
                            # Add an action-based transition from state1 to state2
                          ]],
                     []]
                  ]]

            )

    def add_action(self, state):
        """Add a transition from one state to another."""
        if not isinstance(state, (list, tuple)):
            raise TypeError("State must be of type list or tuple")

        for i in range(len(state)):
            self.add_transition([
                [
                    [
                      [
                        [
                            # Add an action-based transition from state1 to state2
                          ]],
                     []]
                  ]]

            )