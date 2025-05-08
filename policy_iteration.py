from typing import Sequence

import numpy as np

MAX_CARS = 20
MAX_MOVES = 5

MOVE_COST = -2
RENT_CREDIT = 10

State = tuple[int, int]
"""The MDP state is an (i, j) pair of the number of cars at each location."""

Action = int
"""The MDP action is an integer indicating the movement of cars.

A positive value indicates a flow from the first location to the second.
A negative value indicates a flow from the second location to the first.
A value of zero indicates no movement of cars.
"""


def states() -> Sequence[State]:
    """Return a collection of all possible states.

    Returns:
        Sequence[State]: a list of all possible (i, j) state pairs.
    """
    return [(i, j) for i in range(0, MAX_CARS + 1) for j in range(0, MAX_CARS + 1)]


def actions(state: State) -> Sequence[Action]:
    """Return a collection of all possible actions from the given state.

    Args:
        state (State): the starting state.

    Returns:
        Sequence[Action]: a list of all possible (and valid) actions.
    """
    # the total number of cars moveable from either location is bounded by:
    #   - the MAX_MOVES parameter
    #   - the number of cars at the source location
    #   - the number of spots at the target location
    upper_bound = min(MAX_MOVES, state[0], (MAX_CARS - state[1]))
    lower_bound = min(MAX_MOVES, state[1], (MAX_CARS - state[0]))
    return range(-lower_bound, upper_bound + 1)


class PolicyIteration:
    def __init__(self):
        self.pi = np.zeros((MAX_CARS + 1, MAX_CARS + 1), dtype=int)
        self.V = np.zeros((MAX_CARS + 1, MAX_CARS + 1), dtype=float)

    def expected_return(self, state: State, action: Action):
        pass

    def policy_evaluation(self, theta: float = 0.1):
        """Run policy evaluation."""
        while True:
            delta = 0
            for state in states():
                v = self.V[state]
                # note: asynchronous update
                self.V[state] = self.expected_return(state, self.pi[state])
                delta = max(delta, abs(v - self.V[state]))
            if delta < theta:
                return

    def policy_improvement(self) -> bool:
        """Run policy improvement."""
        policy_stable = True
        for state in states():
            best_action, best_v = self.pi[state], self.V[state]
            for action in actions(state):
                if action == self.pi[state]:
                    continue
                v = self.expected_return(state, action)
                if v > best_v:
                    best_action, best_v = action, v
            if best_action != self.pi[state]:
                policy_stable = False
                self.pi[state] = best_action
        return policy_stable

    def solve(self, max_iter: int = 5):
        """Run policy iteration."""
        yield self.pi, self.V
        for i in range(max_iter + 1, start=1):
            self.policy_evaluation()
            policy_stable = self.policy_improvement()
            yield self.pi, self.V
            if policy_stable:
                return
