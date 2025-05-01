import numpy as np
import matplotlib.pyplot as plt

from poisson import TruncatedPoisson


class WorldModel:
    def __init__(
        self,
        # state/action parameters
        max_cars: int = 20,
        max_moves: int = 5,
        # reward parameters
        move_cost: int = -2,
        rent_credit: int = 10,
        # poisson parameters
        expected_requests_1: float = 3,
        expected_returns_1: float = 3,
        expected_requests_2: float = 4,
        expected_returns_2: float = 2,
    ):
        # store init parameters
        self._max_cars = max_cars
        self._max_moves = max_moves
        self._move_cost = move_cost
        self._rent_credit = rent_credit

        # cache PMF values for relevant values of k
        # (i.e. they're likely enough to happen)
        self._requests_poisson_1 = TruncatedPoisson(expected_requests_1)
        self._returns_poisson_1 = TruncatedPoisson(expected_returns_1)
        self._requests_poisson_2 = TruncatedPoisson(expected_requests_2)
        self._returns_poisson_2 = TruncatedPoisson(expected_returns_2)

        # since r is a function of s', s, and a, calculate all possible rewards
        # once and store them in a lookup table
        self._R, self._T = self._init_R_and_T()
        print("WorldModel fitted")

    def _init_R_and_T(self):
        R = np.zeros(
            shape=(
                self._max_cars + 1,  # s1'
                self._max_cars + 1,  # s2'
                self._max_cars + 1,  # s1
                self._max_cars + 1,  # s2
                (self._max_moves * 2) + 1,  # a
            ),
            dtype=int,
        )
        T = np.zeros(
            shape=(
                self._max_cars + 1,  # s1'
                self._max_cars + 1,  # s2'
                self._max_cars + 1,  # s1
                self._max_cars + 1,  # s2
                (self._max_moves * 2) + 1,  # a
            ),
            dtype=float,
        )

        # loop through all possible (s', s, a) combinations
        for s in self.states():
            s1, s2 = s
            for a in self.actions(s):
                # move the cars during the night
                s1_next = s1 - a
                s2_next = s2 + a
                assert 0 <= s1_next <= self._max_cars
                assert 0 <= s2_next <= self._max_cars

                for req_1, ret_1, req_2, ret_2 in self._rv_observations():
                    # get number of requests that can be fulfilled
                    fulfilled_1 = min(s1_next, req_1)
                    fulfilled_2 = min(s2_next, req_2)

                    # assign reward for car rentals, and also cost for moving cars
                    r = (
                        self._move_cost * abs(a)
                        + fulfilled_1 * self._rent_credit
                        + fulfilled_2 * self._rent_credit
                    )

                    # adjust next state (make sure cars do not go over max)
                    s1_next = min(s1_next - fulfilled_1 + ret_1, self._max_cars)
                    s2_next = min(s2_next - fulfilled_2 + ret_2, self._max_cars)
                    R[s1_next, s2_next, s1, s2, a] = r

                    # keep track of probability of transition occuring:
                    # - all random variables are independent of each other,
                    #   so we compute their product
                    joint_probability = (
                        self._requests_poisson_1.pmf(req_1)
                        * self._returns_poisson_1.pmf(ret_1)
                        * self._requests_poisson_2.pmf(req_2)
                        * self._returns_poisson_2.pmf(ret_2)
                    )
                    T[s1_next, s2_next, s1, s2, a] = joint_probability

        return R, T

    def _rv_observations(self):
        """Return an iterator over all joint observations of the Poisson random variables."""
        for req_1 in self._requests_poisson_1.k():
            for ret_1 in self._returns_poisson_1.k():
                for req_2 in self._requests_poisson_2.k():
                    for ret_2 in self._returns_poisson_2.k():
                        yield req_1, ret_1, req_2, ret_2

    def max_cars(self):
        return self._max_cars

    def states(self):
        return [
            (s1, s2)
            for s1 in range(self._max_cars + 1)
            for s2 in range(self._max_cars + 1)
        ]

    def actions(self, s: tuple[int, int]):
        s1, s2 = s
        lower_bound = max(-self._max_moves, -s2, -(self._max_cars - s1))
        upper_bound = min(+self._max_moves, +s1, +(self._max_cars - s2))
        return [a for a in range(lower_bound, upper_bound + 1)]

    def R(self):
        """Reward function.

        Models the function: r(s', s, a)
        """
        return self._R

    def T(self):
        """Transition function.

        Models the function: p(s', r | s, a)
        """
        return self._T


class PolicyIterationSolver:
    def __init__(
        self,
        model: WorldModel,
        gamma: float = 0.9,
    ):
        self._model = model
        self._gamma = gamma
        self._V = np.zeros(
            (self._model.max_cars() + 1, self._model.max_cars() + 1),
            dtype=float,
        )
        self._pi = np.zeros(
            (self._model.max_cars() + 1, self._model.max_cars() + 1),
            dtype=int,
        )

    def expected_return(self, s: tuple[int, int], a: int) -> float:
        # (s1', s2', s1, s2, a)
        T = self._model.T()
        T = T[:, :, *s, a]
        R = self._model.R()
        R = R[:, :, *s, a]
        ret = np.sum(T * (R + self._gamma * self._V))
        return ret

    def evaluation(self, max_iter: int = 10, theta: float = 0.1):
        """Run policy evaluation against the current policy.

        Args:
            max_iter (int): Max number of iterations.
            theta (float): Early stopping criteria.
        """
        for _ in range(max_iter):
            delta = 0
            for s in self._model.states():
                v = self._V[s]
                self._V[s] = self.expected_return(s, self._pi[s])
                delta = max(delta, abs(v - self._V[s]))
            if delta < theta:
                return

    def improvement(self):
        """Run policy improvement against the current value function."""
        policy_stable = True
        for s in self._model.states():
            old_a = self._pi[s]
            best_a, best_v = self._pi[s], self._V[s]
            for a in self._model.actions(s):
                if a == old_a:
                    continue
                v = self.expected_return(s, a)
                if v > best_v:
                    best_a, best_v = a, v
            self._pi[s] = best_a
            if self._pi[s] != old_a:
                policy_stable = False
        return policy_stable

    def solve(self):
        """Solve for the optimal policy using policy iteration."""
        for _ in range(10):
            # pi -> v
            self.evaluation()
            self.plot_V()

            # v -> pi
            policy_stable = self.improvement()
            self.plot_pi()
            if policy_stable:
                return

    def plot_V(self):
        plt.imshow(self._V)
        plt.title("V")
        plt.xlabel("Cars at location 0")
        plt.ylabel("Cars at location 1")
        plt.colorbar()
        plt.show()

    def plot_pi(self):
        plt.imshow(self._pi)
        plt.title("$\pi$")
        plt.xlabel("Cars at location 0")
        plt.ylabel("Cars at location 1")
        plt.colorbar()
        plt.show()


def main():
    model = WorldModel()
    solver = PolicyIterationSolver(model=model)
    solver.solve()


if __name__ == "__main__":
    main()
