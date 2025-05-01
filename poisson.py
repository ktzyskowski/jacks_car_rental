from itertools import count
from typing import Sequence

import numpy as np
from scipy.stats import poisson


class TruncatedPoisson:
    def __init__(
        self,
        mu: float,
        eps: float = 1e-2,
    ):
        self._mu = mu
        self._eps = eps
        self._lower = None
        self._upper = None
        self._pmf = []

        for k in count(start=0):
            p = poisson.pmf(k, self._mu)
            if p > self._eps and self._lower is None:
                self._lower = k
                self._pmf.append(p)
            elif p < self._eps:
                break
            else:
                self._upper = k
                self._pmf.append(p)

        # normalize so that truncated PMF sums to 1
        self._pmf = np.array(self._pmf)
        self._pmf /= self._pmf.sum()
        self._k = np.arange(self._lower, self._upper + 1)

        print(
            f"TruncatedPoisson fitted: (mu={mu}, eps={eps}, lower={self._lower}, upper={self._upper})"
        )

    def k(self) -> Sequence[int]:
        return self._k

    def pmf(self, k) -> float:
        return self._pmf[k]
