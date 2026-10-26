"""
Exact optimal policy for PrivacyEnv via backward induction (dynamic programming).

Why: (1) it tells you the CEILING for any RL agent in your environment, so you can
see whether "RL ~ Always City" is the agents' fault or the environment's, and
(2) it is a strong upper-bound reference line for the report
("our agent recovers X% of the achievable improvement over Always City").

Not an RL agent -- use it only as a reference, never as a submission.
Assumes integer costs and an integer max_budget. Keep utility()/COST in sync with
PrivacyEnv.step().
"""
import numpy as np
from privacy_env import PrivacyEnv

COST = {0: 10.0, 1: 2.0, 2: 0.0}


def utility(a, u, rho):
    if a == 0:
        return u * (6.0 if rho == 0 else 10.0)
    if a == 1:
        return u * (10.0 if rho == 0 else 4.0)
    return -2.0 * u


class OptimalPolicy:
    def __init__(self, n_u=2000):
        env = PrivacyEnv()
        self.B = int(env.max_budget)
        self.T = env.max_steps
        self.w = env.privacy_weight
        self.pen = env.overdraw_penalty
        self.u_grid = (np.arange(n_u) + 0.5) / n_u
        self.V = np.zeros((self.T + 1, self.B + 1))     # V[t, b]: value with t steps left, budget b
        self._solve()

    def _q(self, a, u, rho, b, t):
        c = COST[a]
        if c > b:                                        # unaffordable -> penalty, episode ends
            return np.full_like(np.asarray(u, dtype=float), -self.pen)
        bn = b - c
        future = 0.0 if bn <= 0 else self.V[t - 1, int(bn)]   # budget exhausted -> episode ends
        return utility(a, np.asarray(u, dtype=float), rho) - self.w * c + future

    def _solve(self):
        for t in range(1, self.T + 1):
            for b in range(1, self.B + 1):
                total = 0.0
                for rho in (0, 1):
                    q = np.stack([self._q(a, self.u_grid, rho, b, t) for a in range(3)])
                    total += 0.5 * q.max(axis=0).mean()
                self.V[t, b] = total

    def fixed_action_value(self, a):
        """Expected return of always playing action a (e.g. a=1 -> Always City)."""
        V = np.zeros((self.T + 1, self.B + 1))
        for t in range(1, self.T + 1):
            for b in range(1, self.B + 1):
                c = COST[a]
                if c > b:
                    V[t, b] = -self.pen
                    continue
                r = 0.5 * (utility(a, 0.5, 0) + utility(a, 0.5, 1)) - self.w * c
                V[t, b] = r + (0.0 if b - c <= 0 else V[t - 1, int(b - c)])
        return V[self.T, self.B]

    def __call__(self, obs):
        u, rho, bf, tf = obs[:4]
        b, t = int(round(bf * self.B)), int(round(tf * self.T))
        qs = [float(self._q(a, u, int(rho), b, t)) for a in range(3)]
        return int(np.argmax(qs))


if __name__ == "__main__":
    pol = OptimalPolicy()
    opt, city = pol.V[pol.T, pol.B], pol.fixed_action_value(1)
    print(f"max_budget={pol.B}")
    print(f"Optimal expected return : {opt:7.2f}")
    print(f"Always City expected    : {city:7.2f}")
    print(f"Headroom                : {opt - city:+7.2f} ({100 * (opt - city) / city:+.1f}%)")
