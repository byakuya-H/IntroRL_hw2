from typing import Hashable, Union
from abc import abstractmethod

import numpy as np

from env import Make_Env


class Agent:
    def __init__(self, env: Make_Env, default_action: int = 0):
        self.env = env
        self.default_action = default_action

    @abstractmethod
    def _select_action(self, state) -> int:
        return self.default_action

    def select_action(self, state, epsilon: Union[float, None] = None) -> int:
        """`\\varepsilon`-greedy policy."""
        if epsilon is None or np.random.rand() >= epsilon:
            return self._select_action(state)
        else:
            return self.env.action_sample()


class QAgent(Agent):
    def __init__(
        self,
        env: Make_Env,
        lr: float = 1.0,
        discount_factor: float = 0.8,
        default_action: int = 0,
    ):
        super(QAgent, self).__init__(env, default_action)
        self._Q, self._policy, self.lr, self.discount_factor = (
            dict(),
            dict(),
            lr,
            discount_factor,
        )
        self.Q = type(
            "_QAgent__Q",
            (),
            dict(
                __getitem__=lambda _, k: self._get_q(k[0], k[1]),
                __setitem__=lambda _, k, v: self._set_q(k[0], k[1], v),
                states=lambda _: self._Q.keys(),
            ),
        )()

    def _get_q(self, state: tuple, action: int):
        try:
            return self._Q[state][action]
        except Exception as _:
            return 0

    def _set_q(self, state: tuple, action: int, value: float):
        if self._Q.get(state, None) is None:
            self._Q[state] = dict()
        self._Q[state][action] = value

    def _select_action(self, state: tuple) -> int:
        """greedy policy, return default action if not set."""
        return self._policy.get(state, self.default_action)

    def eval_pol(
        self,
        state: tuple,
        next_state: tuple,
        act: int,
        reward: Union[float, int],
    ):
        """update the Q function."""
        self.Q[state, act] += self.lr * (  # type: ignore[reportIndexIssue]
            reward
            + self.discount_factor * self.Q[next_state, self._select_action(state)]  # type: ignore[reportIndexIssue]
            - self.Q[state, act]  # type: ignore[reportIndexIssue]
        )

    def update_pol(
        self, state: Union[tuple, None] = None, act: Union[int, None] = None
    ):
        """
		update the policy given Q function. only update the relevant part of the policy
		if given `state` and `action`, else update the whole policy.
        """
        if state and act is None:
            for s in self.Q.states():  # type: ignore[reportIndexIssue]
                self._policy[s] = max(list(range(4)), key=lambda i: self.Q[s, i])  # type: ignore[reportIndexIssue]
        elif self.Q[state, self._select_action(state)] < self.Q[state, act]:  # type: ignore[reportIndexIssue]
            self._policy[state] = act  # type: ignore[reportIndexIssue]

    def Qmodel(self):
        return self._Q

    def __str__(self):
        # prec, leng, act_sym = 4, self.env.grid_size, ["->", "<-", "\\/", "^ "]
        prec, leng, act_sym = 4, self.env.grid_size, ["", "", "", ""]
        Q, policy = (
            [[[] for _ in range(leng)] for _ in range(leng)],
            [["" for _ in range(leng)] for _ in range(leng)],
        )
        for x in range(leng):
            for y in range(leng):
                if self.env.not_wall_position(np.array([x, y])):
                    if (np.array(self.env.pos_door) == np.array([x, y])).all():
                        Q[x][y] = [
                            "d" * (len(act_sym[0]) + prec),
                            "d" * (len(act_sym[0]) + prec),
                            "d" * (len(act_sym[0]) + prec),
                            "d" * (len(act_sym[0]) + prec),
                        ]
                        policy[x][y] = "d" * len(act_sym[0])
                    else:
                        Q[x][y] = [
                            f"{act_sym[0]}{self.Q[(x, y), 0]}"[: len(act_sym[0]) + prec]  # type: ignore[reportIndexIssue]
                            + " " * (prec - len(str(self.Q[(x, y), 0])[:prec])),  # type: ignore[reportIndexIssue]
                            f"{act_sym[1]}{self.Q[(x, y), 1]}"[: len(act_sym[0]) + prec]  # type: ignore[reportIndexIssue]
                            + " " * (prec - len(str(self.Q[(x, y), 1])[:prec])),  # type: ignore[reportIndexIssue]
                            f"{act_sym[2]}{self.Q[(x, y), 2]}"[: len(act_sym[0]) + prec]  # type: ignore[reportIndexIssue]
                            + " " * (prec - len(str(self.Q[(x, y), 2])[:prec])),  # type: ignore[reportIndexIssue]
                            f"{act_sym[3]}{self.Q[(x, y), 3]}"[: len(act_sym[0]) + prec]  # type: ignore[reportIndexIssue]
                            + " " * (prec - len(str(self.Q[(x, y), 3])[:prec])),  # type: ignore[reportIndexIssue]
                        ]
                        policy[x][y] = act_sym[self._select_action((x, y))]
                else:
                    Q[x][y] = ["#" * (len(act_sym[0]) + prec) for _ in range(4)]
                    policy[x][y] = "#" * len(act_sym[0])
        Q, policy = (
            "\n\n".join(
                "\n".join("  ".join(Q[x][y][i] for y in range(leng)) for i in range(4))
                for x in range(leng)
            ),
            "\n\n".join(
                "  ".join(policy[x][y] for y in range(leng)) for x in range(leng)
            ),
        )
        return f"\n---\nQ:\n{Q}\npolicy:\n{policy}\n---\n"
