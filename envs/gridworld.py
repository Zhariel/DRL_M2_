from envs.deepsingleagentenv import DeepSingleAgentEnv

import math
import numpy as np
import numba


class GridWorld(DeepSingleAgentEnv):
    def __init__(self):
        self.nb_cells = 25
        self.current_cell = math.floor(25 / 2)
        self.step_count = 0
        self.S = np.arange(self.nb_cells)
        self.A = np.array([0, 1, 2, 3])
        self.R = np.array([-1.0, 0.0, 1.0])
        self.P = np.zeros((len(self.S), len(self.A), len(self.S)+5))

        self.a_effects = {0: 5, 1: -5, 2: 1, 3: -1}

        self.index = lambda x, y: x + y * 5

        # ↓
        for i in range(5):
            for j in range(4):
                s = self.index(i, j)
                print(s)
                self.P[s, 0, s + 5] = 1

        # ↑
        for i in range(5):
            for j in range(1, 5):
                s = self.index(i, j)
                self.P[s, 1, s - 5] = 1

        # →
        for i in range(4):
            for j in range(5):
                s = self.index(i, j)
                self.P[s, 2, s + 1] = 1

        # ←
        for i in range(1, 5):
            for j in range(5):
                s = self.index(i, j)
                self.P[s, 3, s - 1] = 1

        # 🡭
        self.P[3, 2, 4] = 1
        self.P[9, 1, 4] = 1

        # 🡮
        self.P[23, 2, 24] = 1
        self.P[19, 0, 24] = 1

    def available_actions_ids(self) -> np.ndarray:
        s = self.current_cell
        acts = self.A
        aa = []
        for a in self.A:
            if self.P[s, a, s + self.a_effects[a]] == 1:
                aa.append(a)

        return np.array(aa)

    def max_action_count(self) -> int:
        return 4

    def state_description(self) -> np.ndarray:
        return np.array([self.current_cell / (self.nb_cells - 1) * 2.0 - 1.0])

    def state_dim(self) -> int:
        return 1

    def is_game_over(self) -> bool:
        if self.step_count > self.nb_cells:
            return True
        return self.current_cell == 0 or self.current_cell == self.nb_cells - 1

    def act_with_action_id(self, action_id: int):
        self.step_count += 1
        self.current_cell = self.a_effects[action_id]

    def score(self) -> float:
        if self.current_cell == 0:
            return -1.0
        elif self.current_cell == self.nb_cells - 1:
            return 1.0
        else:
            return 0.0

    def reset(self):
        self.current_cell = math.floor(self.nb_cells / 2)
        self.step_count = 0

    def view(self):
        print(f'Game Over: {self.is_game_over()}')
        print(f'score : {self.score()}')
        for i in range(self.nb_cells):
            if i == self.current_cell:
                print("X", end='')
            else:
                print("_", end='')
        print()
