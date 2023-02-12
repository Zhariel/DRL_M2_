import random
import itertools
import numpy as np
from typing import List


class Bomberman:
    def __init__(self):
        self.b = BombermanGame()

    def max_action_count(self) -> int:
        return 5

    def state_description(self) -> np.ndarray:
        return np.array([self.b.position / (self.b.game_size - 1) * 2.0 - 1.0])

    def state_dim(self) -> int:
        return 1

    def is_game_over(self) -> bool:
        if self.b.step >= 300 or self.b.won != 0:
            return True
        return False

    def act_with_action_id(self, action_id: int):
        self.b.frame(action_id)

    def score(self) -> float:
        return float(self.b.won)

    def available_actions_ids(self) -> np.ndarray:
        actions = []
        pos = self.b.position
        for i in range(5):
            s_p = i + self.b.a_effects[i]
            if self.b.P[pos, i, s_p]:
                actions.append(i)

    def reset(self):
        self.b = BombermanGame()

    def view(self):
        self.b.display()


class Bomb:
    def __init__(self, pos):
        self.pos = pos
        self.ticks = 3


class BombermanGame:
    def __init__(self, size=5):
        self.game_size = 5 if size < 5 else size
        # breakable_or_air = 0 if random.random() < 0.95 else 3
        self.ennemy_count = 0
        self.bomb_strength = 1
        self.bombs: List[Bomb] = []
        self.board = [
            [1 if h % 2 == 1 and w % 2 == 1 else (0 if random.random() < 0.85 else 2) for h in range(self.game_size)]
            for w in range(self.game_size)]

        self.blocks = dict({0: "⬛", 1: "⬜", 2: "🔹", 3: "🧑", 4: "💣", 5: "💯", 6: "👽"})
        self.blocks_code = dict({"air": 0, "wall": 1, "breakable": 2, "bomber": 3, "bomb": 4, "power": 5, "ennemy": 6})

        scaling_factor = self.game_size // 5

        nb_powerups = 2

        for i in range(scaling_factor * 2):
            # ennemy
            chosen_col = random.choice(list(range(self.game_size)))
            idx = random.choice(list(range(0, self.game_size, 2)))
            self.board[chosen_col][idx] = 6

        for i in range(nb_powerups):
            # power
            chosen_col = random.choice(list(range(self.game_size)))
            idx = random.choice(list(range(0, self.game_size, 2)))
            self.board[chosen_col][idx] = 5

        chosen_col = random.choice(list(range(self.game_size)))
        idx = random.choice(list(range(0, self.game_size, 2)))
        self.board[chosen_col][idx] = 3

        appears = False
        for sublist in self.board:
            if 6 in sublist:
                appears = True
                break

        if not appears:
            chosen_col = random.choice(list(range(self.game_size)))
            idx = random.choice(list(range(0, self.game_size, 2)))
            while self.board[chosen_col][idx] == 3:
                chosen_col = random.choice(list(range(self.game_size)))
                idx = random.choice(list(range(0, self.game_size, 2)))
            self.board[chosen_col][idx] = 6

        self.won = 0
        self.step = 0
        self.board = list(itertools.chain(*self.board))
        self.position = self.board.index(3)
        self.P = np.zeros((self.game_size ** 2, 5, self.game_size ** 2 + 5))
        self.a_effects = {0: 5, 1: -5, 2: 1, 3: -1, 4: 0}
        self.index = lambda x, y: x + y * 5

        # ↓
        for i in range(5):
            for j in range(4):
                s = self.index(i, j)
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

        # bomb
        for i in range(1, 5):
            for j in range(5):
                s = self.index(i, j)
                self.P[s, 4, s] = 1

        # 🡭
        self.P[3, 2, 4] = 1
        self.P[9, 1, 4] = 1

        # 🡮
        self.P[23, 2, 24] = 1
        self.P[19, 0, 24] = 1

    def place_bomb(self):
        bomb_positions = [i.pos for i in self.bombs]
        if len(self.bombs) < 3 and self.position not in bomb_positions:
            self.bombs.append(Bomb(self.position))

    def bomb_explode(self, origin):
        for i in range(4):
            for _ in range(self.bomb_strength):
                s_p = origin + self.a_effects[i]
                if self.P[origin, i, s_p] == 1:
                    if self.board[s_p] == 1:
                        break
                    if self.board[s_p] == 3 or self.board[s_p] == 6 or self.board[s_p] == 2:
                        self.board[s_p] = 0

    def frame(self, player_a):
        if 3 not in self.board:
            self.won = -1
            return
        if 6 not in self.board:
            self.won = 1
            return

        if player_a == 4:
            self.place_bomb()
        else:  # move player
            s_p = self.position + self.a_effects[player_a]
            if self.P[self.position, player_a, s_p] == 1:
                if self.board[s_p] == 6:
                    self.won = -1
                if self.board[s_p] == 0:
                    self.position = s_p

        if self.step % 4 == 0:  # move ennemy
            for e in self.board:
                if self.board[e] == 6:
                    direction = random.randint(0, 4)
                    s_p = e + self.a_effects[direction]
                    if self.P[e, direction, s_p] == 1:
                        if self.board[s_p] == 0:
                            self.board[e] = 0
                            self.board[s_p] = 6
                        if self.board[s_p] == 3:
                            self.won = -1

    def display(self):
        for i in range(self.game_size):
            for j in range(self.game_size):
                piece = self.board[i * self.game_size + j]
                print(self.blocks[piece], end="")
            print()
