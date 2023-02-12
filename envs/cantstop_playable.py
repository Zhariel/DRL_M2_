import random
from copy import deepcopy
import numpy as np

class CantStopGame:
    def __init__(self, nb_players: int):
        self.col_lens = [3, 5, 7, 9, 11, 13, 11, 9, 7, 5, 3]
        self.b = [[-1 if i + 1 < 7 - j // 2 or i + 1 > 7 + j // 2 else 0 for i in range(max(self.col_lens))] for j in
                  self.col_lens]
        self.b_previous = self.b.copy()
        self.display_code = {-1: "   ", 0: " | ", 1: " $ ", 2: " O ", 11: " . ", 22: " . "}
        self.display_color = {-1: "", 0: "", 1: "\x1b[91m", 2: "\x1b[92m", 11: "\x1b[91m", 22: "\x1b[92m"}
        self.bust_status = {0: "stopped", 1: "continues", 2: "bust"}
        self.available_cols = [[] for _ in range(nb_players + 1)]
        self.neutral = {i: i * 10 + i for i in range(1, nb_players + 1)}
        self.won_col = [0 for i in range(len(self.col_lens))]
        self.winner = 0
        self.won = False
        self.steps = 0
        self.current_decision = "dice"
        self.current_actions = []

    def bind(self, player):
        neutral = self.neutral[player]
        for c in self.b:
            if player in c and neutral in c:
                for i in range(len(c)):
                    if c[i] == player:
                        c[i] = 0
            for i in range(len(c)):
                if c[i] == neutral:
                    c[i] = player

    def pairs(self):
        r = [random.randint(1, 6) for _ in range(4)]

        return [
            [r[0] + r[1], r[2] + r[3]],
            [r[0] + r[2], r[1] + r[3]],
            [r[0] + r[3], r[1] + r[2]],
        ]

    def choose_pairs(self):
        pairs = self.pairs()
        print(f"Pairs are : {pairs[0]}, {pairs[1]}, {pairs[2]}. Enter a number from 1 to 3 to make your choice.")
        choice = int(input().strip())
        if 1 <= choice <= 3:
            return pairs[choice - 1]
        else:
            raise ValueError("Value needs to be between 1 and 3")

    def prompt_continue(self):
        print("Continue ? 1 / 0")
        continues = int(input().strip())
        if continues == 0:
            return False
        elif continues == 1:
            return True
        else:
            raise ValueError("Value needs to be 0 or 1")

    def play(self, player, pair, continues):
        if not continues:
            self.bind(player)
            self.b_previous = deepcopy(self.b)
            return 0  # STOPPED

        self.steps += 1
        cant_play_count = 0
        pair_f = [pair[0] - 2, pair[1] - 2]
        for p in pair_f:
            if self.can_play_col(p, player):
                self.take_action(p, player)
                self.check_won_column(p, player)
                self.update_winner(player)
                continue
            cant_play_count += 1

        if cant_play_count == 2:
            self.b = deepcopy(self.b_previous)
            return 2  # BUST
        return 1  # CAN CONTINUE

    def regen_cols(self):
        for x in self.available_cols:
            x = []

    def can_play_col(self, col_idx, player):
        a = len(self.available_cols[player]) >= 3
        b = col_idx not in self.available_cols[player]
        if self.won_col[col_idx] != 0 or a and b:
            return False
        return True

    def take_action(self, col_idx, player):
        if player in self.b[col_idx] or self.neutral[player] in self.b[col_idx]:
            self.advance_column(col_idx, player)
        else:
            self.place_piece(col_idx, player)

    def find_player_pos(self, col_idx: int, player: int) -> int:
        neutral = self.neutral[player]
        if neutral in self.b[col_idx]:
            return self.b[col_idx].index(neutral)
        else:
            return self.b[col_idx].index(player)

    def place_neutral(self, col_idx: int, player: int, tile_idx: int, save: bool):
        self.b[col_idx][tile_idx] = self.neutral[player]
        if save:
            self.available_cols[player].append(col_idx)

    def advance_column(self, col_idx: int, player: int):
        save = False if self.neutral[player] in self.b[col_idx] else True
        pos = self.find_player_pos(col_idx, player)
        step = self.b[col_idx][pos:].index(0)
        for i, x in enumerate(self.b[col_idx]):
            if x == self.neutral[player]:
                self.b[col_idx][i] = 0
        self.place_neutral(col_idx, player, step + pos, save)

    def place_piece(self, col_idx: int, player: int):
        pos = self.b[col_idx].index(0)
        self.place_neutral(col_idx, player, pos, True)

    def check_won_column(self, col_idx, player):
        rem = (max(self.col_lens) - self.col_lens[col_idx]) // 2
        last_slot = rem + self.col_lens[col_idx] - 1
        if self.b[col_idx][last_slot] != 0:
            self.won_col[col_idx] = player

    def update_winner(self, player):
        count = len([x for x in self.won_col if x == player])
        if count >= 3:
            self.winner = player
            self.won = True
