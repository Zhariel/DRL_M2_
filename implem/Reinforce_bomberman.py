import math
import matplotlib.pyplot as plt
import numpy as np
import numpy.random
import tensorflow as tf
from numba import jit
from tqdm import tqdm
from random import *
from implem import write_log
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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

        return np.array(actions)

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
            [1 if h % 2 == 1 and w % 2 == 1 else (0 if random() < 0.85 else 2) for h in range(self.game_size)]
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
            print()

    def bomb_explode(self, origin):
        self.board[origin] = 0
        for i in range(4):
            for _ in range(self.bomb_strength):
                s_p = origin + self.a_effects[i]
                if self.P[origin, i, s_p] == 1:
                    if self.board[s_p] == 1:
                        break
                    if self.board[s_p] == 3 or self.board[s_p] == 6 or self.board[s_p] == 2:
                        self.board[s_p] = 0

    def frame(self, player_a):
        for b in self.bombs:
            b.ticks -= 1

        if 3 not in self.board:
            self.won = -1
            return
        if 6 not in self.board:
            self.won = 1
            return

        for b in range(len(self.bombs)-1, -1, -1):
            if self.bombs[b].ticks == 0:
                self.bomb_explode(origin=self.bombs[b].pos)
                self.bombs.pop(b)

        if player_a == 4:
            self.place_bomb()

        else:  # move player
            old_pos = self.position
            s_p = self.position + self.a_effects[player_a]
            if self.P[self.position, player_a, s_p] == 1:
                if self.board[s_p] == 6:
                    self.won = -1
                    return
                if self.board[s_p] == 0 or self.board[s_p] == 5:
                    if self.board[s_p] == 5:
                        self.bomb_strength += 1
                    self.board[self.position] = 0
                    self.position = s_p
                    self.board[s_p] = 3
                    for b in self.bombs:
                        if b.pos == old_pos:
                            self.board[old_pos] = 4
            else:
                print("CANT MOVE")

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

def REINFORCE(env: Bomberman, max_iter_count: int = 10000,
              gamma: float = 0.99,
              alpha: float = 0.1):
    pi = tf.keras.models.Sequential()
    pi.add(tf.keras.layers.Dense(env.max_action_count(),
                                 activation=tf.keras.activations.softmax,
                                 use_bias=True
                                 ))

    ema_score = 0.0
    ema_nb_steps = 0.0
    first_episode = True

    step = 0
    ema_score_progress = []
    ema_nb_steps_progress = []

    episode_states_buffer = []
    episode_actions_buffer = []
    episode_rewards_buffer = []

    for _ in tqdm(range(max_iter_count)):
        if env.is_game_over():
            ### TRAINING TIME !!!
            G = 0.0

            for t in reversed(range(0, len(episode_states_buffer))):
                G = episode_rewards_buffer[t] + gamma * G

                with tf.GradientTape() as tape:
                    pi_s_a_t = pi(np.array([episode_states_buffer[t]]))[0][episode_actions_buffer[t]]
                    log_pi_s_a_t = tf.math.log(pi_s_a_t)

                grads = tape.gradient(log_pi_s_a_t, pi.trainable_variables)

                for (v, g) in zip(pi.trainable_variables, grads):
                    if g is not None:
                        v.assign_add(alpha * (gamma ** t) * G * g)

            if first_episode:
                ema_score = env.score()
                ema_nb_steps = step
                first_episode = False
            else:
                ema_score = (1 - 0.9) * env.score() + 0.9 * ema_score
                ema_nb_steps = (1 - 0.9) * step + 0.9 * ema_nb_steps
                ema_score_progress.append(ema_score)
                ema_nb_steps_progress.append(ema_nb_steps)

            env.reset()
            episode_states_buffer.clear()
            episode_actions_buffer.clear()
            episode_rewards_buffer.clear()
            step = 0

        s = env.state_description()

        episode_states_buffer.append(s)

        aa = env.available_actions_ids()

        pi_s = pi(np.array([s]))[0].numpy()
        allowed_pi_s = pi_s[aa]
        sum_allowed_pi_s = np.sum(allowed_pi_s)
        if sum_allowed_pi_s == 0.0:
            probs = np.ones((len(aa),)) * 1.0 / (len(aa))
        else:
            probs = allowed_pi_s / sum_allowed_pi_s

        a = np.random.choice(aa, p=probs)

        episode_actions_buffer.append(a)

        old_score = env.score()
        env.act_with_action_id(a)
        new_score = env.score()
        r = new_score - old_score

        episode_rewards_buffer.append(r)

        step += 1

    return pi, ema_score_progress, ema_nb_steps_progress

gamma = 0.95
alpha = 0.05
max_iter_count = 10000
pi, scores, steps = REINFORCE(BombermanGame(10), max_iter_count=10000)
print(pi.weights)
plt.plot(scores)
plt.show()
plt.plot(steps)
plt.show()
scores = [str(x) for x in scores]
steps = [str(x) for x in steps]
write_log(f"REINFORCE_gamma{gamma}_alpha{alpha}_itercount{max_iter_count}.txt",
                  "score progress," + ",".join(scores) + "\n" + "steps progress," + ",".join(
                      steps))