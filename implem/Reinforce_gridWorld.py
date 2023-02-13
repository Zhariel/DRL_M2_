import math
import matplotlib.pyplot as plt
import numpy as np
import numpy.random
import tensorflow as tf
from numba import jit
from tqdm import tqdm
from implem import write_log
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class DeepSingleAgentEnv:
    def max_action_count(self) -> int:
        pass

    def state_description(self) -> np.ndarray:
        pass

    def state_dim(self) -> int:
        pass

    def is_game_over(self) -> bool:
        pass

    def act_with_action_id(self, action_id: int):
        pass

    def score(self) -> float:
        pass

    def available_actions_ids(self) -> np.ndarray:
        pass

    def reset(self):
        pass

    def view(self):
        pass


class GridWorld(DeepSingleAgentEnv):
    def __init__(self, nb_cells = 25):
        self.nb_cells = nb_cells
        self.current_cell = math.floor(nb_cells / 2)
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


def REINFORCE(env: DeepSingleAgentEnv, max_iter_count: int = 10000,
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
max_iter_count = 100
pi, scores, steps = REINFORCE(GridWorld(25), max_iter_count=100, gamma=0.95, alpha=0.05)
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