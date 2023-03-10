import math
import matplotlib.pyplot as plt
import numpy as np
import numpy.random
import tensorflow as tf
from tqdm import tqdm
from implem import write_log
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"




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


class LineWorld(DeepSingleAgentEnv):
    def __init__(self, nb_cells: int = 5):
        self.nb_cells = nb_cells
        self.current_cell = math.floor(nb_cells / 2)
        self.step_count = 0

    def max_action_count(self) -> int:
        return 2

    def state_description(self) -> np.ndarray:
        return np.array([self.current_cell / (self.nb_cells - 1) * 2.0 - 1.0])

    def state_dim(self) -> int:
        return 1

    def is_game_over(self) -> bool:
        if self.step_count > self.nb_cells ** 2:
            return True
        return self.current_cell == 0 or self.current_cell == self.nb_cells - 1

    def act_with_action_id(self, action_id: int):
        self.step_count += 1
        if action_id == 0:
            self.current_cell -= 1
        else:
            self.current_cell += 1

    def score(self) -> float:
        if self.current_cell == 0:
            return -1.0
        elif self.current_cell == self.nb_cells - 1:
            return 1.0
        else:
            return 0.0

    def available_actions_ids(self) -> np.ndarray:
        return np.array([0, 1])

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


def REINFORCE_with_learned_baseline(env: DeepSingleAgentEnv, max_iter_count: int = 10000,
                                    gamma: float = 0.99,
                                    alpha_pi: float = 0.01,
                                    alpha_v: float = 0.01):
    pi = tf.keras.models.Sequential()
    pi.add(tf.keras.layers.Dense(env.max_action_count(),
                                 activation=tf.keras.activations.softmax,
                                 use_bias=True
                                 ))

    v = tf.keras.models.Sequential()
    v.add(tf.keras.layers.Dense(1,
                                activation=tf.keras.activations.linear,
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

                with tf.GradientTape() as tape_v:
                    v_s_pred = v(np.array([episode_states_buffer[t]]))[0][0]

                delta = G - tf.constant(v_s_pred)

                grad_v_s_pred = tape_v.gradient(v_s_pred, v.trainable_variables)
                for (var, grad) in zip(v.trainable_variables, grad_v_s_pred):
                    if grad is not None:
                        var.assign_add(alpha_v * delta * grad)

                with tf.GradientTape() as tape_pi:
                    pi_s_a_t = pi(np.array([episode_states_buffer[t]]))[0][episode_actions_buffer[t]]
                    log_pi_s_a_t = tf.math.log(pi_s_a_t)

                grads = tape_pi.gradient(log_pi_s_a_t, pi.trainable_variables)

                for (var, grad) in zip(pi.trainable_variables, grads):
                    if grad is not None:
                        var.assign_add(alpha_pi * (gamma ** t) * delta * grad)

            if first_episode:
                ema_score = env.score()
                ema_nb_steps = step
                first_episode = False
            else:
                ema_score = (1 - 0.95) * env.score() + 0.95 * ema_score
                ema_nb_steps = (1 - 0.95) * step + 0.95 * ema_nb_steps
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
    return pi, v, ema_score_progress, ema_nb_steps_progress

gamma = 0.95
alpha = 0.05
max_iter_count = 10000
pi, v, scores, steps = REINFORCE_with_learned_baseline(LineWorld(10), max_iter_count=10000)
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
