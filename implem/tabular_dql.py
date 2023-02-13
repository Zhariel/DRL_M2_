import numpy as np
import random
import matplotlib.pyplot as plt

# Paramètres de l'environnement
nb_state = 6 # Nombre d'états
nb_action = 2 # Nombre d'actions
terminal_states = [0, nb_state-1] # États terminaux

# Paramètres de l'algorithme
alpha = 0.1 # Taux d'apprentissage
gamma = 0.9 # Taux d'actualisation
epsilon = 0.3 # Taux de la stratégie epsilon-greedy
n_episodes = 1000 # Nombre d'épisodes

# Initialisation de la table Q
Q = np.zeros((nb_state, nb_action))

def get_action(state, epsilon):
    """Sélection d'action avec la stratégie epsilon-greedy"""
    if random.uniform(0, 1) < epsilon:
        # Choisir aléatoirement une action
        action = random.choice(range(nb_action))
    else:
        # Choisir l'action avec la valeur Q maximale
        action = np.argmax(Q[state, :])
    return action

def update_state(state, action):
    reward = 0
    """Mettre à jour l'état en fonction de l'action choisie"""
    if action == 0:
        # Action gauche
        next_state = max(state-1, 0)
    else:
        # Action droite
        next_state = min(state+1, nb_state-1)
    if (next_state == 0) :
        # Récompense supplémentaire pour les états terminaux
        reward = -1
    if (next_state== nb_state-1):
        # Récompense supplémentaire pour les états terminaux
        reward = 1


    return next_state, reward

# Initialisation des tableaux pour stocker les scores et le nombre d'étapes
scores = []
steps = []
ema_score = 0.0
# Boucle principale de l'algorithme
for episode in range(n_episodes):
    state = random.choice(range(1, nb_state-1)) # Choisir aléatoirement un état initial
    score = 0
    step = 0
    while state not in terminal_states:
        action = get_action(state, epsilon) # Sélectionner une action
        next_state, reward = update_state(state, action) # Mettre à jour l'état
        score += reward # Mettre à jour le score
        if (episode ==0) :
            ema_score = 0.0
            ema_nb_steps = step
        else :
            ema_score = (1 - 0.999) * reward  + 0.999 * ema_score
            ema_nb_steps = (1 - 0.999) * step + 0.999 * ema_nb_steps
        scores.append(ema_score)
        steps.append(ema_nb_steps)

        Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action]) # Mettre à jour la table Q
        state = next_state # Passer à l'état suivant
        step += 1 # Mettre à jour le nomb


plt.plot(scores)
plt.show()
plt.plot(steps)
plt.show()