from envs.gridworld import GridWorld

g = GridWorld()

for i in range(25):
    g.current_cell = i
    print(i)
    print(g.available_actions_ids())