import numpy as np
import copy
from scipy import stats
from tqdm import tqdm

def collect_episodes(mdp, policy=None, horizon=None, n_episodes=1, render=False):
    paths = []

    for _ in range(n_episodes):
        observations = np.zeros(horizon)
        actions = np.zeros(horizon)
        rewards = np.zeros(horizon)
        next_states = np.zeros(horizon)

        state = mdp.reset()
        for i in range(horizon):
            action = policy.draw_action(state)
            next_state, reward, terminal, _ = mdp.step(action)
            if render:
                mdp.render()
            observations[i] = state
            actions[i] = action
            rewards[i] = reward
            next_states[i] = next_state
            state = copy.copy(next_state)
            if terminal:
                # Finish rollout if terminal state reached
                break
                # We need to compute the empirical return for each time step along the
                # trajectory

        paths.append(dict(
            states=np.array(observations[:i+1]),
            actions=np.array(actions[:i+1]),
            rewards=np.array(rewards[:i+1]),
            next_states=np.array(next_states[:i+1])
        ))
    return paths


def estimate_performance(mdp, policy=None, horizon=None, n_episodes=1, gamma=0.9):
    paths = collect_episodes(mdp, policy, horizon, n_episodes)

    J = 0.
    for p in tqdm(paths):
        df = 1
        sum_r = 0.
        for r in p["rewards"]:
            sum_r += df * r
            df *= gamma
        J += sum_r
    return J / n_episodes


def discretization_2d(x, y, binx, biny):
    _, _, _, binid = stats.binned_statistic_2d(x, y, None, 'count', bins=[binx, biny])
    return binid
