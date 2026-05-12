import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper

def make_env(render_mode=None):
    env = gym.make("MiniGrid-DoorKey-8x8-v0", render_mode=render_mode)
    # DQN usually works better with a flattened observation if we're not using a CNN
    # For MiniGrid, FlatObsWrapper provides a fully observable view as a flat vector
    # However, standard MiniGrid is partially observable. 
    # Let's start with FlatObsWrapper for simplicity.
    env = FlatObsWrapper(env)
    return env

if __name__ == "__main__":
    env = make_env(render_mode="human")
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    # Take a few random steps
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action}, Reward: {reward}, Terminated: {terminated}")
        if terminated or truncated:
            break
    env.close()
