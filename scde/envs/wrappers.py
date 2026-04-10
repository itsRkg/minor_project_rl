import gymnasium as gym
from gymnasium import spaces
from minigrid.wrappers import RGBImgObsWrapper
import numpy as np
import cv2

class ImageOnlyWrapper(gym.ObservationWrapper):
    """
    Extract only the 'image' key from MiniGrid dict observation.
    """
    def __init__(self, env): 
        super().__init__(env) 

        # Extract image space from dict 
        self.observation_space = env.observation_space["image"]

    def observation(self, obs):
        return obs["image"]

class ResizeWrapper(gym.ObservationWrapper):
    """
    Resize observation to fixed size (e.g., 64x64).
    This ensures consistency across environments.
    """

    def __init__(self, env, size=(64, 64)):
        super().__init__(env)
        self.size = size

        # Update observation space
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(size[0], size[1], 3),
            dtype=np.uint8
        )

    def observation(self, obs): 
        obs = cv2.resize(obs, self.size, interpolation=cv2.INTER_AREA)
        return obs

def make_env(env_id: str, seed: int):
    """
    Create a single MiniGrid environment with proper wrappers.

    ```
    Args:
        env_id: name of the environment
        seed: random seed for reproducibility

    Returns:
        Callable that creates a wrapped env (needed for vector env)
    """

    def _init():
        # 1. Create base env
        env = gym.make(env_id)

        # 2. Apply RGB wrapper
        # Converts symbolic obs → RGB image (64×64×3 uint8)
        env = RGBImgObsWrapper(env)

        # 3. Extract only image (remove dict) 
        env = ImageOnlyWrapper(env)

        # 4. Resize observation 64 x 64 
        env = ResizeWrapper(env, size=(64, 64))

        # 5. Set seed
        env.reset(seed=seed)

        return env

    return _init

