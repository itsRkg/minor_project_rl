from gymnasium.vector import SyncVectorEnv
from envs.wrappers import make_env

def make_vec_env(env_id: str, n_envs: int, seed: int = 0):
    """
    Create vectorized environment with SyncVectorEnv.

    ```
    Args:
        env_id: environment name
        n_envs: number of parallel envs
        seed: base seed

    Returns:
        vectorized environment
    """

    env_fns = []

    for i in range(n_envs):
        # Each env gets a different seed
        env_seed = seed + i
        env_fns.append(make_env(env_id, env_seed))

    # Create vectorized env
    vec_env = SyncVectorEnv(env_fns)

    return vec_env
