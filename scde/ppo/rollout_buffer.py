import torch


class RolloutBuffer:
    def __init__(self, n_steps, n_envs, device="cpu"):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.device = device
        self.reset()

    def reset(self):
        self.obs = []
        self.actions = []
        self.log_probs = []

        self.r_ext = []
        self.r_int = []

        self.v_ext = []
        self.v_int = []

        self.dones = []

    def add(self, obs, action, log_prob, r_ext, r_int, v_ext, v_int, done):
        # Move to device immediately (prevents CPU/GPU mismatch later)
        self.obs.append(obs.to(self.device)) # pyright: ignore[reportAttributeAccessIssue]
        self.actions.append(action.to(self.device)) # pyright: ignore[reportAttributeAccessIssue]
        self.log_probs.append(log_prob.to(self.device)) # pyright: ignore[reportAttributeAccessIssue]

        self.r_ext.append(r_ext.to(self.device))
        self.r_int.append(r_int.to(self.device))

        self.v_ext.append(v_ext.to(self.device))
        self.v_int.append(v_int.to(self.device))

        self.dones.append(done.to(self.device))

    def compute_returns_and_advantages(
        self,
        last_v_ext,
        last_v_int,
        gamma=0.999,
        lam=0.95,
    ):
        # Convert lists → tensors (already on correct device)
        obs = torch.stack(self.obs)              # pyright: ignore[reportArgumentType] # (T, B, ...)
        actions = torch.stack(self.actions)      # pyright: ignore[reportArgumentType] # (T, B)
        log_probs = torch.stack(self.log_probs)  # pyright: ignore[reportArgumentType] # (T, B)

        r_ext = torch.stack(self.r_ext)          # (T, B)
        r_int = torch.stack(self.r_int)          # (T, B)

        v_ext = torch.stack(self.v_ext)          # (T, B)
        v_int = torch.stack(self.v_int)          # (T, B)

        dones = torch.stack(self.dones)          # (T, B)

        T = self.n_steps

        adv_ext = torch.zeros_like(r_ext, device=self.device)
        adv_int = torch.zeros_like(r_int, device=self.device)

        last_adv_ext = torch.zeros(self.n_envs, device=self.device)
        last_adv_int = torch.zeros(self.n_envs, device=self.device)

        # Ensure last values are on correct device
        last_v_ext = last_v_ext.to(self.device)
        last_v_int = last_v_int.to(self.device)

        for t in reversed(range(T)):
            if t == T - 1:
                next_v_ext = last_v_ext
                next_v_int = last_v_int
            else:
                next_v_ext = v_ext[t + 1]
                next_v_int = v_int[t + 1]

            # 🔵 Extrinsic (episodic)
            mask = 1.0 - dones[t].float()

            delta_ext = r_ext[t] + gamma * next_v_ext * mask - v_ext[t]
            last_adv_ext = delta_ext + gamma * lam * mask * last_adv_ext
            adv_ext[t] = last_adv_ext

            # 🔴 Intrinsic (non-episodic → NO mask)
            delta_int = r_int[t] + gamma * next_v_int - v_int[t]
            last_adv_int = delta_int + gamma * lam * last_adv_int
            adv_int[t] = last_adv_int

        # Returns
        returns_ext = adv_ext + v_ext
        returns_int = adv_int + v_int

        # Total advantage
        adv_total = adv_ext + adv_int

        # Normalize advantage
        adv_total = (adv_total - adv_total.mean()) / (adv_total.std() + 1e-8)

        # Flatten (T, B) → (T*B)
        self.obs = obs.reshape(-1, *obs.shape[2:])
        self.actions = actions.reshape(-1)
        self.log_probs = log_probs.reshape(-1)

        self.returns_ext = returns_ext.reshape(-1)
        self.returns_int = returns_int.reshape(-1)

        self.adv = adv_total.reshape(-1)

    def get_batches(self, batch_size):
        n = self.obs.size(0) # pyright: ignore[reportAttributeAccessIssue]
        indices = torch.randperm(n, device=self.device)

        for start in range(0, n, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]

            yield {
                "obs": self.obs[batch_idx],
                "actions": self.actions[batch_idx],
                "log_probs": self.log_probs[batch_idx],
                "returns_ext": self.returns_ext[batch_idx],
                "returns_int": self.returns_int[batch_idx],
                "adv": self.adv[batch_idx],
            }