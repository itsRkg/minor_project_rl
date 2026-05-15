import torch


class RolloutBuffer:
    def __init__(
        self,
        n_steps,
        n_envs,
        device="cpu",
    ):
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

        # NEW
        self.h = []

        self.dones = []

    def add(
        self,
        obs,
        action,
        log_prob,
        r_ext,
        r_int,
        v_ext,
        v_int,
        h,
        done,
    ):

        self.obs.append(obs.to(self.device)) # type: ignore

        self.actions.append(action.to(self.device)) # type: ignore

        self.log_probs.append( # type: ignore
            log_prob.to(self.device)
        )

        self.r_ext.append(r_ext.to(self.device))
        self.r_int.append(r_int.to(self.device))

        self.v_ext.append(v_ext.to(self.device))
        self.v_int.append(v_int.to(self.device))

        # NEW
        self.h.append(h.to(self.device)) # type: ignore

        self.dones.append(done.to(self.device))

    def compute_returns_and_advantages(
        self,
        last_v_ext,
        last_v_int,
        gamma=0.999,
        gamma_int=0.99,
        lam=0.95,
    ):

        obs = torch.stack(self.obs) # type: ignore

        actions = torch.stack(self.actions) # type: ignore

        log_probs = torch.stack(self.log_probs) # type: ignore

        r_ext = torch.stack(self.r_ext)
        r_int = torch.stack(self.r_int)

        v_ext = torch.stack(self.v_ext)
        v_int = torch.stack(self.v_int)

        h = torch.stack(self.h) # type: ignore

        dones = torch.stack(self.dones)

        T = self.n_steps

        adv_ext = torch.zeros_like(
            r_ext,
            device=self.device,
        )

        adv_int = torch.zeros_like(
            r_int,
            device=self.device,
        )

        last_adv_ext = torch.zeros(
            self.n_envs,
            device=self.device,
        )

        last_adv_int = torch.zeros(
            self.n_envs,
            device=self.device,
        )

        last_v_ext = last_v_ext.to(self.device)
        last_v_int = last_v_int.to(self.device)

        for t in reversed(range(T)):

            if t == T - 1:
                next_v_ext = last_v_ext
                next_v_int = last_v_int
            else:
                next_v_ext = v_ext[t + 1]
                next_v_int = v_int[t + 1]

            # -----------------------
            # EXTRINSIC GAE
            # -----------------------
            mask = 1.0 - dones[t].float()

            delta_ext = (
                r_ext[t]
                + gamma * next_v_ext * mask
                - v_ext[t]
            )

            last_adv_ext = (
                delta_ext
                + gamma * lam * mask * last_adv_ext
            )

            adv_ext[t] = last_adv_ext

            # -----------------------
            # INTRINSIC GAE
            # NO DONE MASK
            # -----------------------
            delta_int = (
                r_int[t]
                + gamma_int * next_v_int
                - v_int[t]
            )

            last_adv_int = (
                delta_int
                + gamma_int * lam * last_adv_int
            )

            adv_int[t] = last_adv_int

        returns_ext = adv_ext + v_ext
        returns_int = adv_int + v_int

        # Combined advantage
        adv_total = adv_ext + adv_int

        # Normalize
        adv_total = (
            (adv_total - adv_total.mean())
            / (adv_total.std() + 1e-8)
        )

        # Flatten
        self.obs = obs.reshape(
            -1,
            *obs.shape[2:],
        )

        self.actions = actions.reshape(-1)

        self.log_probs = log_probs.reshape(-1)

        self.returns_ext = returns_ext.reshape(-1)
        self.returns_int = returns_int.reshape(-1)

        self.h = h.reshape(
            -1,
            h.shape[-1],
        )

        self.adv = adv_total.reshape(-1)

    def get_batches(
        self,
        batch_size,
    ):

        n = self.obs.size(0) # type: ignore

        indices = torch.randperm(
            n,
            device=self.device,
        )

        for start in range(
            0,
            n,
            batch_size,
        ):

            end = start + batch_size

            batch_idx = indices[start:end]

            yield {
                "obs": self.obs[batch_idx],
                "actions": self.actions[batch_idx],
                "log_probs": self.log_probs[batch_idx],
                "returns_ext": self.returns_ext[batch_idx],
                "returns_int": self.returns_int[batch_idx],
                "adv": self.adv[batch_idx],

                # NEW
                "h": self.h[batch_idx],
            }