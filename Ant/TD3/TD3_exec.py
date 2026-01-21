import gymnasium as gym
import torch 
import torch.nn as nn
import torch.optim as optim 
import numpy as np

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # GPU 사용할 시
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size=100000, device=DEVICE):
        self.device = device
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.acts_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.max_size = size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size=128):
        idxs = np.random.randint(0, self.size, size=batch_size)
        obs = torch.as_tensor(self.obs_buf[idxs], dtype=torch.float32, device=self.device)
        acts = torch.as_tensor(self.acts_buf[idxs], dtype=torch.float32, device=self.device)
        rews = torch.as_tensor(self.rews_buf[idxs], dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_obs = torch.as_tensor(self.next_obs_buf[idxs], dtype=torch.float32, device=self.device)
        done = torch.as_tensor(self.done_buf[idxs], dtype=torch.float32, device=self.device).unsqueeze(-1)
        return obs, acts, rews, next_obs, done
    
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(256, 256), act=nn.ReLU):
        super().__init__()
        layers = []
        last = in_dim #입력 차원
        for h in hidden: #히든 리스트에서 하나씩 빼서 입력, 출력에 맞게 신경망 층 생성 
            layers += [nn.Linear(last, h), act()] # 층 생성과 동시에 활성화 함수를 붙임
            last = h
        layers += [nn.Linear(last, out_dim)] #마지막은 출력 차원과 맞게 출력층 추가
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
class Actor(nn.Module):
    def __init__(self, obs_dim, hidden=(256, 256)):
        super().__init__()
        self.body = MLP(obs_dim, 1, hidden)
        self.tanh = nn.Tanh()

    def forward(self, obs):
        return self.tanh(self.body(obs))  #[-1,1]
class Critic(nn.Module): 
    """Q(s, a_aim) where a_aim is 1-D (dx)"""
    def __init__(self, obs_dim, hidden=(256, 256)):
        super().__init__()
        self.q = MLP(obs_dim + 1, 1, hidden)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1) # 두 벡터를 하나로 합쳐서 새로운 입력으로 만듦
        return self.q(x)
class OUNoise:
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.zeros(size, dtype=np.float32)

    def reset(self):
        self.state[:] = 0.0

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.size).astype(np.float32)
        self.state += dx
        return self.state

class TD3Agent:
    def __init__(
        self,
        obs_dim: int,
        actor_hidden=(256, 256),
        critic_hidden=(256, 256),
        actor_lr=1e-3,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.005,
        device: torch.device = DEVICE,
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau

        self.actor = Actor(obs_dim, actor_hidden).to(device)
        self.actor_target = Actor(obs_dim, actor_hidden).to(device)
        self.critic1 = Critic(obs_dim, critic_hidden).to(device)
        self.critic2 = Critic(obs_dim, critic_hidden).to(device)
        self.critic1_target = Critic(obs_dim, critic_hidden).to(device)
        self.critic2_target = Critic(obs_dim, critic_hidden).to(device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=critic_lr)

        self.policy_delay = 2
        self.total_it = 0

    @torch.no_grad()
    def act(self, obs: np.ndarray, noise: np.ndarray | None = None): #행동을 뽑는 단계
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0) #관측값이 numpy 여서 tensor로 전환
        a = self.actor(o).squeeze(0)                 # shape (1,)
        a = a.clamp(-1.0, 1.0)
        a_np = a.detach().cpu().numpy() # 데이터 numpy로 전환
        if noise is not None:
            a_np = np.clip(a_np + noise, -1.0, 1.0) #
        return a_np.astype(np.float32)               # (dx,)

    def update(self, replay: ReplayBuffer, batch_size=128):
        self.total_it += 1
        obs, action, rew, next_obs, done = replay.sample(batch_size)

        # --- Critic update ---
        with torch.no_grad():
            # Target policy smoothing
            next_action = self.actor_target(next_obs)
            noise = (0.2 * torch.randn_like(next_action)).clamp(-0.5, 0.5)
            next_action = (self.actor_target(next_obs) + noise).clamp(-1.0, 1.0)
            # Clipped Double Q
            target_q1 = self.critic1_target(next_obs, next_action)
            target_q2 = self.critic2_target(next_obs, next_action)
            target_q = torch.min(target_q1, target_q2)
            y = rew + (1.0 - done) * self.gamma * target_q

        q1 = self.critic1(obs, action)
        q2 = self.critic2(obs, action)
        critic1_loss = nn.functional.mse_loss(q1, y)
        critic2_loss = nn.functional.mse_loss(q2, y)

        self.critic1_opt.zero_grad()
        critic1_loss.backward()
        self.critic1_opt.step()
        self.critic2_opt.zero_grad()
        critic2_loss.backward()
        self.critic2_opt.step()

        actor_loss = None
        # --- Policy delay ---
        if self.total_it % self.policy_delay == 0:
            actor_loss = -self.critic1(obs, self.actor(obs)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            # --- Soft target updates for all targets ---
            with torch.no_grad():
                for p, p_t in zip(self.actor.parameters(), self.actor_target.parameters()):
                    p_t.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
                for p, p_t in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                    p_t.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
                for p, p_t in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                    p_t.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

        return {
            "critic1_loss": float(critic1_loss.detach().cpu().item()),
            "critic2_loss": float(critic2_loss.detach().cpu().item()),
            "actor_loss": float(actor_loss.detach().cpu().item()) if actor_loss is not None else None,
        }

def train_td3(env, agent, replay, episodes=300, start_steps=10000, batch_size=128, noise_scale=0.3, render=False):
    returns = []
    ou_noise = OUNoise(size=1)
    # total_Step = 0
    obs, _ = env.reset()
    for ep in range(episodes):
        ep_return = 0
        ou_noise.reset()
        for step in range(1000):
            # total_Step += 1
            # if total_Step % 1000 == 0:
                # print(f"[{total_Step}]")
            if replay.size < start_steps:
                action = env.action_space.sample()
            else:
                noise = ou_noise.sample() * noise_scale
                action = agent.act(obs, noise=noise)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            replay.store(obs, action, reward, next_obs, done)
            obs = next_obs
            ep_return += reward
            if replay.size > batch_size:
                agent.update(replay, batch_size=batch_size)
            if done:
                break
        print(f"[Episode {ep+1}] Return: {ep_return:.2f}")
        obs, _ = env.reset()
        returns.append(ep_return)
        # Save episode returns progressively
        import pandas as pd
        df = pd.DataFrame({"episode": list(range(1, len(returns)+1)), "reward": returns})
        df.to_excel("./mountain_car_continuous/TD3/episode_rewards.xlsx", index=False)
    return returns

if __name__ == "__main__":
    env = gym.make("MountainCarContinuous-v0", render_mode=None)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = TD3Agent(obs_dim=obs_dim)
    replay = ReplayBuffer(obs_dim, act_dim, size=200000)
    train_td3(env, agent, replay, episodes=200, start_steps=10000, batch_size=128, noise_scale=1.0, render=False)

    # --- Save trained parameters ---
    torch.save(agent.actor.state_dict(), "./mountain_car_continuous/TC3/actor.pth")
    # Save critic1 only for compatibility (or save both if needed)
    torch.save(agent.critic1.state_dfict(), "./mountain_car_continuous/TC3/critic.pth")
    print("모델 저장 완료: actor.pth , critic.pth")

    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_excel("episode_rewards.xlsx")
    plt.plot(df["episode"], df["reward"])
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("TD3 Training – Episode Rewards")
    plt.grid(True)
    plt.show()