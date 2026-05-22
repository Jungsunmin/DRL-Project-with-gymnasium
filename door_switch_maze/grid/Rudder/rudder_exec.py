# ================================================================
# RUDDER DQN on MiniGrid-DoorKey-8x8-v0
# Paper-aligned implementation (Arjona-Medina et al., NeurIPS 2019):
#   (I)   safe exploration,
#   (II)  lessons replay buffer (prioritized by LSTM prediction error),
#   (III) LSTM return prediction + method (A): differences of predictions,
#         with return-equivalence compensation (Eq. 4).
# ================================================================

import argparse
import os
import random
from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import minigrid  # noqa: F401  # ensures MiniGrid env registration
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ================================================================
# Device Selection
# ================================================================
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)
print("==========================================")
print(f"📌 Using Device: {DEVICE}")
print("==========================================")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def preprocess_obs(obs):
    img = obs["image"]
    return img.flatten().astype(np.float32) / 255.0


def state_to_tensor(state):
    if isinstance(state, np.ndarray):
        return torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
    tensor = state.to(DEVICE)
    return tensor if tensor.dim() > 1 else tensor.unsqueeze(0)


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def compute_discounted_return(rewards, gamma=1.0):
    total = 0.0
    discount = 1.0
    for reward in rewards:
        total += discount * float(reward)
        discount *= gamma
    return total


def one_hot_action(action, action_dim):
    action_vec = torch.zeros(action_dim, dtype=torch.float32, device=DEVICE)
    action_vec[int(action)] = 1.0
    return action_vec


def soft_update(target, source, tau=0.005):
    for tp, p in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)


@dataclass
class EpisodeLesson:
    trajectory: list
    episode_return: float
    prediction_error: float


class LessonsReplayBuffer:
    """Prioritized replay of episodes with high LSTM prediction error."""

    def __init__(self, capacity=2000, alpha=0.6, eps=1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.eps = eps
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def add(self, trajectory, episode_return, prediction_error):
        lesson = EpisodeLesson(
            trajectory=trajectory,
            episode_return=float(episode_return),
            prediction_error=float(abs(prediction_error)),
        )
        if len(self.buffer) >= self.capacity:
            min_idx = int(np.argmin([item.prediction_error for item in self.buffer]))
            self.buffer[min_idx] = lesson
        else:
            self.buffer.append(lesson)

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return []
        priorities = np.array(
            [(item.prediction_error + self.eps) ** self.alpha for item in self.buffer],
            dtype=np.float64,
        )
        probs = priorities / priorities.sum()
        indices = np.random.choice(
            len(self.buffer),
            size=min(batch_size, len(self.buffer)),
            replace=len(self.buffer) < batch_size,
            p=probs,
        )
        return [self.buffer[int(i)] for i in indices]


class RUDDERStateEncoder(nn.Module):
    def __init__(self, input_dim, encoded_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, encoded_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class RUDDER_DQN(nn.Module):
    def __init__(self, input_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, action_size),
        )

    def forward(self, x):
        return self.net(x)


class LSTMReturnPredictor(nn.Module):
    """
    LSTM g that predicts sequence-wide return at every prefix (paper Sec. 3).
    Input per step: [encoded_state, one_hot_action].
    """

    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out).squeeze(-1)


def safe_epsilon_greedy(model, state, epsilon, action_space, safe_margin=0.10):
    with torch.no_grad():
        q_values = model(state).squeeze(0)

    if random.random() >= epsilon:
        return int(torch.argmax(q_values).item())

    max_q = torch.max(q_values)
    safe_mask = q_values >= (max_q - safe_margin)
    safe_actions = (
        torch.nonzero(safe_mask, as_tuple=False)
        .flatten()
        .detach()
        .cpu()
        .numpy()
        .tolist()
    )
    if len(safe_actions) == 0:
        return random.randrange(action_space)
    return int(random.choice(safe_actions))


def build_return_predictor_sequence(encoder, states, actions, action_dim):
    seq_items = []
    with torch.no_grad():
        for state, action in zip(states, actions):
            encoded_state = encoder(state_to_tensor(state)).squeeze(0)
            action_vec = one_hot_action(action, action_dim)
            seq_items.append(torch.cat([encoded_state, action_vec], dim=0))
    return torch.stack(seq_items, dim=0).unsqueeze(0).to(DEVICE)


def get_episode_return_prediction(return_predictor, encoder, trajectory, action_dim):
    if len(trajectory) == 0:
        return 0.0
    states = [item[0] for item in trajectory]
    actions = [item[1] for item in trajectory]
    x_seq = build_return_predictor_sequence(encoder, states, actions, action_dim)
    with torch.no_grad():
        pred = return_predictor(x_seq)
        return float(pred[0, -1].item())


def train_return_predictor_on_lessons(
    return_predictor,
    return_optimizer,
    encoder,
    lessons_buffer,
    action_dim,
    batch_size=8,
    gamma=1.0,
    continuous_pred_factor=0.5,
):
    """
    JKU demo style: main loss on final prefix + auxiliary loss on all prefixes.
    """
    if len(lessons_buffer) == 0:
        return 0.0

    lessons = lessons_buffer.sample(batch_size)
    losses = []

    for lesson in lessons:
        trajectory = lesson.trajectory
        states = [item[0] for item in trajectory]
        actions = [item[1] for item in trajectory]
        rewards = [item[2] for item in trajectory]

        if len(states) == 0:
            continue

        episode_return = compute_discounted_return(rewards, gamma=gamma)
        x_seq = build_return_predictor_sequence(encoder, states, actions, action_dim)

        target_return = torch.tensor(
            [[episode_return]], dtype=torch.float32, device=DEVICE
        )
        pred = return_predictor(x_seq)

        all_timestep_loss = F.mse_loss(
            pred, target_return.expand_as(pred), reduction="none"
        )
        main_loss = all_timestep_loss[0, -1]
        aux_loss = continuous_pred_factor * all_timestep_loss.mean()
        losses.append(main_loss + aux_loss)

    if len(losses) == 0:
        return 0.0

    loss = torch.stack(losses).mean()
    return_optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(return_predictor.parameters(), max_norm=1.0)
    return_optimizer.step()

    return float(loss.item())


def redistribute_rewards_by_prediction_differences(
    return_predictor,
    encoder,
    trajectory,
    action_dim,
    gamma=1.0,
    warmup=False,
):
    """
    Contribution analysis method (A): R_t = g(prefix_t) - g(prefix_{t-1}),
    with g(prefix_{-1}) = 0. Return-equivalence via last-step compensation (Eq. 4).
    """
    original_rewards = [float(item[2]) for item in trajectory]

    if warmup or len(trajectory) == 0:
        return original_rewards

    states = [item[0] for item in trajectory]
    actions = [item[1] for item in trajectory]
    episode_return = compute_discounted_return(original_rewards, gamma=gamma)

    x_seq = build_return_predictor_sequence(encoder, states, actions, action_dim)

    with torch.no_grad():
        pred = return_predictor(x_seq)
        zero = torch.zeros(1, 1, dtype=pred.dtype, device=pred.device)
        pred_padded = torch.cat([zero, pred], dim=1)
        redistributed = pred_padded[:, 1:] - pred_padded[:, :-1]
        redistributed = redistributed.squeeze(0).clone()

        residual = float(episode_return) - float(redistributed.sum().item())
        redistributed[-1] += residual

    return redistributed.cpu().numpy().astype(np.float32).tolist()


def update_dqn_from_replay(
    replay_buffer,
    batch_size,
    model,
    target_model,
    encoder,
    target_encoder,
    model_optimizer,
    encoder_optimizer,
    dqn_gamma,
):
    if len(replay_buffer) < batch_size:
        return None

    batch = random.sample(replay_buffer, batch_size)
    states, actions, rewards_b, next_states, dones = zip(*batch)

    states = torch.FloatTensor(np.stack(states)).to(DEVICE)
    next_states = torch.FloatTensor(np.stack(next_states)).to(DEVICE)
    actions = torch.LongTensor(actions).to(DEVICE)
    rewards_b = torch.FloatTensor(rewards_b).to(DEVICE)
    dones = torch.FloatTensor(dones).to(DEVICE)

    s_enc = encoder(states)
    with torch.no_grad():
        ns_enc_online = encoder(next_states)
        ns_enc_target = target_encoder(next_states)
        next_q_online = model(ns_enc_online)
        next_actions = torch.argmax(next_q_online, dim=1)
        next_q_target = target_model(ns_enc_target)
        next_values = next_q_target.gather(1, next_actions.view(-1, 1)).squeeze(1)
        targets = rewards_b + dqn_gamma * next_values * (1 - dones)

    q_vals = model(s_enc).gather(1, actions.view(-1, 1)).squeeze(1)
    dqn_loss = F.mse_loss(q_vals, targets)

    model_optimizer.zero_grad()
    encoder_optimizer.zero_grad()
    dqn_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3)
    torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=0.3)
    model_optimizer.step()
    encoder_optimizer.step()

    soft_update(target_model, model, tau=0.005)
    soft_update(target_encoder, encoder, tau=0.005)

    return float(dqn_loss.item())


def train_dqn(
    env,
    model,
    target_model,
    encoder,
    target_encoder,
    return_predictor,
    encoder_optimizer,
    model_optimizer,
    return_optimizer,
    num_episodes=1000,
    env_gamma=1.0,
    dqn_gamma=0.0,
    epsilon_start=1.0,
    epsilon_min=0.05,
    batch_size=128,
    replay_start_size=256,
    dqn_updates_per_episode=4,
    rudder_warmup_episodes=100,
    lessons_capacity=2000,
    lessons_batch_size=8,
    return_train_updates=4,
    continuous_pred_factor=0.5,
    safe_margin=0.10,
):
    replay_buffer = deque(maxlen=50000)
    lessons_buffer = LessonsReplayBuffer(capacity=lessons_capacity)
    all_rewards, dqn_losses, return_losses = [], [], []
    epsilon = epsilon_start
    current_dir = os.path.dirname(os.path.abspath(__file__))

    for episode in range(num_episodes):
        obs, _ = env.reset()
        state_arr = preprocess_obs(obs)
        done, total_reward = False, 0.0
        episode_trajectory = []

        while not done:
            state_tensor = state_to_tensor(state_arr)
            with torch.no_grad():
                encoded_state = encoder(state_tensor)

            action = safe_epsilon_greedy(
                model=model,
                state=encoded_state,
                epsilon=epsilon,
                action_space=env.action_space.n,
                safe_margin=safe_margin,
            )

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += float(reward)
            next_state_arr = preprocess_obs(next_obs)

            episode_trajectory.append(
                (
                    state_arr.copy(),
                    int(action),
                    float(reward),
                    next_state_arr.copy(),
                    float(done),
                )
            )
            state_arr = next_state_arr

        episode_return = compute_discounted_return(
            [item[2] for item in episode_trajectory], gamma=env_gamma
        )
        pred_before = get_episode_return_prediction(
            return_predictor=return_predictor,
            encoder=encoder,
            trajectory=episode_trajectory,
            action_dim=env.action_space.n,
        )
        prediction_error = episode_return - pred_before

        lessons_buffer.add(
            trajectory=episode_trajectory,
            episode_return=episode_return,
            prediction_error=prediction_error,
        )

        update_losses = []
        for _ in range(return_train_updates):
            loss_val = train_return_predictor_on_lessons(
                return_predictor=return_predictor,
                return_optimizer=return_optimizer,
                encoder=encoder,
                lessons_buffer=lessons_buffer,
                action_dim=env.action_space.n,
                batch_size=lessons_batch_size,
                gamma=env_gamma,
                continuous_pred_factor=continuous_pred_factor,
            )
            update_losses.append(loss_val)
        return_loss = float(np.mean(update_losses)) if update_losses else 0.0
        return_losses.append(return_loss)

        use_warmup = episode < rudder_warmup_episodes
        redistributed_rewards = redistribute_rewards_by_prediction_differences(
            return_predictor=return_predictor,
            encoder=encoder,
            trajectory=episode_trajectory,
            action_dim=env.action_space.n,
            gamma=env_gamma,
            warmup=use_warmup,
        )

        for transition, redistributed_reward in zip(
            episode_trajectory, redistributed_rewards
        ):
            state, action, _, next_state, done_flag = transition
            replay_buffer.append(
                (state, action, float(redistributed_reward), next_state, done_flag)
            )

        if len(replay_buffer) >= replay_start_size:
            for _ in range(dqn_updates_per_episode):
                loss_val = update_dqn_from_replay(
                    replay_buffer=replay_buffer,
                    batch_size=batch_size,
                    model=model,
                    target_model=target_model,
                    encoder=encoder,
                    target_encoder=target_encoder,
                    model_optimizer=model_optimizer,
                    encoder_optimizer=encoder_optimizer,
                    dqn_gamma=dqn_gamma,
                )
                if loss_val is not None:
                    dqn_losses.append(loss_val)

        all_rewards.append(total_reward)
        epsilon = max(
            epsilon_min,
            epsilon_start - (epsilon_start - epsilon_min) * (episode / max(1, num_episodes)),
        )

        if episode % 10 == 0:
            mean_reward = (
                np.mean(all_rewards[-50:]) if len(all_rewards) >= 50 else np.mean(all_rewards)
            )
            mean_dqn_loss = np.mean(dqn_losses[-100:]) if dqn_losses else 0.0
            mean_return_loss = np.mean(return_losses[-50:]) if return_losses else 0.0
            mode = "ENV" if use_warmup else "RUDDER-A"
            print(
                f"[Ep {episode:03d}] Reward: {total_reward:.2f} | Mean(50): {mean_reward:.2f} | "
                f"Eps: {epsilon:.3f} | DQN Loss: {mean_dqn_loss:.4f} | "
                f"Return Loss: {mean_return_loss:.4f} | Lessons: {len(lessons_buffer)} | "
                f"PredErr: {prediction_error:.4f} | RewardMode: {mode}"
            )

        if (episode + 1) % 50 == 0:
            df = pd.DataFrame(
                {"episode": range(1, len(all_rewards) + 1), "score": all_rewards}
            )
            df.to_excel(
                os.path.join(current_dir, "episode_rewards_rudder_lstm.xlsx"),
                index=False,
            )

            torch.save(model.state_dict(), os.path.join(current_dir, "rudder_lstm_model.pth"))
            torch.save(
                encoder.state_dict(),
                os.path.join(current_dir, "rudder_lstm_encoder.pth"),
            )
            torch.save(
                target_encoder.state_dict(),
                os.path.join(current_dir, "rudder_lstm_target_encoder.pth"),
            )
            torch.save(
                return_predictor.state_dict(),
                os.path.join(current_dir, "rudder_lstm_return_predictor.pth"),
            )

            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(all_rewards, alpha=0.3, label="Raw Environment Reward")
            if len(all_rewards) >= 10:
                smooth_scores = pd.Series(all_rewards).rolling(window=10).mean()
                plt.plot(smooth_scores, linewidth=2, label="Smoothed Reward (MA 10)")
            plt.title("RUDDER Training Scores")
            plt.xlabel("Episode")
            plt.ylabel("Score")
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(dqn_losses, label="DQN Mini-Batch Loss")
            if return_losses:
                plt.plot(
                    return_losses,
                    label="LSTM Return Predictor Loss",
                    alpha=0.7,
                )
            plt.title("Losses During Training")
            plt.xlabel("Training Step / Episode")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(current_dir, "scores_rudder_lstm.png"))
            plt.close()

    df = pd.DataFrame({"episode": range(1, len(all_rewards) + 1), "score": all_rewards})
    df.to_excel(
        os.path.join(current_dir, "episode_rewards_rudder_lstm.xlsx"), index=False
    )

    return all_rewards, dqn_losses, return_losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--epsilon_min", type=float, default=0.05)
    parser.add_argument(
        "--env_gamma",
        type=float,
        default=1.0,
        help="Discount for episode return (paper finite-horizon experiments use gamma=1).",
    )
    parser.add_argument(
        "--dqn_gamma",
        type=float,
        default=0.0,
        help="Bootstrap gamma for Q-learning on redistributed rewards (0 matches RUDDER type-C).",
    )
    parser.add_argument("--lessons_capacity", type=int, default=2000)
    parser.add_argument("--lessons_batch_size", type=int, default=8)
    parser.add_argument("--return_train_updates", type=int, default=4)
    parser.add_argument(
        "--continuous_pred_factor",
        type=float,
        default=0.5,
        help="Weight for auxiliary LSTM loss on all prefixes (JKU demo).",
    )
    parser.add_argument("--replay_start_size", type=int, default=256)
    parser.add_argument("--dqn_updates_per_episode", type=int, default=4)
    parser.add_argument("--safe_margin", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    env = gym.make("MiniGrid-DoorKey-8x8-v0")
    input_dim = 7 * 7 * 3
    action_dim = int(env.action_space.n)
    encoded_dim = 64

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "rudder_lstm_model.pth")
    encoder_path = os.path.join(current_dir, "rudder_lstm_encoder.pth")
    target_encoder_path = os.path.join(current_dir, "rudder_lstm_target_encoder.pth")
    return_predictor_path = os.path.join(
        current_dir, "rudder_lstm_return_predictor.pth"
    )

    encoder = RUDDERStateEncoder(input_dim=input_dim, encoded_dim=encoded_dim).to(
        DEVICE
    )
    target_encoder = RUDDERStateEncoder(
        input_dim=input_dim, encoded_dim=encoded_dim
    ).to(DEVICE)
    model = RUDDER_DQN(input_size=encoded_dim, action_size=action_dim).to(DEVICE)
    target_model = RUDDER_DQN(input_size=encoded_dim, action_size=action_dim).to(
        DEVICE
    )
    return_predictor = LSTMReturnPredictor(
        input_dim=encoded_dim + action_dim, hidden_dim=128
    ).to(DEVICE)

    encoder.apply(init_weights)
    target_encoder.load_state_dict(encoder.state_dict())
    model.apply(init_weights)
    target_model.load_state_dict(model.state_dict())
    return_predictor.apply(init_weights)

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=1e-4)
    model_optimizer = optim.Adam(model.parameters(), lr=1e-4)
    return_optimizer = optim.Adam(return_predictor.parameters(), lr=1e-4)

    rewards, dqn_losses, return_losses = train_dqn(
        env=env,
        model=model,
        target_model=target_model,
        encoder=encoder,
        target_encoder=target_encoder,
        return_predictor=return_predictor,
        encoder_optimizer=encoder_optimizer,
        model_optimizer=model_optimizer,
        return_optimizer=return_optimizer,
        num_episodes=args.episodes,
        env_gamma=args.env_gamma,
        dqn_gamma=args.dqn_gamma,
        epsilon_min=args.epsilon_min,
        rudder_warmup_episodes=args.warmup,
        lessons_capacity=args.lessons_capacity,
        lessons_batch_size=args.lessons_batch_size,
        return_train_updates=args.return_train_updates,
        continuous_pred_factor=args.continuous_pred_factor,
        replay_start_size=args.replay_start_size,
        dqn_updates_per_episode=args.dqn_updates_per_episode,
        safe_margin=args.safe_margin,
    )

    final_window = 50
    mean_final_reward = np.mean(rewards[-final_window:]) if rewards else 0.0
    success_threshold = 0.8
    episodes_to_threshold = next(
        (i + 1 for i, r in enumerate(rewards) if r >= success_threshold), len(rewards)
    )
    dqn_loss_variance = float(np.var(dqn_losses)) if dqn_losses else 0.0
    return_loss_variance = float(np.var(return_losses)) if return_losses else 0.0
    reward_std = float(np.std(rewards)) if rewards else 0.0
    auc_reward = float(np.trapezoid(rewards, dx=1)) if len(rewards) > 1 else 0.0

    print("===== METRICS =====")
    print(f"Mean reward in final {final_window} episodes: {mean_final_reward:.2f}")
    print(f"Episodes to first success (>{success_threshold}): {episodes_to_threshold}")
    print(f"DQN loss variance: {dqn_loss_variance:.6f}")
    print(f"Return predictor loss variance: {return_loss_variance:.6f}")
    print(f"Reward standard deviation: {reward_std:.2f}")
    print(f"AUC: {auc_reward:.2f}")
    print("===================")

    torch.save(model.state_dict(), model_path)
    torch.save(encoder.state_dict(), encoder_path)
    torch.save(target_encoder.state_dict(), target_encoder_path)
    torch.save(return_predictor.state_dict(), return_predictor_path)

    df = pd.DataFrame({"episode": range(1, len(rewards) + 1), "score": rewards})
    excel_path = os.path.join(current_dir, "episode_rewards_rudder_lstm.xlsx")
    df.to_excel(excel_path, index=False)

    print("Training finished.")
    print(f"DQN model saved to {model_path}")
    print(f"Encoder saved to {encoder_path}")
    print(f"Target encoder saved to {target_encoder_path}")
    print(f"Return predictor saved to {return_predictor_path}")
    print(f"Episode rewards saved to {excel_path}")


if __name__ == "__main__":
    main()
