import gymnasium as gym
import numpy as np
import cv2
import os
import ale_py
from collections import deque
from agent import DQNAgent
from memory import Replay

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0

def stack_frames(stacked_frames, frame, is_new_episode):
    processed = preprocess_frame(frame)
    if is_new_episode:
        stacked_frames = deque([processed] * 4, maxlen=4)
    else:
        stacked_frames.append(processed)
    return np.stack(stacked_frames, axis=-1), stacked_frames

def main():
    os.makedirs("videos", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    env = gym.make("ALE/KungFuMaster-v5", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder="videos",
        episode_trigger=lambda ep: ep % 50 == 0,
        name_prefix="agent_jackie"
    )

    input_shape = (84, 84, 4)
    num_actions = env.action_space.n
    agent = DQNAgent(input_shape=input_shape, num_actions=num_actions, learning_rate=0.00025)
    memory = Replay(capacity=100000)

    episodes = 1000
    max_steps = 10000
    target_update_freq = 10
    best_reward = float("-inf")

    for episode in range(1, episodes + 1):
        obs, _ = env.reset()
        state, stacked_frames = stack_frames(None, obs, is_new_episode=True)
        total_reward = 0

        for step in range(max_steps):
            action = agent.act(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_state, stacked_frames = stack_frames(stacked_frames, next_obs, is_new_episode=False)
            memory.add(state, action, reward, next_state, done)
            agent.learn(memory)
            state = next_state
            total_reward += reward

            if done:
                break

        if episode % target_update_freq == 0:
            agent.update_target_network()

        if total_reward > best_reward:
            best_reward = total_reward
            agent.model.save("checkpoints/best_model.h5")
            print(f"💾 Best model saved with reward {best_reward:.2f}")

        if episode % 100 == 0:
            agent.model.save(f"checkpoints/agent_jackie_ep{episode}.h5")
            print(f"📦 Checkpoint saved at episode {episode}")

        print(f"🎮 Episode {episode} | Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.3f}")

    agent.model.save("models/agent_jackie_model.keras")
    print("🏁 Training complete.")
    env.close()

if __name__ == "__main__":
    main()
