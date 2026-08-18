import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import os
import gymnasium as gym
from torch.distributions import Categorical
from tqdm import tqdm


#设置随机种子
def set_seed(seed, env=None):
    if env is not None:
        env.action_space.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


#定义策略梯度网络
class PolicyGradientNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 4)

    def forward(self, state):
        hid = torch.tanh(self.fc1(state))
        hid = torch.tanh(self.fc2(hid))
        return F.softmax(self.fc3(hid), dim=-1)


#定义策略梯度Agent
class PolicyGradientAgent():
    def __init__(self, network):
        self.network = network
        self.optimizer = optim.SGD(self.network.parameters(), lr=0.001)

    def learn(self, log_probs, rewards):
        loss = (-log_probs * rewards).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sample(self, state):
        action_prob = self.network(torch.FloatTensor(state))
        action_dist = Categorical(action_prob)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob


#定义超参数
config = {
    "seed": 543,
    "num_epochs": 500,        #总共更新 500 次策略
    "episodes_per_epoch": 5,  #每次更新前收集 5 条轨迹
    "gamma": 0.99,           #折扣因子
    "lr": 0.001,
}


if __name__ == '__main__':
    #创建环境
    env = gym.make('LunarLander-v3')
    set_seed(config["seed"], env)

    #初始化网络和Agent
    network = PolicyGradientNetwork()
    agent = PolicyGradientAgent(network)

    #训练循环
    os.makedirs('../models', exist_ok=True)
    os.makedirs('../output', exist_ok=True)

    agent.network.train()
    avg_total_rewards, avg_final_rewards = [], []

    for epoch in tqdm(range(config["num_epochs"]), desc="Training"):
        log_probs, rewards = [], []
        total_rewards, final_rewards = [], []

        #收集多条轨迹
        for episode in range(config["episodes_per_epoch"]):
            state, _ = env.reset()
            total_reward = 0
            episode_rewards = []

            while True:
                action, log_prob = agent.sample(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                log_probs.append(log_prob)
                episode_rewards.append(reward)
                total_reward += reward
                state = next_state

                if done:
                    final_rewards.append(reward)
                    total_rewards.append(total_reward)

                    #计算折扣累计回报（从后往前反向累加，最后翻转回来对齐 log_probs）
                    discounted = 0
                    episode_returns = []
                    for r in reversed(episode_rewards):
                        discounted = r + config["gamma"] * discounted
                        episode_returns.append(discounted)
                    rewards.extend(reversed(episode_returns))
                    break

        #记录训练过程
        avg_total_reward = sum(total_rewards) / len(total_rewards)
        avg_final_reward = sum(final_rewards) / len(final_rewards)
        avg_total_rewards.append(avg_total_reward)
        avg_final_rewards.append(avg_final_reward)

        #奖励标准化（多条轨迹一起标准化，降低方差）
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-9)
        agent.learn(torch.stack(log_probs), torch.from_numpy(rewards))

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{config['num_epochs']} | Total: {avg_total_reward: 4.1f} | Final: {avg_final_reward: 4.1f}")

    #保存训练结果
    torch.save(network.state_dict(), '../models/policy_gradient.pth')
    print(f'Model saved to ../models/policy_gradient.pth')

    #绘制训练曲线
    import matplotlib.pyplot as plt
    plt.plot(avg_total_rewards)
    plt.title("Total Rewards")
    plt.savefig('../output/total_rewards.png')
    plt.close()

    plt.plot(avg_final_rewards)
    plt.title("Final Rewards")
    plt.savefig('../output/final_rewards.png')
    plt.close()

    print('Training completed!')