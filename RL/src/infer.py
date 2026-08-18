import torch
import numpy as np
import os
import gymnasium as gym
from train import config, PolicyGradientNetwork, set_seed


if __name__ == '__main__':
    #创建环境并设置随机种子
    env = gym.make('LunarLander-v3')
    set_seed(config["seed"], env)

    #加载模型
    network = PolicyGradientNetwork()
    model_path = '../models/policy_gradient.pth'
    if not os.path.exists(model_path):
        print(f'Model not found: {model_path}')
        print('Please run train.py first to train the model.')
        exit(1)

    network.load_state_dict(torch.load(model_path, map_location='cpu'))
    network.eval()
    print(f'Model loaded from {model_path}')

    #测试推理
    NUM_OF_TEST = 5
    test_total_reward = []
    action_list = []

    for i in range(NUM_OF_TEST):
        actions = []
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action_prob = network(torch.FloatTensor(state))
            action = torch.argmax(action_prob).item()
            actions.append(action)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        print(f'Test {i+1}: Total Reward = {total_reward:.2f}')
        test_total_reward.append(total_reward)
        action_list.append(actions)

    print(f'\nAverage Total Reward: {np.mean(test_total_reward):.2f}')

    #保存action list（提交用）
    os.makedirs('../output', exist_ok=True)
    output_path = '../output/Action_List.npy'
    np.save(output_path, np.array(action_list))
    print(f'Action list saved to {output_path}')

    #验证：加载保存的action list重新计算得分
    print('\n--- Verifying saved action list ---')
    saved_actions = np.load(output_path, allow_pickle=True)
    if len(saved_actions) != 5:
        print('Wrong format of file !!!')
        exit(0)

    verify_rewards = []
    for actions in saved_actions:
        state, _ = env.reset()
        total_reward = 0
        for action in actions:
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            if done:
                break
        print(f'Verify: Reward = {total_reward:.2f}')
        verify_rewards.append(total_reward)

    print(f'Final Reward (Verified): {np.mean(verify_rewards):.2f}')