from env import Env
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from Agent import REINFORCE
import matplotlib.pyplot as plt

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

evaluate_rewards = []  # Record the rewards during the evaluating

def evaluate_policy(env, agent):
    times = 3  # Perform three evaluations and calculate the average
    evaluate_reward = 0
    for _ in range(times):

        s = env.reset()
        done = False
        episode_reward = 0
        while not done:
            a = agent.choose_action(s, deterministic=False)
            s_, r, done, _, _ = env.step(a, MinCost, REWARD, episode)
            episode_reward += r
            s = s_
        evaluate_reward += episode_reward

    return int(evaluate_reward / times)

def plot_reward():
    plt.figure(1)
    plt.clf()
    loss_t = evaluate_rewards
    plt.title('Train')
    plt.xlabel('100Episodes')
    plt.ylabel('reward')
    plt.plot(evaluate_rewards)
    plt.pause(0.001)  # pause a bit so that plots are updated

if __name__ == '__main__':
    # env_name = ['CartPole-v0', 'CartPole-v1']
    # env_index = 0  # The index of the environments above
    env = Env()
    env_evaluate = Env()  # When evaluating the policy, we need to rebuild an environment
    number = 1
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    state_dim = env.state_dim
    action_dim = env.action_dim
    container_num = env.containernum

    agent = REINFORCE(state_dim, action_dim, container_num)
    writer = SummaryWriter(log_dir='runs/number_{}_seed_{}'.format(number, seed))  # build a tensorboard

    max_train_episode = 1e5  # Maximum number of training steps
    evaluate_freq = 1e2  # Evaluate the policy every 'evaluate_freq' steps
    evaluate_num = 0  # Record the number of evaluations


    print("state_dim={}".format(state_dim))
    print("action_dim={}".format(action_dim))
    print('max_train_episode={}'.format(max_train_episode))
    print('evaluate_freq={}'.format(evaluate_freq))

    episode = 0
    MinCost = 0
    REWARD = 0

    while episode < max_train_episode:
        s = env.reset()
        done = False
        episode += 1
        while not done:
            a = agent.choose_action(s, deterministic=False)
            s_, r, done, M, R = env.step(a, MinCost, REWARD, episode)
            agent.store(s, a, r, done)
            s = s_
            if done:
                MinCost = M
                REWARD = R

        # An episode is over,then update
        loss = agent.learn()
        print("episode:{} \t reward:{} loss:{}\t".format(episode, r, loss))
        # Evaluate the policy every 'evaluate_freq' steps
        # if (episode + 1) % evaluate_freq == 0:
        #     evaluate_reward = evaluate_policy(env_evaluate, agent)
        #     evaluate_rewards.append(evaluate_reward)
        #     print("episode:{} \t reward:{} \t".format(episode, evaluate_reward))
        #     plot_reward()


