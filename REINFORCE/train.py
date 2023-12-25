import gym
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from Algorithm import REINFORCE
def evaluate_policy(env, agent):
    times = 3  # Perform three evaluations and calculate the average
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset()
        done = False
        episode_reward = 0
        while not done:
            a = agent.choose_action(s, deterministic=True)  # We use the deterministic policy during the evaluating
            s_, r, done, _ = env.step(a)
            episode_reward += r
            s = s_
        evaluate_reward += episode_reward

    return int(evaluate_reward / times)


if __name__ == '__main__':
    env_name = ['CartPole-v0', 'CartPole-v1']
    env_index = 0  # The index of the environments above
    env = gym.make(env_name[env_index])
    env_evaluate = gym.make(env_name[env_index])  # When evaluating the policy, we need to rebuild an environment
    number = 1
    seed = 0
    env.seed(seed)
    env_evaluate.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode
    print("state_dim={}".format(state_dim))
    print("action_dim={}".format(action_dim))
    print("max_episode_steps={}".format(max_episode_steps))

    agent = REINFORCE(state_dim, action_dim)
    writer = SummaryWriter(log_dir='runs/REINFORCE/REINFORCE_env_{}_number_{}_seed_{}'.format(env_name[env_index], number, seed))  # build a tensorboard

    max_train_steps = 1e5  # Maximum number of training steps
    evaluate_freq = 1e3  # Evaluate the policy every 'evaluate_freq' steps
    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    total_steps = 0  # Record the total steps during the training

    while total_steps < max_train_steps:
        episode_steps = 0
        s = env.reset()
        done = False
        while not done:
            episode_steps += 1
            a = agent.choose_action(s, deterministic=False)
            s_, r, done, _ = env.step(a)
            agent.store(s, a, r)
            s = s_

            # Evaluate the policy every 'evaluate_freq' steps
            if (total_steps + 1) % evaluate_freq == 0:
                evaluate_num += 1
                evaluate_reward = evaluate_policy(env_evaluate, agent)
                evaluate_rewards.append(evaluate_reward)
                print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward))
                writer.add_scalar('step_rewards_{}'.format(env_name[env_index]), evaluate_reward, global_step=total_steps)
                if evaluate_num % 10 == 0:
                    np.save('./data_train/REINFORCE_env_{}_number_{}_seed_{}.npy'.format(env_name[env_index], number, seed), np.array(evaluate_rewards))

            total_steps += 1

        # An episode is over,then update
        agent.learn()
