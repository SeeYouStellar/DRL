#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in temppliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required  by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#-*- coding: utf-8 -*-

import os
from typing import Container
import numpy as np
import logging
from model import Model
from algorithm import DQN  # from parl.algorithms import DQN  # parl >= 1.3.1
from agent import Agent
from env import Env
import paddle
from replay_memory import ReplayMemory

import env
from agent import flag
from agent import flag_temp
from env import ContainerNumber
from env import NodeNumber
import matplotlib.pyplot as plt

LEARN_FREQ = 8  # learning frequency
MEMORY_SIZE = 20000  # size of replay memory
MEMORY_WARMUP_SIZE = 200  
BATCH_SIZE = 32  
LEARNING_RATE = 0.001
GAMMA = 0.9  

sc_comm = 0
sc_var = 0
flag1 = 1  # 记录这是不是第一个episode
ep = 0 
allCost = []

for i in range(ContainerNumber):
    allCost.append([])
REWARD = 0
test_evareward = 0
episode_loss = []
episode_cost = []
episode_reward = []
def plot_loss():
    plt.figure(1)
    plt.clf()
    loss_t = episode_loss
    plt.title('Train')
    plt.xlabel('50Episodes')
    plt.ylabel('loss')
    plt.plot(loss_t)
    plt.pause(0.001)  # pause a bit so that plots are updated

def plot_cost():
    plt.figure(2)
    plt.clf()
    loss_t = episode_cost
    plt.title('Train')
    plt.xlabel('50Episodes')
    plt.ylabel('cost')
    plt.plot(loss_t)
    plt.pause(0.001)  # pause a bit so that plots are updated

def plot_reward():
    plt.figure(2)
    plt.clf()
    loss_t = episode_reward
    plt.title('Train')
    plt.xlabel('50Episodes')
    plt.ylabel('reward')
    plt.plot(loss_t)
    plt.pause(0.001)  # pause a bit so that plots are updated

def run_episode(env, agent, rpm):

    global flag1  # 记录这是第几个episode
    global allCost  # 每次episode所有step的cost
    global ep
    global REWARD  # 论文中的REWARD 训练过程中最大奖励

    # 临时存储每一轮中每一个transition，在最后一个transition时，一并放入rpm
    obs_list = []
    next_obslist = []
    action_list = []
    done_list = []

    total_reward = 0
    total_cost = 0
    ep += 1
    obs, action = env.reset()

    step = 0
    minCost = -1
    co = 0
    for o in range(ContainerNumber * NodeNumber):
        flag_temp[o] = 0
        flag[o] = 0
    flag1 -= 1

    loss = []
    while True:
        reward = 0
        step += 1  
        obs_list.append(obs)

        '''
        e-greedy choose action:
        1) ramdom choose
        2) argmax(Q(s,a:θ))
        '''
        action = agent.sample(obs)
        # print('e-greedy get action = {}'.format(action))
        action_list.append(action)

        '''
        获得St+1, cost, done
        '''
        next_obs, cost, done, _, _ = env.step(action)
        next_obslist.append(next_obs)
        done_list.append(done)
        
        if allCost[step-1]:
            minCost = min(allCost[step-1])
        # ep(13)
        if flag1 == 0:
            if cost > 0:  # 第一种情况 i=1
                allCost[step - 1].append(cost)
                reward = 0
                co += 1
            else:         # 第一种情况并且有容器部署不了（资源请求无法满足）
                flag1 += 1
                for i in range(co):
                    allCost[step - 1-(i+1)].clear()
                break
        else:
            if cost > 0:
                if step == ContainerNumber:   # 每次随机选择的action里的部署容器能保证不重复吗（agent.sample里实现）
                    if abs(min(allCost[step-1]) - cost) < 0.0000001:  # 第四种情况
                        reward = REWARD
                    elif (min(allCost[step-1]) - cost) > 0:  # 第三种情况
                        REWARD = REWARD + 100
                        reward = REWARD    
                    else:  # 第五种情况
                        reward = 10 * (min(allCost[step-1]) - cost)
                    for i in range(ContainerNumber):
                        rpm.append((obs_list[i], action_list[i], reward, next_obslist[i], done_list[i]))
                    allCost[step - 1].append(cost)
            else:  # 第二种情况
                reward = -100
                rpm.append((obs, action, reward, next_obs, done))
    
        root_logger = logging.getLogger()
        for h in root_logger.handlers[:]:
            root_logger.removeHandler(h)
        logging.basicConfig(level=logging.INFO,filename='details.log')
        logging.info('episode:{}  step:{} Cost:{} min Cost:{} Reward:{} global reward:{} Action:{}'.format(
            ep, step, cost, minCost, reward, REWARD, env.index_to_act(action)))
        
        # train model
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            (batch_obs, batch_action, batch_reward, batch_next_obs,
             batch_done) = rpm.sample(BATCH_SIZE)

            train_loss = agent.learn(batch_obs, batch_action, batch_reward,
                                     batch_next_obs,
                                     batch_done)  # s,a,r,s',done
            loss.append(train_loss)

        total_reward += reward
        total_cost += cost
        
        obs = next_obs
        if done:
            break

    avg_loss = np.mean(loss)
    return total_reward, total_cost, avg_loss


# evaluate agent
def evaluate(env, agent):
    global sc_comm,sc_var
    eval_totalCost = []
    eval_totalReward = []
    reward = 0
    test_evareward = 0
    for i in range(1):
        env.prepare()
        obs = env.update()
        for o in range(ContainerNumber * NodeNumber):
            flag_temp[o] = 0
            flag[o] = 0

        episode_cost = 0
        episode_reward = 0
        step = 0
        while True:
            step +=1
            action = agent.predict(obs)  
            obs, cost, done, comm, var = env.step(action)
            if cost > 0:
                if step == ContainerNumber:
                    if abs(min(allCost[step-1]) - cost) < 0.0000001:
                        reward = test_evareward
                    elif min(allCost[step-1]) - cost > 0:
                        test_evareward += 100
                        reward = test_evareward
                    else:
                        reward = 10*(min(allCost[step-1]) - cost)
            else:
                reward = -100
            episode_cost = cost
            episode_reward = reward
            sc_comm = comm
            sc_var = var
            if done:
                break
        eval_totalCost.append(episode_cost)
        eval_totalReward.append(episode_reward)
    return eval_totalCost,eval_totalReward,sc_comm,sc_var


def main():
    global sc_comm,sc_var 
    env = Env()
    action_dim = ContainerNumber * NodeNumber 
    obs_shape = ContainerNumber * 3 + NodeNumber * (ContainerNumber + 2)  
    rpm = ReplayMemory(MEMORY_SIZE)  
    model = Model(act_dim=action_dim)
    algorithm = DQN(model, act_dim=action_dim, gamma=GAMMA, lr=LEARNING_RATE)
    agent = Agent(
        algorithm,
        obs_dim=obs_shape,
        act_dim=action_dim,
        e_greed=0.2,  
        e_greed_decrement=1e-6)  
    # load model
    # save_path = './dqn_model.ckpt'
    # agent.restore(save_path)

    while len(rpm) < MEMORY_WARMUP_SIZE:
        run_episode(env, agent, rpm)
    max_episode = 10000

    # start train
    episode = 0

    while episode < max_episode: 
        # train part
        avg_loss = []
        for i in range(0, 50):
            total_reward, total_cost, loss = run_episode(env, agent, rpm)
            episode += 1
            with open("reward.txt", "a") as q:  
                q.write("%05d,%.3f \n" % ( episode , total_reward))
            avg_loss.append(loss)

        episode_loss.append(np.mean(avg_loss))

        # test part 50轮测试一次
        eval_totalCost,eval_totalReward,sc_comm,sc_var = evaluate(env, agent)
        episode_cost.append(np.mean(eval_totalCost))
        episode_reward.append(np.mean(eval_totalReward))
        with open("cost.txt", "a") as f:  
                f.write("%d,%.6f \n" %(episode, np.mean(eval_totalCost)))
        root_logger = logging.getLogger()
        for h in root_logger.handlers[:]:
            root_logger.removeHandler(h)

        logging.basicConfig(level=logging.INFO,filename='a.log')
        logging.info('episode:{} e_greed:{} Cost: {} Reward:{} Action:{}'.format(
            episode, agent.e_greed, np.mean(eval_totalCost),np.mean(eval_totalReward), env.action_queue))

    # After training, save the model
    save_path = './dqn_model.ckpt'
    agent.save(save_path)
    return sc_comm,sc_var


if __name__ == '__main__':
    main()
    plot_loss()
    plot_cost()
    plot_reward()
