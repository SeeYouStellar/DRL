import numpy as np
import pandas as pd
import pandas as ps
import time

np.random.seed(2)

N_STATE = 6
ACTIONS = ['LEFT', 'RIGHT']  # 向左或向右一步
EPSILON = 0.9  # 贪心因子
ALPHA = 0.1    # 学习率
LAMBDA = 0.9   # 折扣因子
MAX_EPISODES = 10  # 训练批次
FRESH_TIME = 0.01

# 初始化Q表
def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),
        columns=actions,
    )
    return table

def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):
        action_name = np.random.choice(ACTIONS)
    else:
        idx = state_actions.argmax()
        action_name = ACTIONS[idx]
    return action_name

def get_env_feedback(S, A):
    # 假设环境的反馈都是确定的，即每种状态下采取每种行动的状态转换是确定的
    if A == 'LEFT':
        R = 0
        if S == 0:
            S_ = S
        else:
            S_ = S-1
    else:
        if S == N_STATE-2:
            S_ = 'terminal'
            R = 1
        else:
            S_ = S+1
            R = 0
    return S_, R

def update_env(S, episode, step_counter):
    env_list = ['-']*(N_STATE-1) + ['T']
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                            ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)

def RL():
    # 初始化Q表
    q_table = build_q_table(N_STATE, ACTIONS)
    for episode in range(MAX_EPISODES):

        # episode初始化
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S, episode, step_counter)
        while not is_terminated:
            # print(q_table)
            A = choose_action(S, q_table)
            # print('\n')
            # print(A)
            # print('\n')
            S_, R = get_env_feedback(S, A)
            Q_predict = q_table.loc[S, A]
            if S_ != 'terminal':
                Q_target = R + LAMBDA*q_table.iloc[S_, :].max()
            else:
                Q_target = R
                is_terminated = True
            q_table.loc[S, A] += ALPHA*(Q_target-Q_predict)
            S = S_
            update_env(S, episode, step_counter+1)
            step_counter += 1
    return q_table

if __name__ == "__main__":
    q_table = RL()
    print('\r\nQ-table:\n')
    print(q_table)


