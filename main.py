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

q = build_q_table(N_STATE, ACTIONS)

state_actions = q.iloc[5, :]
print(state_actions)
re = state_actions.argmax()