#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#-*- coding: utf-8 -*-

import copy
import parl
# from parl import layers
import paddle.fluid as fluid
import paddle.fluid.layers as layers
class DQN(parl.Algorithm):
    def __init__(self, model, act_dim=None, gamma=None, lr=None):
        """ DQN algorithm
        
        Args:
            model (parl.Model): the network forwarding structure of the Q function
            act_dim (int): dimensions of action
            gamma (float): attenuation factor of reward
            lr (float): learning_rate
        """
        self.policy_model = model
        self.target_model = copy.deepcopy(model)

        assert isinstance(act_dim, int)
        assert isinstance(gamma, float)
        assert isinstance(lr, float)
        self.act_dim = act_dim
        self.gamma = gamma
        self.lr = lr

    def predict(self, obs):
        """ use value network of self.policy-model to get [Q(s,a1),Q(s,a2),...]
        """
        return self.policy_model.value(obs)

    def learn(self, obs, action, reward, next_obs, terminal):
        """ use DQN algorithm to update value network of self.policy_model
        """

        # 计算Q-target
        next_pred_value = self.target_model.value(next_obs)
        best_v = layers.reduce_max(next_pred_value, dim=1)
        best_v.stop_gradient = True  # prevent gradient
        terminal = layers.cast(terminal, dtype='float32')# convert to float，true equals 1, false equals 0
        target = reward + (1.0 - terminal) * self.gamma * best_v

        # 获取policy-model的输出，为一组Q值向量，每一个对应每个动作的Q值
        pred_value = self.policy_model.value(obs)
        # Convert action to onehot vector
        action_onehot = layers.one_hot(action, self.act_dim)
        action_onehot = layers.cast(action_onehot, dtype='float32')

        # 计算action对应的Q-predict
        pred_action_value = layers.reduce_sum(
            layers.elementwise_mul(action_onehot, pred_value), dim=1)

        # get loss
        loss = layers.square_error_cost(pred_action_value, target)
        loss = layers.reduce_mean(loss)
        optimizer = fluid.optimizer.Adam(learning_rate=self.lr)  # use Adam optimizer
        optimizer.minimize(loss)
        return loss

    def sync_target(self):
        """ Synchronize the model parameter values of self.policy_model to self.target_model
        """
        self.policy_model.sync_weights_to(self.target_model)
