import torch
from model import Policy
import numpy as np

class REINFORCE(object):
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_width = 64  # The number of neurons in hidden layers of the neural network
        self.lr = 4e-4  # learning rate
        self.GAMMA = 0.99  # discount factor
        self.episode_s, self.episode_a, self.episode_r = [], [], []

        self.policy = Policy(state_dim, action_dim, self.hidden_width)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

    def choose_action(self, s, deterministic):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        prob_weights = self.policy(s).detach().numpy().flatten()  # probability distribution(numpy)
        if deterministic:  # We use the deterministic policy during the evaluating
            a = np.argmax(prob_weights)  # Select the action with the highest probability
            return a
        else:  # We use the stochastic policy during the training
            a = np.random.choice(range(self.action_dim), p=prob_weights)  # Sample the action according to the probability distribution
            return a

    def store(self, s, a, r):
        self.episode_s.append(s)
        self.episode_a.append(a)
        self.episode_r.append(r)

    def learn(self, ):
        G = []
        g = 0
        for r in reversed(self.episode_r):  # calculate the return G reversely
            g = self.GAMMA * g + r
            G.insert(0, g)

        for t in range(len(self.episode_r)):
            s = torch.unsqueeze(torch.tensor(self.episode_s[t], dtype=torch.float), 0)
            a = self.episode_a[t]
            g = G[t]

            a_prob = self.policy(s).flatten()
            policy_loss = -pow(self.GAMMA, t) * g * torch.log(a_prob[a])  # 因为要梯度上升，所以误差求反
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

        # Clean the buffer
        self.episode_s, self.episode_a, self.episode_r = [], [], []

