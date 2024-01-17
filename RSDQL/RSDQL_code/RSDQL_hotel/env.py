alpha = 0.5  # reward weighting factor
beta = [0.5, 0.5]  # 资源亲和度
count = 0  # 容器已部署的个数

from MyRL.REINFORCE_RSDQL.Dataset.data import Data

import numpy as np

data = Data()
NodeNumber = data.NodeNumber
ContainerNumber = data.ContainerNumber
ServiceNumber = data.ServiceNumber
ResourceType = data.ResourceType
service_containernum = data.service_containernum  # 每个服务所需容器数列表

service_container = data.service_container  # 每个服务所启动的容器列表
service_container_relationship = data.service_container_relationship  # 微服务和容器的映射
container_state1 = data.container_state_queue[:]  # 容器状态（cpu,mem）队列
class Env():
    def __init__(self):
        # State
        self.State = []
        self.node_state_queue = []
        self.container_state_queue = []
        self.action_queue = []
        self.prepare()

    def prepare(self):
        self.container_state_queue = container_state1[:]
        for i in range(NodeNumber):
            '''[[1, 0, ..., 1, cpu, mem], [], []], 前面containernum个元素代表该container是否部署在该节点上，最后两个属性是cpu和mem'''
            for j in range(ContainerNumber + 2):
                self.node_state_queue.append(0)
        # print(self.container_state_queue)
        # print(self.node_state_queue)
        self.State = self.container_state_queue + self.node_state_queue
        # print(self.State)
        self.action = [-1,-1]
        self.action_queue = [-1,-1]
        # Communication weight between microservices
        self.service_weight = data.service_weight
        # Communication distance between nodes
        self.Dist = data.Dist

    def containerDis(self,i,j):
        # to calculate the distance between container i and j
        m = -1
        n = -1
        m = self.container_state_queue[i*3]
        n = self.container_state_queue[j*3]
        # print(self.container_state_queue)
        p = service_container_relationship[i]
        q = service_container_relationship[j]

        # 判断部署后，这两个容器是否在同一个节点上，并且这两个容器不能是同一个微服务的（这里的微服务指的是最小的那种，不会再有子微服务）
        if self.Dist[m][n] != 0 and (p != q):
            container_dist = self.Dist[m][n]
        else:
            container_dist = 0
        return container_dist

    def serviceComCost(self,i,j):
        cost = 0
        interaction = self.service_weight[i][j] / (service_containernum[i] * service_containernum[j])
        # print('CalcuCost({}, {})'.format(i, j))
        # print('service_weight = {}'.format(self.service_weight[i][j]))
        # print('it = {}'.format(interaction))
        # paper eq(3) 计算每两个微服务之间的所有容器的comCost
        for k in range(len(service_container[i])):
            for l in range(len(service_container[j])):
                # paper eq(2) 计算某两个容器的comCost
                cost += self.containerDis(service_container[i][k], service_container[j][l]) * interaction
                # print('container {} container {} , cost = {}'.format(service_container[i][k], service_container[j][l], cost))
        return cost

    def ComCost(self):
        # paper eq(3) 计算整个应用中所有微服务之间的comCost
        Cost = 0
        for i in range (ServiceNumber):
            for j in range (ServiceNumber):
                Cost += self.serviceComCost(i,j)
        # paper eq(4)
        return 0.5 * Cost

    def usageVar(self):
        NodeCPU = []
        NodeMemory = []
        Var = 0
        for i in range(NodeNumber):
            '''[[1, 0, ..., 1, cpu, mem], [], []], 前面containernum个元素代表该container是否部署在该节点上，最后两个属性是cpu和mem资源占用'''
            U = self.node_state_queue[i*(ContainerNumber+2)+ContainerNumber] 
            M = self.node_state_queue[i*(ContainerNumber+2)+(ContainerNumber+1)]
            NodeCPU.append(U)
            NodeMemory.append(M)
            if NodeCPU[i] > 1 or NodeMemory[i] > 1:
                # 如果没有资源限制，那么都部署到同一个节点上，通信成本就降为0了。0<资源利用率<1
                Var = -10
        # Variance of node load
        # paper eq(7)
        Var += beta[0] * np.var(NodeCPU) + beta[1] * np.var(NodeMemory)
        return Var

    def cost(self):
        re = 0
        g1 = self.ComCost()
        g1 = g1 / 371.5
        g2 = self.usageVar()
        if g2 < 0:
            g2 = -100
        g2 = g2 / 6.002500000000001

        re += alpha * g1 + (1-alpha) * g2
        return 100*re, g1, g2

    def state_update(self,container_state_queue,node_state_queue):
        self.State = container_state_queue + node_state_queue

    def update(self):
        # 根据 action 来 更新 state

        if self.action[0] >= 0 and self.action[1] >= 0:
            print("excute action, action = [{}, {}]".format(self.action[0], self.action[1]))
            # update container state
            self.container_state_queue[self.action[1] * 3] = self.action[0]
            # update node state
            self.node_state_queue[self.action[0] * (ContainerNumber+2) + self.action[1]] = 1  # 代表容器该部署在该节点上
            # 把容器的资源需求加到节点的资源占用上
            self.node_state_queue[self.action[0] * (ContainerNumber+2) + ContainerNumber] += self.container_state_queue[self.action[1] * 3 + 1]
            self.node_state_queue[self.action[0] * (ContainerNumber+2) + (ContainerNumber + 1)] += self.container_state_queue[self.action[1] * 3 + 2]
            self.action_queue.append(self.action)
        else:
            # print("invalid action, action = [{}, {}]".format(self.action[0], self.action[1]))
            self.node_state_queue = []
            self.container_state_queue = []
            self.action_queue = []
            self.prepare()

        self.state_update(self.container_state_queue,self.node_state_queue)
        # print(self.State)
        return self.State

    def step(self,action):
    # input: action(Targetnode，ContainerIndex)
    # output: next state, cost, done
        global count
        self.action = self.index_to_act(action)  # action = containernum*nodenum+x
        self.update()  # 因为action是self的成员变量，所以不用传参
        cost, comm, var = self.cost()
        # print(cost, comm, var)
        done = False 
        count = 0
        
        for i in range(ContainerNumber):
            if self.container_state_queue[3*i] != -1:  # 初始化是（-1,cpu,mem）,用-1判断该容器是否部署
                count += 1
        if count == ContainerNumber:  # 已全部部署完毕
            done = True
        return self.State, cost, done, comm, var

    def reset(self):
        self.node_state_queue = []
        self.container_state_queue = []
        self.prepare() 
        return self.State, self.action
    
    def index_to_act(self, index):
        act = [-1, -1]
        act[0] = int(index / ContainerNumber)  # 节点
        act[1] = index % ContainerNumber  # 容器
        return act

# env = Env()
# env.step(0)
# env.step(1)
# env.step(2)
# env.step(3)
# env.step(4)
# env.step(5)
# env.step(6)
# env.step(7)
"""
action=20
self.action=[2, 4]
node_state_queue = 
[-1, 0.2857142857142857, 0.015625, -1, 0.42857142857142855, 0.03125, -1, 0.2857142857142857, 0.03125, -1, 0.42857142857142855, 0.03125, 2, 0.2857142857142857, 0.03125, -1, 0.42857142857142855, 0.015625, -1, 0.14285714285714285, 0.015625, -1, 0.2857142857142857, 0.015625, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0.2857142857142857, 0.03125, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
"""