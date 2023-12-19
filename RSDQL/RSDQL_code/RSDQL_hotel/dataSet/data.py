#-*- coding: utf-8 -*-
import xml.dom.minidom
from xml.etree import ElementTree as ET

# 每个节点的资源数
CPUnum = 7
Mem = 8*1024

dom1 = xml.dom.minidom.parse('.\\dataSet\\data.xml')
root = dom1.documentElement
dom2 = ET.parse('.\\dataSet\\data.xml')

class Data():
    def __init__(self):
        self.service_containernum = []  # 每个服务的需启动的容器数 [1, 1, 1, 1, 1, 1, 1, 1] 这里每个服务都需要启动一个容器
        self.service_container = []  # 每个服务所启动的容器列表 [[0], [1], [2], [3], [4], [5], [6], [7]]
        self.service_container_relationship = []  # 微服务和容器的映射（假设这个是事先预定的，即哪个微服务在在哪个容器启用）
        self.container_state_queue = []  # 容器状态队列
        self.NodeNumber = int(root.getElementsByTagName('nodeNumber')[0].firstChild.data)  # 5
        self.ContainerNumber = int(root.getElementsByTagName('containerNumber')[0].firstChild.data)  # 8
        self.ServiceNumber = int(root.getElementsByTagName('serviceNumber')[0].firstChild.data)  # 8
        self.ResourceType = int(root.getElementsByTagName('resourceType')[0].firstChild.data)  # 2

        for oneper in dom2.findall('./number/containerNumber'):
            for child in oneper:
                self.service_container_relationship.append(int(child.text))
                # print(self.service_container_relationship)

        for oneper in dom2.findall('./number/serviceNumber'):
            for child in oneper:
                self.service_containernum.append(int(child.text))
                self.service_container.append([int(child[0].text)])
                # 容器状态三元组 部署节点，cpu需求，内存需求
                self.container_state_queue.append(-1)  # 还未部署，所以这里是-1
                self.container_state_queue.append(int(child[0][0].text)/CPUnum)
                self.container_state_queue.append(int(child[0][1].text)/Mem)
                # print(self.service_containernum)
                # print(self.service_container)
                # print(self.container_state_queue)

        Dist_temp = []
        for oneper in dom2.findall('./distance'):
            for child in oneper:
                Dist_temp.append((float(child.text)))
        # 节点之间的距离，用来计算Ccom
        self.Dist = [ Dist_temp[ i : i + self.NodeNumber] for i in range(0,len(Dist_temp),self.NodeNumber)]

        weight_temp = []
        for oneper in dom2.findall('./weight'):
            for child in oneper:
                weight_temp.append((float(child.text)))
        # 微服务之间的交互权重w，用于计算Vusage
        self.service_weight = [ weight_temp[ i : i + self.ServiceNumber] for i in range(0,len(weight_temp),self.ServiceNumber)]



# data = Data()



