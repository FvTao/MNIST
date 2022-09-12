# -*- coding: utf-8 -*-
# @Time    : 2022/9/9 18:06
# @Author  : LuMing
# @File    : bpnn.py
# @Software: PyCharm 
# @Comment : python3.10
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def derived_sigmoid(x):
    return x * (1 - x)


def make_matrix(m, n, fill=0.0):  # m行n列
    a = []
    for i in range(m):
        a.append([fill] * n)
    return np.array(a)


def softmax(array):  # 这个函数的作用是将一组数据转化为概率的形式
    n = 0
    for i in array:
        n += np.exp(i)
    return array / n


def to_result(vector):
    max_index = 0
    max_elem = -1.0
    for i in range(len(vector)):
        if max_elem < vector[i]:
            max_index = i
            max_elem = vector[i]
    return max_index


class Bpnn:  # BP neural network 一个三层的BP神经网络
    def __init__(self, x_num, h_num, y_num):
        self.x_num = x_num + 1  # 添加一个偏置
        self.h_num = h_num
        self.y_num = y_num

        # 初始化向量
        self.x_vector = np.array([0.0] * self.x_num)
        self.hi_vector = np.array([0.0] * self.h_num)
        self.ho_vector = np.array([0.0] * self.h_num)
        self.yi_vector = np.array([0.0] * self.y_num)
        self.yo_vector = np.array([0.0] * self.y_num)

        # 初始化权值矩阵
        self.weight_xh = (np.random.random([self.x_num, self.h_num]) - 0.51)  # 输入数据到隐藏层输入的变换矩阵
        self.weight_hy = (np.random.random([self.h_num, self.y_num]) - 0.51)  # 隐藏层输出到输出层输入的变换矩阵
        # 学习率
        self.lr = 0.1

        # 动量因子
        self.input_correction = make_matrix(self.x_num, self.h_num)
        self.output_correction = make_matrix(self.h_num, self.y_num)

    def forward_propagation(self, x_vector):  # 正向传播
        if len(x_vector) != self.x_num - 1:
            raise ValueError("输入数据与输入结点数量不同")

        # 简单的处理一下输入数据
        self.x_vector[1:self.x_num] = x_vector
        self.x_vector = np.array(self.x_vector)

        # 输入层->隐藏层
        self.hi_vector = np.dot(self.x_vector, self.weight_xh)
        # print(self.hi_vector)

        # 激活隐藏层神经元
        self.ho_vector = np.array(sigmoid(self.hi_vector))
        # print(self.ho_vector)

        # 隐藏层->输出层
        self.yi_vector = np.dot(self.ho_vector, self.weight_hy)
        # print(self.yi_vector)

        # 激活输出层神经元
        self.yo_vector = np.array(sigmoid(self.yi_vector))
        # print(self.yo_vector)

        return self.yo_vector

        # for i in range(self.x_num - 1):
        #     self.x_vector[i] = x_vector[i]
        #     # 激活隐含层神经元
        # for j in range(self.h_num):
        #     total = 0.0
        #     for i in range(self.x_num):
        #         total += self.x_vector[i] * self.weight_xh[i][j]
        #     self.ho_vector[j] = sigmoid(total)
        #     # 激活输出层神经元（即输出结果）
        # for k in range(self.y_num):
        #     total = 0.0
        #     for j in range(self.h_num):
        #         total += self.ho_vector[j] * self.weight_hy[j][k]
        #     self.yo_vector[k] = sigmoid(total)
        # return self.yo_vector[:]

    def backward_propagation(self, labels, correct):
        if len(labels) != self.y_num:
            raise ValueError("标记数量与输出数量不符")

        targets = np.array(labels)  # 简单处理输入

        # 计算误差
        error = 0.5 * np.dot((targets - self.yo_vector).T,
                             (targets - self.yo_vector))

        # 计算残差
        delta_hy = np.array((targets - self.yo_vector) * derived_sigmoid(self.yo_vector))
        delta_xh = np.array(np.dot(delta_hy, self.weight_hy.T) * derived_sigmoid(self.ho_vector))

        # 更新权值
        # print(self.weight_xh)
        self.weight_hy += self.lr * np.dot(delta_hy.reshape(-1, 1),
                                           self.ho_vector.reshape(1, -1)).T + correct * self.output_correction

        self.weight_xh += self.lr * np.dot(delta_xh.reshape(-1, 1),
                                           self.x_vector.reshape(1, -1)).T + correct * self.input_correction

        # 更新
        self.output_correction = self.lr * np.dot(delta_hy.reshape(-1, 1), self.ho_vector.reshape(1, -1)).T
        self.input_correction = self.lr * np.dot(delta_xh.reshape(-1, 1), self.x_vector.reshape(1, -1)).T
        return error

        # # 获取输出层误差及误差相对于输出层神经元的偏导数
        # output_deltas = [0.0] * self.y_num
        # for o in range(self.y_num):
        #     error = label[o] - self.yo_vector[o]
        #     output_deltas[o] = derived_sigmoid(self.yo_vector[o]) * error
        # # 获取隐含层的相对误差及其相对于隐含层神经元的偏导数
        # hidden_deltas = [0.0] * self.h_num
        # for h in range(self.h_num):
        #     error = 0.0
        #     for o in range(self.y_num):
        #         error += output_deltas[o] * self.weight_hy[h][o]
        #     hidden_deltas[h] = derived_sigmoid(self.ho_vector[h]) * error
        # # 更新隐含层到输出层的权值
        # for h in range(self.h_num):
        #     for o in range(self.y_num):
        #         change = output_deltas[o] * self.ho_vector[h]
        #         self.weight_hy[h][o] += self.lr * change + correct * self.output_correction[h][o]
        #         self.output_correction[h][o] = change
        # # 更新输入层到输出层的权重
        # for i in range(self.x_num):
        #     for h in range(self.h_num):
        #         change = hidden_deltas[h] * self.x_vector[i]
        #         self.weight_xh[i][h] += self.lr * change + correct * self.input_correction[i][h]
        #         self.input_correction[i][h] = change
        # # 获取全局误差
        # error = 0.0
        # for o in range(len(label)):
        #     error += 0.5 * (label[o] - self.yo_vector[o]) ** 2
        # return error

    def train(self, train_data, train_label):
        if len(train_label) != len(train_data):
            raise ValueError("训练数据与标签数量不符")

        for i in range(len(train_data)):
            self.forward_propagation(train_data[i])
            error = self.backward_propagation(train_label[i], 0.1)
            if i % 100 == 0:
                print('########################误差 %-.5f######################第%d次迭代' % (error, i))

    def test(self, test_data, test_label):
        if len(test_label) != len(test_data):
            raise ValueError("测试数据与标签数量不符")
        miss_num = 0
        for i in range(len(test_data)):
            yo_vector = self.forward_propagation(test_data[i])
            result_num = to_result(yo_vector)
            # print(test_label[i])
            # print(yo_vector)
            # print(result_num)
            if result_num != test_label[i]:
                miss_num += 1
            if i % 1000 == 0:
                print(yo_vector)
                print(result_num, test_label[i])
                print("miss rate: %f%%" % (100.0 * miss_num / len(test_label)))
        print("测试完成")

    def new_train(self, case, label):
        error = 0
        for k in range(100000):
            for i in range(len(case)):
                self.forward_propagation(case[i])
                error = self.backward_propagation(label[i], 0.1)
            if k % 10 == 0:
                print('########################误差 %-.5f######################第%d次迭代' % (error, k))


def my_test():
    my_bpnn = Bpnn(2, 5, 1)
    cases = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ]
    labels = [[0], [1], [1], [0]]
    my_bpnn.new_train(cases, labels)
    print(my_bpnn.forward_propagation([0, 0]))
    print(my_bpnn.forward_propagation([0, 1]))
    print(my_bpnn.forward_propagation([1, 0]))
    print(my_bpnn.forward_propagation([1, 1]))
