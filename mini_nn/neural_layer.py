# -*- coding: UTF-8 -*-
"""神经网络层模块"""
from mini_nn.neuron import Neuron


class NeuralLayer(object):
  """神经网络层类

  计算网络层的前向输出和梯度

  """
  def __init__(self, weights, biases, activation='sigmoid'):
    """创建一个神经网络层

    :param weights: 权重，[m, n]，其中m为前一层神经元数，n为后一层神经元数
    :param biases: 偏移，[n]
    :param activation: 激活函数名，默认为'sigmoid'
    """
    if len(weights[0]) != len(biases):
      raise Exception('weights and biases length is not equal')

    self.weights = weights
    self.biases = biases
    self.neurons_num = len(biases)
    self.prev_neurons_num = len(self.weights)
    self.neurons = []

    for weight, bias in zip(zip(*self.weights), self.biases):
      self.neurons.append(Neuron(list(weight), bias, activation))

    self.input = None
    self.weighted_sums = None
    self.output = None
    self.back_propagation = None
    self.diff_weighted_sum = None

  def feed_forward(self, input):
    """前向计算，即给定输入值，计算网络输出值

    :param input: 当前层输入，[m]
    :return: 当前层输出, [n]
    """
    self.input = input
    self.weighted_sums = [0] * self.neurons_num
    self.output = [0] * self.neurons_num

    for neuron_i, neuron in enumerate(self.neurons):
      self.weighted_sums[neuron_i] = neuron.calculate_weighted_sum(input)
      self.output[neuron_i] = neuron.calculate_output(self.weighted_sums[neuron_i])
    return self.output

  def diff_layer(self, back_propagation):
    """计算当前层权重和偏移的梯度值

    :param back_propagation: 后面层的梯度
    :return: 当前层权重和偏移对损失函数的梯度值组成的元组
    """
    if len(back_propagation) != self.neurons_num:
      raise Exception('back propagation dimension error')
    if not self.weighted_sums:
      raise Exception('neurons feed forward did\'t execute previously')

    self.back_propagation = [0] * self.prev_neurons_num
    self.diff_weighted_sum = []
    diff_weights = self.__diff_weights(back_propagation)
    diff_bias = [a * b for a, b in zip(back_propagation, self.diff_weighted_sum)]
    return diff_weights, diff_bias

  def update_layer(self, weights, biases, lr):
    """更新层权重和偏移

    :param weights: 变化的权重
    :param biases: 变化的偏移
    :param lr: 学习率
    :return: 无返回
    """
    for j in range(self.neurons_num):
      for i in range(self.prev_neurons_num):
        self.weights[i][j] -= lr * weights[i][j]
        self.neurons[j].weights[i] -= lr * weights[i][j]
      self.biases[j] -= lr * biases[j]
      self.neurons[j].bias -= lr * biases[j]

  def __diff_weights(self, back_propagation):
    for neuron_i, neuron in enumerate(self.neurons):
      self.diff_weighted_sum.append(neuron.diff_activation(self.weighted_sums[neuron_i]))

    diff_weights = [[0] * self.neurons_num] * self.prev_neurons_num
    for i in range(self.prev_neurons_num):
      for j in range(self.neurons_num):
        diff_weights[i][j] = back_propagation[j] * self.diff_weighted_sum[j] * self.input[i]
        self.back_propagation[i] += back_propagation[j] * self.diff_weighted_sum[j] * self.weights[i][j]

    return diff_weights
