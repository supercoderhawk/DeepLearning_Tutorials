# -*- coding: UTF-8 -*-
"""神经元模块"""
import math


class Neuron(object):
  """神经元类

  执行计算加权和与激活函数值

  """
  def __init__(self, weights, bias, activation='sigmoid'):
    """创建神经元

    :param weights: 权重参数，[m]，其中m代表前一层网络神经元个数，也是输入数据的维度
    :param bias: 偏移参数，浮点数
    :param activation: 激活函数名
    :raises
      Exception: 激活函数不是`sigmoid`、`tanh`或者`linear`引发异常
    """
    if activation == 'sigmoid':
      self.activation = Neuron.sigmoid
      self.diff_activation = Neuron.diff_sigmoid
    elif activation == 'tanh':
      self.activation = Neuron.tanh
      self.diff_activation = Neuron.diff_tanh
    elif activation == 'linear':
      self.activation = Neuron.linear
      self.diff_activation = Neuron.diff_linear
    else:
      raise Exception('activation method name error')

    self.weights = weights
    self.bias = bias
    self.input_nums = len(weights)

  def calculate_weighted_sum(self, input):
    """计算输入的加权和

    :param input: 输入参数，[m]
    :return: 加权和，浮点数
    :raises: Exception: 输入数据维度不正确
    """
    if len(input) != self.input_nums:
      raise Exception('input size error, expect {0}, input {1}'.format(self.input_nums, len(input)))

    return sum([a * b for a, b in zip(self.weights, input)]) + self.bias

  def calculate_output(self, weighted_sum):
    """给定加权和计算激活函数输出

    :param weighted_sum: 加权和
    :return: 激活函数输出，浮点数
    :raises: Exception: 输入的加权和不是浮点数
    """
    if not isinstance(weighted_sum, float):
      raise Exception('weighted sum must be float')

    return self.activation(weighted_sum)

  @staticmethod
  def sigmoid(input):
    return 1 / (math.exp(-input) + 1)

  @staticmethod
  def diff_sigmoid(input):
    return Neuron.sigmoid(input) * (1 - Neuron.sigmoid(input))

  @staticmethod
  def tanh(input):
    return 2 * Neuron.sigmoid(2 * input)

  @staticmethod
  def diff_tanh(input):
    return 1 - Neuron.tanh(input) ** 2

  @staticmethod
  def linear(input):
    return input

  @staticmethod
  def diff_linear(input):
    return 1
