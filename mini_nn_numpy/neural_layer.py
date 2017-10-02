# -*- coding: UTF-8 -*-
"""神经网络层"""
import numpy as np
from mini_nn_numpy.activation import Activation


class NeuralLayer(object):
  def __init__(self, weight, bias, activation='sigmoid'):
    """创建一个神经网络层

    :param weights: 权重，[n, m]维numpy数组，其中n为后一层神经元数，m为前一层神经元数
    :param biases: 偏移，[n,1]维numpy数组
    :param activation: 激活函数名，字符串，可选值可选值为'sigmoid'，'tanh'，'relu'和'linear'，默认为'sigmoid'
    :raises
      Exception: 激活函数名不是字符串或者不在四个可选函数中
    """
    if not isinstance(activation, str):
      raise Exception('activation function name is not a string')
    if not Activation.has(activation):
      raise Exception('activation function name is not allowed')

    self.__weight = weight
    self.__bias = bias
    self.__activation, self.__diff_activation = Activation.get(activation)
    self.__weight_shape = self.__weight.shape
    self.__bias_shape = self.__bias.shape
    self.__neuron_count = self.__weight_shape[0]

    self.__input = None
    self.__weighted_sum = None
    self.__output = None
    self.__back_propagation = None

  def feed_forward(self, input):
    self.__input = input
    self.__weighted_sum = np.matmul(self.__weight , input) + self.__bias
    self.__output = self.__activation(self.__weighted_sum)
    return self.__output

  def diff_layer(self, back_propagation):
    """计算当前层权重和偏移的梯度值

    :param back_propagation: 后面层的梯度
    :return: 当前层权重和偏移对损失函数的梯度值组成的元组
    """
    if not isinstance(self.__input, np.ndarray):
      raise Exception('feed forward hasn\'t be executed')

    diff_layer_weight = self.__diff_activation(self.__weighted_sum)
    diff_base = diff_layer_weight * back_propagation
    diff_weight = np.sum(np.expand_dims(self.__input, 0) * np.expand_dims(diff_base, 1),-1)
    new_back_propagation = np.matmul(self.__weight.T, diff_base)
    return diff_weight, np.expand_dims(np.sum(diff_base,1),1), new_back_propagation

  def update_layer(self, delta_weight, delta_bias):
    self.__weight += delta_weight
    self.__bias += delta_bias

  @property
  def weight_shape(self):
    return self.__weight_shape

  @property
  def bias_shape(self):
    return self.__bias_shape
