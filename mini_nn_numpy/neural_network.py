# -*- coding: UTF-8 -*-
"""神经网络核心类"""
import numpy as np
from mini_nn_numpy.neural_layer import NeuralLayer
from utils.check import check_dimension


class NeuralNetwork:
  def __init__(self, layers_count, *, learning_rate=0.2, activation='sigmoid'):
    """创建一个神经网络

    :param layers_count:  网络尺寸，即每一层网络神经元个数，一维列表，均为正整数
    :param learning_rate: 学习率
    :param activation: 隐藏层激活函数名，可选值为'sigmoid'，'tanh'，'relu'和'linear'，默认'sigmoid'
    :raises
      Exception:
        1. 网络尺寸列表错误，包括非1维列表，元素不是正整数，元素个数小于三个（至少有输入、输出和一个隐藏层）
        2. 学习率不是整数或浮点数
    """
    if not check_dimension(layers_count, 1, type=int, func=lambda i: i > 0):
      raise Exception('layers count error')
    elif len(layers_count) < 3:
      raise Exception('layers count less than two')

    if not isinstance(learning_rate, (float, int)):
      raise Exception('learning rate is not int or float')

    self.__learning_rate = learning_rate

    # 创建神经网络层
    self.__layers = []
    # 创建隐藏层
    for prev, curr in zip(layers_count[:-2], layers_count[1:-1]):
      weight = NeuralNetwork.__get_init([curr, prev])
      bias = NeuralNetwork.__get_init([curr, 1])
      self.__layers.append(NeuralLayer(weight=weight, bias=bias, activation=activation))
    # 创建输出层
    weight = NeuralNetwork.__get_init([layers_count[-1], layers_count[-2]])
    bias = NeuralNetwork.__get_init([layers_count[-1], 1])
    self.__layers.append(NeuralLayer(weight=weight, bias=bias, activation='linear'))

  def fit(self, data_inputs, data_outputs, *, epochs=10000, interval=1000):
    """使用mini batch的随机梯度下降训练网络

    :param data_inputs: 训练数据输入部分，[count, batch_size, n_i]，count代表batch个数，batch_size代表mini batch的大小，n_i代表输入维度
    :param data_outputs: 训练数据输入部分对应的输出，[count, batch_size, n_o]，count代表batch个数，batch_size代表mini batch的大小，n_o代表输出维度
    :param epochs: 迭代次数，为正整数
    :param interval: 打印次数和loss值的间隔，为正整数
    :return: 无返回值
    :raises:
      Exception:
        1. 训练数据的输入或者输出部分不是numpy数组
        2. 训练数据输入或者输出部分维度不是三维
        3. 训练数据的输入输出个数不同
        4. 迭代次数或者间隔次数不是正整数
    """
    if not isinstance(data_inputs, np.ndarray):
      raise Exception('training input data is not numpy array')
    if not isinstance(data_outputs, np.ndarray):
      raise Exception('training output data is not numpy array')
    input_shape = data_inputs.shape
    output_shape = data_outputs.shape
    if len(input_shape) != 3:
      raise Exception('training input data size is not three')
    if len(output_shape) != 3:
      raise Exception('training output data size is not three')
    if input_shape[:-1] != output_shape[:-1]:
      raise Exception('batch count and size of training input and output are not equal')

    if not isinstance(epochs, int) or epochs <= 0:
      raise Exception('epoch is not positive integer')
    elif not isinstance(interval, int) or interval <= 0:
      raise Exception('interval is not positive integer')

    for epoch in range(1, epochs + 1):
      for batch_input, batch_output in zip(data_inputs, data_outputs):
        batch_input = batch_input.T
        batch_output = batch_output.T
        # 前向
        output = self.__feed_forward(batch_input)
        # 后向
        back_propagation = output - batch_output
        for layer in self.__layers[::-1]:
          diff_weight, diff_bias, back_propagation = layer.diff_layer(back_propagation)
          layer.update_layer(-self.__learning_rate * diff_weight, -self.__learning_rate * diff_bias)

      if epoch % interval == 0:
        print('epoch: {0}, loss: {1}'.format(epoch, self.calculate_loss(data_inputs, data_outputs)))

  def predict(self, data_input):
    """在当前参数下。给定输入值，求输出值

    :param data_input: 输入数据，[x, n_i]，其中x代表输入数据个数，n_i为网络的输入维度
    :return: 网络输出值 [x, n_o]，x尾输入数据个数，n_o为网络的输出维度
    :raises:
      Execption: 输入数据不是numpy二维数组
    """
    if not isinstance(data_input, np.ndarray) or len(data_input.shape) != 3:
      raise Exception('input data format error')

    return self.__feed_forward(data_input.T)

  def calculate_loss(self, data_inputs, data_outputs):
    """根据给定的训练数据计算网络loss值

    :param data_inputs:  输入部分，[count, batch_size, n_i]，count代表batch个数，batch_size代表mini batch的大小，n_i代表输入维度
    :param data_outputs: 输出部分，[count, batch_size, n_o]，count代表batch个数，batch_size代表mini batch的大小，n_o代表输出维度
    :return: 网络loss值，浮点数
    :raises:
      Exception: 训练数据的输入或者输出部分不是numpy数组或者个数不同
    """
    if not isinstance(data_inputs, np.ndarray) or not isinstance(data_outputs, np.ndarray):
      raise Exception('data is not numpy array')
    input_shape = data_inputs.shape
    output_shape = data_outputs.shape
    if len(input_shape) != len(output_shape) or input_shape[:-1] != output_shape[:-1]:
      raise Exception('data count is not equal')

    output = []

    for input in data_inputs:
      output.append(self.__feed_forward(input.T))

    return np.sum(np.square(np.array(output) - data_outputs)) / len(data_inputs)

  def __feed_forward(self, data_input):
    for layer in self.__layers:
      data_input = layer.feed_forward(data_input)
    return data_input

  @staticmethod
  def __get_init(size):
    return np.random.normal(0.0, 1.0 / size[-1], size)
