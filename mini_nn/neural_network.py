# -*- coding: UTF-8 -*-
"""神经网络模块"""
import random
import math
from mini_nn.neural_layer import NeuralLayer
from utils.check import check_dimension, check_dimension_size


class NeuralNetwork(object):
  """神经网络核心模块

  包括网络构建。计算前向输出和使用输入数据训练网络。

  """

  def __init__(self, layers_count, learning_rate, *, activation='sigmoid'):
    """创建一个神经网络

    :param layers_count: 网络尺寸，即每一层网络神经元个数，一维列表，列表元素均为正整数
    :param learning_rate: 学习率，浮点数
    :param activation: 隐藏层激活函数名
    :raises
      Exception:
        1. 网络尺寸列表错误，包括非1维列表，元素不是正整数，元素个数小于三个（至少有输入、输出和一个隐藏层）
        2. 学习率不是浮点数
    """
    if not check_dimension(layers_count, 1, type=int, func=lambda i: i > 0):
      raise Exception('layers count error')
    elif len(layers_count) < 3:
      raise Exception('layers count less than two')

    if not isinstance(learning_rate, float):
      raise Exception('learning rate is not float')

    self.__learning_rate = learning_rate
    self.__input_nums = layers_count[0]
    self.__output_nums = layers_count[-1]

    # 创建网络
    self.__layers = []
    # 创建隐藏层
    for prev, curr in zip(layers_count[:-2], layers_count[1:-1]):
      self.__layers.append(NeuralLayer(
        weights=self.__get_init_weights(prev, curr),
        biases=self.__get_randoms(curr),
        activation=activation))
    # 创建输出层
    self.__layers.append(NeuralLayer(
      weights=self.__get_init_weights(layers_count[-2], layers_count[-1]),
      biases=self.__get_randoms(layers_count[-1]),
      activation='linear'))

  def fit(self, training_data_input, training_data_output, *, epoch=100, interval=10):
    """使用数据训练网络

    :param training_data_input: 训练数据输入部分，[x, n]_i，x代表训练数据个数，n_i代表输入维度
    :param training_data_output: 训练数据输入部分对应的输出，[x, n_o]，x代表训练数据个数，n_o代表输出维度
    :param epoch: 迭代次数，为正整数
    :param interval: 打印次数和loss值的间隔，为正整数
    :return: 无返回值
    :raises:
      Exception:
        1. 训练数据的输入或者输出部分不是列表
        2. 训练数据输入或者输出部分不是类矩阵的列表
        3. 训练数据的输入输出个数不同
        4. 迭代次数或者间隔次数不是正整数
    """
    if not isinstance(training_data_input, list):
      raise Exception('training input data is not list')
    elif not isinstance(training_data_output, list):
      raise Exception('traing output data is not list')
    elif len(training_data_input) != len(training_data_output):
      raise Exception('training input and output data items not equal')

    if not check_dimension_size(training_data_input, [len(training_data_input), self.__input_nums]):
      raise Exception('training input data size error')

    if not check_dimension_size(training_data_output, [len(training_data_output), self.__output_nums]):
      raise Exception('training output data size error')

    if not epoch and (not isinstance(epoch, int) or not epoch <= 0):
      raise Exception('epoch is not positive integer')
    elif not interval and (not isinstance(interval, int) or not interval <= 0):
      raise Exception('interval is not positive integer')

    for i in range(1, epoch + 1):
      for input, output in zip(training_data_input, training_data_output):
        diff_weights = []
        diff_biases = []
        # 前向传播
        for layer in self.__layers:
          input = layer.feed_forward(input)

        # 反向传播，计算每层梯度
        back_propagation = [(pred - real) for pred, real in zip(self.__layers[-1].output, output)]
        for layer_i, layer in enumerate(self.__layers[::-1]):
          diff_weight, diff_bias = layer.diff_layer(back_propagation)
          diff_weights.append(diff_weight)
          diff_biases.append(diff_bias)
          back_propagation = layer.back_propagation

        # 更新权值
        for layer, diff_weight, diff_bias in zip(self.__layers[::-1], diff_weights, diff_biases):
          layer.update_layer(diff_weight, diff_bias, self.__learning_rate)

      if i % interval == 0:
        loss = self.calculate_loss(training_data_input, training_data_output, check=False)
        print('epoch: {0}, loss: {1}'.format(i, loss))

  def predict(self, test_data):
    """在当前参数下。给定输入值，求输出值

    :param test_data: 输入数据，[x, n_i]，其中x代表输入数据个数，n_i为网络的输入维度
    :return: 网络输出值 [x, n_o]，x尾输入数据个数，n_o为网络的输出维度
    :raises:
      Execption:
        1. 输入数据不是列表类型
        2. 输入数据第二维和网络输入维度不同
    """
    if not isinstance(test_data, list):
      raise Exception('training input data is not list')

    if not check_dimension_size(test_data, [len(test_data), self.__input_nums]):
      raise Exception('training input data size error')

    result = []
    for test_input in test_data:
      result.append(self.__feed_forward_network(test_input))

    return result

  def calculate_loss(self, training_data_input, training_data_output, *, check=True):
    """根据给定的训练数据计算网络loss值

    :param training_data_input:  训练数据输入部分，[x, n]_i，x代表训练数据个数，n_i代表输入维度
    :param training_data_output: 训练数据输入部分对应的输出，[x, n_o]，x代表训练数据个数，n_o代表输出维度
    :param check: 是否检查输入参数，默认为True
    :return: 网络loss值
    :raises:
      Exception:
        check为True:
        1. 训练数据的输入或者输出部分不是列表
        2. 训练数据输入或者输出部分不是类矩阵的列表
        3. 训练数据的输入输出个数不同
    """
    if check:
      if not isinstance(training_data_input, list):
        raise Exception('training input data is not list')
      elif not isinstance(training_data_output, list):
        raise Exception('traing output data is not list')
      elif len(training_data_input) != len(training_data_output):
        raise Exception('training input and output data items not equal')

      if not check_dimension_size(training_data_input, [len(training_data_input), self.__input_nums]):
        raise Exception('training input data size error')

      if not check_dimension_size(training_data_output, [len(training_data_output), self.__output_nums]):
        raise Exception('training output data size error')

    loss = 0.0
    for input, output in zip(training_data_input, training_data_output):
      loss += sum([(a - b) ** 2 for a, b in zip(self.__feed_forward_network(input), output)])
    return loss / len(training_data_input)

  def __feed_forward_network(self, input):
    # 前向传播
    for layer in self.__layers:
      input = layer.feed_forward(input)
    return self.__layers[-1].output

  def __get_init_weights(self, row, column):
    weights = []

    for i in range(row):
      weights.append(self.__get_randoms(column))

    return weights

  def __get_randoms(self, count):
    randoms = []

    for i in range(count):
      randoms.append(random.gauss(0, 1.0 / math.sqrt(count)))

    return randoms
