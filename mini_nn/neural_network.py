# -*- coding: UTF-8 -*-
import random
import math
from mini_nn.neural_layer import NeuralLayer
from utils.check import check_dimension, check_dimension_size


class NeuralNetwork:
  def __init__(self, layers_count, learning_rate):
    if not check_dimension(layers_count, 1, type=int, func=lambda i: i > 0):
      raise Exception('layers count error')
    elif len(layers_count) < 2:
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
        biases=self.__get_randoms(curr)))
    # 创建输出层
    self.__layers.append(NeuralLayer(
      weights=self.__get_init_weights(layers_count[-2], layers_count[-1]),
      biases=self.__get_randoms(layers_count[-1]),
      activation='linear'))

  def fit(self, training_data_input, training_data_output, *, epoch=100, interval=10):
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

    for i in range(epoch):
      for input, output in zip(training_data_input, training_data_output):
        diff_weights = []
        diff_biases = []
        # 前向传播
        for layer in self.__layers:
          input = layer.feed_forward(input)

        # 反向传播
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
    if not isinstance(test_data, list):
      raise Exception('training input data is not list')

    if not check_dimension_size(test_data, [len(test_data), self.__input_nums]):
      raise Exception('training input data size error')

    result = []
    for test_input in test_data:
      result.append(self.__feed_forward_network(test_input))

    return result

  def calculate_loss(self, training_data_input, training_data_output, *, check=True):
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
