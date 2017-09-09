# -*- coding: UTF-8 -*-
"""运行神经网络"""
from mini_nn.neural_network import NeuralNetwork


def runner():
  """运行神经网络示例函数

  :return: 无返回值
  """
  training_data_input = [[0, 0], [0, 1], [1, 0], [1, 1]]
  training_data_output = [[0], [1], [1], [0]]
  nn = NeuralNetwork([2, 2, 1], 0.5)
  nn.fit(training_data_input, training_data_output, epoch=100000, interval=1000)
  print(nn.predict(training_data_input))


if __name__ == '__main__':
  runner()
