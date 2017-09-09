# -*- coding: UTF-8 -*-
import math


class Neuron:
  def __init__(self, weights, bias, activation='sigmoid'):
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
    if len(input) != self.input_nums:
      raise Exception('input size error, expect {0}, input {1}'.format(self.input_nums, len(input)))

    return sum([a * b for a, b in zip(self.weights, input)]) + self.bias

  def calculate_output(self, weighted_sum):
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
