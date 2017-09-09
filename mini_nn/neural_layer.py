# -*- coding: UTF-8 -*-
from mini_nn.neuron import Neuron


class NeuralLayer:
  def __init__(self, weights, biases, activation='sigmoid'):
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
    self.input = input
    self.weighted_sums = [0] * self.neurons_num
    self.output = [0] * self.neurons_num

    for neuron_i, neuron in enumerate(self.neurons):
      self.weighted_sums[neuron_i] = neuron.calculate_weighted_sum(input)
      self.output[neuron_i] = neuron.calculate_output(self.weighted_sums[neuron_i])
    return self.output

  def diff_layer(self, back_propagation):
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
