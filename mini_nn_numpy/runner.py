#-*- coding: UTF-8 -*-
import numpy as np
from mini_nn_numpy.neural_network import NeuralNetwork

def runner():
  data_input = np.expand_dims(np.array([[0,0],[0,1],[1,0],[1,1]]),0)
  data_output = np.expand_dims(np.array([[0],[1],[1],[1]]),0)
  nn = NeuralNetwork([2,3,1], learning_rate=0.5)
  nn.fit(data_input, data_output)
  print(nn.predict(np.squeeze(data_input)))
  print(nn.predict(np.array([[1,1]])))

if __name__ == '__main__':
  runner()