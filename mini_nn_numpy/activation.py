# -*- coding: UTF-8 -*-
"""激活函数模块"""
import numpy as np


class Activation(object):
  """激活函数类，包括常见的激活函数及其导数
  Attributes:
    __ACTIVATION: 当前类中所有的激活函数名

  """
  __ACTIVATION = ['sigmoid', 'tanh', 'relu', 'linear']

  @staticmethod
  def has(name):
    """判断当前类中是否存在给定名字的激活函数

    :param name: 激活函数名
    :return: 存在，返回True，否则返回False
    :raises:
      Exception: 输入的名字不是一个字符串
    """
    if not isinstance(name, str):
      raise Exception('name is not a string')
    return name in Activation.__ACTIVATION

  @staticmethod
  def get(item):
    """返回给定名字的激活函数及其导数

    例：
      acti, diff_acti = Activation['sigmoid']    sigmoid函数及其导数
      acti2, diff_acti2 = Activation['fx']  均为None
      acti3, diff_acti3 = Activation[123]  引发异常

    :param item: 激活函数名
    :return: 激活函数及其导数组成的元组
    :raises:
      Exception: 输入的名字不是字符串
    """
    if not isinstance(item, str):
      raise Exception('item is not a string')
    if item == 'sigmoid':
      return Activation.__sigmoid, Activation.__diff_sigmoid
    elif item == 'tanh':
      return Activation.__tanh, Activation.__diff_tanh
    elif item == 'relu':
      return Activation.__relu, Activation.__diff_relu
    elif item == 'linear':
      return Activation.__linear, Activation.__diff_linear
    else:
      return None, None

  @staticmethod
  def __sigmoid(input):
    return 1 / (1 + np.exp(-input))

  @staticmethod
  def __diff_sigmoid(input):
    output = Activation.__sigmoid(input)
    return output * (1 - output)

  @staticmethod
  def __tanh(input):
    return 2 * Activation.__sigmoid(2 * input) - 1

  @staticmethod
  def __diff_tanh(input):
    return 1 - Activation.__tanh(input) ** 2

  @staticmethod
  def __relu(input):
    return np.maximum(input, 0)

  @staticmethod
  def __diff_relu(input):
    return input > 0

  @staticmethod
  def __linear(input):
    return input

  @staticmethod
  def __diff_linear(input):
    return np.ones(input.shape, input.dtype)
