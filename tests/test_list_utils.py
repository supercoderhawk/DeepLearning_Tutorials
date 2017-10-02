from unittest import TestCase
from utils.list_utils import *


# -*- coding: UTF-8 -*-
class TestListUtils(TestCase):
  def test_flatten(self):
    a = [1, 2, 3, [4, 5]]
    self.assertCountEqual(flatten(a), [1, 2, 3, 4, 5])

    b = [[1, 2, 3], [], [4, [5]]]
    self.assertCountEqual(flatten(b), [1, 2, 3, 4, 5])

    self.assertEqual(flatten([]), [])
    try:
      flatten(1)
    except Exception as e:
      self.assertEqual(e.args[0], 'input is not a list')

  def test_list_reshape(self):
    a = [1, 2, 3, 4, 5, 6, 7, 8]
    a1 = [[1, 2], [3, 4], [5, 6], [7, 8]]
    self.assertEqual(list_reshape(a, [4, 2]), a1)
    self.assertEqual(list_reshape([1], [1, 1]), [[1]])
    try:
      list_reshape(a, [4, 1])
    except Exception as e:
      self.assertEqual(e.args[0], 'element count is not equal to shape')
    try:
      list_reshape(a, [-1, 1])
    except Exception as e:
      self.assertEqual(e.args[0], 'dimension element is not positive integer')
    try:
      list_reshape(1, [1, 1])
    except Exception as e:
      self.assertEqual(e.args[0], 'input content is not list type')
    try:
      list_reshape(a, 1)
    except Exception as e:
      self.assertEqual(e.args[0], 'input content is not list type')

  def test_list_size(self):
    self.assertEqual(list_size([]), [])
    self.assertEqual(list_size([1]), [1])
    self.assertEqual(list_size([[1]]), [1, 1])
    try:
      list_size([1, [2]])
    except Exception as e:
      self.assertEqual(e.args[0], 'input argument is not a matrix-liked list')
    try:
      list_size(1)
    except Exception as e:
      self.assertEqual(e.args[0], 'input argument is not list')

  def test_list_add(self):
    a = [[1, 2], [3, 4]]
    b = [[11, 12], [13, 14]]
    ab = [[12, 14], [16, 18]]
    self.assertEqual(list_add(a, b), ab)
    self.assertEqual(list_add([], []), [])
    self.assertEqual(list_add([1.0], [10]), [11])
    try:
      list_add([1, 2], [[3, 4], [1, 2]])
    except Exception as e:
      self.assertEqual(e.args[0], 'two arguments size length is not equal')

    try:
      list_add([1, 2], 1)
    except Exception as e:
      self.assertEqual(e.args[0], 'second argument is not a list')

    try:
      list_add([(1, 2)], [11])
    except Exception as e:
      self.assertEqual(e.args[0], 'can only concatenate tuple (not "int") to tuple')
