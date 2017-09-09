from unittest import TestCase
from utils.check import *


# -*- coding: UTF-8 -*-
class TestCheck(TestCase):
  def test_check_dimension(self):
    a = [1, 2, 3]
    self.assertEqual(check_dimension(a, 1, type=int), True)
    self.assertEqual(check_dimension(a, 1, type=float), False)
    self.assertEqual(check_dimension(a, 2, type=int), False)
    self.assertEqual(check_dimension(a, 1), True)
    self.assertEqual(check_dimension(a, 1, func=lambda i: i > 0), True)
    self.assertEqual(check_dimension(a, 1, type=int, func=lambda i: i > 0), True)
    try:
      check_dimension(a, 1, type=list)
    except Exception as e:
      self.assertEqual(e.args[0], 'type is not allowed')

    b = [[-1, 2, 3], [4, 5, 6]]
    self.assertEqual(check_dimension(b, 2, type=int), True)
    self.assertEqual(check_dimension(b, 2, type=int, func=lambda i: i > 0), False)
    self.assertEqual(check_dimension(b, 2, func=lambda i: i > 0), False)

    c = [1, 2, 3, [4, 5]]
    self.assertEqual(check_dimension(c, 1, type=int), False)

    d = []
    self.assertEqual(check_dimension(d, 1, type=int), False)

    e = [1.2, 3, 4, 5]
    self.assertEqual(check_dimension(e, 1), True)
    self.assertEqual(check_dimension(e, 1, type=int), False)

  def test_check_dimension_size(self):
    a = [1, 2, 3]
    self.assertEqual(check_dimension_size(a, [3]), True)
    self.assertEqual(check_dimension_size(a, [3, 1]), False)
    self.assertEqual(check_dimension_size(a, [2]), False)
    try:
      check_dimension_size(a, [[2]])
    except Exception as e:
      self.assertEqual(e.args[0], 'dimension size error')
    try:
      check_dimension_size(a, [])
    except Exception as e:
      self.assertEqual(e.args[0], 'dimension size empty')

    b = [[1, 2], [3, 4]]
    self.assertEqual(check_dimension_size(b, [2, 2]), True)
    self.assertEqual(check_dimension_size(b, [2]), False)

  def test_is_matrix_list(self):
    self.assertEqual(is_matrix_list([1,2,3]),True)
    self.assertEqual(is_matrix_list([]),True)
    self.assertEqual(is_matrix_list([1,[2]]),False)
    self.assertEqual(is_matrix_list([[1], [2]]), True)
    self.assertEqual(is_matrix_list([[1,2,3,4], [2,2,3,4]]), True)
    self.assertEqual(is_matrix_list([[[1]],[[2]]]), True)
    self.assertEqual(is_matrix_list([[1,2,3],[[1,2,3]]]), False)
    self.assertEqual(is_matrix_list([[1, 2, 3], [1,2,[1, 2, 3]]]), False)