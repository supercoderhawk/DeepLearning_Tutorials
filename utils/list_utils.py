# -*- coding: UTF-8 -*-
"""列表的一些常用操作封装"""
from utils.check import is_matrix_list, check_dimension
from itertools import starmap
from collections import deque


def flatten(a):
  """将嵌套列表展平

  :param a: 输入的列表
  :return: 返回一维的列表
  :raise: Exception: 输入不是列表

  """
  if not isinstance(a, list):
    raise Exception('input is not a list')

  result = a
  check = True

  while check:
    tmp = []
    for r in result:
      if isinstance(r, list):
        tmp.extend(r)
      else:
        tmp.append(r)

    # 检查是否还有嵌套列表
    check = any(map(lambda l: isinstance(l, list), tmp))
    result = tmp

  return result


def list_reshape(a, dims):
  """将一维列表转换成指定维度

  :param a: 待转换的一维列表
  :param dims: 维度列表，正整数
  :return: 转换后的列表
  :raises
    1. 输入参数a不是一维列表
    2. 输入参数dim不是一维列表，或元素不是正整数
    3. a的元素个数和维度dim代表的个数不同
  """
  try:
    if not check_dimension(a, 1):
      raise Exception('input is not an one dimension list')
    if not check_dimension(dims, 1, type=int, func=lambda i: i > 0):
      raise Exception('dimension element is not positive integer')
  except Exception as e:
    raise e

  dim_count = 1
  for dim in dims:
    dim_count *= dim
  if dim_count != len(a):
    raise Exception('element count is not equal to shape')

  queue = deque(a)
  for dim_size in dims[:0:-1]:
    dim_count //= dim_size
    for i in range(dim_count):
      queue.append([queue.popleft() for _ in range(dim_size)])

  return list(queue)


def list_size(a):
  """返回嵌套列表每一维的大小

  例：
    a=[] -> []
    a=[1,2,3] -> [3]
    a=[[1,2],[3,4]] -> [2,2]

    a=None, a=1, a=(1,3), a={1:2} 等引发异常
  :param a: 输入的嵌套列表
  :return: 输入列表每一维的长度
  :raise: Exception: 若输入不是列表或者输入的列表有没对齐的维度，引发异常。
  注：形如[[1,2],[3,4],5] 的列表为没对齐的嵌套列表

  """

  if not isinstance(a, list):
    raise Exception('input argument is not list')

  if not is_matrix_list(a):
    raise Exception('input argument is not a matrix-liked list')

  size = []

  if not a:
    return size

  size.append(len(a))

  while True:
    a = a[0]

    if not isinstance(a, list):
      return size

    size.append(len(a))


def list_add(a, b, *, elem=False, check=True):
  """类矩阵列表相加，类似于numpy的加运算符

  :param a: 待相加的参数
  :param b: 待相加的参数
  :param elem: 是否为元素相加（即是否为最内层的相加）
  :param check: 是否检查元素类型及维度，只在非最内层相加时起作用
  :return: 相加的结果
  :raises
    Exception:
      1. 待相加的参数不是类矩阵的列表或维度不同
      2. 元素为None或者对应元素无法相加
  """
  if elem:
    if a is None or b is None:
      raise Exception('none type can\'t be added')
    if isinstance(a, list) or isinstance(b, list):
      raise Exception('element is list')
    try:
      return a + b
    except Exception as e:
      raise e
  else:
    if check:
      if not isinstance(a, list):
        raise Exception('first argument is not a list')
      elif not isinstance(b, list):
        raise Exception('second argument is not a list')

      if not is_matrix_list(a):
        raise Exception('first argument is not a matrix-liked list')
      elif not is_matrix_list(b):
        raise Exception('second argument is not a matrix-liked list')

    size_a = list_size(a)
    size_b = list_size(b)

    if len(size_a) != len(size_b):
      raise Exception('two arguments size length is not equal')

    if not all(starmap(lambda a, b: a == b, zip(size_a, size_b))):
      raise Exception('two arguments size is not strictly equal')

    res = []
    for a_elem, b_elem in zip(a, b):
      if isinstance(a_elem, list):
        res.append(list_add(a_elem, b_elem, check=False))
      else:
        res.append(list_add(a_elem, b_elem, elem=True, check=False))

    return res
