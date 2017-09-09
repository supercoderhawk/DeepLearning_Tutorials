# -*- coding: UTF-8 -*-
"""常用的检查函数"""

def check_dimension(a, dim, *, type=None, func=None):
  """检查列表是否是指定维度

  例：
    [1,3] 为一维
    [[1,2],[3,4,5]] 为二维

    [1,[2]] 为非法输入，会引发异常

  :param a: 待判断的列表
  :param dim: 维度
  :param type: 数据类型，对每个元素进行检查，有一个不满足返回False, 默认为None，即不检查，仅可选int、float、str、tuple和dict
  :param func: 对每个数据进行判断的函数，返回值布尔类型，如果有一个返回值为False，函数返回False
  :return: 布尔值，其中，维度、类型和函数检查都通过返回True，否则返回False
  :raise
    Exception:
      1. 类型在指定返回
      2. 函数不是可调用的
      3. 维度dim不是正整数

  """
  if not a:
    return False

  if type:
    types = [int, float, str, tuple, dict]
    if type not in types:
      raise TypeError('type is not allowed')

  if func:
    if not callable(func):
      raise Exception('function can\'t be called')

  if not isinstance(dim, int):
    raise TypeError('dimension type is not integer')
  elif dim <= 0:
    raise Exception('dimension is not a positive integer')

  result = a

  for i in range(dim - 1):
    check = all(map(lambda l: isinstance(l, list), result))
    if not check:
      return False
    result = [a for l in result for a in l]

  if func and type:
    # 既要检查类型又有附加函数
    check = all(map(lambda i: isinstance(i, type) and func(i), result))
  elif func:
    # 仅有附加函数
    check = all(map(lambda i: not isinstance(i, list) and func(i), result))
  elif type:
    # 仅要检查类型
    check = all(map(lambda i: isinstance(i, type), result))
  else:
    # 两者都没有，仅保证不是列表即可
    check = all(map(lambda i: not isinstance(i, list), result))

  return check


def check_dimension_size(a, size):
  """检查列表维度是否等于给定值，类似于numpy中shape的比较

  例：
    a=[1,2], size=[2]   -> True
    a=[1,2], size=[2,1] -> False
    a=[1,2], size=[3]   -> False
    a=[[1,2],[3,4]], size=[2,2] -> True
    a=[[1,2],[3,4,5]], size=[2,2] -> False
    a=[[1,2],[3,[4]]], size=[2,2] -> False

    a = [1,2] size=2 引发异常

  :param a: 待判断的的列表
  :param size: 列表每一维的大小
  :return: 布尔值，若每维均相同，返回True，否则返回False
  :raise:
    Exception:
      1. 维度不是一维列表或列表元素不是正整数

  """
  if not size:
    raise Exception('dimension size empty')
  if not check_dimension(size, 1, type=int, func=lambda i: i > 0):
    raise Exception('dimension size error')

  if len(a) != size[0]:
    return False

  flatten_list = a
  for s in size[1:]:
    for item in flatten_list:
      if not isinstance(item, list):
        return False
      if len(item) != s:
        return False
    flatten_list = [a for l in flatten_list for a in l]

  return all(map(lambda i: not isinstance(i, list), flatten_list))


def is_matrix_list(a):
  """判断列表是否矩阵

  判断列表是否是矩阵，即各维大小是够相同，矩阵维度从1~n。元素类型除列表类型外均可。

  例：
    a=[] -> True
    a=[1,2] -> True
    a=[[1],2] -> False
    a=[[1],[2]] -> True
    a=[[1],[2,3]] -> False

  :param a: 待判断的列表
  :return: 若列表为矩阵，返回True，否则返回False
  :raise:
    Exception:
      1. 输入参数不是列表。
  """
  if not isinstance(a, list):
    raise Exception('input data is not array')

  # a为空列表
  if not a:
    return True

  result = a
  while True:
    if isinstance(result[0], list):
      for l in result[1:]:
        if not isinstance(l, list):
          return False
        if len(result[0]) != len(l):
          return False
      result = [a for l in result for a in l]
    else:
      for num in result[1:]:
        if isinstance(num, list):
          return False
      break

  return True
