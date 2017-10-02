# 深度学习练习库

本项目是学习深度学习时写的一些程序和笔记。代码均有详细的注释。

## 目录

### mini_nn

不依赖任何第三方库实现的最简单的多层感知机神经网络。

说明：[mini_nn.ipynb（例子和说明）](https://github.com/supercoderhawk/DeepLearning_Tutorials/blob/master/mini_nn/mini_nn.ipynb)

### mini_nn_numpy

仅依赖`numpy`实现的简单多层感知机，代码比`mini_nn`简单很多。

### tensorflow

`tensorflow`例子

### utils

一些常用函数的封装

其中

#### check.py

* `check_dimension`：检查列表维度
* `check_dimension_size`：检查列表维度是否为给定的维度
* `is_matrix_list`：判断列表是否为类矩阵列表

#### list_utils.py

主要是实现列表的一些矩阵操作

* `list_flatten`：展平列表
* `list_reshape`：转换列表网维度
* `list_size`：获得列表维度
* `list_add`：列表相加

### tests

单元测试