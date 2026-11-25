"""Задача Лапласа на прямоугольнике и её аналитическое решение."""

from math import pi

import numpy as np

LX = 1.0
LY = pi / 2


def exact_solution(x, y):
    return np.exp(x) * np.cos(y)


def left_value(y):
    return np.cos(y)


def right_value(y):
    return np.e * np.cos(y)


def bottom_derivative(x):
    return np.zeros_like(x)


def top_derivative(x):
    return -np.exp(x)
