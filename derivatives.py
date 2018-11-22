#!/usr/bin/env python3
# encoding: utf-8

"""
@Project: MATH548_Project1
@FileName: derivatives.py
@Author: Kim Ki Hyeon, Lu Weikun, Peng Yixin, Zhou Nan
@Date: 2018/10/25
@Description：Include derivatives types such as options and forwards
@File URL: https://github.com/ppttzhu/MATH548_Project1/import
"""

from enum import Enum
import datetime


class CallPutType(Enum):
    call = 1
    put = 2


class ExerciseType(Enum):
    european = 1
    american = 2


class Option:
    def __init__(self, product_id: str, strike: float, maturity=datetime, call_put_type=CallPutType,
                 exercise_type=ExerciseType):
        """
        :param product_id: the ID of option
        :param strike: Strike price of option
        :param maturity: Maturity date of option
        :param call_put_type: Enum type indicator: 1 for call, 2 for put
        :param exercise_type: Enum type indicator: 1 for European, 2 for American
        """
        self.product_id = product_id
        self.strike = strike
        self.maturity = maturity
        self.call_put_type = call_put_type
        self.exercise_type = exercise_type


class Forward:
    def __init__(self, product_id: str, strike: float, maturity=datetime):
        """
        :param product_id: the ID of forward
        :param strike: Strike price of forward
        :param maturity: Maturity date of forward
        """
        self.product_id = product_id
        self.strike = strike
        self.maturity = maturity
