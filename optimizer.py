#!/usr/bin/env python3
# encoding: utf-8

"""
@Project: MATH548_Project2
@FileName: optimizer.py
@Author: Peng, Yixin
@Date: 2018/11/22
@Description：Perform Portfolio Optimization with Dynamic Programming Principle and Martingale Method
@File URL: https://github.com/ppttzhu/MATH548_Project2
"""

import numpy as np
import math
import scipy.optimize as optimize

GAMMA = 1
UTILITY_FUNCTION = 1  # 1:x, 2:log(x), 3:1 − exp(−γx), 4:x^γ/γ


class PortfolioOptimizer:

    def __init__(self, v_0: float, u: float, d: float, r: float, t: int, p_up: float):
        """
        :param v_0: the initial capital
        :param u: the range if stock price goes up
        :param d: the range if stock price goes down
        :param r: interest rate paid by bank
        :param t: period of investment
        :param p_up: statistic probability of stock price going up
        :return: self object
        """
        self.v_0 = v_0
        self.u = u
        self.d = d
        self.r = r
        self.t = t
        self.p_up = p_up
        
        self.delta_t = 1        
        self.q_up = self.rn_probability(self.u, self.d, self.r, self.delta_t)
        self.p_down = 1 - self.p_up
        self.q_down = 1 - self.q_up

    def build_stock_tree(self) -> list:
        """
        Construct binomial tree, don't combine middle branches
        :return: stock price tree
        """
        s_t_list = [[1]]  # scaled s_0
        for time_step in range(1, self.t + 1):
            s_t = []
            for branch in range(int(math.pow(2, time_step))):
                pre_branch = int(branch / 2)
                s_pre_branch = s_t_list[time_step - 1][pre_branch]
                up_down = branch % 2  # 0 for up, 1 for down
                if up_down:
                    s_t. append(s_pre_branch * self.d)
                else:
                    s_t. append(s_pre_branch * self.u)
            s_t_list.append(s_t)
        
        return s_t_list

    def build_stock_tree_combine(self) -> list:
        """
        Construct binomial tree, combine middle branches
        :return: stock price tree
        """
        s_t_list = []
        for time_step in range(self.t + 1):
            s_t = []
            for branch in range(time_step + 1):
                s_t. append(math.pow(self.u, time_step - branch) * math.pow(self.d, branch))
            s_t_list.append(s_t)
        
        return s_t_list

    def dynamic_programming_principle(self) -> list:
        """
        Dynamic Programming Principle to find trading strategy
        :return: stock trading strategy and portfolio value tree: [h1, v]
        """
        s_t_list = self.build_stock_tree()
        h1_list = []  # percentage of stock in portfolio
        v_list = [[self.v_0]]  # value of portfolio
        for time_step in range(len(s_t_list) - 1):
            
            # Calculate h(t)
            h1_list_ = []
            for branch in range(len(s_t_list[time_step])):
                s_t = s_t_list[time_step][branch]
                s_up = s_t_list[time_step + 1][2 * branch]
                s_down = s_t_list[time_step + 1][2 * branch + 1]
                v_t = v_list[time_step][branch]
                h1 = optimize.minimize(self.utility_maximizer, 1, args=(v_t, s_t, s_up, s_down)).x[0]
                h1_list_.append(h1)
            h1_list.append(h1_list_)
            
            # Calculate v(t+1)
            v_list_ = []
            for branch in range(len(s_t_list[time_step + 1])):
                pre_branch = int(branch / 2)
                s_pre_branch = s_t_list[time_step][pre_branch]
                h_pre_branch = h1_list[-1][pre_branch]
                v_pre_branch = v_list[-1][pre_branch]
                s_branch = s_t_list[time_step + 1][branch]
                v_branch = h_pre_branch * v_pre_branch / s_pre_branch * s_branch + (1 - h_pre_branch) * v_pre_branch * (1 + self.r)
                v_list_.append(v_branch)
            v_list.append(v_list_)
            
        return [h1_list, v_list]
    
    def utility_maximizer(self, h1: float, v_t: float, s_t: float, s_up: float, s_down: float):
        # if h1 is a percentage of v_t
        v_up = h1 * v_t / s_t * s_up + (1 - h1) * v_t * (1 + self.r)
        v_down = h1 * v_t / s_t * s_down + (1 - h1) * v_t * (1 + self.r)
#         # if h1 is the number of stock holding
#         v_up = v_t * (1 + self.r) + (s_up - s_t * (1 + self.r)) * h1
#         v_down = v_t * (1 + self.r) + (s_down - s_t * (1 + self.r)) * h1
        max_util = self.p_up * utility_function(v_up) + self.p_down * utility_function(v_down)
        return -max_util
    
    def martingale_method(self) -> list:
        """
        Martingale Method to find trading strategy
        :return: stock trading strategy and portfolio value tree: [h1, v]
        """
        # Solve for Lagrange multiplier lambda
        q_list = self.generate_probablity_tree(self.q_up, self.t)
        p_list = self.generate_probablity_tree(self.p_up, self.t)
        l_lambda = optimize.fsolve(self.lambda_solver, 1, args=(q_list, p_list))
        print(l_lambda)
        
#         return [h1_list, v_list]
    
    def lambda_solver(self, l_lambda: float, q_list: list, p_list: list) -> float:
        expect = 0
        b_t = math.pow(1 + self.r, self.t)
        for i in range(len(q_list)):
            expect += utility_function_prime_inverse(l_lambda * q_list[i] / p_list[i] / b_t) / b_t * q_list[i]
        return expect - self.v_0

    def generate_probablity_tree(self, p_up: float, step: int) -> float:
        tree = []
        for i in range(step + 1):
            p = math.pow(p_up, i) * math.pow(1 - p_up, step - i) * yang_hui_triangle(i , step)
            tree.append(p)
        return tree

    # Supporting static functions

    @staticmethod
    def rn_probability(up: float, down: float, r: float, delta_t: float) -> list:
        """
        calculate risk neutual probability
        :param up: calibrated up range
        :param down: calibrated down range
        :param r: risk free rate (annual rate, expressed in terms of compounding)
        :param delta_t: time interval of one branch (expressed in years)
        :return: up probability
        """

        # risk neutral probability
        q_up = (math.pow(1 + r, delta_t) - down) / (up - down)

        return q_up


def utility_function(x: float) -> float:
    """
    utility functions used in this program, selected from utility_function1-4
    """
    if UTILITY_FUNCTION == 1:
        return utility_function1(x)
    elif UTILITY_FUNCTION == 2:
        return utility_function2(x)
    elif UTILITY_FUNCTION == 3:
        return utility_function3(x, GAMMA)
    elif UTILITY_FUNCTION == 4:
        return utility_function4(x, GAMMA)
    else:
        print("Unsupported utility function.")
        return -1


def utility_function_prime_inverse(x: float) -> float:
    """
    Prime and inverted utility functions, selected from utility_function_prime_inverse1-4
    """
    if UTILITY_FUNCTION == 2:
        return utility_function_prime_inverse2(x)
    elif UTILITY_FUNCTION == 3:
        return utility_function_prime_inverse3(x, GAMMA)
    elif UTILITY_FUNCTION == 4:
        return utility_function_prime_inverse4(x, GAMMA)
    else:
        print("Unsupported utility function.")
        return -1


def utility_function1(x: float) -> float:
    """
    u(x) = x
    """
    return x


def utility_function2(x: float) -> float:
    """
    u(x) = log(x)
    """
    return math.log(x)


def utility_function_prime_inverse2(y: float) -> float:
    """
    u(x) = log(x)
    """
    return 1 / y


def utility_function3(x: float, gamma: float) -> float:
    """
    u(x) = 1 − exp(−γx)
    """
    return 1 - math.exp(-gamma * x)


def utility_function_prime_inverse3(y: float, gamma: float) -> float:
    """
    u(x) = 1 − exp(−γx)
    """
    return -math.log(y / gamma) / gamma


def utility_function4(x: float, gamma: float) -> float:
    """
    u(x) = x^γ/γ
    """
    return math.pow(x, gamma) / gamma


def utility_function_prime_inverse4(y: float, gamma: float) -> float:
    """
    u(x) = x^γ/γ
    """
    return math.pow(gamma * y, 1 / gamma)


def yang_hui_triangle(n: int, k: int) -> int:
    """
    Calculate C(n, k) using Yang Hui's (Pascal's) triangle, n <= k
    https://en.wikipedia.org/wiki/Pascal%27s_triangle#Binomial_expansions
    :param n: select n
    :param k: total number k
    :return: the number of way to select n from k
    """

    # parameters out of range
    if n > k or k < 0 or n < 0:
        print('Error: Parameters out of range.')
        return -1

    list1 = []
    for i in range(k + 1):
        list0 = list1
        list1 = []
        for j in range(i + 1):
            if j == 0 or j == i:
                list1.append(1)
            else:
                list1.append(list0[j - 1] + list0[j])

    return list1[n]
