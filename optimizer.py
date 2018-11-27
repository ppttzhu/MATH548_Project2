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

import math
import scipy.optimize as optimize

GAMMA3 = 1  # GAMMA3 > 0
GAMMA4 = 0.5  # GAMMA4 < 1
UTILITY_FUNCTION = 2  # 1:x, 2:log(x), 3:1 − exp(−γx), 4:x^γ/γ


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
        self.q_up = (math.pow(1 + self.r, self.delta_t) - self.d) / (self.u - self.d)
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
                mini = optimize.minimize(self.utility_maximizer, 1, args=(v_t, s_t, s_up, s_down))
                if not mini.success:
                    print("utility_maximizer failed.")
                h1_list_.append(mini.x[0])
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
        q_list = generate_probablity_tree(self.q_up, self.t)
        p_list = generate_probablity_tree(self.p_up, self.t)
        
#         # Solve for Lagrange multiplier lambda
#         root = optimize.root(self.lambda_solver, 1, args=(q_list, p_list))
#         if not root.success:
#             print("lambda_solver failed.")
#         l_lambda = root.x[0]
#         print(l_lambda)
#          
#         # Solve for optimal attainable wealth at maturity 
#         w_list = []
#         for branch in range(self.t + 1):
#             b_t = math.pow(1 + self.r, self.t)
#             w = utility_function_prime_inverse(l_lambda * q_list[branch] / p_list[branch] / b_t)
#             w = utility_function_prime_inverse(l_lambda * q_list[branch] / p_list[branch] / b_t)
#             w_list.append(w)
        
        # Calculate optimal attainable wealth by analytic solution
        w_list = self.w_analytic_solution(q_list, p_list)
         
        # Solve for optimal attainable wealth and trading strategy at each time
        s_t_list = self.build_stock_tree_combine()
        h1_list = []
        v_list = [w_list]
        for time_step in range(self.t, 0, -1):
            v_list_ = []
            h1_list_ = []
            for branch in range(time_step):
                # Solve for trading strategy
                s = [s_t_list[time_step][branch], s_t_list[time_step][branch + 1]]
                x = [v_list[0][branch], v_list[0][branch + 1]]
                h0h1 = binomial_tree_hedging(s, x, self.r, time_step)
                # Solve for attainable wealth at t-1
                mma_part = h0h1[0] * math.pow(1 + self.r, time_step - 1)
                stock_part = h0h1[1] * s_t_list[time_step - 1][branch]
                v = mma_part + stock_part
                h = stock_part / v
                v_list_.append(v)
                h1_list_.append(h)
            v_list.insert(0, v_list_) 
            h1_list.insert(0, h1_list_) 
                            
        return [h1_list, v_list]
    
    def lambda_solver(self, l_lambda: float, q_list: list, p_list: list) -> float:
        """
        Calculate Lagrange multiplier lambda by iteration/solver
        """
        expect = 0
        b_t = math.pow(1 + self.r, self.t)
        if l_lambda < 0 and UTILITY_FUNCTION == 3:  # set constraints for solver (lambda > 0)
            l_lambda = abs(l_lambda) + 100
        for i in range(len(q_list)):
            expect += utility_function_prime_inverse(l_lambda * q_list[i] / p_list[i] / b_t) / b_t * q_list[i]
        return expect - self.v_0
    
    def w_analytic_solution(self, q_list: list, p_list: list) -> list:
        """
        Calculate optimal attainable wealth by analytic solution
        """
        if UTILITY_FUNCTION == 2:
            return self.w_analytic_solution2(q_list, p_list)
        elif UTILITY_FUNCTION == 3:
            if GAMMA3 <= 0:
                print("GAMMA of utility function3 should be positive.")
            return self.w_analytic_solution3(q_list, p_list, GAMMA3)
        elif UTILITY_FUNCTION == 4:
            if GAMMA4 >= 1:
                print("GAMMA of utility function4 should be smaller than 1.")
            return self.w_analytic_solution4(q_list, p_list, GAMMA4)
        else:
            print("Unsupported utility function.")
            return -1

    def w_analytic_solution2(self, q_list: list, p_list: list) -> float:
        """
        u(x) = log(x)
        """
        w_list = []
        b_t = math.pow(1 + self.r, self.t)
        for i in range(len(p_list)):
            l = q_list[i] / p_list[i]
            w_list.append(b_t / l * self.v_0)
        return w_list

    def w_analytic_solution3(self, q_list: list, p_list: list, gamma: float) -> list:
        """
        u(x) = 1 − exp(−γx), γ > 0
        """
        e1 = 0
        e2 = 0
        b_t = math.pow(1 + self.r, self.t)
        for i in range(len(p_list)):
            l = q_list[i] / p_list[i]
            e1 += p_list[i] * (l / b_t * math.log(l / b_t / gamma))
            e2 += p_list[i] * (l / b_t)
        w_list = []
        for i in range(len(p_list)):
            l = q_list[i] / p_list[i]
            w_list_ = ((self.v_0 * gamma + e1) / e2 - math.log(l / b_t / gamma)) / gamma 
            w_list.append(w_list_)
        return w_list

    def w_analytic_solution4(self, q_list: list, p_list: list, gamma: float) -> list:
        """
        u(x) = x^γ/γ, γ < 1
        """
        e = 0
        b_t = math.pow(1 + self.r, self.t)
        for i in range(len(p_list)):
            l = q_list[i] / p_list[i]
            e += p_list[i] * math.pow(l / b_t, -gamma / (1 - gamma))
        w_list = []
        for i in range(len(p_list)):
            l = q_list[i] / p_list[i]
            w_list_ = self.v_0 * math.pow(l / b_t, -1 / (1 - gamma)) / e
            w_list.append(w_list_)
        return w_list


def binomial_tree_hedging(s: list, x: list, r: float, t: float) -> list:
    """
    binomial tree support function to calculate hedging ratio on each node
    :param s: list of spot price of the underlying asset. Length 2, upward and downward side.
    :param x: list of value of the contingent claim. Length 2, upward and downward side.
    :param r: risk free rate (annual rate, expressed in terms of compounding)
    :param t: time of latter time step
    :return: list of hedging strategy of the contingent claim. Length 2, (H0, H1)
    """
    h1 = (x[1] - x[0]) / (s[1] - s[0])
    h0 = (x[1] - s[1] * h1) / math.pow(1 + r, t)

    hedging = [h0, h1]

    return hedging


def utility_function(x: float) -> float:
    """
    utility functions used in this program, selected from utility_function1-4
    """
    if UTILITY_FUNCTION == 1:
        return utility_function1(x)
    elif UTILITY_FUNCTION == 2:
        return utility_function2(x)
    elif UTILITY_FUNCTION == 3:
        if GAMMA3 <= 0:
            print("GAMMA of utility function3 should be positive.")
        return utility_function3(x, GAMMA3)
    elif UTILITY_FUNCTION == 4:
        if GAMMA4 >= 1:
            print("GAMMA of utility function4 should be smaller than 1.")
        return utility_function4(x, GAMMA4)
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
        if GAMMA3 <= 0:
            print("GAMMA of utility function3 should be positive.")
        return utility_function_prime_inverse3(x, GAMMA3)
    elif UTILITY_FUNCTION == 4:
        if GAMMA4 >= 1:
            print("GAMMA of utility function4 should be smaller than 1.")
        return utility_function_prime_inverse4(x, GAMMA4)
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
    u(x) = 1 − exp(−γx), γ > 0
    """
    return 1 - math.exp(-gamma * x)


def utility_function_prime_inverse3(y: float, gamma: float) -> float:
    """
    u(x) = 1 − exp(−γx), γ > 0
    """
    return -math.log(y / gamma) / gamma


def utility_function4(x: float, gamma: float) -> float:
    """
    u(x) = x^γ/γ, γ < 1
    """
    return math.pow(x, gamma) / gamma


def utility_function_prime_inverse4(y: float, gamma: float) -> float:
    """
    u(x) = x^γ/γ, γ < 1
    """
    return math.pow(y, 1 / (gamma - 1))


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


def generate_probablity_tree(p_up: float, step: int) -> float:
    tree = []
    for i in range(step + 1):
        p = math.pow(p_up, step - i) * math.pow(1 - p_up, i) * yang_hui_triangle(i , step)
        tree.append(p)
    return tree
