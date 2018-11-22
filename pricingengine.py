#!/usr/bin/env python3
# encoding: utf-8

"""
@Project: MATH548_Project1
@FileName: pricingengine.py
@Author: Kim Ki Hyeon, Lu Weikun, Peng Yixin, Zhou Nan
@Date: 2018/10/25
@Description：Include derivatives pricing types and method
@File URL: https://github.com/ppttzhu/MATH548_Project1/import
https://github.com/ppttzhu/MATH548_Project2.git
"""

import numpy as np
import math
import scipy.optimize as optimize
from scipy.stats import norm
from derivatives import *


class PricingMethod(Enum):
    binomial_tree_model = 1
    bs_baw_benchmarking_model = 2


class CompoundingMethod(Enum):
    discrete_compounded = 1
    continuous_compounded = 2


class TreeAssumption(Enum):
    p_half = 1
    ud_1 = 2
    ud_calibrated = 3


class VolatilityCalculation(Enum):
    estimation = 1
    calibration = 2


PRICING_METHOD = PricingMethod.binomial_tree_model
COMPOUNDING_METHOD = CompoundingMethod.continuous_compounded
TREE_ASSUMPTION = TreeAssumption.ud_1
VOLATILITY_CALIBRATION = VolatilityCalculation.calibration
BUSINESS_DAYS_PER_YEAR = 252
BINOMIAL_TREE_STEP = 20


class OptionPricingEngine:
    def __init__(self, pricing_date=datetime, option=Option):
        """
        :param pricing_date: Date to calculate fair value
        :param option: Option object
        :return:
        """
        self.option = option
        self.pricing_date = pricing_date

        # cp: indicator 1 for call, -1 for put
        if option.call_put_type is CallPutType.call:
            self.cp = 1
        else:
            self.cp = -1

        # t: time to maturity(expressed in years, assume act/365 daycounter)
        self.t = (option.maturity - pricing_date).days / 365

    def calibrate(self, s: float, r_curve: list, b: float, s_history: list = [], options: list = [],
                  market_price: list = []) -> list:
        """
        Calibrate pricing models
        :param s: spot price of the underlying asset (ex-dividend)
        :param r_curve: risk free rate curve(annual rate, expressed in terms of continuous compounding)
        :param b: dividend rate of underlying asset (annual rate)
        :param s_history: history price of the underlying asset
        :param options: list of options objects for calibration
        :param market_price: the market price of options
        :return: a list of calibrated parameters: [sigma_log_return, up_down]
        """

        # Interpolate risk free rate
        r = interpol(r_curve[0], r_curve[1], self.t)

        delta_t = self.t / BINOMIAL_TREE_STEP

        # Calculate volatility
        if PRICING_METHOD is PricingMethod.bs_baw_benchmarking_model or \
                (PRICING_METHOD is PricingMethod.binomial_tree_model and TREE_ASSUMPTION is TreeAssumption.ud_1):
            if VOLATILITY_CALIBRATION is VolatilityCalculation.estimation:
                sigma_log_return = self.volatility_log_return(s_history, BUSINESS_DAYS_PER_YEAR)
            elif VOLATILITY_CALIBRATION is VolatilityCalculation.calibration:
                guess = 0.1
                sigma_bounds = ((0, 10),)
                optimize_result = optimize.minimize(self.volatility_optimizer, guess,
                                                    args=(options, market_price, s, r_curve, b),
                                                    bounds=sigma_bounds)
                sigma_log_return = optimize_result.x[0]
        else:
            sigma_log_return = 0

        # Calculate up and down range
        if PRICING_METHOD is PricingMethod.binomial_tree_model:
            if TREE_ASSUMPTION is TreeAssumption.ud_1:
                up_down_p = self.ud_1_list(sigma_log_return, r, delta_t)
            elif TREE_ASSUMPTION is TreeAssumption.p_half:
                sigma_return = self.volatility_return(s_history, BUSINESS_DAYS_PER_YEAR)
                miu_return = self.expectation_return(s_history)
                up_down_p = self.p_half_list(sigma_return, miu_return, delta_t)
            elif TREE_ASSUMPTION is TreeAssumption.ud_calibrated:

                guess = [math.exp(r * delta_t), math.exp(-r * delta_t)]
                ud_bounds = ((1, 10), (0, 1))  # bound for up is (1, 10) and bound for down is (0, 1)
                optimize_result = optimize.minimize(self.up_down_optimizer, guess,
                                                    args=(options, market_price, s, r_curve, b),
                                                    bounds=ud_bounds)
                # print(optimize_result)
                up_down = optimize_result.x
                up_down_p = self.ud_calibrated_list(up_down[0], up_down[1], r, delta_t)
        else:
            up_down_p = []

        return [sigma_log_return, up_down_p]

    def npv(self, s: float, r_curve: list, b: float, sigma: float = 0, up_down_p: list = []) -> list:
        """
        The summary of pricing models
        :param s: spot price of the underlying asset (ex-dividend)
        :param r_curve: risk free rate curve(annual rate, expressed in terms of continuous compounding)
        :param b: dividend rate of underlying asset (annual rate)
        :param sigma: calibrated sigma
        :param up_down_p: calibrated up and down
        :return: if binomial model: the output is: [npv, h0_tree, h1_tree, s_tree, bond_tree, option_tree], else, [npv]
        """

        # Interpolate risk free rate
        r = interpol(r_curve[0], r_curve[1], self.t)

        if PRICING_METHOD is PricingMethod.binomial_tree_model:
            return self.binomial_tree_backstep(s, r, b, up_down_p, BINOMIAL_TREE_STEP)

        elif PRICING_METHOD is PricingMethod.bs_baw_benchmarking_model:
            # For benchmark

            if self.option.exercise_type is ExerciseType.european:
                npv = self.bs_formula(s, r, b, sigma)
            elif self.option.exercise_type is ExerciseType.american:
                npv = self.baw_formula(s, r, b, sigma)
            return [npv[0]]

        print('Error: Method not supported.')
        return -1

    def binomial_tree_european_analytic(self, s: float, r: float, b: float, up_down_p: list, n: int) -> float:
        """
        Binomial tree method to price European call and put option (Just for benchmarking)
        We can get same result with binomial_tree_backstep function
        """

        up = up_down_p[0]
        down = up_down_p[1]
        q_up = up_down_p[2]
        q_down = up_down_p[3]

        npv = 0
        # price by adding discounted payoff
        for i in range(n + 1):
            payoff = 0
            if self.cp == 1:
                payoff = self.call_payoff((s * compounding_factor(self.t, - b, COMPOUNDING_METHOD)
                                           * math.pow(up, i) * math.pow(down, n - i)), self.option.strike)
            elif self.cp == -1:
                payoff = self.put_payoff((s * compounding_factor(self.t, - b, COMPOUNDING_METHOD)
                                          * math.pow(up, i) * math.pow(down, n - i)), self.option.strike)
            npv += discount_factor(self.t, r, COMPOUNDING_METHOD) * yang_hui_triangle(i, n) \
                   * payoff * math.pow(q_up, i) * math.pow(q_down, n - i)

        return npv

    def binomial_tree_backstep(self, s: float, r: float, b: float, up_down_p: list, n: int) -> list:
        """
        Binomial tree method to price American/European, call/put option step by step and derive hedging strategy
        """

        delta_t = self.t / n

        up = up_down_p[0]
        down = up_down_p[1]
        q_up = up_down_p[2]
        q_down = up_down_p[3]

        h0_tree = []
        h1_tree = []
        s_tree = []
        bond_tree = []
        option_tree = []

        # price by rolling back the payoff step by step
        z_t_1_discounted = []
        for time_step in range(n, -1, -1):
            z_t = []
            s_t_list = []
            b_t_list = []
            h0_t_list = []
            h1_t_list = []
            for branch in range(time_step + 1):
                # calculate intrinsic value
                payoff = 0

                s_t = s * compounding_factor(time_step * delta_t, - b, COMPOUNDING_METHOD) \
                      * math.pow(up, time_step - branch) * math.pow(down, branch)

                s_t_list.append(s_t)
                b_t_list.append(compounding_factor(time_step * delta_t, r, COMPOUNDING_METHOD))

                if self.option.exercise_type is ExerciseType.american:
                    # American option: buyer can choose to exercise before or after dividend
                    s_t_ex_div = s * compounding_factor(max((time_step - 1), 0) * delta_t, - b, COMPOUNDING_METHOD) \
                                 * math.pow(up, time_step - branch) * math.pow(down, branch)
                else:
                    s_t_ex_div = s_t

                if self.cp == 1:
                    payoff = max(self.call_payoff(s_t_ex_div, self.option.strike),
                                 self.call_payoff(s_t, self.option.strike))
                elif self.cp == -1:
                    payoff = max(self.put_payoff(s_t_ex_div, self.option.strike),
                                 self.put_payoff(s_t, self.option.strike))

                # calculate max of Y(t-1) and Z*(t-1)
                if time_step == n:  # last step Z_T = Y_T
                    z_t.append(payoff)
                else:
                    if self.option.exercise_type is ExerciseType.american:
                        # American option
                        z_t.append(max(payoff, z_t_1_discounted[branch]))
                    else:
                        # European option
                        z_t.append(z_t_1_discounted[branch])

            # Calculate hedging strategy
            if time_step != 0:
                for branch in range(time_step):
                    s_up_down = [s_t_list[branch], s_t_list[branch + 1]]
                    x_up_down = [z_t[branch], z_t[branch + 1]]
                    hedging = binomial_tree_hedging(s_up_down, x_up_down, r, delta_t * time_step)
                    h0_t_list.append(hedging[0])
                    h1_t_list.append(hedging[1])
                h0_tree.insert(0, h0_t_list)
                h1_tree.insert(0, h1_t_list)

            s_tree.insert(0, s_t_list)
            bond_tree.insert(0, b_t_list)
            option_tree.insert(0, z_t)

            # Discount z_t to z_t_1_discounted
            z_t_1_discounted = []
            if time_step != 0:
                for i in range(time_step):
                    z_t_1_discounted.append(discount_factor(delta_t, r, COMPOUNDING_METHOD)
                                            * (q_up * z_t[i] + q_down * z_t[i + 1]))
            else:
                npv = z_t[0]

        return [npv, h0_tree, h1_tree, s_tree, bond_tree, option_tree]

    # Black–Scholes formula for benchmarking European Option

    def bs_formula(self, s: float, r: float, b: float, sigma: float) -> float:
        """
        Black–Scholes formula to price European call and put option
        https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model
        """
        d1 = self.bs_formula_d1(s, self.option.strike, self.t, r, b, sigma)
        d2 = self.bs_formula_d2(s, self.option.strike, self.t, r, b, sigma)

        npv = self.cp * (norm.cdf(self.cp * d1) * s * math.exp(-b * self.t)
                         - norm.cdf(self.cp * d2) * self.option.strike * math.exp(-r * self.t))

        return npv

    # Barone-Adesi and Whaley formula for benchmarking American Option

    def baw_formula(self, s: float, r: float, b: float, sigma: float) -> float:
        """
        Barone-Adesi and Whaley formula to price American call and put option
        https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model#American_options
        http://finance.bi.no/~bernt/gcc_prog/algoritms_v1/algoritms/node24.html
        """
        # set the accuracy requirement for quadratic approximation
        tolerance = 1.0e-6  # set the accuracy requirement for quadratic approximation
        max_iterations = 500  # set the max iterations number

        cp = self.cp
        t = self.t
        k = self.option.strike

        # to do: add dividend
        european = self.bs_formula(s, r, b, sigma)
        # If dividend rate is zero and call option, never early exercised
        if (b - 0.0) < tolerance and cp == 1:
            return european

        nn = 2.0 * b / (sigma * sigma)
        m = 2.0 * r / (sigma * sigma)
        k_cap = 1.0 - math.exp(-r * t)
        q = (-(nn - 1) + cp * math.sqrt(math.pow((nn - 1), 2.0) + (4 * m / k_cap))) * 0.5

        # seed value from paper
        qu = 0.5 * ((-nn - 1.0) + cp * math.sqrt(math.pow((nn - 1), 2.0) + 4.0 * m))
        su = k / (1.0 - 1.0 / qu)
        h2 = - (b * t + cp * 2.0 * sigma * math.sqrt(t)) * (k / (su - k))
        s_seed = k + (su - k) * (1.0 - math.exp(h2))

        # Using Newton Raphson algorithm to find critical price Si
        no_iterations = 0
        si = s_seed
        g = 1
        gprime = 1.0

        while math.fabs(g) > tolerance and math.fabs(
                gprime) > tolerance and no_iterations < max_iterations and si > 0.0:
            e = self.bs_formula(si, r, b, sigma)
            d1 = self.bs_formula_d1(si, k, t, r, b, sigma)
            g = cp * (si - k - (1.0 / q) * si * (1 - math.exp((b - r) * t) * norm.cdf(cp * d1))) - e
            gprime = cp * (1.0 - 1.0 / q) * (1.0 - math.exp((b - r) * t) * norm.cdf(cp * d1)) \
                     + (1.0 / q) * math.exp((b - r) * t) * norm.pdf(cp * d1) * (1.0 / (sigma * math.sqrt(t)))
            si = si - (g / gprime)
            no_iterations = no_iterations + 1

        if math.fabs(g) > tolerance:
            s_star = s_seed  # did not converge
        else:
            s_star = si

        if s * cp >= s_star * cp:
            american = (s - k) * cp
        else:
            d1 = self.bs_formula_d1(si, k, t, r, b, sigma)
            a = (cp * s_star / q) * (1.0 - math.exp((b - r) * t) * norm.cdf(cp * d1))
            american = european + a * math.pow((s / s_star), q)

        return max(american, european)

    # Optimizers for calibration

    def volatility_optimizer(self, sigma: float, options: list, market_price: list, s: float, r_curve: list,
                             b: float) -> float:
        """
        Find the optimal volatility to minimize the squared sum of error between model price and market price
        """
        r = interpol(r_curve[0], r_curve[1], self.t)
        delta_t = self.t / BINOMIAL_TREE_STEP
        up_down_p = self.ud_1_list(sigma, r, delta_t)

        ess = 0
        for i in range(len(options)):
            pricing_engine = OptionPricingEngine(self.pricing_date, options[i])
            model_price = pricing_engine.npv(s, r_curve, b, sigma, up_down_p)
            ess += math.pow(model_price[0] - market_price[i], 2)
        return ess

    def up_down_optimizer(self, up_down_q: list, options: list, market_price: list, s: float, r_curve: list,
                          b: float) -> float:
        """
        Find the optimal up and down range to minimize the squared sum of error between model price and market price
        """
        up, down = up_down_q
        r = interpol(r_curve[0], r_curve[1], self.t)
        delta_t = self.t / BINOMIAL_TREE_STEP
        up_down_p = self.ud_calibrated_list(up, down, r, delta_t)

        ess = 0
        for i in range(len(options)):
            pricing_engine = OptionPricingEngine(self.pricing_date, options[i])
            model_price = pricing_engine.npv(s, r_curve, b, 0, up_down_p)
            ess += math.pow(model_price[0] - market_price[i], 2)
        return ess

    # Supporting static functions

    @staticmethod
    def ud_1_list(sigma: float, r: float, delta_t: float) -> list:
        """
        calculate up and down range and probability for ud_1 tree assumption
        :param sigma: standard deviation of logged stock return
        :param r: risk free rate (annual rate, expressed in terms of compounding)
        :param delta_t: time interval of one branch (expressed in years)
        :return: up and down range and probability
        """
        up = math.exp(sigma * math.sqrt(delta_t))
        down = math.exp(-sigma * math.sqrt(delta_t))

        # risk neutral probability
        q_up = (compounding_factor(delta_t, r, COMPOUNDING_METHOD) - down) / (up - down)
        q_down = (up - compounding_factor(delta_t, r, COMPOUNDING_METHOD)) / (up - down)

        return [up, down, q_up, q_down]

    @staticmethod
    def ud_calibrated_list(up: float, down: float, r: float, delta_t: float) -> list:
        """
        calculate up and down range and probability for ud_calibrated tree assumption
        :param up: calibrated up range
        :param down: calibrated down range
        :param r: risk free rate (annual rate, expressed in terms of compounding)
        :param delta_t: time interval of one branch (expressed in years)
        :return: up and down range and probability
        """

        # risk neutral probability
        q_up = (compounding_factor(delta_t, r, COMPOUNDING_METHOD) - down) / (up - down)
        q_down = (up - compounding_factor(delta_t, r, COMPOUNDING_METHOD)) / (up - down)

        return [up, down, q_up, q_down]

    @staticmethod
    def p_half_list(sigma: float, miu: float, delta_t: float) -> list:
        """
        calculate up and down range and probability for p_half tree assumption
        :param sigma: standard deviation of stock return (annualized)
        :param miu: expectation of stock return (annualized)
        :param delta_t: time interval of one branch (expressed in years)
        :return: up and down range and probability
        """
        up = 1 + miu * delta_t + sigma * math.sqrt(delta_t)
        down = 1 + miu * delta_t - sigma * math.sqrt(delta_t)

        # risk neutral probability
        q_up = 1 / 2
        q_down = 1 / 2

        return [up, down, q_up, q_down]

    @staticmethod
    def bs_formula_d1(s: float, k: float, t: float, r: float, b: float, sigma: float) -> float:
        """
        Black–Scholes formula support function
        """
        d1 = (math.log(s / k) + (r - b + sigma * sigma / 2) * t) / (sigma * math.sqrt(t))

        return d1

    @staticmethod
    def bs_formula_d2(s: float, k: float, t: float, r: float, b: float, sigma: float) -> float:
        """
        Black–Scholes formula support function
        """
        d2 = (math.log(s / k) + (r - b - sigma * sigma / 2) * t) / (sigma * math.sqrt(t))

        return d2

    @staticmethod
    def call_payoff(s: float, k: float) -> float:
        """
        Calculate the payoff of call option
        :param s: spot price
        :param k: strike price
        :return: payoff of call option
        """
        payoff = 0
        if s > k:
            payoff = s - k

        return payoff

    @staticmethod
    def put_payoff(s: float, k: float) -> float:
        """
        Calculate the payoff of put option
        :param s: spot price
        :param k: strike price
        :return: payoff of put option
        """
        payoff = 0
        if s < k:
            payoff = k - s

        return payoff

    @staticmethod
    def volatility_log_return(s: list, multiplied_factor: int):
        """
        Calculate volatility of log return
        :param s: list of historical spot price of the underlying asset.
        :param multiplied_factor: number of business days for the corresponding frequency of historical data.
        E.g. for daily, 250 or 260 or 252. For weekly, 52.
        :return: annualized volatility of log return
        """

        # insufficient length of data
        if len(s) < 2:
            print('Error: Insufficient length of data.')
            return -1

        # logarithm of historical price
        log_s_return = []
        for i in range(1, len(s)):
            log_s_return.append(math.log(s[i] / s[i - 1]))

        vol = np.std(log_s_return, ddof=1) * math.sqrt(multiplied_factor)

        return vol

    @staticmethod
    def volatility_return(s: list, multiplied_factor: int):
        """
        Calculate volatility of return
        :param s: list of historical spot price of the underlying asset.
        :param multiplied_factor: number of business days for the corresponding frequency of historical data.
        E.g. for daily, 250 or 260 or 252. For weekly, 52.
        :return: annualized volatility of return
        """

        # insufficient length of data
        if len(s) < 2:
            print('Error: Insufficient length of data.')
            return -1

        s_return = []
        for i in range(1, len(s)):
            s_return.append(s[i] / s[i - 1])

        vol = np.std(s_return, ddof=1) * math.sqrt(multiplied_factor)

        return vol

    @staticmethod
    def expectation_return(s: list):
        """
        Calculate expectation of return
        :param s: list of historical spot price of the underlying asset.
        :return: expectation of return
        """

        # insufficient length of data
        if len(s) < 2:
            print('Error: Insufficient length of data.')
            return -1

        s_return = []
        for i in range(1, len(s)):
            s_return.append(s[i] / s[i - 1])

        exp = np.mean(s_return)

        return exp


class ForwardPricingEngine:
    def __init__(self, pricing_date=datetime, forward=Forward):
        """
        :param pricing_date: Date to calculate fair value
        :param forward: Forward object
        :return:
        """
        self.forward = forward
        self.pricing_date = pricing_date

        # t: time to maturity(expressed in years, assume act/365 daycounter)
        self.t = (forward.maturity - pricing_date).days / 365

    def npv(self, s: float, r_curve: list, b: float, sigma: float = 0, up_down_p: list = []) -> list:
        """
        Calculate the fair value and hedging strategy of forward by rolling back the payoff step by step
        :param s: spot price of the underlying asset (ex-dividend)
        :param r_curve: risk free rate curve(annual rate, expressed in terms of continuous compounding)
        :param b: dividend rate of underlying asset (annual rate)
        :param sigma: calibrated sigma
        :param up_down_p: calibrated up and down
        :return: the price of forward [npv, h0_tree, h1_tree, s_tree, bond_tree, option_tree]
        """
        # Interpolate risk free rate
        r = interpol(r_curve[0], r_curve[1], self.t)
        n = BINOMIAL_TREE_STEP
        delta_t = self.t / n

        up = up_down_p[0]
        down = up_down_p[1]
        q_up = up_down_p[2]
        q_down = up_down_p[3]

        h0_tree = []
        h1_tree = []
        s_tree = []
        bond_tree = []
        option_tree = []

        # price by rolling back the payoff step by step
        x_t_1_discounted = []
        for time_step in range(n, -1, -1):
            x_t = []
            s_t_list = []
            b_t_list = []
            h0_t_list = []
            h1_t_list = []
            for branch in range(time_step + 1):
                # calculate intrinsic value

                s_t = s * compounding_factor(time_step * delta_t, - b, COMPOUNDING_METHOD) \
                      * math.pow(up, time_step - branch) * math.pow(down, branch)

                s_t_list.append(s_t)
                b_t_list.append(compounding_factor(time_step * delta_t, r, COMPOUNDING_METHOD))

                if time_step == n:  # last step Z_T = Y_T
                    payoff = s_t - self.forward.strike
                    x_t.append(payoff)
                else:
                    x_t.append(x_t_1_discounted[branch])

            # Calculate hedging strategy
            if time_step != 0:
                for branch in range(time_step):
                    s_up_down = [s_t_list[branch], s_t_list[branch + 1]]
                    x_up_down = [x_t[branch], x_t[branch + 1]]
                    hedging = binomial_tree_hedging(s_up_down, x_up_down, r, delta_t * time_step)
                    h0_t_list.append(hedging[0])
                    h1_t_list.append(hedging[1])
                h0_tree.insert(0, h0_t_list)
                h1_tree.insert(0, h1_t_list)

            s_tree.insert(0, s_t_list)
            bond_tree.insert(0, b_t_list)
            option_tree.insert(0, x_t)

            # Discount x_t to x_t_1_discounted
            x_t_1_discounted = []
            if time_step != 0:
                for i in range(time_step):
                    x_t_1_discounted.append(discount_factor(delta_t, r, COMPOUNDING_METHOD)
                                            * (q_up * x_t[i] + q_down * x_t[i + 1]))
            else:
                npv = x_t[0]

        return [npv, h0_tree, h1_tree, s_tree, bond_tree, option_tree]

    def forward_analytic(self, s: float, r_curve: list, b: float) -> list:
        """
        Calculate the fair value of forward (For benchmarking)
        :param s: spot price of the underlying asset (ex-dividend)
        :param r_curve: risk free rate curve(annual rate, expressed in terms of continuous compounding)
        :param b: dividend rate of underlying asset (annual rate)
        :return: the price of forward
        """
        # Interpolate risk free rate
        r = interpol(r_curve[0], r_curve[1], self.t)

        return [s - self.forward.strike * discount_factor(self.t, (r - b), COMPOUNDING_METHOD)]


def compounding_factor(t: float, r: float, compounding=CompoundingMethod) -> float:
    """
    Calculate compounding factor based on time, discount rate and compounding method
    :param t: time
    :param r: discount rate
    :param compounding: Enum type indicator: 1 for discrete_compounded, 2 for continuous_compounded
    :return: compounding factor
    """
    if compounding is CompoundingMethod.discrete_compounded:
        return math.pow(1 + r, t)
    elif compounding is CompoundingMethod.continuous_compounded:
        return math.exp(r * t)

    # Exception
    return 1


def discount_factor(t: float, r: float, compounding=CompoundingMethod) -> float:
    """
    Calculate discount factor based on time, discount rate and compounding method
    :param t: time
    :param r: discount rate
    :param compounding: Enum type indicator: 1 for discrete_compounded, 2 for continuous_compounded
    :return: discount factor
    """
    return 1 / compounding_factor(t, r, compounding)


def yang_hui_triangle(n: int, k: int) -> int:
    """
    Calculate C(n, k) using Yang Hui's (Pascal's) triangle
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


def interpol(x_vector: list, y_vector: list, point: float) -> float:
    """
    linear interpolation method
    :param x_vector: x data set
    :param y_vector: f(x) data set
    :param point: sample object
    :return: f(point)
    """
    num = len(y_vector)
    if point <= x_vector[0]:
        # for smaller value set equal to smallest value in Y vector
        return y_vector[0]
    else:
        i = 1
        # Find out where point is
        while x_vector[i] < point:
            i = i + 1
            if i > num:
                # for larger values set equal to largest value in Y vector
                return y_vector[num]
        weight = (x_vector[i] - point) / (x_vector[i] - x_vector[i - 1])
        return weight * y_vector[i - 1] + (1 - weight) * y_vector[i]


def binomial_tree_hedging(s: list, x: list, r: float, t: float) -> list:
    """
    binomial tree support function to calculate hedging ratio on each node
    :param s: list of spot price of the underlying asset. Length 2, upward and downward side.
    :param x: list of value of the contingent claim. Length 2, upward and downward side.
    :param r: risk free rate (annual rate, expressed in terms of compounding)
    :param t: time interval of binomial tree (expressed in years)
    :return: list of hedging strategy of the contingent claim. Length 2, (H0, H1)
    """
    h1 = (x[1] - x[0]) / (s[1] - s[0])
    h0 = (x[1] - s[1] * h1) * discount_factor(t, r, COMPOUNDING_METHOD)

    hedging = [h0, h1]

    return hedging