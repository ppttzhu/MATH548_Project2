#!/usr/bin/env python3
# encoding: utf-8

"""
@Project: MATH548_Project2
@FileName: testing.py
@Author: Peng, Yixin
@Date: 2018/11/22
@Description：Testing with inputs and outputs
@File URL: https://github.com/ppttzhu/MATH548_Project2
"""

from optimizer import PortfolioOptimizer
import csv


def main():
    
    # ----------------inputs---------------------
    
    v_0 = 1
    u = 9 / 8
    d = 6 / 8
    r = 0.00
    t = 3
    p_up = 0.5
    
    # ----------------process---------------------
    
    portfolio_optimizer = PortfolioOptimizer(v_0, u, d, r, t, p_up)
    s_tree = portfolio_optimizer.build_stock_tree()
    dynamic_programming_principle = portfolio_optimizer.dynamic_programming_principle()
    trading_strategy = dynamic_programming_principle[0]
    portfolio_value = dynamic_programming_principle[1]
    print(trading_strategy[0][0])
    
    # ----------------outputs---------------------
    
    with open("[output]s_tree.csv", "w", newline='') as csvfile:
        result_writer = csv.writer(csvfile, delimiter=',', quotechar=',')
        for line in s_tree:
            result_writer.writerow(line)
    with open("[output]trading_strategy.csv", "w", newline='') as csvfile:
        result_writer = csv.writer(csvfile, delimiter=',', quotechar=',')
        for line in trading_strategy:
            result_writer.writerow(line)
    with open("[output]portfolio_value.csv", "w", newline='') as csvfile:
        result_writer = csv.writer(csvfile, delimiter=',', quotechar=',')
        for line in portfolio_value:
            result_writer.writerow(line)
    print("Done")


if __name__ == "__main__":
    main()
