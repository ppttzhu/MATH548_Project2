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
    
    # u(2)
    v_0 = 100
    u = 2
    d = 0.5
    r = 0.00
    t = 2
    p_up = 0.5
  
#     # u(3)
#     v_0 = 1  # 10
#     u = 9 / 8
#     d = 6 / 8
#     r = 0.00
#     t = 2
#     p_up = 0.5
    
#     # u(4)
#     v_0 = 10
#     u = 1.1
#     d = 0.85
#     r = 0.00
#     t = 2
#     p_up = 0.5
    
    # ----------------process---------------------
    
    portfolio_optimizer = PortfolioOptimizer(v_0, u, d, r, t, p_up)
    
    # Dynamic Programming Principle
    s_tree_d = portfolio_optimizer.build_stock_tree()
    dynamic_programming_principle = portfolio_optimizer.dynamic_programming_principle()
    trading_strategy_d = dynamic_programming_principle[0]
    portfolio_value_d = dynamic_programming_principle[1]

    # Martingale Method
    s_tree_m = portfolio_optimizer.build_stock_tree_combine()
    martingale_method = portfolio_optimizer.martingale_method()
    trading_strategy_m = martingale_method[0]
    portfolio_value_m = martingale_method[1]
    
    # ----------------outputs---------------------
    
    with open("[output]s_tree_d.csv", "w", newline='') as csvfile:
        result_writer = csv.writer(csvfile, delimiter=',', quotechar=',')
        for line in s_tree_d:
            result_writer.writerow(line)
    with open("[output]trading_strategy_d.csv", "w", newline='') as csvfile:
        result_writer = csv.writer(csvfile, delimiter=',', quotechar=',')
        for line in trading_strategy_d:
            result_writer.writerow(line)
    with open("[output]portfolio_value_d.csv", "w", newline='') as csvfile:
        result_writer = csv.writer(csvfile, delimiter=',', quotechar=',')
        for line in portfolio_value_d:
            result_writer.writerow(line)
            
    with open("[output]s_tree_m.csv", "w", newline='') as csvfile:
        result_writer = csv.writer(csvfile, delimiter=',', quotechar=',')
        for line in s_tree_m:
            result_writer.writerow(line)
    with open("[output]trading_strategy_m.csv", "w", newline='') as csvfile:
        result_writer = csv.writer(csvfile, delimiter=',', quotechar=',')
        for line in trading_strategy_m:
            result_writer.writerow(line)
    with open("[output]portfolio_value_m.csv", "w", newline='') as csvfile:
        result_writer = csv.writer(csvfile, delimiter=',', quotechar=',')
        for line in portfolio_value_m:
            result_writer.writerow(line)
            
    print("Done")


if __name__ == "__main__":
    main()
