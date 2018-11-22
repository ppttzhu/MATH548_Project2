#!/usr/bin/env python3
# encoding: utf-8

"""
@Project: MATH548_Project1
@FileName: testing.py
@Author: Kim Ki Hyeon, Lu Weikun, Peng Yixin, Zhou Nan
@Date: 2018/10/25
@Description：
@File URL: https://github.com/ppttzhu/MATH548_Project1/import
"""

from pricingengine import *
from pandas_datareader import data
from pinance import Pinance
from dateutil.relativedelta import relativedelta
import csv
from tkinter import *
from tkinter import messagebox


def main():
    # ----------------pre-defined parameters---------------------

    stock_name = 'TWTR'
    data_source_price = 'yahoo'
    data_source_dividend = 'yahoo-dividends'
    maturity_string_test = "2019-03-15"
    maturity_test = datetime.datetime.strptime(maturity_string_test, "%Y-%m-%d")
    pricing_date_string_test = "2018-11-05"
    pricing_date_test = datetime.datetime.strptime(pricing_date_string_test, "%Y-%m-%d")
    is_calculator = 1  # use a calculator or not

    # ----------------product data---------------------

    # Get options for calibration, source: bbg
    csv_file = open("[input]Options_for_calibrate_" + pricing_date_string_test + ".csv", "r")
    reader = csv.reader(csv_file)

    options_for_calibrate_list = []
    options_for_calibrate_price_list = []
    options_list = []
    options_price_list = []

    for item in reader:
        # Drop first line
        if reader.line_num == 1:
            continue

        maturity_date = datetime.datetime.strptime(item[2], "%Y-%m-%d")
        option = Option(item[0], float(item[1]), maturity_date, CallPutType(int(item[3])), ExerciseType(int(item[4])))
        options_for_calibrate_list.append(option)
        options_for_calibrate_price_list.append(float(item[5]))
        if item[2] == maturity_string_test:
            options_list.append(option)
            options_price_list.append(float(item[5]))

    csv_file.close()

    # ----------------market data---------------------

    # Minimum 1 year of historical data
    historical_start_date2 = pricing_date_test - (maturity_test - pricing_date_test)
    historical_start_date1 = pricing_date_test - relativedelta(years=1)
    historical_start_date_test = min(historical_start_date1, historical_start_date2)

    stock_price_list = data.DataReader(name=stock_name, data_source=data_source_price, start=historical_start_date_test,
                                       end=pricing_date_test)
    s_history_test = stock_price_list['Adj Close']
    s_test = stock_price_list['Adj Close'][pricing_date_test]

    dividend_list = data.DataReader(name=stock_name, data_source=data_source_dividend, start=historical_start_date1,
                                    end=pricing_date_test)
    if not dividend_list.empty:
        b_test = sum(list(dividend_list['value'])) / s_test
    else:
        b_test = 0

    # Daily Treasury Yield Curve Rates for Risk Free Rate, Data source:
    # https://www.treasury.gov/resource-center/data-chart-center/interest-rates/Pages/TextView.aspx?data=yieldYear&year=2018
    csv_file = open("[input]RiskFreeRate_" + pricing_date_string_test + ".csv", "r")
    reader = csv.reader(csv_file)

    risk_free_rate_x = []
    risk_free_rate_y = []
    for item in reader:
        # Drop first line
        if reader.line_num == 1:
            continue
        risk_free_rate_x.append(float(item[0]))
        risk_free_rate_y.append(float(item[1]))

    risk_free_rate = [risk_free_rate_x, risk_free_rate_y]
    csv_file.close()

    # ----------------performing pricing---------------------

    global npv
    npv = []

    if is_calculator:
        def click():
            global npv
            if product_type.get():
                option = Option("Calculator", float(strike.get()),
                                datetime.datetime.strptime(maturity.get(), "%Y-%m-%d"),
                                CallPutType(call_put_type.get()),
                                ExerciseType(exercise_type.get()))

                pricing_engine = OptionPricingEngine(pricing_date_test, option)
                parameters = pricing_engine.calibrate(s_test, risk_free_rate, b_test, s_history_test,
                                                      options_for_calibrate_list,
                                                      options_for_calibrate_price_list)

                npv = pricing_engine.npv(s_test, risk_free_rate, b_test, parameters[0], parameters[1])
                npv_print = '%.4f' % npv[0]

                stock = Pinance(stock_name)
                if option.call_put_type == CallPutType.call:
                    stock.get_options(maturity_string_test, 'C', option.strike)
                else:
                    stock.get_options(maturity_string_test, 'P', option.strike)
                try:
                    market_price_today = stock.options_data['lastPrice']
                    market_price_today_print = '%.4f' % market_price_today
                    difference_print = '%.4f' % (npv[0] - market_price_today)
                except TypeError as te:
                    market_price_today_print = "Not Found"
                    difference_print = "N/A"

                if parameters[0] != 0:
                    sigma_print = '%.4f' % parameters[0]
                else:
                    sigma_print = "N/A"

                if len(parameters[1]) > 0:
                    up_print = '%.4f' % parameters[1][0]
                    down_print = '%.4f' % parameters[1][1]
                    q_up_print = '%.4f' % parameters[1][2]
                    q_down_print = '%.4f' % parameters[1][3]
                else:
                    up_print = "N/A"
                    down_print = "N/A"
                    q_up_print = "N/A"
                    q_down_print = "N/A"
                message = 'model price: ' + npv_print + '\nmarket price: ' + market_price_today_print \
                          + '\n(model-market): ' + difference_print + '\n\nsigma: ' + sigma_print \
                          + '\n\nup: ' + up_print + '\ndown: ' + down_print \
                          + '\nq_up: ' + q_up_print + '\nq_down: ' + q_down_print

            else:
                option = Option("Calculator", float(strike.get()),
                                datetime.datetime.strptime(maturity.get(), "%Y-%m-%d"),
                                CallPutType(1),
                                ExerciseType(1))

                pricing_engine_option = OptionPricingEngine(pricing_date_test, option)
                parameters = pricing_engine_option.calibrate(s_test, risk_free_rate, b_test, s_history_test,
                                                             options_for_calibrate_list,
                                                             options_for_calibrate_price_list)

                forward = Forward("Calculator", float(strike.get()),
                                  datetime.datetime.strptime(maturity.get(), "%Y-%m-%d"))
                pricing_engine_forward = ForwardPricingEngine(pricing_date_test, forward)
                npv = pricing_engine_forward.npv(s_test, risk_free_rate, b_test, parameters[0], parameters[1])
                npv_print = '%.4f' % npv[0]

                if parameters[0] != 0:
                    sigma_print = '%.4f' % parameters[0]
                else:
                    sigma_print = "N/A"

                if len(parameters[1]) > 0:
                    up_print = '%.4f' % parameters[1][0]
                    down_print = '%.4f' % parameters[1][1]
                    q_up_print = '%.4f' % parameters[1][2]
                    q_down_print = '%.4f' % parameters[1][3]
                else:
                    up_print = "N/A"
                    down_print = "N/A"
                    q_up_print = "N/A"
                    q_down_print = "N/A"
                message = 'model price: ' + npv_print + '\n\nsigma: ' + sigma_print \
                          + '\n\nup: ' + up_print + '\ndown: ' + down_print \
                          + '\nq_up: ' + q_up_print + '\nq_down: ' + q_down_print

            messagebox.showinfo('Price', message)  # place holder

        # ----------------pop-out windows setting---------------------

        font = ('TIMES NEW ROMAN', 20)

        window = Tk()
        window.title('Derivatives Pricing Tool')

        lbl = Label(window, text='Strike Price', font=font)
        lbl.grid(column=0, row=0)
        # strike = Entry(window, width=10, font=font, textvariable=StringVar(window, value='34.314944'))
        strike = Entry(window, width=10, font=font)
        strike.grid(column=1, row=0)

        lbl = Label(window, text='Maturity Date', font=font)
        lbl.grid(column=0, row=1)
        # maturity = Entry(window, width=10, font=font, textvariable=StringVar(window, value=maturity_string_test))
        maturity = Entry(window, width=10, font=font)
        maturity.grid(column=1, row=1)

        product_type = IntVar()
        ra1 = Radiobutton(window, text='Option', value=1, font=font, variable=product_type)
        ra2 = Radiobutton(window, text='Forward', value=0, font=font, variable=product_type)
        ra1.grid(column=0, row=2)
        ra2.grid(column=1, row=2)

        call_put_type = IntVar()
        rad1 = Radiobutton(window, text='Call', value=1, font=font, variable=call_put_type)
        rad2 = Radiobutton(window, text='Put', value=2, font=font, variable=call_put_type)
        # call_put_type.set(1)
        rad1.grid(column=0, row=3)
        rad2.grid(column=1, row=3)

        exercise_type = IntVar()
        radn1 = Radiobutton(window, text='European', value=1, font=font, variable=exercise_type)
        radn2 = Radiobutton(window, text='American', value=2, font=font, variable=exercise_type)
        # exercise_type.set(1)
        radn1.grid(column=0, row=4)
        radn2.grid(column=1, row=4)

        btn = Button(window, text='Get Price', font=font, command=click)
        btn.grid(column=0, row=6)

        window.geometry('500x350')
        window.mainloop()

    else:

        pricing_engine = OptionPricingEngine(pricing_date_test, options_list[0])
        parameters = pricing_engine.calibrate(s_test, risk_free_rate, b_test, s_history_test,
                                              options_for_calibrate_list,
                                              options_for_calibrate_price_list)

        npv_list = []
        for option in options_list:
            pricing_engine = OptionPricingEngine(pricing_date_test, option)
            npv = pricing_engine.npv(s_test, risk_free_rate, b_test, parameters[0], parameters[1])
            npv_list.append(npv)

        # ----------------printing results---------------------

        if parameters[0] != 0:
            print("sigma = %f" % parameters[0])
        if parameters[1]:
            print("up = %f" % parameters[1][0])
            print("down = %f" % parameters[1][1])
            print("q_up = %f" % parameters[1][2])
            print("q_down = %f" % parameters[1][3])

        with open("[output]npv_list.csv", "w", newline='') as csvfile:
            result_writer = csv.writer(csvfile, delimiter=',', quotechar=',')
            for i in range(len(options_list)):
                result_writer.writerow([options_list[i].product_id, npv_list[i][0], options_price_list[i],
                                        npv_list[i][0] - options_price_list[i]])

    # export hedging strategy in csv files
    if len(npv) > 1 and is_calculator:
        with open("[output]h0_tree.csv", "w", newline='') as csvfile:
            result_writer = csv.writer(csvfile, delimiter=',', quotechar=',')
            for line in npv[1]:
                result_writer.writerow(line)
        with open("[output]h1_tree.csv", "w", newline='') as csvfile:
            result_writer = csv.writer(csvfile, delimiter=',', quotechar=',')
            for line in npv[2]:
                result_writer.writerow(line)
        with open("[output]s_tree.csv", "w", newline='') as csvfile:
            result_writer = csv.writer(csvfile, delimiter=',', quotechar=',')
            for line in npv[3]:
                result_writer.writerow(line)
        with open("[output]bond_tree.csv", "w", newline='') as csvfile:
            result_writer = csv.writer(csvfile, delimiter=',', quotechar=',')
            for line in npv[4]:
                result_writer.writerow(line)
        with open("[output]derivative_tree.csv", "w", newline='') as csvfile:
            result_writer = csv.writer(csvfile, delimiter=',', quotechar=',')
            for line in npv[5]:
                result_writer.writerow(line)


if __name__ == "__main__":
    main()
