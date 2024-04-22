# API获取数据+leverage

import functools
import pandas as pd
import numpy as np
import tushare as ts
from scipy.optimize import minimize


# 定义缓存装饰器
def cache(func):
    cached_data = {}
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = args + tuple(kwargs.items())
        if key not in cached_data:
            cached_data[key] = func(*args, **kwargs)
        return cached_data[key]
    return wrapper


# 假设get_daily_price已经实现，用于获取每个资产每天的收盘价
@cache
def get_daily_price(ts_code, date):
    pro = ts.pro_api('20231208200557-b4f921df-1bab-430e-ab72-9d36e3a2ecbe')
    pro._DataApi__http_url = 'http://tsapi.majors.ltd:7000'

    # 尝试获取指定日期的数据
    if ts_code.endswith(('.SHF', '.CFX', '.INE')):
        data = pro.fut_daily(ts_code=ts_code, trade_date=date)
    elif ts_code.startswith('5'):
        data = pro.fund_daily(ts_code=ts_code, trade_date=date)
    elif ts_code.startswith('.CB'):
        data = pro.yc_cb(ts_code=ts_code, curve_type='0', trade_date=date)
    else:
        data = pro.daily(ts_code=ts_code, trade_date=date)

    # 如果指定日期没有数据，则尝试获取之前最近的交易日数据
    if data.empty:
        # 往前查询最多尝试30天，防止无限循环
        for i in range(1, 31):
            prev_date = pd.to_datetime(date) - pd.Timedelta(days=i)
            prev_date_str = prev_date.strftime('%Y%m%d')

            if ts_code.endswith(('.SHF', '.CFX', '.INE')):
                data = pro.fut_daily(ts_code=ts_code, trade_date=prev_date_str)
            elif ts_code.startswith('5'):
                data = pro.fund_daily(ts_code=ts_code, trade_date=prev_date_str)
            elif ts_code.startswith('2'):
                data = pro.repo_daily(ts_code=ts_code, trade_date=prev_date_str)
            else:
                data = pro.daily(ts_code=ts_code, trade_date=prev_date_str)

            if not data.empty:
                break

    if data.empty:
        return np.nan  # 如果仍然没有数据，返回np.nan
    else:
        data.index = pd.to_datetime(data['trade_date'], format='%Y%m%d')
        close = data['close'].iloc[0]  # 获取最近交易日的收盘价
        return close


# 获取收益率数据的函数
@cache
def get_ret(ts_code, start, end):
    # 实现get_ret函数的代码
    pro = ts.pro_api('20231208200557-b4f921df-1bab-430e-ab72-9d36e3a2ecbe')
    pro._DataApi__http_url = 'http://tsapi.majors.ltd:7000'

    if ts_code.endswith(('.SHF', '.CFX', '.INE')):
        data = pro.fut_daily(ts_code=ts_code, start_date=start, end_date=end)
    elif ts_code.startswith('5'):
        data = pro.fund_daily(ts_code=ts_code, start_date=start, end_date=end)
    elif ts_code.startswith('2'):
        data = pro.repo_daily(ts_code=ts_code, start_date=start, end_date=end)
    else:
        data = pro.daily(ts_code=ts_code, start_date=start, end_date=end)

    data.index = pd.to_datetime(data['trade_date'], format='%Y%m%d')  # 设置日期索引
    close = data['close']  # 日收盘价
    ret = np.log(close / close.shift(1))  # 日收益率
    ret = ret.dropna()  # 删除NaN值

    # 根据不同的资产类型调整收益率
    if ts_code.endswith('.SHF') or ts_code.endswith('.INE'):
        ret *= 5  # 对于'.SHF', '.INE'的收益率结果乘以10
    elif ts_code.endswith('.CFX'):
        ret *= 10  # 对于'.CFX'的收益率结果乘以20

    return ret


# 获取gc007:隔夜国债收益率(年化)
@cache
def get_gc007(ts_code, curve_type, trade_date):
    pro = ts.pro_api('20231208200557-b4f921df-1bab-430e-ab72-9d36e3a2ecbe')
    pro._DataApi__http_url = 'http://tsapi.majors.ltd:7000'
    gc007 = pro.yc_cb(ts_code=ts_code, curve_type=curve_type, trade_date=trade_date).sort_values(by='curve_term').iloc[0]['yield']
    return gc007


# 获取权重的函数
def calculate_monthly_weights(end_date, assets):
    ret = pd.DataFrame()
    for asset in assets:
        # 修改为从五年前开始的日期
        five_years_ago = (pd.to_datetime(end_date) - pd.DateOffset(years=5)).strftime('%Y%m%d')
        ret_ = get_ret(asset, five_years_ago, end_date)
        ret = pd.concat([ret, ret_], axis=1)
    ret = ret.dropna()

    R_cov = ret.cov()  # 计算协方差
    cov = np.array(R_cov)

    def risk_budget_objective(weights, cov):
        weights = np.array(weights)
        sigma = np.sqrt(np.dot(weights, np.dot(cov, weights)))
        MRC = np.dot(cov, weights) / sigma
        TRC = weights * MRC
        delta_TRC = [sum((i - TRC) ** 2) for i in TRC]
        return sum(delta_TRC)

    x0 = np.ones(cov.shape[0]) / cov.shape[0]
    bnds = tuple((0, None) for x in x0)
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})
    options = {'disp': False, 'maxiter': 1000, 'ftol': 1e-20}

    solution = minimize(risk_budget_objective, x0, args=cov, bounds=bnds, constraints=cons, method='SLSQP', options=options)

    final_weights = solution.x  # 最终的权重
    return final_weights


# 获取交易日期
@cache
def get_trade_days(start_date, end_date):
    pro = ts.pro_api('20231208200557-b4f921df-1bab-430e-ab72-9d36e3a2ecbe')
    pro._DataApi__http_url = 'http://tsapi.majors.ltd:7000'
    df = pro.trade_cal(exchange='', start_date=start_date, end_date=end_date)
    df = df[df.is_open == 1]  # 筛选出开市的日期
    df.sort_values('cal_date', inplace=True)# 确保日期是升序的
    trade_days = pd.to_datetime(df.cal_date).dt.strftime('%Y%m%d')
    return trade_days


# 计算510300.SH的交易费率
def calculate_trading_cost_510300_sh(amount):
    cost = abs(amount) * 0.0003  # 交易佣金0.03%，最低5元
    if cost < 5:
        cost = 5
    if amount < 0:  # 仅在卖出时收取印花税
        cost = cost + abs(amount) * 0.0005  # 印花税0.05%
    return cost


# 计算513500.SH的交易费率
def calculate_trading_cost_513500_sh(amount):
    cost = abs(amount) * 0.0003  # 交易佣金0.03%，最低5元
    if cost < 5:
        cost = 5
    if amount < 0:  # 仅在卖出时收取印花税
        cost = cost + abs(amount) * 0.0005  # 印花税0.05%
    return cost


# 计算TL.CFX的交易费率
def calculate_trading_cost_tl_cfx(quantity):
    cost = abs(quantity) * 3.01  # 这里假设期货公司收取0.01元/手的开平仓费用
    return cost


# 计算AU.SHF的交易费率
def calculate_trading_cost_au_shf(quantity):
    cost = abs(quantity) * 10.01  # 这里假设期货公司收取0.01元/手的开平仓费用
    return cost


# 计算CU.SHF的交易费率
def calculate_trading_cost_cu_shf(amount):
    cost = abs(amount) * 0.00005 + 0.01  # 这里假设期货公司收取0.01元/手的开平仓费用
    return cost


# 计算SC.INE的交易费率
def calculate_trading_cost_sc_ine(quantity):
    cost = abs(quantity) * 20.01  # 这里假设期货公司收取0.01元/手的开平仓费用
    return cost


# 净值回测函数
def net_value_backtest(start_date, end_date, init_fund, ts_code, weights_df):
    # 获取交易日历
    trade_days = list(get_trade_days(start_date, end_date))

    net_values = pd.Series(index=pd.to_datetime(trade_days), dtype=float)
    current_fund = init_fund
    holdings = {code: 0 for code in ts_code}
    last_month = None  # 用于跟踪上一个月份

    # 假设开始时，所有的持仓都是空的
    holdings_value = {code: 0 for code in ts_code}  # 当前持仓的市值

    for day in trade_days:
        date = pd.to_datetime(day)
        month = date.strftime('%Y-%m')  # 当前日期的月份
        daily_value = 0  # 每天开始计算之前重置daily_value
        surplus_value = 0  # 每天开始计算之前重置surplus的每日收益 (除去保证金外的账户余额的利息收益)

        # 检查是否需要更新权重：如果是新的一个月或者是回测的第一天
        if month != last_month or day == trade_days[0]:
            # 将当前月份字符串转换为日期格式，并计算上个月的日期
            prev_month_date = pd.to_datetime(month) - pd.DateOffset(months=1)
            # 将上个月的日期转换回字符串格式
            prev_month_str = prev_month_date.strftime('%Y-%m')
            # 使用上个月的权重
            if prev_month_str in weights_df.index:
                weights = weights_df.loc[prev_month_str]
                # 更新持仓
                for code in ts_code:
                    price = get_daily_price(code, day)
                    if not np.isnan(price):  # 确保价格有效
                        # 直接访问权重数值，而不是使用包含日期索引的Series
                        weight = weights[code].values[0]  # 假设weights是Series，获取其数值
                        print(f"{day} - {code}: price = {price}, weight = {weight}")

        # 计算当天的净值
        for code in ts_code:
            price = get_daily_price(code, day)
            surplus_value_spilt = 0
            if not np.isnan(price):  # 确保价格有效
                # 根据不同的资产代码设置不同的杠杆倍数
                if code == 'TL.CFX':
                    leverage = 10  # TL.CFX 使用10倍杠杆
                elif code in ['AU.SHF', 'CU.SHF', 'SC.INE']:
                    leverage = 5  # AU.SHF, CU.SHF, SC.INE 使用5倍杠杆
                else:
                    leverage = 1  # 其他资产使用1倍杠杆
                target_value = current_fund * weights[code].values[0]  # 根据权重计算目标持仓市值
                current_value = holdings_value.get(code, 0)  # 获取当前持仓市值
                if month != last_month or day == trade_days[0]:
                    buy_amount = target_value - current_value  # 计算需要买入的金额
                else:
                    buy_amount = 0

                # 计算交易成本(交易所手续费+证券/期货公司佣金)
                if code == '510300.SH':
                    if buy_amount != 0:
                        trading_cost = calculate_trading_cost_510300_sh(buy_amount)
                    else:
                        trading_cost = 0
                    buy_amount -= trading_cost  # 考虑交易成本后的实际买入金额
                elif code == '513500.SH':
                    if buy_amount != 0:
                        trading_cost = calculate_trading_cost_513500_sh(buy_amount)
                    else:
                        trading_cost = 0
                    buy_amount -= trading_cost  # 考虑交易成本后的实际买入金额
                elif code == 'TL.CFX':  # 这里有一个重要问题，即持有金额变化可能小于当期交易量，因此前后持有的期货可能期限不同，不是同一个期货
                    if buy_amount / price != 0:  # 这里不存在平今操作，因此买卖均需手续费，此外这里计算的是交易量而非交易金额
                        trading_cost = calculate_trading_cost_tl_cfx(buy_amount*leverage/1000000)  # 每手十年期国债期货名义本金100万元，并考虑杠杆
                    else:
                        trading_cost = 0
                    buy_amount -= trading_cost  # 考虑交易成本后的实际买入金额
                elif code == 'AU.SHF':
                    if buy_amount / price != 0:
                        trading_cost = calculate_trading_cost_au_shf(buy_amount*leverage/(price*1000))  # 每手黄金1000克，并考虑杠杆
                    else:
                        trading_cost = 0
                    buy_amount -= trading_cost  # 考虑交易成本后的实际买入金额
                elif code == 'CU.SHF':
                    if buy_amount / price != 0:  # 铜期货的交易费率是按照总金额的百分比计算的
                        trading_cost = calculate_trading_cost_cu_shf(buy_amount*leverage)  # 每手铜5吨，但铜期货的交易费率并不是按每手多少钱计算
                    else:
                        trading_cost = 0
                    buy_amount -= trading_cost  # 考虑交易成本后的实际买入金额
                elif code == 'SC.INE':
                    if buy_amount / price != 0:
                        trading_cost = calculate_trading_cost_sc_ine(buy_amount*leverage/(price*1000))  # 每手原油1000桶，并考虑杠杆
                    else:
                        trading_cost = 0
                    buy_amount -= trading_cost  # 考虑交易成本后的实际买入金额
                else:
                    trading_cost = 0  # 对于其他资产，这里假设没有交易成本

                # 更新持仓量，这里需要考虑实际买入或卖出的金额
                if buy_amount != 0:
                    holdings[code] += buy_amount / price  # 基础持仓，这里不考虑期货的杠杆效应，在下面考虑杠杆效应

                # 设置前一个交易日prev_day
                prev_day_date = date - pd.Timedelta(days=1)
                prev_day_str = prev_day_date.strftime('%Y%m%d')

                if code in ('510300.SH', '513500.SH'):
                    holdings_value[code] = holdings[code] * price  # 更新持仓市值
                elif code in ('TL.CFX', 'AU.SHF', 'CU.SHF', 'SC.INE'):
                    if day == trade_days[0]:
                        holdings_value[code] = holdings[code] * price  # 第一天不产生杠杆损益
                    else:
                        prev_price = get_daily_price(code, prev_day_str)
                        holdings_value[code] = holdings[code] * price + (leverage-1) * holdings[code] * (price - prev_price)  # 考虑杠杆

                # 计算除去期货保证金和追保金额外的闲置资金的每日收益
                if code in ('TL.CFX', 'AU.SHF', 'CU.SHF', 'SC.INE'):
                    gc007 = get_gc007('1001.CB', '0', day)  # gc007:隔夜国债收益率(年化)
                    surplus_value_spilt = holdings_value[code] * 0.5 * gc007/100 * 1/365  # 这里假设期货用于保证金和追保金额占面值的10%
                    surplus_value += surplus_value_spilt

                print(f"{day} - {code}: price = {price:.3f}, target_value = {target_value:.2f}, holding_value = {holdings_value.get(code, 0):.2f}, buy_amount = {buy_amount:.2f}, cost = {trading_cost:.2f}, surplus_value_spilt = {surplus_value_spilt:.2f}")

                daily_value += holdings_value[code] + surplus_value_spilt

        if month != last_month or day == trade_days[0]:
            last_month = month  # 更新月份跟踪器

        if daily_value > 0:  # 确保只在有有效净值时更新current_fund
            current_fund = daily_value
        net_values.at[date] = daily_value
        print(f"{date} 更新后的资金总额是 {current_fund:.2f}, 其中包括货币收益 {surplus_value:.2f}")

    return net_values


# 主函数
if __name__ == "__main__":
    # 参数设置
    ts_code = ['510300.SH', '513500.SH', 'TL.CFX', 'AU.SHF', 'CU.SHF', 'SC.INE']
    start_date = '20181231'
    end_date = '20190325'
    init_fund = 100000000

    monthly_weights = []
    months = pd.date_range(start=start_date, end=end_date, freq='M')

    for i in range(len(months)):
        month_start = months[i].strftime('%Y%m01')
        month_end = months[i].strftime('%Y%m%d')
        weights = calculate_monthly_weights(month_end, ts_code)
        monthly_weights.append(weights)

    weights_df = pd.DataFrame(monthly_weights, columns=ts_code)
    weights_df.index = pd.date_range(start=start_date, end=end_date, freq='M')
    print(weights_df)

    # 进行净值回测
    net_values = net_value_backtest(start_date, end_date, init_fund, ts_code, weights_df)

    # 假设get_trade_days返回的是'YYYYMMDD'格式的交易日列表
    trade_days = get_trade_days(start_date, end_date)

    # 将trade_days转换为datetime格式，以便与net_values的索引保持一致
    trade_dates = pd.to_datetime(trade_days, format='%Y%m%d')
    print(len(trade_dates))
    print(len(net_values))

    # 将net_values转换为数值类型
    net_values = [float(value) for value in net_values]

    print(net_values)

    df = pd.DataFrame({'Trade Date': trade_dates, 'Net Value': net_values})
    df.to_excel('net_value_backtest_4.xlsx', index=False)