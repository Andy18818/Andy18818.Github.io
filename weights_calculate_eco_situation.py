# 同时考虑近期和长期波动率，优化组合权重

# 在我们之前的Risk_Parity和Risk_Parity2中，我们使用的是tushare的API接口来调取数据，但一旦任务量提升就面临速度很慢的问题，同时有些数据调用还有权限限制
# 因此我们接下来都直接使用本地数据进行回测，数据可以从Choice和Wind下载
# 具体数据格式可以从代码中推测，或者可以看我上传的Excel文件示例

import pandas as pd
import numpy as np
from scipy.optimize import minimize


# 获取交易日期
def get_trade_days(start_date, end_date):
    try:
        # 尝试从预加载的数据中获取交易日数据
        data_file_path = 'C:\\Python程序设计基础\\pythonProject1\\日线数据.xls'
        # 加载Excel文件
        data_sheets = pd.read_excel(data_file_path, sheet_name=None, index_col='trade_date', parse_dates=['trade_date'])
        trade_days_data = data_sheets["TradeDays"]
        # 筛选指定时间范围内的交易日
        filtered_trade_days = trade_days_data[(trade_days_data.index >= start_date) & (trade_days_data.index <= end_date)]
        # 返回格式化后的交易日列表
        return filtered_trade_days.index.strftime('%Y%m%d').tolist()
    except KeyError:
        # 如果工作表不存在，返回空列表
        return []


# 获取收益率数据的函数
def get_ret_from_excel(economic_situation, start_date, end_date, excel_path='C:\\Python程序设计基础\\pythonProject1\\日线数据.xls'):
    # 从指定的Excel文件和sheet中读取数据
    data = pd.read_excel(excel_path, sheet_name='六种资产', index_col=0, parse_dates=True)
    # 确保开始日期和结束日期在数据范围内，使得我们在计算收益率时不会出现问题（rets可以和weights对齐）
    data = data.loc[start_date:end_date]

    ts_code = ['510300.SH', '513500.SH', 'TL.CFX', 'AU.SHF', 'CU.SHF', 'SC.INE']
    rets = pd.DataFrame()

    for code in ts_code:
        # 获取指定标的的收盘价数据
        close = data[code]

        # 计算日收益率
        ret = np.log(close / close.shift(1))

        # 根据不同的资产类型调整收益率
        if code.endswith('.SHF') or code.endswith('.INE'):
            ret *= 1  # 对于'.SHF', '.INE'的收益率结果乘以3
        elif code.endswith('.CFX'):
            ret *= 1  # 对于'.CFX'的收益率结果乘以5

        # 将计算出的日收益率保存到字典中
        rets[code] = ret

    # 根据经济情况读取权重数据
    excel_path = ''  # 根据economic_situation确定具体sheet_name
    if economic_situation == '01':
        excel_path = 'C:\\Python程序设计基础\\pythonProject1\\monthly_weights_AU_CU_SC.xlsx'
    elif economic_situation == '02':
        excel_path = 'C:\\Python程序设计基础\\pythonProject1\\monthly_weights_300_500_SC.xlsx'
    elif economic_situation == '03':
        excel_path = 'C:\\Python程序设计基础\\pythonProject1\\monthly_weights_TL_AU_CU.xlsx'
    elif economic_situation == '04':
        excel_path = 'C:\\Python程序设计基础\\pythonProject1\\monthly_weights_300_500_TL.xlsx'
    else:
        print("经济情况不在01-04之间")

    # 读取权重数据
    weights_data = pd.read_excel(excel_path, index_col=0, engine='openpyxl')
    # 假设 get_trade_days 是一个函数，返回指定日期范围内的交易日列表
    # 注意：这里将start_date_base和end_date_base转换为实际的日期格式，以确保与权重数据的日期格式一致
    trade_days = get_trade_days(start_date, end_date)

    # 初始化一个空的DataFrame，以存储每日权重
    daily_weights = pd.DataFrame(index=pd.to_datetime(trade_days), columns=weights_data.columns)

    # 转换权重数据的索引为日期格式，确保它们与交易日对齐
    # 假设权重数据的索引已经是'YYYY-MM'格式，我们需要将它们转换为每个月的第一天
    weights_data.index = weights_data.index.to_period('M').to_timestamp()

    # 填充每日权重
    for date in trade_days:
        # 找到不晚于当前日期的最近的月末权重日期
        prev_month_end = weights_data.index[weights_data.index <= date][-1]
        # 使用这个月末的权重
        daily_weights.loc[date] = weights_data.loc[prev_month_end]

    # 由于直接赋值可能会导致数据类型不一致（赋值的是对象而不是数值），需要转换类型
    daily_weights = daily_weights.astype(float)

    # 现在daily_weights包含了每一天的权重，每个月的权重等于上个月月末的权重

    # 注意：这里假设weights_data至少有一个月末的权重数据，且start_date_base之前的最后一个月有权重数据

    # 计算加权收益率
    portfolio_return = (rets * daily_weights).sum(axis=1)

    return portfolio_return


# 获取权重的函数
def calculate_monthly_weights(economic_situations, start_date, end_date):
    ret = pd.DataFrame()
    for economic_situation in economic_situations:
        ret_ = get_ret_from_excel(economic_situation, start_date, end_date)
        ret = pd.concat([ret, ret_], axis=1)
    ret = ret.dropna()

    R_cov = ret.cov()  # 计算协方差
    cov = np.array(R_cov)

    def risk_budget_objective(weights, cov):
        weights = np.array(weights)  # weights为一维数组
        sigma = np.sqrt(np.dot(weights, np.dot(cov, weights)))  # 获取组合标准差
        MRC = np.dot(cov, weights) / sigma  # MRC = cov@weights/sigma
        TRC = weights * MRC
        delta_TRC = [sum((i - TRC) ** 2) for i in TRC]
        return sum(delta_TRC)

    x0 = np.ones(cov.shape[0]) / cov.shape[0]
    bnds = tuple((0, None) for x in x0)
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})
    options = {'disp': False, 'maxiter': 1000, 'ftol': 1e-20}

    solution = minimize(risk_budget_objective, x0, args=cov, bounds=bnds, constraints=cons, method='SLSQP',
                        options=options)

    final_weights = solution.x  # 权重
    return final_weights


# 主函数
if __name__ == "__main__":
    economic_situations = ['01', '02', '03', '04']
    start_date = '20181231'
    end_date = '20240325'
    start_date_base = '20181231'  # 备份日期
    end_date_base = '20240325'


    monthly_weights = []
    monthly_weights_2 = []
    months = pd.date_range(start=start_date, end=end_date, freq='M')

    # 考虑该时间点上，前一段时间的波动率
    for i in range(len(months)):
        month_end = months[i].strftime('%Y%m%d')
        # 设置month_start
        month_start = (pd.to_datetime(month_end) - pd.DateOffset(months=3)).strftime('%Y%m%d')  # 这里的问题是，实际上有些资产在五年前没有数据
        weights = calculate_monthly_weights(economic_situations, month_start, month_end)
        monthly_weights.append(weights)
        print(weights)

        # 考虑该时间点上，前一段时间的波动率
        # 设置month_start
        month_start_2 = (pd.to_datetime(month_end) - pd.DateOffset(months=1)).strftime('%Y%m%d')
        weights_2 = calculate_monthly_weights(economic_situations, month_start_2, month_end)
        monthly_weights_2.append(weights_2)
        print(weights_2)

    # 计算average_weights
    average_weights = [(np.array(w1) * 0.3 + np.array(w2) * 0.7) for w1, w2 in zip(monthly_weights, monthly_weights_2)]

    weights_df1 = pd.DataFrame(monthly_weights, columns=economic_situations)
    weights_df1.index = pd.date_range(start=start_date, end=end_date, freq='M')
    weights_df2 = pd.DataFrame(monthly_weights_2, columns=economic_situations)
    weights_df2.index = pd.date_range(start=start_date, end=end_date, freq='M')
    weights_df = pd.DataFrame(average_weights, columns=economic_situations)
    weights_df.index = pd.date_range(start=start_date, end=end_date, freq='M')
    weights_df.to_excel('monthly_weights_eco_situation.xlsx')
    print(weights_df1)
    print(weights_df2)
    print(weights_df)
