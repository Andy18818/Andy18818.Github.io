# 同时考虑近期和长期波动率，优化组合权重

# 在我们之前的Risk_Parity和Risk_Parity2中，我们使用的是tushare的API接口来调取数据，但一旦任务量提升就面临速度很慢的问题，同时有些数据调用还有权限限制
# 因此我们接下来都直接使用本地数据进行回测，数据可以从Choice和Wind下载
# 具体数据格式可以从代码中推测，或者可以看我上传的Excel文件示例

import pandas as pd
import numpy as np
from scipy.optimize import minimize


# 获取收益率数据的函数
def get_ret_from_excel(ts_code, start_date, end_date, excel_path='C:\\Python程序设计基础\\pythonProject1\\日线数据.xls', sheet_name='300+500+原油'):
    # 从指定的Excel文件和sheet中读取数据
    data = pd.read_excel(excel_path, sheet_name=sheet_name, index_col=0, parse_dates=True)

    # 确保开始日期和结束日期在数据范围内
    data = data.loc[start_date:end_date]

    # 获取指定标的的收盘价数据
    close = data[ts_code]

    # 计算日收益率
    ret = np.log(close / close.shift(1))

    # 根据不同的资产类型调整收益率
    if ts_code.endswith('.SHF') or ts_code.endswith('.INE'):
        ret *= 1  # 对于'.SHF', '.INE'的收益率结果乘以3
    elif ts_code.endswith('.CFX'):
        ret *= 1  # 对于'.CFX'的收益率结果乘以5

    return ret


# 获取权重的函数
def calculate_monthly_weights(start_date, end_date, assets):
    ret = pd.DataFrame()
    for asset in assets:
        ret_ = get_ret_from_excel(asset, start_date, end_date)
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
    ts_code = ['510300.SH', '513500.SH', 'SC.INE']
    start_date = '20180630'
    end_date = '20240325'

    monthly_weights = []
    monthly_weights_2 = []
    months = pd.date_range(start=start_date, end=end_date, freq='M')

    # 考虑该时间点上，前一段时间的波动率
    for i in range(len(months)):
        month_end = months[i].strftime('%Y%m%d')
        # 设置month_start
        month_start = (pd.to_datetime(month_end) - pd.DateOffset(months=3)).strftime('%Y%m%d')  # 这里的问题是，实际上有些资产在五年前没有数据
        weights = calculate_monthly_weights(month_start, month_end, ts_code)
        monthly_weights.append(weights)
        print(weights)

        # 考虑该时间点上，前一段时间的波动率
        # 设置month_start
        month_start_2 = (pd.to_datetime(month_end) - pd.DateOffset(months=1)).strftime('%Y%m%d')
        weights_2 = calculate_monthly_weights(month_start_2, month_end, ts_code)
        monthly_weights_2.append(weights_2)
        print(weights_2)

    # 计算average_weights
    average_weights = [(np.array(w1) * 0.3 + np.array(w2) * 0.7) for w1, w2 in zip(monthly_weights, monthly_weights_2)]

    weights_df1 = pd.DataFrame(monthly_weights, columns=ts_code)
    weights_df1.index = pd.date_range(start=start_date, end=end_date, freq='M')
    weights_df2 = pd.DataFrame(monthly_weights_2, columns=ts_code)
    weights_df2.index = pd.date_range(start=start_date, end=end_date, freq='M')
    weights_df = pd.DataFrame(average_weights, columns=ts_code)
    weights_df.index = pd.date_range(start=start_date, end=end_date, freq='M')
    weights_df.to_excel('monthly_weights_300_500_SC.xlsx')
    print(weights_df1)
    print(weights_df2)
    print(weights_df)
