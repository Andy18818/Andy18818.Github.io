import pandas as pd
import numpy as np
import tushare as ts
from scipy.optimize import minimize

# 读入3支股票 2023-04-10 到 2024-04-09 日收盘价数据，并计算对数收益率

ts.set_token('your tusahre token') #这里我使用tushare接口调取数据，这里需要替换成你自己的tushare token，或者直接从本地读取收益率数据

# 初始化pro接口
pro = ts.pro_api()


def get_ret(ts_code):
    data = pro.daily(ts_code=ts_code, start_date='20230410', end_date='20240409')
    data.index = pd.to_datetime(data['trade_date'],format='%Y-%m-%d') #设置日期索引
    close = data['close'] #日收盘价
    close.name = ts_code
    ret = np.log(close/close.shift(1)) #日收益率
    return ret

ts_code = ['002531.SZ', '603259.SH', '603501.SH']
ret = pd.DataFrame()
for ts_code in ts_code:
 ret_ = get_ret(ts_code)
 ret = pd.concat([ret,ret_],axis=1)
ret = ret.dropna()

R_cov = ret.cov() #计算协方差
cov = np.array(R_cov)


def risk_budget_objective(weights, cov):
    weights = np.array(weights)  # weights为一维数组
    sigma = np.sqrt(np.dot(weights, np.dot(cov, weights)))  # 获取组合标准差
    # sigma = np.sqrt(weights@cov@weights)
    MRC = np.dot(cov, weights) / sigma  # MRC = cov@weights/sigma
    # MRC = np.dot(weights,cov)/sigma
    TRC = weights * MRC
    delta_TRC = [sum((i - TRC) ** 2) for i in TRC]
    return sum(delta_TRC)


'''
#若将权重weights转化为二维数组的形式，则编制函数如下：
def risk_budget_objective(weights,cov):
    weights = np.matrix(weights) #weights为二维数组
    sigma = np.sqrt(np.dot(weights, np.dot(cov, weights.T))[0,0]) #获取组合标准差   
    #sigma = np.sqrt((weights@cov@weights.T)[0,0])   
    MRC = np.dot(cov,weights.T).A1/sigma
    #MRC = np.dot(weights,cov).A1/sigma
    TRC = weights.A1 * MRC
    delta_TRC = [sum((i - TRC)**2) for i in TRC]
    return sum(delta_TRC)
'''


def total_weight_constraint(x):
    return np.sum(x) - 1.0


x0 = np.ones(cov.shape[0]) / cov.shape[0]
bnds = tuple((0, None) for x in x0)
cons = ({'type': 'eq', 'fun': total_weight_constraint})
# cons = ({'type':'eq', 'fun': lambda x: sum(x) - 1})
options = {'disp': False, 'maxiter': 1000, 'ftol': 1e-20}

solution = minimize(risk_budget_objective, x0, args=cov, bounds=bnds, constraints=cons, method='SLSQP',
                    options=options)

# 求解出权重
final_weights = solution.x  # 权重
for i in range(len(final_weights)):
    print(f'{final_weights[i]:.1%}投资于{R_cov.columns[i]}')
