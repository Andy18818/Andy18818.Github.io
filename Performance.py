import pandas as pd
import tushare as ts

# 定义文件路径
file_path = 'C:\\Python程序设计基础\\pythonProject1\\Performance.xlsx'

# 读取数据
df = pd.read_excel(file_path, engine='openpyxl')

# 计算年化收益率
days = (df['Trade Date'].iloc[-1] - df['Trade Date'].iloc[0]).days
annual_return = ((df['Net Value'].iloc[-1] / df['Net Value'].iloc[0]) ** (365.0 / days)) - 1

# 计算最大回撤
rolling_max = df['Net Value'].cummax()
daily_drawdown = df['Net Value'] / rolling_max - 1.0
max_drawdown = daily_drawdown.min()

# 计算夏普比率
pro = ts.pro_api('20231208200557-b4f921df-1bab-430e-ab72-9d36e3a2ecbe')
pro._DataApi__http_url = 'http://tsapi.majors.ltd:7000'
rf = pro.yc_cb(ts_code='1001.CB', curve_type='0', trade_date='20240325').sort_values(by='curve_term').iloc[0]['yield']  # 无风险利率
daily_returns = df['Net Value'].pct_change()  # 计算日收益率
std_dev = daily_returns.std() * (365 ** 0.5)  # 年化标准差
sharpe_ratio = (annual_return - rf/100) / std_dev

# 在DataFrame中添加计算结果
df['Annual Return'] = None
df.loc[0, 'Annual Return'] = annual_return

df['Max Drawdown'] = None
df.loc[0, 'Max Drawdown'] = max_drawdown

df['Sharpe Ratio'] = None  # 添加一列，但只在第一行填入计算的夏普比率
df.loc[0, 'Sharpe Ratio'] = sharpe_ratio

# 保存更新后的DataFrame回Excel
df.to_excel(file_path, index=False)

print(f"年化收益率: {annual_return*100:.2f}%, 最大回撤: {max_drawdown*100:.2f}%, 夏普比率: {sharpe_ratio:.2f}")