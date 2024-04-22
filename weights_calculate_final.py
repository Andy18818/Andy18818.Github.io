import pandas as pd

# 读取子策略权重
strategy_weights = pd.read_excel('C:\\Python程序设计基础\\pythonProject1\\monthly_weights_eco_situation.xlsx', index_col=0, engine='openpyxl')

# 读取各个子策略中资产的权重
weights_AU_CU_SC = pd.read_excel('C:\\Python程序设计基础\\pythonProject1\\monthly_weights_AU_CU_SC.xlsx', index_col=0, engine='openpyxl')
weights_300_500_SC = pd.read_excel('C:\\Python程序设计基础\\pythonProject1\\monthly_weights_300_500_SC.xlsx', index_col=0, engine='openpyxl')
weights_TL_AU_CU = pd.read_excel('C:\\Python程序设计基础\\pythonProject1\\monthly_weights_TL_AU_CU.xlsx', index_col=0, engine='openpyxl')
weights_300_500_TL = pd.read_excel('C:\\Python程序设计基础\\pythonProject1\\monthly_weights_300_500_TL.xlsx', index_col=0, engine='openpyxl')

# 确保所有权重数据的索引格式一致
strategy_weights.index = pd.to_datetime(strategy_weights.index)
weights_AU_CU_SC.index = pd.to_datetime(weights_AU_CU_SC.index)
weights_300_500_SC.index = pd.to_datetime(weights_300_500_SC.index)
weights_TL_AU_CU.index = pd.to_datetime(weights_TL_AU_CU.index)
weights_300_500_TL.index = pd.to_datetime(weights_300_500_TL.index)

# 创建一个空的DataFrame来存储最终的资产权重
asset_weights = pd.DataFrame(index=strategy_weights.index, columns=['510300.SH', '513500.SH', 'TL.CFX', 'AU.SHF', 'CU.SHF', 'SC.INE'])

# 计算每个资产在整个资产组合中的权重
for date in strategy_weights.index:
    for col in asset_weights.columns:
        asset_weight = 0
        if col in weights_AU_CU_SC.columns:
            asset_weight += strategy_weights.loc[date, '01'] * weights_AU_CU_SC.loc[date, col]
        if col in weights_300_500_SC.columns:
            asset_weight += strategy_weights.loc[date, '02'] * weights_300_500_SC.loc[date, col]
        if col in weights_TL_AU_CU.columns:
            asset_weight += strategy_weights.loc[date, '03'] * weights_TL_AU_CU.loc[date, col]
        if col in weights_300_500_TL.columns:
            asset_weight += strategy_weights.loc[date, '04'] * weights_300_500_TL.loc[date, col]
        asset_weights.loc[date, col] = asset_weight

# 将计算结果保存到Excel文件
print(asset_weights)
asset_weights.to_excel('C:\\Python程序设计基础\\pythonProject1\\final_asset_weights.xlsx')
