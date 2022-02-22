import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from matplotlib.pylab import style                                   # 自定义图表风格
style.use('ggplot')
# 解决中文的显示问题
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf         # 自相关图、偏自相关图
from statsmodels.tsa.stattools import adfuller as ADF                # 平稳性检验
from statsmodels.stats.diagnostic import acorr_ljungbox              # 白噪声检验
import statsmodels.api as sm                                         # D-W检验,一阶自相关检验
from statsmodels.graphics.api import qqplot                          # 画QQ图,检验一组数据是否服从正态分布
from statsmodels.tsa.arima.model import ARIMA
import pyflux as pf
sale = pd.read_csv('C:/Users/derrick/Desktop/bit.csv', index_col='日期', encoding='gbk')
sale.head()



print('-----')
sale.info()
plt.figure(figsize=(10, 5))
sale.plot()
plt.show()

# 解读：具有单调递增趋势，则是非平稳序列。

plot_acf(sale, lags=35).show()

# 解读：自相关系数长期大于零，没有趋向于零，说明序列间具有很强的长期相关性。

# 方法：单位根检验

print('原始序列的ADF检验结果为：', ADF(sale.价格))

# 解读：P值大于显著性水平α（0.05），接受原假设（非平稳序列），说明原始序列是非平稳序列。

d1_sale = sale.diff(periods=1, axis=0).dropna()

# 时序图
plt.figure(figsize=(10, 5))
d1_sale.plot()
plt.show()
# 解读：在均值附件比较平稳波动

# 自相关图
plot_acf(d1_sale, lags=34).show()
# 解读：有短期相关性，但趋向于零。

# 平稳性检验
print('原始序列的ADF检验结果为：', ADF(d1_sale.价格))

# 解读：P值小于显著性水平α（0.05），拒绝原假设（非平稳序列），说明一阶差分序列是平稳序列。

print('一阶差分序列的白噪声检验结果为：', acorr_ljungbox(d1_sale, lags=1))  # 返回统计量、P值

# 解读：p值小于0.05，拒绝原假设（纯随机序列），说明一阶差分序列是非白噪声序列。

d1_sale = sale.diff(periods=1, axis=0).dropna()

# 自相关图
plot_acf(d1_sale, lags=34).show()

# 解读：有短期相关性，但趋向于零。

# 偏自相关图
plot_pacf(d1_sale, lags=10).show()

# 偏自相关图
plot_pacf(d1_sale, lags=17).show()

# 解读：自相关图，1阶截尾；偏自相关图，拖尾。则ARIMA(p,d,q)=ARIMA(0,1,1)
pmax = int(len(d1_sale) / 10)  # 一般阶数不超过length/10
qmax = int(len(d1_sale) / 10)  # 一般阶数不超过length/10


aic_matrix = []
for p in range(pmax + 1):
    tmp = []
    for q in range(qmax + 1):
        try:
            tmp.append(pf.ARIMA(data=sale, ar=p, ma=q, target='价格', family=pf.Normal()).fit().aic)
        except:
            tmp.append(None)
    aic_matrix.append(tmp)
aic_matrix = pd.DataFrame(aic_matrix)
p, q = aic_matrix.stack().idxmin()  # 最小值的索引
#创建模型
model = pf.ARIMA(data=sale, ar=p, ma=q, target='价格', family=pf.Normal()).fit("MLE")
#查看模型报告

model.summary()
