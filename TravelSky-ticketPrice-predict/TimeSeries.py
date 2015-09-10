import pandas as pd
import numpy as np

#读入数据
raw_data = pd.read_csv('RawPrices.csv')
#创造新列
raw_data['TF'] = pd.Series(0,index = raw_data.index)

#更新新列
weights = np.arange(1,57)/float(sum(range(1,57)))
for i in range(56,len(raw_data)):
		 if np.inner(raw_data.ix[i-56:i-1,0],weights)>raw_data.ix[i,0]:
		 	raw_data.ix[i,1] = True
		 else:
		 	raw_data.ix[i,1] = False
#将结果输出为新的csv文件
raw_data.to_csv('RawPrices_Treated.csv')