import numpy as np 
import pandas as pd 
from numpy.random import uniform
#读入csv文件
raw_data = pd.read_csv('RawPrices_QLearning.csv')#

#创造新列
raw_data['Q(W,S)'] = pd.Series(0,index = raw_data.index)
raw_data['Q(B,S)'] = pd.Series(0,index = raw_data.index)
raw_data['Q(S)'] = pd.Series(0,index = raw_data.index)
raw_data['B/W'] = pd.Series(0,index = raw_data.index)

#更新新列
for i in range(len(raw_data)-1):
    
    min_sum = 0
    for j in range(1,8):
        min_sum = min_sum + min(raw_data.ix[i+1:,j])
    raw_data.ix[i,'Q(W,S)'] = -min_sum/7
    #raw_data.ix[i,'Q(W,S)'] = -(min(raw_data.ix[i+1:,'Jun9'])+min(raw_data.ix[i+1:,'Jun10'])+min(raw_data.ix[i+1:,'Jun11'])+min(raw_data.ix[i+1:,'Jun12'])+min(raw_data.ix[i+1:,'Jun13'])+min(raw_data.ix[i+1:,'Jun14'])+min(raw_data.ix[i+1:,'Jun15']))/7
    
    raw_data.ix[i,'Q(B,S)'] = -uniform(500,1500)
    
    raw_data.ix[i,'Q(S)'] = max(raw_data.ix[i,'Q(W,S)'],raw_data.ix[i,'Q(B,S)'])
    
    raw_data.ix[i,'B/W'] = np.where(raw_data.ix[i,'Q(W,S)']>raw_data.ix[i,'Q(B,S)'], 'W', 'B')

#将结果输出为新的csv文件
raw_data.to_csv('RawPrices_QLearning_Treated.csv')