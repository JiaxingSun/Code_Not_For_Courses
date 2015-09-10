import numpy as np
import pandas as pd
from numpy.random import uniform
#read the csv file into raw_data
raw_data = pd.read_csv('ArtificialData.csv')
#create new columns
raw_data['Q(W,S)'] = pd.Series(0,index = raw_data.index)
raw_data['Q(B,S)'] = pd.Series(0,index = raw_data.index)
raw_data['B/W_ratio'] = pd.Series(0,index = raw_data.index)
raw_data['B/W_TimeSeries'] = pd.Series(0,index = raw_data.index)
raw_data['real_tag'] = pd.Series(0,index = raw_data.index)
#update the data
for i in range(240*50):
    #update the 'B/W ratio'
    min_sum = 0
    if (i+1)%240 !=0:
        for j in range(1,8):
            min_sum = min_sum + min(raw_data.ix[(i+j*240+1):((i+j*240)/240*240+239),'price'])
        raw_data.ix[i,'Q(W,S)'] = -min_sum/7
    else:
        raw_data.ix[i,'Q(W,S)'] = -3000000
    raw_data.ix[i,'Q(B,S)'] = -raw_data.ix[i,'price']
    raw_data.ix[i,'B/W_ratio'] = raw_data.ix[i,'Q(B,S)']/float(raw_data.ix[i,'Q(W,S)'])
    
    #update the 'real tag'
    if (i+1)%240 !=0:
        if raw_data.ix[i,'price']< min(raw_data.ix[(i+1):((i+1)/240*240+239),'price'])*1.05:
            raw_data.ix[i,'real_tag'] = 'B'
        else:
            raw_data.ix[i,'real_tag'] = 'W'
    else:
        raw_data.ix[i,'real_tag'] = 'B'
    
    #update the 'B/W_TimeSeries'
    weights = np.arange(1,57)/float(sum(range(1,57)))
    if i%240 >= 56:
        if np.inner(raw_data.ix[i-56:i-1,'price'],weights)>raw_data.ix[i,'price']:
            raw_data.ix[i,'B/W_TimeSeries'] = 'B'
        else:
            raw_data.ix[i,'B/W_TimeSeries'] = 'W'
    
raw_data