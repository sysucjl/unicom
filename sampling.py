import pandas as pd
import numpy as np 

print 'reading training-set...'
train_raw = pd.read_csv('./data/4G/ZDYCL_4G_201507.csv',header=None)

train_at = train_raw[train_raw[45]==0]
train_lt = train_raw[train_raw[45]==1]

radio = 2
sampler = np.random.permutation(len(train_at))
train_at_sample = train_at.take(sampler[:len(train_lt)*radio])

train_sample = pd.concat([train_at_sample,train_lt],join='inner',ignore_index=True)
sampler = np.random.permutation(len(train_sample))
train_sample = train_sample.take(sampler)