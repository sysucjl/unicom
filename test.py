#coding=UTF-8
import pandas as pd

print 'reading predicting-set...'
pred = pd.read_csv('../result/predict_09_10.csv')

print 'reading testing-set...'
test_raw = pd.read_csv('../data/4G/ZDYCL_4G_201510.csv',header=None)
test = test_raw[45]

count = 0
for i in range(len(pred)):
	if pred['label'][i]>0 and test[i]>0:
		count = count + 1

pred_lt = len(pred[pred['label']>0])
total_lt = len(test_raw[test_raw[45]>0])
print '预测离网并实际离网人数：'+str(count)
print '预测离网人数：'+str(pred_lt)
print '实际离网人数：'+str(total_lt)
print '查全率:'+str(count*1.0/total_lt*100) + '%'
print '查准率:'+str(count*1.0/pred_lt*100) + '%'