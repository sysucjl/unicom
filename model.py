# coding=UTF-8
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import pandas as pd
import numpy as np

# 读取数据
print 'reading training-set...'
train_raw = pd.read_csv('../data/4G/ZDYCL_4G_201509.csv',header=None)
print 'reading testing-set...'
test_raw = pd.read_csv('../data/4G/ZDYCL_4G_201510.csv',header=None)

# 取样
train_at = train_raw[train_raw[45]==0]
train_lt = train_raw[train_raw[45]==1]

radio = 1 #取样比
if radio!=0:
	sampler = np.random.permutation(len(train_at))
	train_at_sample = train_at.take(sampler[:len(train_lt)*radio])

	train_sample = pd.concat([train_at_sample,train_lt],join='inner',ignore_index=True)
	sampler = np.random.permutation(len(train_sample))
	train_sample = train_sample.take(sampler)

	train_data = train_sample[list(range(1,45))]
	train_label = train_sample[45]

	print len(train_sample[train_sample[45]>0])

else:
	train_data = train_raw[list(range(1,45))]
	train_label = train_raw[45]

test_data = test_raw[list(range(1,45))]
test_label = test_raw[45]
print len(test_raw[test_raw[45]>0])

# 用于二分类的多层感知器
model = Sequential()
model.add(Dense(64, input_dim=44, init='uniform', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# model = Sequential()

# model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
#                         border_mode='valid',
#                         input_shape=(1, img_rows, img_cols)))
# model.add(Activation('relu'))
# model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
# model.add(Dropout(0.25))

# model.add(Flatten())
# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(nb_classes))
# model.add(Activation('softmax'))

# model.compile(loss='categorical_crossentropy',
#               optimizer='adadelta',
#               metrics=['accuracy'])

x_train = np.array(train_data)
y_train = np.array(train_label)

model.fit(x_train,y_train,nb_epoch=10, batch_size=32)
result = model.predict(np.array(test_data),batch_size=32, verbose=1)
predict_result = pd.DataFrame(result,columns=['label'])
predict_result[predict_result>0]=1
print predict_result.describe()
predict_result.to_csv('../result/predict_09_10.csv',index=False,header=True)  

x_test = np.array(test_data)
y_test = np.array(test_label)
score = model.evaluate(x_test,y_test,verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])

