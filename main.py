import pandas as pd
import numpy as np
import math, sys
import matplotlib.pyplot as plt

plt.rc('font', size=18); plt.rcParams['figure.constrained_layout.use'] = True
# read data. column 1 is date/time, col 6 is #bikes
df = pd.read_csv("Datasets/Dame_Street_Data.csv", usecols = [2,7], parse_dates=[1])
# print(df.head())
# 3rd Feb 2020 is a monday, 10th is following monday
start=pd.to_datetime("01−01−2020",format='%d−%m−%Y')
end=pd.to_datetime("24−02−2020",format='%d−%m−%Y')


# convert date/time to unix timestamp in sec
t_full=pd.array(pd.DatetimeIndex(df.iloc[:,0]).view(np.int64))/1000000000


print(t_full[0])


dt = t_full[1] - t_full[0]
print("data sampling interval is %d secs"%dt)
# extract data between start and end dates
t_start = pd.DatetimeIndex([start]).view(np.int64)/1000000000
t_end = pd.DatetimeIndex([end]).view(np.int64)/1000000000
t = np.extract([(t_full>=t_start) & (t_full<=t_end)], t_full)
t = (t - t[0]/60/60/24) # convert timestamp to days

y = np.extract([(t_full>=t_start) & (t_full<=t_end)], df.iloc[:,1]).astype(np.int64)
# #plot extracted data
# plt.scatter(t,y, color='red', marker='.'); plt.show()








def test_preds(q,dd,lag,plot):
    print('Test Preds function called')
    #q−step ahead prediction
    stride=1
    XX = y[0 : y.size - q - lag * dd : stride]
    for i in range(1,lag):
        X = y[i*dd : y.size - q - (lag-i) * dd : stride]
        XX = np.column_stack((XX, X))
    
    yy=y[lag*dd+q::stride]; tt=t[lag*dd+q::stride]

    from sklearn.model_selection import train_test_split
    train, test =  train_test_split(np.arange(0,yy.size),test_size=0.2)


    from sklearn.linear_model import Ridge
    model = Ridge(fit_intercept=False).fit(XX[train], yy[train])

    print(model.intercept_, model.coef_)

    if plot:
        y_pred = model.predict(XX)
        # print(tt , y_pred[0])
        plt.scatter(t, y, color='black'); plt.scatter(tt, y_pred, color='red')
        plt.xlabel("time (days)"); plt.ylabel("#bikes")
        plt.legend(["training data","predictions"],loc='upper right')
        day=math.floor(24*60*60/dt) # number of samples per day
        # plt.xlim(((lag*dd+q)/day,(lag*dd+q)/day+2))
        plt.show()

# prediction using short−term trend
plot=True
test_preds(q=1,dd=1,lag=3,plot=plot)

#putting it together
q=10
lag=3; stride=1
w=math.floor(7*24*12/dt) # number of samples per week
len = y.size - w - lag*w - q

XX=y[q:q+len:stride]
for i in range(1,lag):
    X=y[i*w+q:i*w+q+len:stride]
    XX=np.column_stack((XX,X))
d=math.floor(24*12/dt) # number of samples per day
for i in range(0,lag):
    X=y[i*d+q:i*d+q+len:stride]
    XX=np.column_stack((XX,X))
for i in range(0,lag):
    X=y[i:i+len:stride]
    XX=np.column_stack((XX,X))
yy=y[lag*w+w+q:lag*w+w+q+len:stride]
tt=t[lag*w+w+q:lag*w+w+q+len:stride]

from sklearn.model_selection import train_test_split
train, test = train_test_split(np.arange(0,yy.size),test_size=0.2)
#train = np.arange(0,yy.size)
from sklearn.linear_model import Ridge
model = Ridge(fit_intercept=False).fit(XX[train], yy[train])
print(model.intercept_, model.coef_)
if plot:
    y_pred = model.predict(XX)
    plt.scatter(t, y, color='black'); plt.scatter(tt, y_pred, color='blue')
    plt.xlabel("time (days)"); plt.ylabel("#bikes")
    plt.legend(["training data","predictions"],loc='upper right')
    day=math.floor(24*12/dt) # number of samples per day
    # plt.xlim((4*7,4*7+4))
    plt.show()