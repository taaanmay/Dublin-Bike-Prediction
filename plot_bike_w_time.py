import pandas as pd
import numpy as np
import math, sys
import matplotlib.pyplot as plt

plt.rc('font', size=18); plt.rcParams['figure.constrained_layout.use'] = True
# read data. column 1 is date/time, col 6 is #bikes
df = pd.read_csv("Datasets/Rathdown_House_Data.csv", usecols = [2,7], parse_dates=[1])
# print(df.head())
# 3rd Feb 2020 is a monday, 10th is following monday
start=pd.to_datetime("27−01−2020",format='%d−%m−%Y')
end=pd.to_datetime("29−01−2020",format='%d−%m−%Y')


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
plt.scatter(t,y, color='red', marker='.'); plt.show()