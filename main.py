import pandas as pd
import numpy as np
import math, sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyClassifier
    


plt.rc('font', size=18); plt.rcParams['figure.constrained_layout.use'] = True
# read data. column 1 is date/time, col 6 is #bikes
# df = pd.read_csv("Datasets/Avondale_Street_Data.csv", usecols = [2,7], parse_dates=[1])
df = pd.read_csv("Datasets/Dame_Street_Data.csv", usecols = [2,7], parse_dates=[1])
print("DF.HEAD ->")
print(df.head())

# DF.HEAD ->
#                   TIME AVAILABLE BIKES
# 0  2020-01-01 06:25:02              15
# 1  2020-01-01 06:30:02              15
# 2  2020-01-01 06:35:02              15
# 3  2020-01-01 06:40:03              15
# 4  2020-01-01 06:45:02              15
# data sampling interval is 300 secs



# Start Date 01 Jan 2020
# End Date 24 Feb 2020
start=pd.to_datetime("01−01−2020",format='%d−%m−%Y')
end=pd.to_datetime("24−02−2020",format='%d−%m−%Y')

# convert date/time to unix timestamp in sec
t_full=pd.array(pd.DatetimeIndex(df.iloc[:,0]).view(np.int64))/1000000000
dt = t_full[1]-t_full[0]
print("data sampling interval is %d secs"%dt)

# extract data between start and end dates
t_start = pd.DatetimeIndex([start]).view(np.int64)/1000000000
t_end = pd.DatetimeIndex([end]).view(np.int64)/1000000000

t = np.extract([(t_full>=t_start) & (t_full<=t_end)], t_full)

t=(t-t[0])/60/60/24 # convert timestamp to days





y = np.extract([(t_full>=t_start) & (t_full<=t_end)], df.iloc[:,1]).astype(np.int64)





def test_preds(q,dd,lag,plot, title_str):
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
        plt.title(title_str)
        plt.scatter(t, y, color='black'); plt.scatter(tt, y_pred, color='red')
        plt.xlabel("time (days)"); plt.ylabel("#bikes")
        plt.legend(["training data","predictions"],loc='upper right')
        day=math.floor(24*60*60/dt) # number of samples per day
        plt.xlim(((lag*dd+q)/day,(lag*dd+q)/day+25))
        plt.show()


def put_it_together():
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
        plt.title('Putting it Together Method : Rathdown House')
        plt.scatter(t, y, color='red'); plt.scatter(tt, y_pred, color='blue')
        plt.xlabel("time (days)"); plt.ylabel("#bikes")
        plt.legend(["training data","predictions"],loc='upper right')
        day=math.floor(24*12/dt) # number of samples per day
        # plt.xlim((4*7,4*7+4))
        plt.show()


def apply_models(q,dt,lag,y,t,c_value,title, plot):
    print(title)
    #q−step ahead prediction
    stride=1
    #set up putting-together features
    # features: [y^(k-3*w), y^(k-2*w), y^(k-1*w),y^(k-3*d), y^(k-2*d), y^(k-1*d), y^(k-3-q), y^(k-2-q), y^(k-1-q)]
    w=math.floor(7*24*60*60/dt) # number of samples per week
    len = y.size-w-lag*w-q
    XX=y[q:q+len:stride]
    # print('HERE\n\n\n')
    # print(XX)
    
    for i in range(1,lag): # set up [y^(k-3*w), y^(k-2*w), y^(k-1*w)] when lag=3
        X=y[i*w+q:i*w+q+len:stride] 
        XX=np.column_stack((XX,X)) #add weekly feature to XX
    d=math.floor(24*60*60/dt) # number of samples per day
    
    for i in range(0,lag): # set up [y^(k-3*d), y^(k-2*d), y^(k-1*d)] when lag=3
        X=y[i*d+q:i*d+q+len:stride]
        XX=np.column_stack((XX,X)) #add daily feature to XX
    
    for i in range(0,lag): # set up [y^(k-3-q), y^(k-2-q), y^(k-1-q)] when lag=3
        X=y[i:i+len:stride]
        XX=np.column_stack((XX,X)) #add short-term feature to XX
    
    
    yy=y[lag*w+w+q:lag*w+w+q+len:stride] # set up yy for putting-together feature
    tt=t[lag*w+w+q:lag*w+w+q+len:stride] # set up tt for putting-together feature

    
    train, test = train_test_split(np.arange(0,yy.size),test_size=0.2)

    
    ai = (1/(2*c_value)) 
    ridge_model = Ridge(fit_intercept=False, alpha=ai).fit(XX[train], yy[train])
    lasso_model = Lasso(fit_intercept=False, alpha=ai).fit(XX[train], yy[train]) 
   
   
    # NEW ADDITION
    decision_tree_model =  DecisionTreeRegressor()
    decision_tree_model.fit(XX[train], yy[train])

    

    #Baseline model
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(XX[train], yy[train])
    
    print("\n")
    print("the value of each parameters in feature")
    print(ridge_model.intercept_, ridge_model.coef_)
    print("\n")

    #preditions by each model
    y_pred_ridge = ridge_model.predict(XX)
    y_pred_lasso = lasso_model.predict(XX)

    y_pred_decision_tree = decision_tree_model.predict(XX)
    y_pred__dummy = dummy_clf.predict(XX)
    #evaluation of the results, using mean square error
    from sklearn.metrics import mean_squared_error
    print("\n")
    print(f"Results for {title}")

    from sklearn.metrics import accuracy_score
    # accuracy = accuracy_score(yy, y_pred_ridge)
    # print('accuracy of ridge = %f' % (accuracy))

    # accuracy = accuracy_score(yy, y_pred_lasso)
    # print('accuracy of lasso = %f' % (accuracy))

    accuracy = accuracy_score(yy, y_pred__dummy)
    print('accuracy of dummy = %f' % (accuracy))

    # score = decision_tree_model.score(yy[train], y_pred__dummy)
    # accuracy = accuracy_score(yy, y_pred_decision_tree)
    # print('accuracy of decision tree = %f' % (score))


    print(f"MSE (ridge model) {(mean_squared_error(yy,y_pred_ridge))}")
    print(f"MSE (lasso model) {(mean_squared_error(yy,y_pred_lasso))}")
    print(f"MSE (decision tree model) {(mean_squared_error(yy,y_pred_decision_tree))}")
    print(f"MSE (baseline model) {(mean_squared_error(yy,y_pred__dummy))}")
    print("\n")

    if plot is True:
        # Plot Ridge Prediction
        plt.rcParams["figure.figsize"] = (13, 4)
        plt.scatter(t, y, color='black'); 
        plt.scatter(tt, y_pred_ridge, color='blue')
        plt.ylabel("No. of bikes")
        plt.xlabel("Time in days)"); 
        
        plt.legend(["Real Data","Ridge Predictions"],loc='upper right')
        plt.title(title)
        plt.xlim((7*7,7*7+5))
        plt.savefig('Final Plots/'+title+' Ridge.png')
        plt.show()

        # Polt Lasso Predictions
        plt.rcParams["figure.figsize"] = (13, 4)
        plt.scatter(t, y, color='black'); 
        plt.scatter(tt, y_pred_lasso, color='yellow', marker='<')
        plt.ylabel("No. of bikes")
        plt.xlabel("Time in days)"); 
        
        plt.legend(["Real data","Lasso Predictions"],loc='upper right')
        plt.title(title)
        plt.xlim((7*7,7*7+5))
        plt.savefig('Final Plots/'+title+' Lasso.png')
        plt.show()
        
        # Plot Decision Tree Predictions
        plt.rcParams["figure.figsize"] = (13, 4)
        plt.scatter(t, y, color='black'); 
        plt.scatter(tt,y_pred_decision_tree, color='green', marker='x')
        plt.ylabel("No. of bikes")
        plt.xlabel("Time in days)"); 
        
        plt.legend(["Real data","Decision Trees Predictions"],loc='upper right')
        plt.title(title)
        plt.xlim((7*7,7*7+5))
        plt.savefig('Final Plots/'+title+' Decision Tree.png')
        plt.show()
        
        # # To plot all the predictions on the same graph, uncomment this
        # plt.xlabel("time (days)"); 
        # plt.ylabel("#bikes")
        # plt.legend(["training data","ridge_predictions", "lasso_predictions", "decision_trees_predictions"],loc='upper left')
        # plt.title(title)
        # plt.show()



# prediction using short−term trend
# q = 1 is 5 mins
# q = 2 is 10 mins
# q = 12 is 60 mins
plot=True
# test_preds(q=2,dd=1,lag=3,plot=plot, title_str = '5-step ahead (10 mins) : Rathdown House')
# test_preds(q=6,dd=1,lag=3,plot=plot, title_str = '5-step ahead (30 mins) : Rathdown House')
# test_preds(q=12,dd=1,lag=3,plot=plot, title_str = '5-step ahead (60 mins) : Rathdown House')



# put_it_together()



# prediction using short−term trend
stand_name = 'Dame Street'

# stand_name = 'Avondale_Street_Data'
apply_models(q=2,dt=dt,lag=3,y=y, t=t, c_value=0.5,title= "Prediction for no. of bikes at " +stand_name + " in 10 mins", plot=True)

# prediction using short−term trend
# features: [y^(k-3-q), y^(k-2-q), y^(k-1-q)]
apply_models(q=6,dt=dt,lag=3,y=y,t=t, c_value=0.5,title= "Prediction for no. of bikes at " +stand_name + " in 30 mins", plot=True)

# prediction using short−term trend
# features: [y^(k-3-q), y^(k-2-q), y^(k-1-q)]
apply_models(q=12,dt=dt,lag=3,y=y, t=t, c_value=0.5,title= "Prediction for no. of bikes at " +stand_name + " in 1 hour", plot=True)



#                   TIME AVAILABLE BIKES
# 0  2020-01-01 06:25:02              15
# 1  2020-01-01 06:30:02              15
# 2  2020-01-01 06:35:02              15
# 3  2020-01-01 06:40:03              15
# 4  2020-01-01 06:45:02              15
# data sampling interval is 300 secs
# Prediction for no. of bikes at Dame Street in 10 mins


# the value of each parameters in feature
# 0.0 [-0.00237841  0.51347205  0.37716833 -0.00237841  0.27792848 -0.08224472
#   0.00108212 -0.00403457 -0.00237841]




# The evaluation of the results
# accuracy of dummy = 0.143424
# square error of ridge model 19.747978 
# square error of lasso model 19.857163 
# square error of decision tree model 1.861937 
# square error of baseline model 76.504957 


# Prediction for no. of bikes at Dame Street in 30 mins


# the value of each parameters in feature
# 0.0 [-0.01192638  0.54639528  0.35230897 -0.01192638  0.26792422 -0.08026728
#   0.06282058 -0.08704942  0.0293456 ]




# The evaluation of the results
# accuracy of dummy = 0.143804
# square error of ridge model 19.787076 
# square error of lasso model 19.927744 
# square error of decision tree model 1.531331 
# square error of baseline model 76.525514 


# Prediction for no. of bikes at Dame Street in 1 hour


# the value of each parameters in feature
# 0.0 [ 0.01432902  0.54699722  0.35288374  0.01432902  0.26382408 -0.05893043
#  -0.1109462   0.1668689  -0.10965035]




# The evaluation of the results
# accuracy of dummy = 0.144378
# square error of ridge model 19.830361 
# square error of lasso model 19.994912 
# square error of decision tree model 1.556443 
# square error of baseline model 76.527611 