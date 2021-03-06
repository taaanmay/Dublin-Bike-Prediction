from matplotlib import colors
import pandas as pd
import numpy as np
import math, sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold    


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

final_compate_std = []
final_compate_mean = []




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

def cross_validation_for_C(XX, yy, title):
    Cs = [0.00001, 0.00005, 0.0001, 0.0005, 0.001,0.005, 0.01,0.05, 0.1, 0.5, 1, 5, 10, 50, 100]

    ridge_means=[]
    lasso_means=[]
    ridge_stds=[]
    lasso_stds=[]
    
    for c_i in Cs:
        ridge_error_mse=[]
        lasso_error_mse=[]


        kf = KFold(n_splits=5)
        
        for train, test in kf.split(yy):
            alp = (1/(2*c_i)) 
            
            # Lasso
            lasso_model = Lasso(fit_intercept=False, alpha=alp)
            lasso_model.fit(XX[train], yy[train]) 
            lasso_y_pred = lasso_model.predict(XX)
            lasso_error_mse.append(mean_squared_error(yy,lasso_y_pred))
    
            
            # Ridge
            ridge_model = Ridge(fit_intercept=False, alpha=alp)
            ridge_model.fit(XX[train], yy[train])
            ridge_y_pred = ridge_model.predict(XX)
            ridge_error_mse.append(mean_squared_error(yy,ridge_y_pred))
            
        ridge_means.append(np.array(ridge_error_mse).mean())
        lasso_means.append(np.array(lasso_error_mse).mean())

        ridge_stds.append(np.array(ridge_error_mse).std())
        lasso_stds.append(np.array(lasso_error_mse).std())

    

    plt.errorbar(Cs,lasso_means,yerr=lasso_stds)
    plt.xlabel('Ci'); 
    plt.ylabel('MSE (Lasso) '+title)
    plt.xlim((0,1.0))
    plt.show()


    plt.errorbar(Cs,ridge_means,yerr=ridge_stds)
    plt.xlabel('Ci'); 
    plt.ylabel('MSE (Ridge) '+title)
    plt.show()

    


def apply_models(q,dt,lag,y,t,c_value,title, plot):
    print(title)
    #q−step ahead prediction
    stride=1
    
    
    w=math.floor(7*24*60*60/dt) # number of samples per week
    len = y.size-w-lag*w-q
    XX=y[q:q+len:stride]
    # print('HERE\n\n\n')
    # print(XX)
    

    # Features =  [y^(k-3*w), y^(k-2*w), y^(k-1*w),y^(k-3*d), y^(k-2*d), y^(k-1*d), y^(k-3-q), y^(k-2-q), y^(k-1-q)]
    for i in range(1,lag): 
        X=y[i*w+q:i*w+q+len:stride] 
        XX=np.column_stack((XX,X)) 
    d=math.floor(24*60*60/dt) 
    
    for i in range(0,lag): 
        X=y[i*d+q:i*d+q+len:stride]
        XX=np.column_stack((XX,X)) 
    
    for i in range(0,lag): 
        X=y[i:i+len:stride]
        XX=np.column_stack((XX,X)) 
    
    
    yy=y[lag*w+w+q:lag*w+w+q+len:stride] 
    tt=t[lag*w+w+q:lag*w+w+q+len:stride] 

    
    train, test = train_test_split(np.arange(0,yy.size),test_size=0.2)

    # # UNCOMMENT BELOW LINE FOR CROSS VALIDATION OF values of C
    # cross_validation_for_C(XX, yy, 'Cross Validation for C')

    ai = (1/(2*0.001)) 
    # Ridge Model
    ridge_model = Ridge(fit_intercept=False, alpha=ai).fit(XX[train], yy[train])
    
    ai = (1/(2*c_value)) 
    # Lasso Model
    lasso_model = Lasso(fit_intercept=False, alpha=ai).fit(XX[train], yy[train]) 
   
   
    # Decision Tree
    decision_tree_model =  DecisionTreeRegressor()
    decision_tree_model.fit(XX[train], yy[train])
    
    

    # Dummy Regressor
    dummy_reg = DummyClassifier(strategy="most_frequent")
    dummy_reg.fit(XX[train], yy[train])
    

    print("\n\nIntercept and Coefficient of Ridge ->")
    print(ridge_model.intercept_, ridge_model.coef_)
    
    print("\n\nIntercept and Coefficient of Lasso ->")
    print(lasso_model.intercept_, lasso_model.coef_)
    

    
    # Predict
    ridge_y_pred = ridge_model.predict(XX)
    lasso_y_pred = lasso_model.predict(XX)
    decision_tree_y_pred = decision_tree_model.predict(XX)
    baseline_y_pred = dummy_reg.predict(XX)

    # la_error.append(metrics.mean_squared_error(yy, lasso_y_pred))

    # Use MSE
    
    print(f"Results for {title}")


    # score = decision_tree_model.score(yy[train], baseline_y_pred)
    # accuracy = accuracy_score(yy, decision_tree_y_pred)
    # print('accuracy of decision tree = %f' % (score))


    print(f"MSE (ridge model) {(mean_squared_error(yy,ridge_y_pred))}")
    print(f"MSE (lasso model) {(mean_squared_error(yy,lasso_y_pred))}")
    print(f"MSE (decision tree model) {(mean_squared_error(yy,decision_tree_y_pred))}")
    print(f"MSE (baseline model) {(mean_squared_error(yy,baseline_y_pred))}")
    print("\n")

    

    if plot is True:
        # Plot Ridge Prediction
        plt.rcParams["figure.figsize"] = (13, 4)
        plt.scatter(t, y, color='black'); 
        plt.scatter(tt, ridge_y_pred, color='blue')
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
        plt.scatter(tt, lasso_y_pred, color='yellow', marker='<')
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
        plt.scatter(tt,decision_tree_y_pred, color='green', marker='x')
        plt.ylabel("No. of bikes")
        plt.xlabel("Time in days)"); 
        
        plt.legend(["Real data","Decision Trees Predictions"],loc='upper right')
        plt.title(title)
        plt.xlim((7*7,7*7+5))
        plt.savefig('Final Plots/'+title+' Decision Tree.png')
        plt.show()
        



# prediction using short−term trend
# q = 1 is 5 mins
# q = 2 is 10 mins
# q = 12 is 60 mins
plot=True
# test_preds(q=2,dd=1,lag=3,plot=plot, title_str = '5-step ahead (10 mins) : Rathdown House')
# test_preds(q=6,dd=1,lag=3,plot=plot, title_str = '5-step ahead (30 mins) : Rathdown House')
# test_preds(q=12,dd=1,lag=3,plot=plot, title_str = '5-step ahead (60 mins) : Rathdown House')



# put_it_together()




stand_name = 'Dame Street'
# stand_name = 'Avondale_Street_Data'

apply_models(q=2,dt=dt,lag=3,y=y, t=t, c_value=0.5,title= "Prediction for no. of bikes at " +stand_name + " in 10 mins", plot=True)


apply_models(q=6,dt=dt,lag=3,y=y,t=t, c_value=0.5,title= "Prediction for no. of bikes at " +stand_name + " in 30 mins", plot=True)

apply_models(q=12,dt=dt,lag=3,y=y, t=t, c_value=0.5,title= "Prediction for no. of bikes at " +stand_name + " in 1 hour", plot=True)

