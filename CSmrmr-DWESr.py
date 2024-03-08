from itertools import product
from sklearn.feature_selection import RFE
from matplotlib import pyplot
from sklearn.ensemble import RandomForestRegressor
from pandas import read_csv
import time
from ypstruct import struct
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import math
from numpy.linalg import norm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, fcluster
import random
import statsmodels.api as sm
from time import time
from sklearn.cluster import KMeans
from statsmodels.tsa.statespace.sarimax import SARIMAX

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")



####### READING DATASET 
df = pd.read_csv('/home/hossein/Downloads/avocado_datasets/avocado_l24.csv', sep=";")
df['KilosEntrados'] = df['KilosEntrados'].clip(lower=0)
data = df
data.index = pd.to_datetime(data['FechaRecepcion'])
data.drop(columns='FechaRecepcion', inplace=True)

# # Dataset 2  mango-l24
# df = pd.read_csv('/home/hossein/mango_l24.csv', sep=";")
# df['KilosEntrados'] = df['KilosEntrados'].clip(lower=0)
# # df.drop(df.tail(12).index,inplace=True)
# data = df


variables = data.loc[:, data.columns != 'KilosEntrados']
var = variables

col = np.zeros(int(var.shape[1]))
col_name = ["" for x in range(int(var.shape[1]))]
for i in range(int(var.shape[1])):
    col[i] = i
    col_name[i] = var.columns[i]

var = variables.rename(columns={x: y for x, y in zip(
    variables.columns, range(0, len(variables.columns)))})


#################### APPLICATION of  CSmrmr

from time import time
t0 = time()
fs = SelectKBest(score_func=f_regression, k='all')
# fs = SelectKBest(score_func=mutual_info_regression,k='all')


fs.fit(var, data.loc[:, 'KilosEntrados'])
df2 = pd.DataFrame(index=range(var.shape[0]), columns=range(var.shape[1]+1))
discarding_size = np.ceil(var.shape[1]*0.1)
corr_mrmr_index = 0
#######
corr_mrmr_value = pd.DataFrame(index=range(
    var.shape[0]), columns=range(int(np.ceil(discarding_size))))
#######
corr_mrmr = np.zeros(int(np.ceil(discarding_size)))
corr_mrmr_names = ['']*(int(np.ceil(discarding_size)))
###########
Normalized_Corr = np.zeros(var.shape[1])
#########
cor_dataframe = pd.DataFrame(fs.scores_)
for g in range(var.shape[1]):
    Normalized_Corr[g] = (cor_dataframe.iloc[g, :] - np.min(cor_dataframe)
                          )/(np.max(cor_dataframe)-np.min(cor_dataframe))

n = 0
Normalized_Corr = pd.DataFrame(Normalized_Corr)
corr_mrmr_index = 0
corr_mrmr[corr_mrmr_index] = np.argmax(Normalized_Corr)
corr_mrmr_names[n] = col_name[int(corr_mrmr[corr_mrmr_index])]

var = var.reset_index(drop=True)
corr_mrmr_value.iloc[:, corr_mrmr_index] = var.iloc[:,
                                                    np.argmax(Normalized_Corr)]

Not_Selected_var = var.iloc[:, var.columns !=
                            var.columns[np.argmax(Normalized_Corr)]]
Normalized_Corr = Normalized_Corr.drop([np.argmax(Normalized_Corr)])
dist = np.zeros(Not_Selected_var.shape[1])
jac = np.zeros(Not_Selected_var.shape[1])


for j in range(1, int(discarding_size)):

    mean_value = corr_mrmr_value.mean(axis=1)

    for g in range(Not_Selected_var.shape[1]):

        AB = abs(np.dot(Not_Selected_var.iloc[:, g], mean_value))
        A = norm(Not_Selected_var.iloc[:, g])**2
        B = norm(mean_value)**2
        jac[g] = (AB/(A+B-AB))
        dist[g] = Normalized_Corr.iloc[g, :] - (AB/(A+B-AB))

    n = n+1
    temp = Not_Selected_var.columns[np.argmax(dist, axis=0)]
    corr_mrmr_index = corr_mrmr_index+1
    # features kept by corr_mrmr
    corr_mrmr[corr_mrmr_index] = temp
    # dataset sliced and kept by corr_mrmr
    corr_mrmr_value.iloc[:, corr_mrmr_index] = Not_Selected_var.loc[:, temp]
    # dataset discarded by corr_mrr
    Not_Selected_var = Not_Selected_var.iloc[:,
                                             Not_Selected_var.columns != temp]
    Normalized_Corr = Normalized_Corr.drop([temp])
    jac = np.zeros(Not_Selected_var.shape[1])
    dist = np.zeros(Not_Selected_var.shape[1])
    corr_mrmr_names[n] = col_name[temp]

t1 = time()
print(t1-t0)

################### FUNCTIONS 
def sarimax_evaluate_models(df1, df2, configs):
    """
    Evaluates all possible SARIMAX parameters.

    Returns the best parameter selection.
    """
    best_cfg = None
    best_score = 0
    for config in configs:
        p, d, q, P_value, D_value, Q_value, seasonality = config
        order = (p, d, q)
        s_order = (P_value, D_value, Q_value, seasonality)
        try:
            r2 = evaluate_sarimax_model_r2(df1, df2, order, s_order)
            print("Configuration: ", config, " has r2 of: ", r2)
            if r2 > best_score:
                best_score, best_cfg = r2, [order, s_order]
        except Exception as err:
            print(f"SARIMAX config {config} has raised an error.")
            print(err)
            continue
    # ###### added for CS    
    if  best_cfg == None:
        best_cfg=[(0, 1, 0), (0, 1, 0, 12)]
    # #####    
    return best_cfg


def evaluate_sarimax_model_r2(df1, df2, order, s_order):
    """
    Evaluates the SARIMAX model given certain parameters using R2.

    Returns the R2 score as a float value
    """
    try:
        model = SARIMAX(
            endog=df1.iloc[:, df1.shape[1]-1],
            exog=df1.iloc[:, 0:df1.shape[1]-1],
            order=order,
            seasonal_order=s_order,
            enforce_invertibility=False,
            enforce_stationarity=False,

        )
        results = model.fit(disp=False)

        
        pred_uc = results.get_forecast(
            exog=df2.iloc[:, 0:df2.shape[1]-1], steps=12
        )
        # pred_uc = results.get_forecast(
        #     exog=df2.iloc[:, 0:df2.shape[1]-1], steps=24
        # )
        # return r2_score(df2["KilosEntrados"].values, pred_uc.predicted_mean)
        return r2_score(df2.iloc[:, df2.shape[1]-1], pred_uc.predicted_mean)
    except Exception as err:
        print(err)
        return float("inf")


def sarimax_configs(seasonality: int = 12):
# def sarimax_configs(seasonality: int = 24):

    """
    Method to get all posible parameter configurations for Grid Search.

    Returns a list of tuples with parameters
    """
    p_params = [0, 1, 2]
    d_params = [1]
    q_params = [0, 1, 2]
    P_params = [0, 1, 2]
    D_params = [1]
    Q_params = [0, 1, 2]
    Q_params = [0, 1, 2]


    configs = product(
        p_params, d_params, q_params, P_params, D_params, Q_params, [
            seasonality]
    )

    return configs


def sarimax_fit(df, config):
    """
    Method to train SARIMA given data and its parameters.

    Returns the fitted model.
    """
    order, sorder = config

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        model = SARIMAX(
            
            endog=df.iloc[:, df.shape[1]-1],
            exog=df.iloc[:, 0:df.shape[1]-1],
            order=order,
            seasonal_order=sorder,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit_model = model.fit(disp=False)

    return fit_model



def theil_index(y, y_est):
    n = len(y)
    num = math.sqrt(np.sum(np.power(y - y_est, 2)) / n)
    den1 = math.sqrt(np.sum(np.power(y, 2)) / n)
    den2 = math.sqrt(np.sum(np.power(y_est, 2)) / n)
    return num / (den1 + den2)

################# TESTING ORIGINAL YEAR############

df = pd.read_csv('/home/hossein/Downloads/avocado_datasets/avocado_l24.csv', sep=";")
df['KilosEntrados'] = df['KilosEntrados'].clip(lower=0)
data3 = df
data3.index = pd.to_datetime(data3['FechaRecepcion'])
data3.drop(columns='FechaRecepcion', inplace=True)
df_train = data3[:-12]
df_test = data3[-12:]
df_train = df_train.reset_index()
df_train.drop(columns='FechaRecepcion', inplace=True)
df_train.drop(columns='KilosEntrados', inplace=True)
df_test = df_test.reset_index()
df_test.drop(columns='FechaRecepcion', inplace=True)
df_test.drop(columns='KilosEntrados', inplace=True)
data3 = df
data33 = data3.reset_index()
data33.drop(columns='FechaRecepcion', inplace=True)
yntrain = data33.loc[0:data33.shape[0]-13, 'KilosEntrados']
yntest = data33.loc[data33.shape[0]-12:data33.shape[0], 'KilosEntrados']
yntest = yntest.reset_index()
yntest.drop(columns='index', inplace=True)
#######################################



###### TESTING DISCARDED YEAR
data3 = data
df_train = data3[:-12]
df_test = data3[-12:]
df_train = df_train.reset_index()
df_train.drop(columns='KilosEntrados', inplace=True)
df_test = df_test.reset_index()
df_test.drop(columns='KilosEntrados', inplace=True)
data33 = data3.reset_index()
yntrain = data33.loc[0:data33.shape[0]-13, 'KilosEntrados']
yntest = data33.loc[data33.shape[0]-12:data33.shape[0], 'KilosEntrados']
yntest = yntest.reset_index()
yntest.drop(columns='index', inplace=True)

#########################





############################# using CSmrmr####################
discarding_size = np.ceil(var.shape[1]*0.1)
# discarding_size=10

d_train = pd.DataFrame(index=range(
    int(df_train.shape[0])), columns=range(int(discarding_size)))
for j in range(int(corr_mrmr.shape[0])):
    d_train.iloc[:, j] = df_train.iloc[:, int(corr_mrmr[j])]
vari = d_train


d_test = pd.DataFrame(index=range(
    int(df_test.shape[0])), columns=range(int(discarding_size)))
for j in range(int(corr_mrmr.shape[0])):
    d_test.iloc[:, j] = df_test.iloc[:, int(corr_mrmr[j])]

 ##########################DWES#############################
f_array = struct(features=None, r2=None, tiempo=None)
f_arrays = f_array.repeat(50)
maeList=np.zeros(50)
rmseList=np.zeros(50)
theilList=np.zeros(50)
for g in range(50):

    

    t0 = time()
    iter = 50
    stop = 0
    accC = np.zeros(iter)
    opt = 0
    no_op = 0
    Xn = vari
    Xnt = Xn.T
    cl_num = 3    # cl_num is number of clusters in DWES
    kmeans = KMeans(n_clusters=cl_num).fit(Xnt)
    cluster_labels = kmeans.labels_
    cln = pd.DataFrame(cluster_labels)
    cln.columns = ['labels']
    cln = cln+1
    alfa = 0.1

    th = np.full((1, cl_num), 0.5)      # setup initial threshold

    th1 = np.zeros(cl_num)

    for i in range(cl_num):       # generate random threshold
        th1[i] = random.uniform(0, 1)

    while all(i >= 0.50 for i in th1):
        for i in range(cl_num):       # generate random threshold
            th1[i] = random.uniform(0, 1)

    # compare with initial threshold and select target clusters
    mask = np.where(th1 <= th)
    mask = mask[1]+1

    subset_size = np.sum(th1 < th)

# construct one solution
    subset = np.zeros(len(mask))

    for i in range(len(mask)):
        subset[i] = cln.labels[cln.labels.eq(mask[i])].sample().index.values

# construct the respective matrix
    matn = np.zeros((d_train.shape[0], len(mask)+1))
    matnT = np.zeros((d_test.shape[0], len(mask)+1))
# df_ = pd.DataFrame(index=61, columns=cl_num+1)
    for i in range(len(mask)):
        matn[:, i] = d_train.iloc[:, int(subset[i])]
    matn[:, len(mask)] = yntrain

    for i in range(len(mask)):
        matnT[:, i] = d_test.iloc[:, int(subset[i])]
    matnT[:, len(mask)] = yntest.squeeze()

    selected_variables = ['']*len(corr_mrmr_names)
    for i in range(len(subset)):
        selected_variables[i] = corr_mrmr_names[int(subset[i])]
# run SVM or DT for new mat
    it = 1

    configs = sarimax_configs()
# grid search
    matn1 = pd.DataFrame(matn)
    matnT1 = pd.DataFrame(matnT)
    best_cfg = sarimax_evaluate_models(matn1, matnT1, configs)
    BEST_CFG = best_cfg

    results = sarimax_fit(matn1, best_cfg)
    forecast = results.get_forecast(exog=matnT1.iloc[:, 0:matnT1.shape[1]-1], steps=12)
    # forecast = results.get_forecast(exog=matnT1.iloc[:, 0:matnT1.shape[1]-1], steps=24)

    forecasted_values = forecast.predicted_mean

    r2_sarimax = r2_score(matnT1.iloc[:, matnT1.shape[1]-1], forecasted_values)

    r2 = round(r2_sarimax, 2)

    FIT = r2
    opt = opt+1
    TRAIN = matn1
    TEST=matnT1
    s_size = np.zeros(iter)
    count = 0
    s_size[count] = len(subset)
    print(" Best R2",  "  ", r2, "      ", "subset_size", "  ", s_size[count], "     ", "iteration", "  ",  it)

    es = 1
    while (es < iter) and (stop < 5):  # for es in range(iter):

        # construct one solution
        th1 = np.zeros(cl_num)

        for i in range(cl_num):       # generate random threshold
            th1[i] = random.uniform(0, 1)
        while all(i >= 0.50 for i in th1):
            for i in range(cl_num):       # generate random threshold
                th1[i] = random.uniform(0, 1)

        # compare with initial threshold and select target clusters
        mask = np.where(th1 <= th)
        mask = mask[1]+1

# construct one solution
        tempset = np.zeros(len(mask))
        for i in range(len(mask)):
            # tempset[i]=np.array(random.choices(cln.labels[cln.labels.eq(mask[i])].index))
            tempset[i] = cln.labels[cln.labels.eq(
                mask[i])].sample().index.values
# construct the respective matrix

        matn = np.zeros((d_train.shape[0], len(mask)+1))
        matnT = np.zeros((d_test.shape[0], len(mask)+1))
        for i in range(len(mask)):
            matn[:, i] = d_train.iloc[:, int(tempset[i])]
        matn[:, len(mask)] = yntrain

        for i in range(len(mask)):
            matnT[:, i] = d_test.iloc[:, int(tempset[i])]
        matnT[:, len(mask)] = yntest.squeeze()

        configs = sarimax_configs()
     # grid search
        matn1 = pd.DataFrame(matn)
        matnT1 = pd.DataFrame(matnT)
        best_cfg = sarimax_evaluate_models(matn1, matnT1, configs)

        results = sarimax_fit(matn1, best_cfg)
        forecast = results.get_forecast(exog=matnT1.iloc[:, 0:matnT1.shape[1]-1], steps=12)
        # forecast = results.get_forecast(exog=matnT1.iloc[:, 0:matnT1.shape[1]-1], steps=24)

        forecasted_values = forecast.predicted_mean

        temp_r2 = r2_score(
            matnT1.iloc[:, matnT1.shape[1]-1], forecasted_values)
        temp_r2 = round(temp_r2, 2)
        temp_FIT = temp_r2
     # tempnm=np.mean(nm)
        it = it+1
        if temp_FIT > FIT:
            opt = opt+1
            FIT = temp_FIT
            r2 = temp_r2
            accC[es] = FIT
            count = count+1
            s_size[count] = len(mask)

            TRAIN = matn1
            TEST=matnT1
            Features = tempset
            stop = 0
            BEST_CFG=best_cfg
            selected_variables = ['']*len(corr_mrmr_names)
            for i in range(len(tempset)):
                selected_variables[i] = corr_mrmr_names[int(tempset[i])]
            m = mask-1
            for i in range(len(m)):       # generate random threshold
                th[0, m[i]] = th[0, m[i]]+(alfa*(1-th[0, m[i]]))
                if (th[0, m[i]] > 1):
                    th[0, m[i]] = 1
            s_size[count] = len(tempset)
            print("Best R2",  "  ", r2, "      ",   "subset_size", "  ", s_size[count], "     ", "iteration", "  ",  it)

        elif  temp_r2 == FIT and len(mask)<s_size[count]:
           
           stop=0
           no_op=no_op+1
           count=count+1   
           s_size[count]=len(mask)
           TRAIN = matn1
           TEST=matnT1
           BEST_CFG=best_cfg
           features=tempset
           accC[es] = r2_sarimax 
           selected_variables=['']*len(corr_mrmr_names)
           for i in range(len(mask)):
              
               selected_variables[i]=corr_mrmr_names[int(tempset[i])]
           m=mask-1    
           for i in range(len(m)):       # generate random threshold 
              th[0,m[i]]=th[0,m[i]]+(alfa*(1-th[0,m[i]]))
              if (th[0,m[i]]>1):
                  th[0,m[i]]=1
           s_size[count]=len(tempset) 
           #    
           print("Best R2",  "  ", r2, "      ", "subset_size", "  ", s_size[count], "     ", "iteration", "  ",  it, " Current run","  ",  g)
        
        else:
            stop = stop+1
            accC[es] = FIT
            print(" Best R2",  "  ", r2, "      ",  "subset_size", "  ", s_size[count], "     ", "iteration", "  ",  it, " Current run","  ",  g)
        es = es+1

    t1 = time()

    
    f_arrays[g].features = selected_variables
    f_arrays[g].r2 = r2
    f_arrays[g].tiempo = t1-t0
    
    results = sarimax_fit(TRAIN, BEST_CFG)
    forecast=results.get_forecast(exog=TEST.drop(TEST.columns[TEST.shape[1]-1], axis=1),steps=12)
    # forecast=results.get_forecast(exog=TEST.drop(TEST.columns[TEST.shape[1]-1], axis=1),steps=24)

    forecasted_values = forecast.predicted_mean
    
    maeList[g] = round(mean_absolute_error(TEST[TEST.shape[1]-1].values, forecasted_values))
    rmseList[g] = round(math.sqrt(mean_squared_error(TEST[TEST.shape[1]-1].values, forecasted_values)))
    theilList[g] = (theil_index(TEST[TEST.shape[1]-1].values, forecasted_values))

##############

# FREQUENCY OF SELECTED FEATURES AFTER 50 RUNS
num = np.zeros(50)
fs = []
summation = 0
ac = 0
for i in range(50):
    fs.append(f_arrays[i].features)
    summation = f_arrays[i].tiempo+summation
    ac = f_arrays[i].r2+ac
    num[i] = len(np.where(np.array(fs[i]) != '')[0])

fs2 = []
for i in range(0, 50):
    for j in range(int(discarding_size)):
        if fs[i][j] != '':
            fs2.append(fs[i][j])

temp = np.unique(fs2)
temp2 = np.zeros(len(temp))
for i in range(len(temp)):
    temp2[i] = len(np.where(np.array(fs2) == temp[i])[0])
##########MEASUREMENT CRITERIA AFTER 50 RUNS

ac/50
summation/50
np.mean(num)
sum(maeList)/len(maeList)
sum(rmseList)/len(rmseList)
sum(theilList)/len(theilList)




##############SAMPLE CODE FOR DRAWING FIGURE FOR FREQUENCY OF SELECTED FEATURES AFTER 50 RUNS

barlist = plt.bar(temp, temp2)
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Frequency')
plt.title('Avocado_L24 (Original)')
plt.show()




