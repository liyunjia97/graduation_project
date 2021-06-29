'''
本code旨在提取轨迹特征完成对ETA的预估
'''
#%%
import pandas as pd
from datetime import datetime
import seaborn as sns 
import random
import shapely.geometry as geo 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, RepeatedKFold
import lightgbm as lgb
from hyperopt import fmin, tpe, hp,space_eval,rand,Trials,partial,STATUS_OK
from sklearn.metrics import r2_score,mean_squared_error, mean_absolute_error
from keras import losses
from sklearn import preprocessing
from keras.layers import  Input, concatenate
from keras.layers.core import Dense, Dropout, Activation, Flatten, Permute, Reshape
from keras.models import Model
from keras.optimizers import RMSprop, Adam, SGD, Nadam
from tqdm import tqdm 
from math import sqrt
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from pyproj import Transformer
#%%
df=pd.read_csv(r'/home/liyunjia/data/AIS_data_process/trajectory_calculate_new/trajectory_01.csv')
BJS_format = "%Y-%m-%d %H:%M:%S" 
df['Receivedtime（UTC+8）'] = df['Receivedtime（UTC+8）'].apply(lambda x: datetime.strptime(x, BJS_format))
lon_lat=df[['Lon_d','Lat_d']]
transformer = Transformer.from_crs("epsg:4326", "epsg:3857")
Lat1, Lon1 = transformer.transform(lon_lat['Lat_d'], lon_lat['Lon_d'])
Lon_lat1=pd.concat([pd.DataFrame(Lat1),pd.DataFrame(Lon1)],axis=1)  
Lon_lat1.columns=['Lat_d1','Lon_d1'] 
df=pd.concat([df,Lon_lat1],axis=1)
scaler = StandardScaler()
scaler_Lat_Lon = pd.DataFrame(scaler.fit_transform(pd.concat([pd.DataFrame(df['Lat_d1']),pd.DataFrame(df['Lon_d1'])],axis=1)))
scaler_Lat_Lon.columns=['Lat','Lon']
Lon_lat2=pd.concat([pd.DataFrame(scaler_Lat_Lon['Lat']),pd.DataFrame(scaler_Lat_Lon['Lon'])],axis=1)  
Lon_lat2.columns=['Lat_d1_scaler','Lon_d1_scaler'] 
df=pd.concat([df,Lon_lat2],axis=1)
#%%
mmsi_list=df['MMSI'].unique()
i=-1
data={}
data_to_up_cluster_after=pd.DataFrame()
for mmsi in tqdm(mmsi_list):
    df1=df[df['MMSI']==mmsi]
    para_list=df1['paragraph'].unique()
    for para in para_list:
        df2=df1[df1['paragraph']==para]
        if (len(df2)>20) and len(df2)<300:
            i+=1
            data[i]=df2
            label_type=labels_035['type'].iloc[i]
            data[i]['label']=label_type
            data[i]['order']=labels_035['order'].iloc[i]
            if label_type==0:
                data_to_up_cluster_after=data_to_up_cluster_after.append(data[i]) 
#%%
df=data_to_up_cluster_after
#%%
BJS_format = "%Y-%m-%d %H:%M:%S"
df['Receivedtime（UTC+8）'] = df['Receivedtime（UTC+8）'].apply(lambda x: datetime.strptime(x, BJS_format))
#小时和周几 进行划分
df['hour'] = df['Receivedtime（UTC+8）'].dt.hour
df['weekday'] = df['Receivedtime（UTC+8）'].dt.weekday
#白天黑夜划分
df['day_nig'] = 0
df.loc[(df['hour'] > 5) & (df['hour'] < 20),'day_nig'] = 1

#%%
#按照df进行提取特征
def time_calculate(df):
    mmsi_list_origin=df['MMSI'].unique()
    time_predict_list=[]
    time_train_list=[]
    mmsi_list=[]
    ShipTypeEN_list=[]
    Length_list=[]
    Draught_list=[]
    Speed_list=[]
    traj_num_list=[]
    break2=False
    Line_length_list=[]
    for mmsi in tqdm(mmsi_list_origin):
        df1=df[df['MMSI']==mmsi]
        paragraph_list=df1['paragraph'].unique()
        for paragraph in paragraph_list:
            df2=df1[df1['paragraph']==paragraph]
            ratio=random.uniform(0.3,0.7)
            traj_num=int(len(df2)*ratio)
            df_train=df2.iloc[:traj_num]
            
            time_predict=(df2['Receivedtime（UTC+8）'].iloc[-1]-df2['Receivedtime（UTC+8）'].iloc[0]).total_seconds()#预测值
            time_train=(df_train['Receivedtime（UTC+8）'].iloc[-1]-df_train['Receivedtime（UTC+8）'].iloc[0]).total_seconds()
            geo_linestring=geo.asLineString(np.array(df_train[['Lon_d','Lat_d']]))
            ShipTypeEN=df_train['ShipTypeEN'].iloc[0]
            Length=df_train['Length'].iloc[0]
            Draught=df_train['Draught'].iloc[0]
            Speed=df_train['Speed'].mean()
            time_predict_list.append(time_predict)
            time_train_list.append(time_train)
            ShipTypeEN_list.append(ShipTypeEN)
            Length_list.append(Length)
            Draught_list.append(Draught)
            Speed_list.append(Speed)
            mmsi_list.append(mmsi)
            traj_num_list.append(traj_num/len(df2))
            Line_length_list.append(geo_linestring.length)
    df_feature=pd.DataFrame()
    df_feature['MMSI']=mmsi_list
    df_feature['Speed']=Speed_list
    df_feature['Draught']=Draught_list
    df_feature['Length']=Length_list
    df_feature['ShipTypeEN']=ShipTypeEN_list
    df_feature['time_train']=time_train_list
    df_feature['traj_num']=traj_num_list
    df_feature['Line_length']=Line_length_list
    df_feature['time_predict']=time_predict_list
    return df_feature
#%%
df_train_forecast=time_calculate(df)
# %%
# colorMap = {elem:index+1 for index,elem in enumerate(set(df["ShipTypeEN"]))}
# df_train_forecast['ShipTypeEN'] = df_train_forecast['ShipTypeEN'].map(colorMap)
# df_train_forecast['ShipTypeEN']=df_train_forecast['ShipTypeEN']-1

#%%
#%%
y=df_train_forecast['time_predict']
X=df_train_forecast.drop(['MMSI','time_predict','ShipTypeEN'],axis=1)
scale_x=preprocessing.MinMaxScaler().fit(X)
X=scale_x.transform(X)
scale_y=preprocessing.MinMaxScaler().fit(np.array(y).reshape(-1, 1))
y=scale_y.transform(np.array(y).reshape(-1, 1))
# X=df_train_forecast.drop(['MMSI','time_predict'],axis=1)
data=pd.DataFrame(df_train_forecast,columns=['ShipTypeEN'])
cat_=pd.get_dummies(df_train_forecast['ShipTypeEN'])
X=pd.concat([cat_,pd.DataFrame(X)],axis=1)
#%%

#%%
X.astype('category')
X_data, X_test, y_data, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
X_data = np.array(X_data)
X_test = np.array(X_test)
Y_data = np.array(y_data)
Y_test = np.array(y_test)
#%%
space = {'units1': hp.choice('units1', [16,64,128,320,512]),
        'units2': hp.choice('units2', [16,64,128,320,512]),
        'units3': hp.choice('units3', [16,64,128,320,512]),
        'lr': hp.choice('lr',[0.01, 0.001, 0.0001]),
        'activation': hp.choice('activation',['relu',
                                                'sigmoid',
                                                'tanh',
                                                'linear']),
        'loss': hp.choice('loss', [losses.logcosh,
                                    losses.mse,
                                    losses.mae,
                                    losses.mape])}
def experiment(params):
    main_input = Input(shape=(16, ), name='main_input')
    x = Dense(params['units1'], activation=params['activation'])(main_input)
    x = Dense(params['units2'], activation=params['activation'])(x)
    x = Dense(params['units3'], activation=params['activation'])(x)
    output = Dense(1, activation = "linear", name = "out")(x)
    final_model = Model(inputs=[main_input], outputs=[output])
    opt = Adam(lr=params['lr'])
    final_model.compile(optimizer=opt,  loss=params['loss'])
    folds = KFold(n_splits=10, shuffle=True, random_state=2018)
    # oof_lgb = np.zeros(len(X_data))
    # predictions_lgb = np.zeros(len(X_test))
    # predictions_train_lgb = np.zeros(len(X_data))
    mse_list=[]
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_data, Y_data)):
        X_trn=X_data[trn_idx]
        Y_trn=Y_data[trn_idx]
        X_val=X_data[val_idx]
        Y_val=Y_data[val_idx]
        history = final_model.fit(X_trn, Y_trn, 
            epochs = 30, 
            batch_size = 256, 
            verbose=0, 
            validation_data=(X_val, Y_val),
            shuffle=True)
        y_test_predict=final_model.predict(X_test)
        mse=mean_squared_error(Y_test,y_test_predict)
        mse_list.append(mse)
    mse = np.mean(mse_list)    
    print('mse',mse)
    return mse
algo = partial(tpe.suggest,n_startup_jobs=1)
best = fmin(experiment,space,algo=algo,max_evals=200)
#%%

X 
#%%
# print(best)
#{'activation': 0, 'loss': 1, 'lr': 1, 'units1': 4, 'units2': 1, 'units3': 1}
main_input = Input(shape=(18, ), name='main_input')
x = Dense(512, activation='relu')(main_input)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
output = Dense(1, activation = "linear", name = "out")(x)
final_model = Model(inputs=[main_input], outputs=[output])
opt = Adam(lr=0.001)
final_model.compile(optimizer=opt,loss=losses.mse)
# %%
history = final_model.fit(X_data, Y_data, 
    epochs = 30, 
    batch_size = 256, 
    verbose=0, 
    validation_data=(X_test, Y_test),
    shuffle=True)
# %%
y_test_predict=final_model.predict(X_test)
#%%
y_data_predict=final_model.predict(X_data)
#%%
def inverse_trans(x):
    max_=df_train_forecast['time_predict'].max()
    min_=df_train_forecast['time_predict'].min()
    return x*(max_-min_)+min_ 
# %%
df1=pd.concat([pd.DataFrame(inverse_trans(Y_test)),pd.DataFrame(inverse_trans(y_test_predict)),pd.DataFrame(inverse_trans(Y_test))-pd.DataFrame(inverse_trans(y_test_predict))],axis=1)
df1.columns=['time_true','time_predict','time_diff']
#%%
df_test=pd.concat([pd.DataFrame(X_test),pd.DataFrame(inverse_trans(Y_test)),pd.DataFrame(inverse_trans(y_test_predict)),pd.DataFrame(inverse_trans(Y_test))-pd.DataFrame(inverse_trans(y_test_predict))],axis=1)
# df_test.columns=['Cargo ship', 'Engaged in dredging or underwater operations', 'Fishing',
#        'HSC', 'Law enforcement vessel', 'Other type of ship', 'Passenger ship',
#        'Tanker', 'Towing', 'Tug','Speed','Draught','Length','time_train','traj_num','Line_length','time_test','time_test_predict','diff_time']
# %%
plt.subplots(figsize=(10,7))
plt.plot(y_test_predict,label='time_prediction')
plt.plot(Y_test,label='time_true')
plt.legend()
# %%
#这样最后求解得到的特征mse是0.09左右。
plt.plot((y_test_predict-Y_test).flatten())
#%%
df_test.sort_values(by='diff_time')

# %%
from sklearn.metrics import r2_score,mean_squared_error, mean_absolute_error
mse = mean_squared_error(df1['time_predict'],  df1['time_true'])
rmse = sqrt(mean_squared_error(df1['time_predict'],  df1['time_true']))
mape = np.mean(np.abs((df1['time_predict'] -  df1['time_true']) / df1['time_predict'])) * 100
mae = mean_absolute_error(df1['time_predict'],  df1['time_true'])
R_2=r2_score(df1['time_true'],df1['time_predict'])
print("mse: {:<8.8f}".format(mse))
print("rmse: {:<8.8f}".format(rmse))
print("mape: {:<8.8f}".format(mape))
print("R_2: {:<8.8f}".format(R_2))
# %%
df1
# %%
