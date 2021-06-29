#%%
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.random_projection import GaussianRandomProjection,SparseRandomProjection
from sklearn.cluster import OPTICS,cluster_optics_dbscan
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from pyproj import Transformer
from tqdm import tqdm
from keplergl import KeplerGl 
from flask import Flask
import geopandas as gpd 
from shapely.geometry import Polygon,LineString
from datetime import datetime, timedelta
from time import time 
#%%
df=pd.read_pickle(r'/home/liyunjia/data/AIS_data_process/trajectory_calculate_new/聚类算法0612/distancemat0611_month1_xgcourse1.pkl')
#%%
transformer = SparseRandomProjection(random_state=1,eps=0.3)#用随机投影对数据进行降维，该方法可以最大程度保持不同数据之间的距离
# clust = OPTICS(min_samples=30,n_jobs=-1,max_eps=1,cluster_method='xi',algorithm= 'kd_tree')#这个运行时间是500秒，也就是8分钟左右
clust = OPTICS(min_samples=30,n_jobs=-1,algorithm= 'kd_tree')#这个的运行时间是703秒
X_new = transformer.fit_transform(np.array(df))
#%%
X_new.shape
#%%

beagin_time=time()
clust.fit(np.array(df))
reachability = clust.reachability_[clust.ordering_]
end_time=time()
print('该程序运行时间',end_time-beagin_time)
#%%

#%%
plt.plot(range(0,len(reachability)), reachability,linewidth=1)
space = np.arange(len(X_new))
plt.plot(space, np.full_like(space, 0.35, dtype=float), 'k-.', alpha=0.35) 
#%%
plt.plot(range(0,len(reachability)-700), reachability[:len(reachability)-700],linewidth=1)
space = np.arange(len(X_new))
plt.plot(space, np.full_like(space, 0.35, dtype=float), 'k-.', alpha=0.35)
#%%
labels_information=pd.concat([pd.DataFrame([i for i in range(len(reachability))]),pd.DataFrame(clust.ordering_),pd.DataFrame(reachability)],axis=1)
labels_information.columns=['order','key','reachability']
# %%
labels_information
# %%
labels_035 = cluster_optics_dbscan(reachability=clust.reachability_,
                                   core_distances=clust.core_distances_,
                                   ordering=clust.ordering_, eps=0.35)
#%%
labels_035=pd.DataFrame(labels_035)
labels_035.columns=['type']
order=clust.ordering_
order=pd.DataFrame(order)
order['order']=order.index 
order=order.set_index(0)
labels_035=pd.concat([labels_035,order],axis=1)
#%%
labels_035=pd.DataFrame(labels_035)
#%%
labels_035['type'].value_counts()

# %%
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
# %%
mmsi_list=df['MMSI'].unique()
i=-1
data={}
for mmsi in tqdm(mmsi_list):
    df1=df[df['MMSI']==mmsi]
    para_list=df1['paragraph'].unique()
    for para in para_list:
        df2=df1[df1['paragraph']==para]
        if (len(df2)>20) and len(df2)<300:
            i+=1
            data[i]=df2

#%%
for i in tqdm(range(len(data))):
    data[i]['label']=labels_035['type'].iloc[i]
    data[i]['order']=labels_035['order'].iloc[i]
#%%
def gen_line_dataframe(df):
    line_list=[]
    for key in tqdm(range(len(df))):
        line_dict={}
        data=df[key]
        para_list=data['paragraph'].unique()
        for para in para_list:
            data1=data[data['paragraph']==para]
            lat_lon=data1[['Lon_d','Lat_d']]
            judge_up_down=lat_lon.iloc[:10].diff(1).mean()
            if judge_up_down['Lon_d']<0 and  judge_up_down['Lat_d']>0:
                 line_dict['up_down']=1#上行
            else:
                 line_dict['up_down']=0#下行
            line=LineString(np.array(lat_lon))
            line_dict['key']=key
            line_dict['order']=data1['order'].iloc[0]
            line_dict['label']=data1['label'].iloc[0]
            line_dict['geometry']=gpd.GeoSeries([line])[0]
            line_list.append(line_dict)
    return gpd.GeoDataFrame(line_list)
#%%
df1=gen_line_dataframe(data)
#%%
#%%
app = Flask(__name__)
map_1 = KeplerGl(height=500)
map_1.add_data(data=df1, name='polygon_predict')
@app.route('/')
def index():
    global map_1
    return map_1._repr_html_()
if __name__ == '__main__':
    app.run(debug=False)

#先去判断一下1和2这两个类什么区别。
#%%
labels_035
# %%
# 这段代码可以看出往上行走的轨迹和往下面行走的轨迹全都分类成功，没有失败的
#这段代码也可以去计算上行轨迹他们的key是什么，然后记住他们，并保存下来用于后续的机器学习训练。
# information_label=[]
# order_list=[]
# for i in range(len(labels_035)):
#     if labels_035.iloc[i]['type']==2:
#         data_try=data[i]
#         if data_try.iloc[0]['Lon_d']-data_try.iloc[-1]['Lon_d']>0:
#             information_label.append(1)
#         else:
#             information_label.append(0)

# pd.DataFrame(information_label).value_counts()
# %%
