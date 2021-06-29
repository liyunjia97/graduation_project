'''
该代码集成了坐标变换、异常剔除、距离矩阵生成、kmeans聚类以及TSNE可视化显示
重新计算fastDTW和DTW时间的对比来进行查看FastDTW是不是可以降低为线性时间
'''
#%%
import pandas as pd
import numpy as np
from pyproj import Transformer
import matplotlib.pyplot as plt 
import math 
import seaborn as sns 
from datetime import datetime 
from keras.callbacks import TensorBoard 
from keras.layers import Dense,Dropout
from keras.layers import LSTM
import os
from keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from tensorflow.python.keras.callbacks import BackupAndRestore
from tqdm import tqdm 
import random
import os
from itertools import combinations
from joblib import Parallel, delayed
import multiprocessing as mp
from dtw import dtw,accelerated_dtw
from scipy.spatial.distance import cdist
import logging
import time,sys
from logging.handlers import TimedRotatingFileHandler 
import traceback
from rdp import rdp 
import time,sys
from scipy.spatial.distance import euclidean
pd.set_option('display.max_columns',None)
begin_time=time.time()
#%%

log = logging.getLogger('yyx')
log.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
log_file_handler = TimedRotatingFileHandler(filename="log", when="H", interval=1, backupCount=7)
log_file_handler.setFormatter(formatter)
log_file_handler.setLevel(logging.INFO)
log.addHandler(log_file_handler)
log_file_handler.suffix = "%Y-%m-%d_%H-%M-%S.log"
log.addHandler(log_file_handler) 
log.info('程序已运行')
df=pd.read_csv(r'/home/liyunjia/data/AIS_data_process/trajectory_calculate_new/trajectory_02.csv')
BJS_format = "%Y-%m-%d %H:%M:%S"
df['Receivedtime（UTC+8）'] = df['Receivedtime（UTC+8）'].apply(lambda x: datetime.strptime(x, BJS_format))

lon_lat=df[['Lon_d','Lat_d']]
transformer = Transformer.from_crs("epsg:4326", "epsg:3857")
Lat1, Lon1 = transformer.transform(lon_lat['Lat_d'], lon_lat['Lon_d'])
Lon_lat1=pd.concat([pd.DataFrame(Lat1),pd.DataFrame(Lon1)],axis=1)  
Lon_lat1.columns=['Lat_d1','Lon_d1'] 
df=pd.concat([df,Lon_lat1],axis=1)
#%%
scaler = StandardScaler()
scaler_Lat_Lon = pd.DataFrame(scaler.fit_transform(pd.concat([pd.DataFrame(df['Lat_d1']),pd.DataFrame(df['Lon_d1'])],axis=1)))
scaler_Lat_Lon.columns=['Lat','Lon']
Lon_lat2=pd.concat([pd.DataFrame(scaler_Lat_Lon['Lat']),pd.DataFrame(scaler_Lat_Lon['Lon'])],axis=1)  
Lon_lat2.columns=['Lat_d1_scaler','Lon_d1_scaler'] 
df=pd.concat([df,Lon_lat2],axis=1)
#%%
#将航向角进行转换
mmsi_list=df['MMSI'].unique()
i=0
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
def anomaly_process_course(data):
    len_data=len(data)
    i=1 
    while i <len_data:      
        j=i-1
        while math.isnan(data.loc[data.iloc[j].name,'Course']):
            j=j-1
        course=(data.iloc[i]['Course']-data.iloc[j]['Course'])    
        num_nan=0
        m=i-1#m是为了判断之前已经删除了几个错误的数据了，如果删除过多，则不再删除新的数据
        while math.isnan(data.loc[data.iloc[m].name,'Course']):
            num_nan+=1 
            m-=1 
        if np.cos(course/180 *np.pi)<0.55 and num_nan<5: #关键的是在于这个阈值怎么确定
            data.loc[data.iloc[i].name,'Course'] =np.nan
        i+=1 
    data['Course']= data['Course'].interpolate()
    return data
def anomaly_process_speed(data):
    data['delta_time'].iloc[0]=0 
    len_data=len(data)
    i=1 
    while i <len_data:      
        j=i-1
        while math.isnan(data.loc[data.iloc[j].name,'Speed']):
            j=j-1
        Speed_diff=(data.iloc[i]['Speed']-data.iloc[j]['Speed'])    
        judge=Speed_diff/(data.iloc[i]['Receivedtime（UTC+8）']-data.iloc[j]['Receivedtime（UTC+8）']).total_seconds()
        if abs(judge)>0.04: #关键点是在于这个阈值怎么确定
            data.loc[data.iloc[i].name,'Speed'] =np.nan
        i+=1 
    data['Speed']= data['Speed'].interpolate()
    return data
#向量夹角异常剔除法
def angle(v1, v2):
    dx1 = v1[2] - v1[0]
    dy1 = v1[3] - v1[1]
    dx2 = v2[2] - v2[0]
    dy2 = v2[3] - v2[1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180/math.pi)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180/math.pi)
    if angle1*angle2 >= 0:
        included_angle = abs(angle1-angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
    if included_angle > 180:
        included_angle = 360 - included_angle
    return included_angle  
def trajectory_process(data):
    if data['Speed'].mean()<1.5:
        return data 
    len_data=len(data)
    i=5 #i是指的现在判断的点
    while i <len_data: 
        j=i-1#j是指的i现在点之前的一个点
        k=j-1 #k是指的j这个点之前的点
        while i>5 and (math.isnan(data.loc[data.iloc[j].name,'Lon_d1']) or data.loc[data.iloc[j].name,'Lon_d1']==data.loc[data.iloc[i].name,'Lon_d1'] ):
            j=j-1
            k=j-1
        while i>5 and (math.isnan(data.loc[data.iloc[k].name,'Lon_d1']) or data.loc[data.iloc[j].name,'Lon_d1']==data.loc[data.iloc[k].name,'Lon_d1']) :
            k=k-1
        a=list(data[['Lon_d1','Lat_d1']].iloc[j])+list(data[['Lon_d1','Lat_d1']].iloc[i])
        b=list(data[['Lon_d1','Lat_d1']].iloc[k])+list(data[['Lon_d1','Lat_d1']].iloc[j])#如果这个b是错误的就麻烦很大
        if i==5:
            b=list(data[['Lon_d1','Lat_d1']].iloc[0])+list(data[['Lon_d1','Lat_d1']].iloc[j])
        anglei=angle(a,b) 
        m=i-1 #m是为了判断一共有i之前一共有几个点是nan值，如果nan值过多则意味着判断出错，即停止判断删除
        num_nan=0
        while math.isnan(data.loc[data.iloc[m].name,'Lon_d1']):
            num_nan+=1 
            m-=1 
        if anglei>30 and num_nan<4:#anglei和num_nan是异常值判断的情况
            data.loc[data.iloc[i].name,'Lon_d1'] =np.nan
            data.loc[data.iloc[i].name,'Lat_d1'] =np.nan
            data.loc[data.iloc[i].name,'Lon_d'] =np.nan
            data.loc[data.iloc[i].name,'Lat_d'] =np.nan
        i+=1 
    data['Lon_d1']= data['Lon_d1'].interpolate()
    data['Lat_d1']= data['Lat_d1'].interpolate()
    data['Lon_d']= data['Lon_d'].interpolate()
    data['Lat_d']= data['Lat_d'].interpolate()
    return data

#下面这个循环是用来做异常检测和航向角处理以及轨迹约简
for i in tqdm(data.keys()):
    data[i]=anomaly_process_course(data[i])
    data[i]=anomaly_process_speed(data[i])
    data[i]=trajectory_process(data[i])
    data[i]['sin_course']=np.sin((data[i]['Course']/180)*np.pi)
    data[i]['cos_course']=np.cos((data[i]['Course']/180)*np.pi)
    lon_lat=data[i][['Lon_d1','Lat_d1']]
    line_simplify=rdp(lon_lat, epsilon=5, return_mask=True)
    data[i]=data[i][line_simplify]

#计算轨迹的距离 这个再去看一看是否可以用fast-DTW来代替
def trajectory_distance(ptSetA,ptSetB):
    x=np.array(ptSetA)
    y=np.array(ptSetB) 
    d, cost_matrix, acc_cost_matrix, path = accelerated_dtw(x, y, dist='euclidean')
    return d 
#%%
def DistanceMat(data,w=[0.7,0.3]):
    '''
    功能：计算轨迹段的距离矩阵
    输出：距离矩阵
    '''
    #要计算的组合
    try:
        ptCom = list(combinations(list(data.keys()),2)) #combinations('ABCD', 2) --> AB AC AD BC BD CD。data.key()即是为了获得data的索引。然后进行互相匹配计算距离.每一个key代表了
        #基于轨迹的距离
        distance_tra = Parallel(n_jobs=mp.cpu_count(),verbose=False)(delayed(trajectory_distance)(
            data[ptSet1][['Lon_d1_scaler','Lat_d1_scaler']],data[ptSet2][['Lon_d1_scaler','Lat_d1_scaler']]#该程序核心思想是计算不同索引轨迹之间的距离，然后使用多核进行计算轨迹之间的距离
            ) for ptSet1,ptSet2 in ptCom)
        distancemat_tra = pd.DataFrame(ptCom)#建立轨迹距离矩阵
        distancemat_tra['distance'] = distance_tra #建立distance这一列
        distancemat_tra = distancemat_tra.pivot(index=0,columns=1,values='distance')#通过长表进行变宽表，也就是原来的0列作为index，原来的1列作为columns，然后value是distance。假设原来的distancemat_tra是n*3,所以现在的是n*n
        for pt1 in data.keys():
            distancemat_tra.loc[pt1,pt1] = 0#这个是将对角的索引和列名相等的元素均化为零 原来代码是distancemat_tra.loc[pt1,pt1]，是否要在pt1外面套一个str(pt1),取决于pt1的类型，需要和原有的类型一直
        distancemat_tra = distancemat_tra.fillna(0)#这个是将NaN所对应的未知均填充为零
        distancemat_tra = distancemat_tra.loc[list(data.keys()),list(data.keys())]#这个是按照原来的keys进行排序
        distancemat_tra = distancemat_tra+distancemat_tra.T#然后两个相加就得到了每两个轨迹线段的距离矩阵
        #基于方向的距离
        distance_direction = Parallel(n_jobs=mp.cpu_count(),verbose=False)(delayed(trajectory_distance)(
                data[ptSet1][['sin_course','cos_course']],data[ptSet2][['sin_course','cos_course']]
                ) for ptSet1,ptSet2 in ptCom)
        distancemat_direction = pd.DataFrame(ptCom)
        distancemat_direction['Course'] = distance_direction 
        distancemat_direction = distancemat_direction.pivot(index=0,columns=1,values='Course')
        for pt1 in data.keys():
            distancemat_direction.loc[pt1,pt1] = 0
        distancemat_direction = distancemat_direction.fillna(0)
        distancemat_direction = distancemat_direction.loc[list(data.keys()),list(data.keys())]
        distancemat_direction = distancemat_direction+distancemat_direction.T
        distancemat_tra = (distancemat_tra-distancemat_tra.min().min())/(distancemat_tra.max().max()-distancemat_tra.min().min()) #第一个min函数是对所有的列取其最小值。然后第二个列是对取其最小值出来的列再取其最小值
        distancemat_direction = (distancemat_direction-distancemat_direction.min().min())/(distancemat_direction.max().max()-distancemat_direction.min().min())
        distancemat = w[0]*distancemat_tra+w[1]*distancemat_direction 
    except Exception as e:
        log.info(traceback.format_exc())
    return distancemat
distancemat = DistanceMat(data,w=[0.7,0.3])

distancemat.to_pickle('distancemat0622_month2_xgcourse1.pkl')
end_time=time.time()
print(end_time-begin_time)
# #%%
# distancemat=pd.read_pickle(r'/home/liyunjia/data/AIS_data_process/trajectory_calculate_new/聚类算法书写/distancemat0602_month1.pkl')
# #%%
# #%%
# #%%
# # 尝试去做kmeans聚类
# class KMeans:
#   def __init__(self,n_clusters=5,Q=74018,max_iter=150):
#      self.n_clusters = n_clusters #聚类数
#      self.Q = Q
#      self.max_iter = max_iter  # 最大迭代数
     
#   def fit(self,distancemat):
#      #选择初始中心
#      best_c = random.sample(distancemat.columns.tolist(),1)   #从生成的距离矩阵里面随机选取一条轨迹
#      for i in range(self.n_clusters-1):
#        best_c += random.sample(distancemat.loc[(distancemat[best_c[-1]]>self.Q)&(~distancemat.index.isin(best_c))].index.tolist(),1) #然后再从这个里面选取其他的轨迹
#      center_init = distancemat[best_c] #选择最小的样本组合为初始质心
#      self._init_center = center_init
#      #迭代停止条件
#      iter_ = 0
#      run = True
#      #开始迭代
#      while (iter_<self.max_iter)&(run==True):
#        #聚类聚类标签更新
#        labels_ = np.argmin(center_init.values,axis=1)#这个是判断每条轨迹距离所选择5条中心轨迹的距离以选择对应的lebel
#        #聚类中心更新
#        best_c_ = [distancemat.iloc[labels_== i,labels_==i].sum().idxmin() for i in range(self.n_clusters)] #该代码是在每个簇里面找一个距离所有点最近的点作为质点。
#        center_init_ = distancemat[best_c_]
#        #停止条件
#        iter_ += 1
#        if best_c_ == best_c:
#           run = False
#        center_init = center_init_.copy()
#        best_c = best_c_.copy()
#      #记录数据
#      self.labels_ = np.argmin(center_init.values,axis=1)
#      self.center_tra = center_init.columns.values
#      self.num_iter = iter_
#      self.sse = sum([sum(center_init.iloc[self.labels_==i,i]) for i in range(self.n_clusters)])

# #%%
# #  聚类，保存不同的sse
# SSE = []
# for i in range(1,30):
#   kmeans = KMeans(n_clusters=i,Q=0.01,max_iter=150)
#   kmeans.fit(distancemat)
#   SSE.append(kmeans.sse)
# # #画图
# #%%
# plt.figure(0)
# plt.plot([i for i in range(1,30)],SSE)
# plt.scatter([i for i in range(1,30)],SSE)
# plt.show()
# # %%
# #使用最好的结果进行聚类
# n_clusters=4
# kmeans = KMeans(n_clusters=n_clusters,Q=0.01,max_iter=150)
# kmeans.fit(distancemat)
# kmeans.sse  #输出sse
# kmeans.labels_  #输出标签
# kmeans.center_tra  #输出聚类中心
# # %%
# plt.plot(data[463]['Lon_d'],data[463]['Lat_d'])
# plt.plot(data[623]['Lon_d'],data[623]['Lat_d'])
# plt.plot(data[324]['Lon_d'],data[324]['Lat_d'])

# # %%
# plt.plot(data[4]['Lon_d'],data[4]['Lat_d'])
# # %%
# plt.figure(1)
# for i in range(n_clusters):
#    for name in distancemat.columns[kmeans.labels_==i]:
#         plt.plot(data[name].loc[:,'Lon_d'],data[name].loc[:,'Lat_d'],c=sns.xkcd_rgb[list(sns.xkcd_rgb.keys())[i]])
# plt.show()
# # %%
# for name in distancemat.columns[kmeans.labels_==0]:
#     plt.plot(data[name].loc[:,'Lon_d'],data[name].loc[:,'Lat_d'],c=sns.xkcd_rgb[list(sns.xkcd_rgb.keys())[i]])
# plt.show()
# # %%
# from MulticoreTSNE import MulticoreTSNE as TSNE
# from sklearn.preprocessing import LabelEncoder
# import matplotlib.pyplot as plt 
# import multiprocessing as mp 
# from mpl_toolkits.mplot3d import Axes3D 
# #%%
# embeddings = TSNE(n_jobs=mp.cpu_count(),n_components=2).fit_transform(distancemat)
# vis_x = embeddings[:, 0]
# vis_y = embeddings[:, 1]
# list_=[]
# for i in kmeans.labels_:
#   if i==0:
#     list_.append(0)
#   elif i==1:
#     list_.append(3)
#   elif i==2:
#     list_.append(7)
#   elif i==3:
#     list_.append(9)

# plt.scatter(vis_x, vis_y,c=list_, marker='.')
# # plt.colorbar(ticks=range(10))
# plt.clim(-0.5, 9.5)
# plt.show()
# # # %%
# # # 从kmeans的聚类效果来看，聚类为3个类别是最适合的
