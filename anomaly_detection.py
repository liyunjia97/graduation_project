import math 
import numpy as np 

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
