import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
print("load success")

def analysisInfo(df, fileName, pred = None): 
    obs = df.shape[0]
    types = df.dtypes
    counts = df.apply(lambda x : x.count())
    nulls = df.apply(lambda x : x.isnull().sum())
    distincts = df.apply(lambda x : x.unique().shape[0])
    missing_ration = (df.isnull().sum() / obs) * 100
    skewness = df.skew()
    kurtosis = df.kurt()
    
    infoTLB = pd.concat([types, counts, nulls, distincts, missing_ration, skewness, kurtosis], axis=1)
    infoTLB.columns = ['types','counts','nulls', 'distincts', 'missing_ration', 'skewness', 'kurtosis']
    print(infoTLB)

map_head = {'N' : 1, 'NE' : 2, 'E' : 3, 'SE' : 4 , 'S' : 5, 'SW' : 6, 'W' : 7, 'NW' : 8 } 

def trimDirection(df) : 
    col = ['EntryHeading']
    df[col] = df[col].applymap(map_head.get)
    df[col] = df['EntryHeading'].apply(lambda x : int( x % 2))
    df[col] = df['Month'].apply(lambda x : int (x % 4))
    return df 

def trimHours(df):
    df['Hour'] = df['Hour'].apply(lambda x : int(x % 6))
    return df

def trimLocation(df):
    df['Longitude'] = df['Longitude'].apply(lambda x : round(x, 2))
    df['Latitude'] = df['Latitude'].apply(lambda x : round(x, 2))
    return df


def composeKey(lat, long, head, weekend, hour, month) : 
    key = "{:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}".format(lat, long, head, weekend, hour, month)
    return key


#training data => make category
trim_train_data = train_data.drop(['ExitStreetName', 'EntryStreetName', 'IntersectionId', 'Path', 'ExitHeading', 'City'], axis = 1)

trim_train_data = trimDirection(trim_train_data)
trim_train_data = trimHours(trim_train_data)
trim_train_data = trimLocation(trim_train_data)
print(trim_train_data.head(10))

trim_train_data = trim_train_data.groupby(['Latitude', 'Longitude', 'EntryHeading', 'Weekend', 'Hour', 'Month']).agg({'TotalTimeStopped_p20' : np.mean, 'TotalTimeStopped_p50' : np.mean, 'TotalTimeStopped_p80' : np.mean, 'DistanceToFirstStop_p20' : np.mean, 'DistanceToFirstStop_p50' : np.mean, 'DistanceToFirstStop_p80' : np.mean})

newColumns = ['TotalTimeStopped_p20', 'TotalTimeStopped_p50', 'TotalTimeStopped_p80','DistanceToFirstStop_p20', 'DistanceToFirstStop_p50', 'DistanceToFirstStop_p80'] 
trim_train_data.columns = newColumns 
trim_train_data = trim_train_data.reset_index()
trim_train_data['key'] = trim_train_data.apply(lambda row : composeKey(row['Latitude'], row['Longitude'], row['EntryHeading'], row['Weekend'], row['Hour'], row['Month']), axis = 1) 
print(trim_train_data.head(10))


train_dict = trim_train_data.set_index('key').T.to_dict('dict')


#test data => make category
trim_test_data = test_data[test_data.RowId < 1920335]
trim_test_data = trim_test_data.drop(['ExitStreetName', 'EntryStreetName', 'IntersectionId', 'Path', 'ExitHeading', 'City'], axis = 1)

trim_test_data = trimDirection(trim_test_data)
trim_test_data = trimHours(trim_test_data)
trim_test_data = trimLocation(trim_test_data)
trim_test_data['key'] = trim_test_data.apply(lambda row : composeKey(row['Latitude'], row['Longitude'], row['EntryHeading'], row['Weekend'], row['Hour'], row['Month']), axis = 1) 


def stop20(key) : 
    if(key in train_dict.keys()) : 
        return train_dict[key]['TotalTimeStopped_p20']
    return 0

def stop50(key) : 
    if(key in train_dict.keys()) : 
        return train_dict[key]['TotalTimeStopped_p50']
    return 0

def stop80(key) : 
    if(key in train_dict.keys()) : 
        return train_dict[key]['TotalTimeStopped_p80']
    return 0

def dist20(key) : 
    if(key in train_dict.keys()) : 
        return train_dict[key]['DistanceToFirstStop_p20']
    return 0

def dist50(key) : 
    if(key in train_dict.keys()) : 
        return train_dict[key]['DistanceToFirstStop_p50']
    return 0

def dist80(key) : 
    if(key in train_dict.keys()) : 
        return train_dict[key]['DistanceToFirstStop_p80']
    return 0

trim_test_data['TotalTimeStopped_p20'] = trim_test_data.apply(lambda row : stop20(row['key'] ), axis = 1) 
trim_test_data['TotalTimeStopped_p50'] = trim_test_data.apply(lambda row : stop50(row['key'] ), axis = 1) 
trim_test_data['TotalTimeStopped_p80'] = trim_test_data.apply(lambda row : stop80(row['key'] ), axis = 1) 
trim_test_data['DistanceToFirstStop_p20'] = trim_test_data.apply(lambda row : dist20(row['key'] ), axis = 1) 
trim_test_data['DistanceToFirstStop_p50'] = trim_test_data.apply(lambda row : dist50(row['key'] ), axis = 1) 
trim_test_data['DistanceToFirstStop_p80'] = trim_test_data.apply(lambda row : dist80(row['key'] ), axis = 1) 
print(trim_test_data.columns)



trim_test_data = trim_test_data.drop(['RowId', 'Latitude', 'Longitude', 'EntryHeading', 'Weekend', 'Hour', 'Month', 'key'], axis = 1)
print(trim_test_data.shape)
trim_test_data = trim_test_data.values.flatten()
print(trim_test_data.shape)
tmp = np.transpose(trim_test_data)
print(tmp.shape)

result = pd.DataFrame()
tid = [] 
for i in range(0, 1920335) : 
    for j in range(0, 6) : 
        tid.append(str(i) + "_" + str(j)) 
result['TargetId'] = tid 
result['Target'] = tmp

result.to_csv('result.csv', index=None)
