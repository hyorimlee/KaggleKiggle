import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

train = pd.read_csv('data\\train.csv')
test = pd.read_csv('data\\test.csv')

def checkDataState(data):
  print(data.shape)
  #print(data.shape)
  #print(data.head())
  print(data.info())
  #print(data['SalePrice'].describe())
  #corr = data.corr()
  #print(corr.shape); 
  #print(corr.iloc[:, 81])

def analysisInfo(df, fileName, pred=None): 
    obs = df.shape[0]
    types = df.dtypes
    counts = df.apply(lambda x: x.count())
    nulls = df.apply(lambda x: x.isnull().sum())
    distincts = df.apply(lambda x: x.unique().shape[0])
    missing_ration = (df.isnull().sum()/ obs) * 100
    skewness = df.skew()
    kurtosis = df.kurt() 

    if pred is None:
        cols = ['types', 'counts', 'distincts', 'nulls', 'missing ration', 'skewness', 'kurtosis']
        infoTLB = pd.concat([types, counts, distincts, nulls, missing_ration, skewness, kurtosis], axis = 1)

    else:
        corr = df.corr()[pred]
        infoTLB = pd.concat([types, counts, distincts, nulls, missing_ration, skewness, kurtosis, corr], axis = 1, sort=False)
        corr_col = 'corr '  + pred
        cols = ['types', 'counts', 'distincts', 'nulls', 'missing_ration', 'skewness', 'kurtosis', corr_col ]
    
    print('=> save ' + str(infoTLB.shape[0])) 

    infoTLB.columns = cols
    if(pred != None) : 
      infoTLB = infoTLB.sort_values(by=corr_col, ascending=False)
    fileName = fileName + '.csv'
    infoTLB.to_csv(fileName, index = True, header = True)
    return

def showCorrelation(data, xValue, yValue): 
  fig, ax = plt.subplots() 
  ax.scatter(x=data[xValue], y=data[yValue])
  plt.ylabel(yValue, fontsize = 13)
  plt.xlabel(xValue, fontsize = 13)
  plt.show()

def fillMissingData(data) :
  # fill missing value
  for col in ('PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2') : 
    data[col] = data[col].fillna('None')

  # fill LotFrontage as Mean value of City
  FrontMean = data.groupby('Neighborhood').mean()["LotFrontage"]
  for index, row in data.iterrows():
    if( np.isnan(row['LotFrontage']) ): 
      data.loc[index, 'LotFrontage'] = FrontMean[row['Neighborhood']]

  for col in ('GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    data[col] = data[col].fillna(0)

  data["MasVnrType"] = data["MasVnrType"].fillna("None")
  data["MasVnrArea"] = data["MasVnrArea"].fillna(0)

  zone = data.groupby(['Neighborhood'])['MSZoning'].agg(pd.Series.mode)
  print(zone)
  for index, row in data.iterrows(): 
    if(row['MSZoning'] not in ('FV', 'RL', 'RM', 'C', 'RH', 'RP')) : 
      data.loc[index, 'MSZoning'] = zone[row['Neighborhood']]
  
  data = data.drop(['Utilities'], axis=1)
  data["Functional"] = data["Functional"].fillna("Typ"); 

  for col in ('KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType', 'Electrical') : 
    print(col)
    data[col]  = data[col].fillna(data[col].mode()[0])
  return data

def preprocessingData(data) : 
  cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',  'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope', 'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')

  for c in cols:
      lbl = LabelEncoder() 
      lbl.fit(list(data[c].values)) 
      data[c] = lbl.transform(list(data[c].values))


def goRegression(data) : 
  for col in data.columns : 
    if col not in ('SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd') : 
      data = data.drop(col, axis = 1)
  return data 


if __name__ == '__main__'  : 
  #showCorrelation(data, 'OverallQual', 'SalePrice')
  #showCorrelation(data, 'GrLivArea', 'SalePrice')

  print("train data : " + str(train.shape[0]))
  print("test data : " + str(test.shape[0])) 
  size = test.shape[0]

  train = fillMissingData(train)
  test = fillMissingData(test)

  preprocessingData(train)
  preprocessingData(test)

  train = goRegression(train)
  test = goRegression(test)

  analysisInfo(train, 'trainAnalysis', 'SalePrice')
  analysisInfo(test, 'testAnalysis')

  mlr = LinearRegression() 
  print(train.shape)
  print(train.head(5))
  x_train = train.iloc[:, :10]
  y_train = train.iloc[:, 10]

  mlr.fit(x_train, y_train)
  x_test = test.iloc[:, :]
  predictValue = mlr.predict(x_test)
  print(predictValue)

  #res = [[0] * 2] * size
  #for i in range(0, size, 1) : 
  #  print(i)
  #  res[i][0] = i + 1460
  #  res[i][1] = predictValue[i]

  resinpd = pd.DataFrame(predictValue) 
  resinpd.to_csv('result.csv', header=None, index=None)

