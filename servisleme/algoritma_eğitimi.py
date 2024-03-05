#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data = pd.read_csv('önişlemesonrası_data.csv')


# In[3]:


data.head()


# In[4]:


data = data[["tipi","net_metrekare","oda_sayısı","bina_yaşı","dairenin_katı","takas","ısıtma","yapı_durumu","yapı_tipi","site_içerisinde",
         "eşya_durumu","banyo_sayısı","wc_sayısı","fiyat"]]


# In[5]:


data.head()


# In[6]:


from sklearn.model_selection import learning_curve, RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


# In[7]:


#Train Test Bölümlemeleri Oluşturalım
from sklearn.model_selection import train_test_split
X = data.drop(['fiyat'], axis=1)
y = data['fiyat']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0 )

print('x_train :', x_train.shape)
print('x_test :', x_test.shape)
print('y_train :', y_train.shape)
print('y_test :', y_test.shape)


# In[8]:


#Normalize etme
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
x_train= scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)


# In[9]:


# Lineer Regresyon 
from sklearn.linear_model import LinearRegression, Lasso

lr = LinearRegression()
lr.fit(x_train,y_train)
y_predict = lr.predict(x_test) # üretilen modeli (lr) test verisi ile deneyelim

print('mean absolute error: ',mean_absolute_error(y_test, y_predict))
print('root mean squared error: ',np.sqrt(mean_squared_error(y_test,y_predict)))
print('Coefficient of determination R^2: ',r2_score(y_test,y_predict))


# In[10]:


from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(criterion='mse',splitter='best')
dt.fit(x_train,y_train)
y_predict_dt = dt.predict(x_test)

print('mean absolute error: ',mean_absolute_error(y_test, y_predict_dt))
print(' root mean squared error: ',np.sqrt(mean_squared_error(y_test,y_predict_dt)))
print('Coefficient of determination R^2: ',r2_score(y_test,y_predict_dt))


# In[11]:


lasso = Lasso(alpha=0.001)
lasso.fit(x_train,y_train)
y_predict_lasso = lasso.predict(x_test) 

print('mean absolute error: ',mean_absolute_error(y_test, y_predict_lasso))
print(' root mean squared error: ',np.sqrt(mean_squared_error(y_test,y_predict_lasso)))
print('Coefficient of determination R^2: ',r2_score(y_test,y_predict_lasso))


# In[12]:


from sklearn.ensemble import RandomForestRegressor 

rf = RandomForestRegressor()
rf.fit(x_train,y_train)
y_predict_rf = rf.predict(x_test)

print('mean absolute error: ',mean_absolute_error(y_test, y_predict_rf))
print('root mean squared error: ',np.sqrt(mean_squared_error(y_test,y_predict_rf)))
print('Coefficient of determination R^2: ',r2_score(y_test,y_predict_rf))


# In[13]:


from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Bağımsız ve bağımlı değişkenleri belirleme
X = data.drop(['fiyat'], axis=1) 
y = data['fiyat']

# Veri setini eğitim ve test olarak bölmek
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 #ElasticNet Regresyon modeli
elasticnet_model = ElasticNet()
elasticnet_model.fit(X_train, y_train)
elasticnet_predictions = elasticnet_model.predict(X_test)

# Metrikleri hesaplama
elasticnet_rmse = mean_squared_error(y_test, elasticnet_predictions, squared=False)
elasticnet_mae = mean_absolute_error(y_test, elasticnet_predictions)
elasticnet_r2 = r2_score(y_test, elasticnet_predictions)

# Sonuçları yazdırma
print("ElasticNet Regresyon:")
print("RMSE:", elasticnet_rmse)
print("MAE:", elasticnet_mae)
print("R²:", elasticnet_r2)


# In[14]:


from sklearn.ensemble import GradientBoostingRegressor

# Gradient Boosting Regresyon modeli
gb_model = GradientBoostingRegressor()
gb_model.fit(X_train, y_train)
gb_predictions = gb_model.predict(X_test)

# Metrikleri hesaplama
gb_rmse = mean_squared_error(y_test, gb_predictions, squared=False)
gb_mae = mean_absolute_error(y_test, gb_predictions)
gb_r2 = r2_score(y_test, gb_predictions)

# Sonuçları yazdırma
print("Gradient Boosting Regresyon:")
print("RMSE:", gb_rmse)
print("MAE:", gb_mae)
print("R²:", gb_r2)


# In[15]:


from sklearn.linear_model import BayesianRidge

# Bayesian Ridge Regresyon modeli
bayesianridge_model = BayesianRidge()
bayesianridge_model.fit(X_train, y_train)
bayesianridge_predictions = bayesianridge_model.predict(X_test)

# Metrikleri hesaplama
bayesianridge_rmse = mean_squared_error(y_test, bayesianridge_predictions, squared=False)
bayesianridge_mae = mean_absolute_error(y_test, bayesianridge_predictions)
bayesianridge_r2 = r2_score(y_test, bayesianridge_predictions)

# Sonuçları yazdırma
print("Bayesian Ridge Regresyon:")
print("RMSE:", bayesianridge_rmse)
print("MAE:", bayesianridge_mae)
print("R²:", bayesianridge_r2)


# In[16]:


from sklearn.svm import SVR
from sklearn.datasets import make_regression
import math

# SVR modeli 
svr = SVR(kernel='rbf') 
svr.fit(X_train, y_train)

# Modeli kullanarak tahmin yapma
y_pred = svr.predict(X_test)

# Model performansını değerlendirme
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = math.sqrt(mse)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
print(f"Root Mean Squared Error: {rmse}")


# In[17]:


#R^2 değeri 1 e daha yakın
#root mean squared error (RMSE) değeri 0 a daha yakın daha küçük


# In[18]:


#Fine Tuning Grid Search
from sklearn.model_selection import GridSearchCV
# tüm parametreler için değil sadece iki temel parametre için arama yapıyoruz!!!
params_grid = {
    'n_estimators':[10,20,50],
    'max_leaf_nodes':list(range(0,5))}


grid_search = GridSearchCV(RandomForestRegressor(min_samples_split=2,bootstrap=False,random_state=42), params_grid, verbose=1, cv=3)

grid_search.fit(x_train, y_train)


# In[19]:


grid_search.best_params_


# In[20]:


# bulduğumuz parametrelerle test edelim
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
rf = RandomForestRegressor(n_estimators=50,max_leaf_nodes=2)
rf.fit(x_train,y_train)
y_predict_rf = rf.predict(x_test)
print('mean absolute error: ',mean_absolute_error(y_test, y_predict_rf))
print('root mean squared error: ',np.sqrt(mean_squared_error(y_test,y_predict_rf)))
print('Coefficient of determination R^2: ',r2_score(y_test,y_predict_rf))


# In[21]:


import joblib
joblib.dump(rf, "../makine_öğrenmesi/servisleme/random_forest_model.pkl")


# In[ ]:




