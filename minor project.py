#!/usr/bin/env python
# coding: utf-8

# In[3]:



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings


# In[4]:


warnings.filterwarnings('ignore')


# In[5]:



data = pd.read_csv('big_mart_Train.csv')
data.head(1)


# In[6]:



data.sample(5)


# In[7]:



data.shape


# In[8]:


data.describe()


# In[9]:


data.isnull().sum()


# In[10]:



per = data.isnull().sum() * 100 / len(data)
print(per)


# In[11]:


data.duplicated().any()


# In[12]:


data['Item_Weight']


# In[13]:


data['Outlet_Size']


# In[14]:


mean_weight = data['Item_Weight'].mean()
median_weight = data['Item_Weight'].median()


# In[15]:



print(mean_weight,median_weight)


# In[16]:



data['Item_Weight_mean']=data['Item_Weight'].fillna(mean_weight)
data['Item_Weight_median']=data['Item_Weight'].fillna(median_weight)


# In[17]:


data.head(1)


# In[18]:


print("Original Weight variable variance",data['Item_Weight'].var())
print("Item Weight variance after mean imputation",data['Item_Weight_mean'].var())
print("Item Weight variance after median imputation",data['Item_Weight_median'].var())


# In[19]:


data['Item_Weight'].plot(kind = "kde",label="Original")

data['Item_Weight_mean'].plot(kind = "kde",label = "Mean")

data['Item_Weight_median'].plot(kind = "kde",label = "Median")

plt.legend()
plt.show()


# In[20]:


data[['Item_Weight','Item_Weight_mean','Item_Weight_median']].boxplot()


# In[21]:


data['Item_Weight_interploate']=data['Item_Weight'].interpolate(method="linear")


# In[22]:



data['Item_Weight'].plot(kind = "kde",label="Original")

data['Item_Weight_interploate'].plot(kind = "kde",label = "interploate")

plt.legend()
plt.show()


# In[23]:


from sklearn.impute import KNNImputer


# In[24]:


knn = KNNImputer(n_neighbors=10,weights="distance")


# In[25]:


data['knn_imputer']= knn.fit_transform(data[['Item_Weight']]).ravel()


# In[26]:



data['Item_Weight'].plot(kind = "kde",label="Original")

data['knn_imputer'].plot(kind = "kde",label = "KNN imputer")

plt.legend()
plt.show()


# In[27]:


data = data.drop(['Item_Weight','Item_Weight_mean','Item_Weight_median','knn_imputer'],axis=1)


# In[28]:



data.head(1)


# In[29]:



data.isnull().sum()


# In[30]:


data['Outlet_Size'].value_counts()


# In[31]:


data['Outlet_Type'].value_counts()


# In[32]:


mode_outlet = data.pivot_table(values='Outlet_Size',columns='Outlet_Type',aggfunc=(lambda x:x.mode()[0]))


# In[33]:


mode_outlet


# In[34]:


missing_values = data['Outlet_Size'].isnull()


# In[35]:


missing_values


# In[36]:


data.loc[missing_values,'Outlet_Size'] = data.loc[missing_values,'Outlet_Type'].apply(lambda x :mode_outlet[x])


# In[37]:



data.isnull().sum()


# In[38]:


data.columns


# In[39]:


data['Item_Fat_Content'].value_counts()


# In[40]:


data.replace({'Item_Fat_Content':{'Low Fat':'LF','low fat':'LF','reg':'Regular'}},inplace=True)


# In[41]:


data['Item_Fat_Content'].value_counts()


# In[42]:


data.columns


# In[43]:


data['Item_Visibility'].value_counts()


# In[44]:


data['Item_Visibility_interpolate']=data['Item_Visibility'].replace(0,np.nan).interpolate(method='linear')


# In[45]:


data.head(1)


# In[46]:


data['Item_Visibility_interpolate'].value_counts()


# In[47]:


data['Item_Visibility'].plot(kind="kde",label="Original")

data['Item_Visibility_interpolate'].plot(kind="kde",color='red',label="Interpolate")

plt.legend()
plt.show()


# In[48]:



data = data.drop('Item_Visibility',axis=1)


# In[49]:


data.head(1)


# In[50]:


data.columns


# In[51]:


data['Item_Type'].value_counts()


# In[52]:


data.columns


# In[53]:


data['Item_Identifier'].value_counts().sample(5)


# In[54]:


data['Item_Identifier'] =data['Item_Identifier'].apply(lambda x : x[:2])


# In[55]:


data['Item_Identifier'].value_counts()


# In[56]:


data.columns


# In[57]:


data['Outlet_Establishment_Year']


# In[58]:



import datetime as dt


# In[59]:



current_year = dt.datetime.today().year


# In[60]:


current_year


# In[61]:



data['Outlet_age']= current_year - data['Outlet_Establishment_Year']


# In[62]:


data.head(1)


# In[63]:


data = data.drop('Outlet_Establishment_Year',axis=1)


# In[64]:



data.head()


# In[65]:



from sklearn.preprocessing import OrdinalEncoder

data_encoded = data.copy()

cat_cols = data.select_dtypes(include=['object']).columns

for col in cat_cols:
    oe = OrdinalEncoder()
    data_encoded[col]=oe.fit_transform(data_encoded[[col]])
    print(oe.categories_)


# In[66]:



data_encoded.head(3)


# In[67]:



X = data_encoded.drop('Item_Outlet_Sales',axis=1)
y = data_encoded['Item_Outlet_Sales']


# In[68]:



y


# In[69]:



from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score

rf = RandomForestRegressor(n_estimators=100,random_state=42)
scores = cross_val_score(rf,X,y,cv=5,scoring='r2')
print(scores.mean())


# In[73]:


get_ipython().system('pip install xgboost')

from xgboost import XGBRFRegressor

xg = XGBRFRegressor(n_estimators=100,random_state=42)
scores = cross_val_score(xg,X,y,cv=5,scoring='r2')
print(scores.mean())


# In[74]:



xg = XGBRFRegressor(n_estimators=100,random_state=42)

xg1 = xg.fit(X,y)
pd.DataFrame({
    'feature':X.columns,
    'XGBRF_importance':xg1.feature_importances_
    
}).sort_values(by='XGBRF_importance',ascending=False)


# In[75]:



['Item_Visibility_interpolate','Item_Weight_interploate',
'Item_Type','Outlet_Location_Type','Item_Identifier','Item_Fat_Content']


# In[76]:



from xgboost import XGBRFRegressor

xg = XGBRFRegressor(n_estimators=100,random_state=42)
scores = cross_val_score(xg1,X.drop(['Item_Visibility_interpolate','Item_Weight_interploate',
'Item_Type','Outlet_Location_Type','Item_Identifier','Item_Fat_Content'],axis=1),y,cv=5,scoring='r2')
print(scores.mean())


# In[78]:



final_data = X.drop(columns=['Item_Visibility_interpolate','Item_Weight_interploate',
'Item_Type','Outlet_Location_Type','Item_Identifier','Item_Fat_Content'],axis=1)


# In[79]:


final_data


# In[80]:


from xgboost import XGBRFRegressor


# In[81]:



xg_final = XGBRFRegressor()


# In[82]:


xg_final.fit(final_data,y)


# In[83]:



from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# In[84]:


X_train,X_test,y_train,y_test = train_test_split(final_data,y,
                                                 test_size=0.20,
                                                 random_state=42)


# In[85]:



xg_final.fit(X_train,y_train)


# In[86]:


y_pred = xg_final.predict(X_test)


# In[87]:



mean_absolute_error(y_test,y_pred)


# In[95]:


pred = xg_final.predict(np.array([[141.6180,9.0,1.0,1.0,24]]))[0]
print(pred)


# In[89]:


print(f"Sales Value is between {pred-714.42} and {pred+714.42}")


# In[90]:



import joblib


# In[91]:


joblib.dump(xg_final,'bigmart_model')


# In[93]:


model = joblib.load('bigmart_model')


# In[94]:


pred = model.predict(np.array([[141.6180,9.0,1.0,1.0,24]]))[0]
print(pred)


# In[96]:


print(f"Sales Value is between {pred-714.42} and {pred+714.42}")


# In[ ]:



import numpy as np
import datetime as dt
from tkinter import *
import joblib
current_year = dt.datetime.today().year
def show_entry_fields():
    p1=float(e1.get())
    #p4=float(e4.get())
    
    text = clicked.get()
    if text == "OUT010":
        p2=0
        print(p2)
    elif text=="OUT013":
        p2=1
        print(p2)
    elif text=="OUT017":
        p2=2
        print(p2)
    elif text=="OUT018":
        p2=3
        print(p2)
    elif text=="OUT019":
        p2=4
        print(p2)
    elif text=="OUT027":
        p2=5
        print(p2)
    elif text=="OUT035":
        p2=6
        print(p2)
    elif text=="OUT045":
        p2=7
        print(p2)
    elif text=="OUT046":
        p2=8
        print(p2)
    elif text=="OUT049":
        p2=9
        print(p2)
    text0 = clicked0.get()
    if text0 == "High":
        p3=0
        print(p3)
    elif text0=="Medium":
        p3=1
        print(p3)
    elif text0=="Small":
        p3=2
        print(p3)
        
    text1 = clicked1.get()
    if text1 == "Supermarket Type1":
        p4=1
        print(p4)
    elif text1=="Supermarket Type2":
        p4=2
        print(p4)
    elif text1=="Supermarket Type3":
        p4=3
        print(p4)
    elif text1=="Grocery Store":
        p4=0
        print(p4)
    
    p5=current_year - int(e5.get())
    print(p5)
    
    model = joblib.load('bigmart_model')
    result=model.predict(np.array([[p1,p2,p3,p4,p5]]))
    Label(master, text="Sales Amount is in between").grid(row=8)
    Label(master, text=float(result) -714.42 ).grid(row=10)
    Label(master, text="and").grid(row=11)
    Label(master, text=float(result) + 714.42) .grid(row=12)
    print("Sales amount", result)
    
master = Tk()
master.title("Big Mart Sales Prediction using Machine Learning")


label = Label(master, text = " Big Mart Sales Prediction using ML"
                          , bg = "black", fg = "white"). \
                               grid(row=0,columnspan=2)

# Item_MRP	Outlet_Identifier	Outlet_Size	Outlet_Type	Outlet_age
Label(master, text="Item_MRP").grid(row=1)
Label(master, text="Outlet_Identifier").grid(row=2)
Label(master, text="Outlet_Size").grid(row=3)
Label(master, text="Outlet_Type").grid(row=4)
Label(master, text="Outlet_Establishment_Year").grid(row=5)


clicked = StringVar()
options = ['OUT010', 'OUT013', 'OUT017', 'OUT018', 'OUT019', 'OUT027',
       'OUT035', 'OUT045', 'OUT046', 'OUT049']

clicked0 = StringVar()

options0 = ['High', 'Medium', 'Small']

clicked1 = StringVar()
options1 = ['Grocery Store', 'Supermarket Type1', 'Supermarket Type2',
       'Supermarket Type3']

e1 = Entry(master)

e2 = OptionMenu(master , clicked , *options )
e2.configure(width=15)


e3 = OptionMenu(master , clicked0 , *options0 )
e3.configure(width=15)


e4 = OptionMenu(master , clicked1 , *options1 )
e4.configure(width=15)

e5 = Entry(master)


e1.grid(row=1, column=1)
e2.grid(row=2, column=1)
e3.grid(row=3, column=1)
e4.grid(row=4, column=1)
e5.grid(row=5, column=1)



Button(master, text='Predict', command=show_entry_fields).grid()

mainloop()


# In[ ]:




