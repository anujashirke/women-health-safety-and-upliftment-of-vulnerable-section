import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt

# ## LOADING FILE
df=pd.read_csv("Strokesdataset.csv")



df1=df.copy()




df1["age"]=df1.age.apply(np.ceil).astype(int)
mean=df.bmi.mean()

df1["bmi"]=df1.bmi.fillna(value=mean)

df1.dropna(axis=0,inplace=True )


male_index=df1[df1["gender"]=="Male"].index
other_index=df1[df1["gender"]=="Other"].index



df1.drop(index=male_index,inplace=True)
df1.drop(index=other_index,inplace=True)



df1["bmi"]=df1["bmi"].round(decimals=2)


# df1[avg_glucose_level]<140 low
# df1[avg_glucose_level]<199 Medium
# df1[avg_glucose_level]>=200 High



# create a list of our conditions
conditions = [
    (df1["avg_glucose_level"]<140),
    (df1["avg_glucose_level"]<199),
    (df1["avg_glucose_level"]>=200)
    ]

# create a list of the values we want to assign for each condition
values = ['Low', 'Medium', 'High']

# create a new column and use np.select to assign values to it using our lists as arguments
df1['Sugar'] = np.select(conditions, values)

# df1[bmi]<18.5 low
# df1[bmi]>18.5 & df1[bmi]<29.9 Medium
# df1[bmi]>30 High

# create a list of our conditions
conditions = [
    (df1["bmi"]<18.5),
    (df1["bmi"]>18.5) & (df1["bmi"]<29.9),
    (df1["bmi"]>30 )
    ]

# create a list of the values we want to assign for each condition
values = ['Low', 'Medium', 'High']

# create a new column and use np.select to assign values to it using our lists as arguments
df1['bmi'] = np.select(conditions, values)


from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

df1["ever_married"]= df1[["ever_married"]].apply(le.fit_transform)
df1["work_type"]= df1[["work_type"]].apply(le.fit_transform)
df1["Residence_type"]= df1[["Residence_type"]].apply(le.fit_transform)
df1["smoking_status"]= df1[["smoking_status"]].apply(le.fit_transform)
df1["Sugar"]= df1[["Sugar"]].apply(le.fit_transform)
df1["bmi"]= df1[["bmi"]].apply(le.fit_transform)



#train_test_split

x=df1.drop(columns=["stroke","id","gender","avg_glucose_level"])
y=df1["stroke"]

from sklearn.model_selection import train_test_split 

x_train,x_test,y_train,y_test=train_test_split(x,
                                               y,
                                               random_state=4,
                                               test_size=0.3)





# ## RESAMPLING

from imblearn.over_sampling import SMOTE


y.value_counts()


sm=SMOTE(sampling_strategy="minority")


x_sm,y_sm=sm.fit_resample(x,y)


y_sm.value_counts()


x_train,x_test,y_train,y_test=train_test_split(x_sm,
                                               y_sm,
                                               random_state=4,
                                               test_size=0.3)


# ## MODEL SELECTION



#sklearn models
from sklearn.neighbors import KNeighborsClassifier




#other gradient boosting
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier




catboost=CatBoostClassifier(thread_count=-1,logging_level='Silent')
lightgbm=LGBMClassifier()
Knearest=KNeighborsClassifier()




from sklearn.metrics import classification_report



lightgbm.fit(x_sm,y_sm)
lig_y=lightgbm.predict(x_test)
print(classification_report(lightgbm.predict(x_test),y_test))


catboost.fit(x_sm,y_sm)
cat_y=catboost.predict(x_test)
print(classification_report(catboost.predict(x_test),y_test))



Knearest.fit(x_sm,y_sm)
kn_y=Knearest.predict(x_test)
print(classification_report(Knearest.predict(x_test),y_test))



from sklearn.metrics import roc_curve, roc_auc_score



lig_auc = roc_auc_score(y_test, lig_y)
cat_auc = roc_auc_score(y_test, cat_y)
kn_auc = roc_auc_score(y_test, kn_y)



print('Light GBM: AUROC = %.3f' % (lig_auc))
print('Cat Boost: AUROC = %.3f' % (cat_auc))
print('K-nn : AUROC = %.3f' % (kn_auc))


lig_fpr, lig_tpr, _ = roc_curve(y_test, lig_y)
cat_fpr, cat_tpr, _ = roc_curve(y_test, cat_y)
kn_fpr, kn_tpr, _ = roc_curve(y_test, kn_y)






# ## Test Data


test_data=pd.read_csv("healthcare-dataset-stroke-data.csv")




male_index=test_data[test_data["gender"]=="Male"].index
other_index=test_data[test_data["gender"]=="Other"].index



test_data.drop(index=male_index,inplace=True)


test_data.drop(index=other_index,inplace=True)



test_data.dropna(axis=0,inplace=True )

# df1[avg_glucose_level]<140 low
# df1[avg_glucose_level]<199 Medium
# df1[avg_glucose_level]>=200 High



# create a list of our conditions
conditions = [
    (test_data["avg_glucose_level"]<140),
    (test_data["avg_glucose_level"]<199),
    (test_data["avg_glucose_level"]>=200)
    ]

# create a list of the values we want to assign for each condition
values = ['Low', 'Medium', 'High']

# create a new column and use np.select to assign values to it using our lists as arguments
test_data['Sugar'] = np.select(conditions, values)

# display updated DataFrame
test_data.head()

# test_data[bmi]<18.5 low
# test_data[bmi]>18.5 & df1[bmi]<29.9 Medium
# test_data[bmi]>30 High

# create a list of our conditions
conditions = [
    (test_data["bmi"]<18.5),
    (test_data["bmi"]>18.5) & (test_data["bmi"]<29.9),
    (test_data["bmi"]>30 )
    ]

# create a list of the values we want to assign for each condition
values = ['Low', 'Medium', 'High']

# create a new column and use np.select to assign values to it using our lists as arguments
test_data['bmi'] = np.select(conditions, values)

# display updated DataFrame
test_data.head()

test_data["ever_married"]= test_data[["ever_married"]].apply(le.fit_transform)
test_data["work_type"]= test_data[["work_type"]].apply(le.fit_transform)
test_data["Residence_type"]= test_data[["Residence_type"]].apply(le.fit_transform)
test_data["smoking_status"]= test_data[["smoking_status"]].apply(le.fit_transform)
test_data["Sugar"]= test_data[["Sugar"]].apply(le.fit_transform)
test_data["bmi"]= test_data[["bmi"]].apply(le.fit_transform)




test_data.drop(columns="id",inplace=True)



test_data.drop(columns="gender",inplace=True)



Xtest=test_data.drop(columns=["stroke","avg_glucose_level"])
Ytest=test_data["stroke"]



testy=lightgbm.predict(Xtest)
print(classification_report(lightgbm.predict(Xtest),Ytest))



cat_Y=catboost.predict(Xtest)
print(classification_report(catboost.predict(Xtest),Ytest))



kn_Y=Knearest.predict(Xtest)
print(classification_report(Knearest.predict(Xtest),Ytest))




sm=SMOTE(sampling_strategy="minority")



x_sm,y_sm=sm.fit_resample(x,y)

lig_y=lightgbm.predict(x_sm)
print(classification_report(lightgbm.predict(x_sm),y_sm))

cat_y=catboost.predict(x_sm)
print(classification_report(catboost.predict(x_sm),y_sm))


kn_y=Knearest.predict(x_sm)
print(classification_report(Knearest.predict(x_sm),y_sm))

lig_auc = roc_auc_score(y_sm, lig_y)
cat_auc = roc_auc_score(y_sm, cat_y)
kn_auc = roc_auc_score(y_sm, kn_y)



print('Light GBM: AUROC = %.3f' % (lig_auc))
print('Cat Boost: AUROC = %.3f' % (cat_auc))
print('K-nn : AUROC = %.3f' % (kn_auc))


# ## Creating a pickle file for the classifier

import pickle



filename = 'Stroke.pkl'
pickle.dump(catboost.fit(x_sm,y_sm), open(filename, 'wb'))







