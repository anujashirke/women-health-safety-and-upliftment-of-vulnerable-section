


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt

# ## LOADING FILE
df=pd.read_csv("Strokesdataset.csv")





corr=df.corr()

sns.heatmap(corr,annot=True)


df.stroke.value_counts()



df.ever_married.value_counts()



df.smoking_status.value_counts()



df.work_type.value_counts()



df.Residence_type.value_counts()



mean=df.bmi.mean()



df["bmi"].isnull().sum()


df1=df.copy()



df1.head()


df1["bmi"]=df1.bmi.fillna(value=mean)
df1.head()



df1["age"]=df1.age.apply(np.ceil).astype(int)
df1.info()



df1.dropna(axis=0,inplace=True )



df1.isnull().sum()



male_index=df1[df1["gender"]=="Male"].index
other_index=df1[df1["gender"]=="Other"].index



df1.drop(index=male_index,inplace=True)
df1.drop(index=other_index,inplace=True)



df1.gender.value_counts()



df1["bmi"]=df1["bmi"].round(decimals=2)


df1.tail()

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()


df1["ever_married"]= df1[["ever_married"]].apply(le.fit_transform)
df1["work_type"]= df1[["work_type"]].apply(le.fit_transform)
df1["Residence_type"]= df1[["Residence_type"]].apply(le.fit_transform)
df1["smoking_status"]= df1[["smoking_status"]].apply(le.fit_transform)
df1.head()


df["hypertension"].value_counts()



df1["work_type"].value_counts()


from matplotlib import pyplot as plt
plt.subplots(figsize=(10,10))
sns.heatmap(corr,annot=True,fmt=".3g",annot_kws={"fontsize":10})


df1.head()


x=df1.drop(columns=["stroke","id","gender"])
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


x_train.head()



y_train.head()


# ## MODEL SELECTION



#sklearn models
from sklearn.neighbors import KNeighborsClassifier



get_ipython().system('pip install catboost')
get_ipython().system('pip install xgboost')
get_ipython().system('pip install lightgbm')




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



plt.plot(lig_fpr, lig_tpr, linestyle='--', label='Light GBM (AUROC = %.3f)' % lig_auc)
plt.plot(cat_fpr, cat_tpr, marker='.', label='Cat Boost (AUROC = %0.3f)' % cat_auc)
plt.plot(kn_fpr, kn_tpr, marker='.', label='K-nn (AUROC = %0.3f)' % kn_auc)

# Title
plt.title('ROC Plot')
# Axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# Show legend
plt.legend() # 
# Show plot
plt.show()


# ## Test Data


test_data=pd.read_csv("healthcare-dataset-stroke-data.csv")



test_data.isna().sum()


test_data.gender.value_counts()


male_index=test_data[test_data["gender"]=="Male"].index
other_index=test_data[test_data["gender"]=="Other"].index



test_data.drop(index=male_index,inplace=True)


test_data.drop(index=other_index,inplace=True)


test_data.gender.value_counts()



test_data.isna().sum()


test_data.dropna(axis=0,inplace=True )



test_data.head()



test_data["ever_married"]= test_data[["ever_married"]].apply(le.fit_transform)
test_data["work_type"]= test_data[["work_type"]].apply(le.fit_transform)
test_data["Residence_type"]= test_data[["Residence_type"]].apply(le.fit_transform)
test_data["smoking_status"]= test_data[["smoking_status"]].apply(le.fit_transform)
test_data.head()



test_data.drop(columns="id",inplace=True)



test_data.drop(columns="gender",inplace=True)



test_data.head(1)


Xtest=test_data.drop(columns="stroke")
Ytest=test_data["stroke"]



testy=lightgbm.predict(Xtest)
print(classification_report(lightgbm.predict(Xtest),Ytest))



cat_Y=catboost.predict(Xtest)
print(classification_report(catboost.predict(Xtest),Ytest))



kn_Y=Knearest.predict(Xtest)
print(classification_report(Knearest.predict(Xtest),Ytest))



test_data.stroke.value_counts()



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







