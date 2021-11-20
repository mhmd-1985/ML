import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('CC GENERAL.csv')
df.head()
df.info()
df.describe(include = 'all').T
df.duplicated().sum()
#Drop unique columns for Analysis
df_clus = df.drop('CUST_ID', axis = 1)
# Checking for Outliers
plt.figure(figsize = (20,16))
Features = df_clus.columns
for i in range(len(Features)):
    plt.subplot(6,3, i + 1)
    sns.boxplot(y = df_clus[Features[i]], data = df_clus)
    plt.title(f"Boxplot of {Features[i]}")
    plt.tight_layout()

def detect_outliers(col):
    Q1, Q3 = col.quantile([0.25,0.75])
    IQR = Q3 - Q1
    lower_range = Q1 - (1.5 * IQR)
    upper_range = Q3 + (1.5 * IQR)
    return lower_range, upper_range
Feature_list = df_clus.columns
for i in Feature_list:
    lr, ur = detect_outliers(df_clus[i])
    df_clus[i] = np.where(df_clus[i] > ur, ur,df_clus[i])
    df_clus[i] = np.where(df_clus[i] < lr, lr,df_clus[i])
plt.figure(figsize = (20,16))
Features = df_clus.columns
for i in range(len(Features)):
    plt.subplot(6,3, i + 1)
    sns.boxplot(y = df_clus[Features[i]], data = df_clus)
    plt.title(f"Boxplot of {Features[i]}")
    plt.tight_layout()
    
df_clus.isnull().sum()
df_clus.CREDIT_LIMIT = df_clus.CREDIT_LIMIT.fillna(df_clus.CREDIT_LIMIT.mean())
df_clus.MINIMUM_PAYMENTS = df_clus.MINIMUM_PAYMENTS.fillna(df_clus.MINIMUM_PAYMENTS.mean())


from sklearn.preprocessing import StandardScaler

 
sta= StandardScaler() 
sta.fit(df_clus)
df_clus_scaled= pd.DataFrame(sta.transform(df_clus),columns = df_clus.columns)

from sklearn.cluster import KMeans

k_Means4 = KMeans(n_clusters = 3,random_state = 123)
k_Means4.fit(df_clus_scaled)
labels = k_Means4.labels_
df['Cluster_Labels_3'] = labels
df.head(3)

df_out = df.groupby(by = 'Cluster_Labels_3').sum()[['PURCHASES','PAYMENTS','TENURE']].reset_index()
df_out.head(3)
sns.set_style("darkgrid")
plt.figure(figsize = (18,4))
plt.subplot(1,3,1)
sns.barplot(x= 'Cluster_Labels_3',y = 'PURCHASES', data = df_out, palette = 'crest', seed = 123);
plt.subplot(1,3,2)
sns.barplot(x= 'Cluster_Labels_3',y = 'PAYMENTS', data = df_out, palette = 'crest', seed = 123);
plt.subplot(1,3,3)
sns.barplot(x= 'Cluster_Labels_3',y = 'TENURE', data = df_out, palette = 'crest', seed = 123);
