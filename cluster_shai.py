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

# =============================================================================
# Observations
# The Outliers are present in all the fields except : PURCHASES_FREQUENCY, PURCHASES_INSTALLMENTS_FREQUENCY.
# Removing Outliers    
# =============================================================================
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
    
# =============================================================================
# Treat Missing Values
# Observations
# The missing values are present in the fields : CREDIT_LIMIT and MINIMUM_PAYMENTS.
# I will use median() to fill the null fields.
# =============================================================================
df_clus.isnull().sum()
df_clus.CREDIT_LIMIT = df_clus.CREDIT_LIMIT.fillna(df_clus.CREDIT_LIMIT.median())
df_clus.MINIMUM_PAYMENTS = df_clus.MINIMUM_PAYMENTS.fillna(df_clus.MINIMUM_PAYMENTS.median())
df_clus.isnull().sum()
#Scaling of Data using z-score method
#Importing the necessary libraries 

from sklearn.preprocessing import StandardScaler
#Intilizing object of StandardScaler
 
zscore = StandardScaler() 
zscore.fit(df_clus)
df_clus_scaled= pd.DataFrame(zscore.transform(df_clus),columns = df_clus.columns)
df_clus_scaled.describe().T
from sklearn.cluster import KMeans
wss = []
for i in range(1,11):
    k_means = KMeans(n_clusters = i)
    k_means.fit(df_clus_scaled)
    wss.append(k_means.inertia_)
    print(f"The inertia of {i} clusters : {k_means.inertia_}")
plt.plot(range(1,11),wss, marker='o', linestyle='dashed',linewidth=2, markersize=8);
    

    
# =============================================================================
#   Observations:
# From the above plot, we can see, there is a sharp decrease in the inertia from cluster = 1 till cluster= 4, Hence we can either choose 3 or 4 clusters. But we will verify the silhouette_score for clusters upto 10.
# 
# Predicting KMean and silhouette_score with n clusters  
# =============================================================================
from sklearn.metrics import silhouette_samples, silhouette_score
sil_score = []
sil_width_min = []
for i in range(2,11):
    k_means = KMeans(n_clusters = i,random_state = 123)
    k_means.fit(df_clus_scaled)
    labels = k_means.labels_
    score = silhouette_score(df_clus_scaled,labels, random_state = 123)
    sil_score.append(score)
    print(f"The Silhouette Score of {i} clusters : {score}")
    min_width = silhouette_samples(df_clus_scaled,labels).min()
    sil_width_min.append(min_width)
    print(f"The Silhouette Width of {i} clusters : {min_width}")
    
plt.figure(figsize = (18,4))
plt.subplot(1,2,1)
plt.plot(range(2,11),sil_score, marker='o', linestyle='dashed',linewidth=2, markersize=8);
plt.subplot(1,2,2)
plt.plot(range(2,11),sil_width_min, marker='o', linestyle='dashed',linewidth=2, markersize=8);
# =============================================================================
# Observations
# The silhouette_score is highest with 3 numbers of clusters and lowest for 9 & 10 number of clusters.
# The Minimum silhouette Width, all the values for all the clusters in analysis are negatives, with minimum value at n = 9 and maximum at n = 2.
# Hence lets form an analysis with n_clusters = 3.
# 
# KMeans with clusters = 3
# =============================================================================
k_Means4 = KMeans(n_clusters = 3,random_state = 123)
k_Means4.fit(df_clus_scaled)
labels = k_Means4.labels_
df['Cluster_Labels_3'] = labels
df.head(3)
silhouette_score(df_clus_scaled,labels,random_state = 123)
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
# =============================================================================
# Observations
# Cluster 0 : It is highest amounts for Tenure, medium for Purchases, whereas lowest for Payments.
# Cluster 1 : It is lowest for Purchases,whereas medium for Tenure & Payments.
# Cluster 2 : It is highest for Purchases & Payments, whereas lowest for Tenure.
# =============================================================================
