#%%
import KMeanData
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# Drop rows with null values and remove Year and Vote_Average column.
data = KMeanData.dataset.dropna()
data = data[['Language', 'Genre']]
# print(data)

# Feature Scaling, apply standard scaling method on the two features.
sc_data = StandardScaler()
data_std = sc_data.fit_transform(data.astype(float))

# Cluster with Kmeans.
kmeans=KMeans(n_clusters=5,random_state=42).fit(data_std)
labels=kmeans.labels_

# Making a new dataframe now.
newDataset=pd.DataFrame(data=data_std, columns=['Language','Genre'])
newDataset['label_kmeans']=labels

# Visualization
fig, ax=plt.subplots()
# ax=fig.add_subplot(111,projection='3d') # Implementation of a 3d model with three features.
plt.scatter(newDataset['Language'][newDataset['label_kmeans']==0],newDataset['Genre'][newDataset['label_kmeans']==0],c='blue',s=100,edgecolor='green',linestyle='--')
plt.scatter(newDataset['Language'][newDataset['label_kmeans']==1],newDataset['Genre'][newDataset['label_kmeans']==1],c='red',s=100,edgecolor='green',linestyle='--')
plt.scatter(newDataset['Language'][newDataset['label_kmeans']==2],newDataset['Genre'][newDataset['label_kmeans']==2],c='green',s=100,edgecolor='green',linestyle='--')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=500)
ax.set_xlabel("Language")
ax.set_ylabel("Genre")
plt.show()

# Make blobs to create artifical data
raw_data = make_blobs(n_samples = 2000, n_features = 2, centers = 4, cluster_std = 1.8)

# Visualization of the raw data.
plt.scatter(raw_data[0][:,0], raw_data[0][:,1], c=raw_data[1])

# Create a model.
model = KMeans(n_clusters=5)
model.fit(raw_data[0])

# Make predictions with kmeans clustering model.
model.labels_
model.cluster_centers_

# Visualizing the accuracy of the model
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(10,6))
ax1.set_title('Our Model')
ax1.scatter(raw_data[0][:,0], raw_data[0][:,1],c=model.labels_)
ax2.set_title('Original Data')
ax2.scatter(raw_data[0][:,0], raw_data[0][:,1],c=raw_data[1])

# %%
mms_data = MinMaxScaler()
mms_data_std = mms_data.fit_transform(data.astype(float))

kmeans=KMeans(n_clusters=5,random_state=42).fit(mms_data_std)
labels=kmeans.labels_

# Making a new dataframe now.
newDataset=pd.DataFrame(data=mms_data_std, columns=['Language','Genre'])
newDataset['label_kmeans']=labels

# Visualization
fig, ax=plt.subplots()
# ax=fig.add_subplot(111,projection='3d') # Implementation of a 3d model with three features.
plt.scatter(newDataset['Language'][newDataset['label_kmeans']==0],newDataset['Genre'][newDataset['label_kmeans']==0],c='blue',s=100,edgecolor='green',linestyle='--')
plt.scatter(newDataset['Language'][newDataset['label_kmeans']==1],newDataset['Genre'][newDataset['label_kmeans']==1],c='red',s=100,edgecolor='green',linestyle='--')
plt.scatter(newDataset['Language'][newDataset['label_kmeans']==2],newDataset['Genre'][newDataset['label_kmeans']==2],c='green',s=100,edgecolor='green',linestyle='--')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=500)
ax.set_xlabel("Language")
ax.set_ylabel("Genre")
plt.show()

# Make blobs to create artifical data
raw_data = make_blobs(n_samples = 2000, n_features = 2, centers = 4, cluster_std = 1.8)

# Visualization of the raw data.
plt.scatter(raw_data[0][:,0], raw_data[0][:,1], c=raw_data[1])

# Create a model.
model = KMeans(n_clusters=5)
model.fit(raw_data[0])

# Make predictions with kmeans clustering model.
model.labels_
model.cluster_centers_

# Visualizing the accuracy of the model
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(10,6))
ax1.set_title('Our Model')
ax1.scatter(raw_data[0][:,0], raw_data[0][:,1],c=model.labels_)
ax2.set_title('Original Data')
ax2.scatter(raw_data[0][:,0], raw_data[0][:,1],c=raw_data[1])
# %%
