import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.cluster import DBSCAN
from sklearn.cluster import dbscan
import matplotlib.pyplot as plt


# X, _ = datasets.make_blobs(n_samples=300, centers=4, random_state=42)
# core_samples, cluster_ids = dbscan(X, eps=1.0, min_samples=5)

X, _ = datasets.make_moons(500, noise=0.1, random_state=1)
# eps为邻域半径，min_samples为最少点数目   # cluster_ids中-1表示对应的点为噪声点
core_samples, cluster_ids = dbscan(X, eps=0.2, min_samples=20, algorithm='ball_tree') # 'ball_tree'最快 'brute'极慢 默认应该是'kd_tree'

# dbscan = DBSCAN(eps=0.2, min_samples=40, algorithm='ball_tree')
# cluster_ids = dbscan.fit_predict(X)


df = pd.DataFrame(X,columns=['feature1','feature2'])
df.plot.scatter('feature1','feature2', s=100,alpha=0.6, title='dataset by make_moon')

df = pd.DataFrame(np.c_[X,cluster_ids],columns=['feature1','feature2','cluster_id'])
df['cluster_id'] = df['cluster_id'].astype('i2')
df.plot.scatter('feature1','feature2', s=100,c=list(df['cluster_id']),cmap='rainbow',colorbar=False, alpha=0.6,title='sklearn DBSCAN cluster result')

plt.show()





