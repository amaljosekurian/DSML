from sklearn.cluster import KMeans
from sklearn.datasets import  load_breast_cancer
import matplotlib.pyplot as plt

br=load_breast_cancer()
x=br.data
y=br.target

k=KMeans(n_clusters=3,random_state=45)
k.fit(x)
clusters=k.labels_
print(clusters)

centeroids=k.cluster_centers_
print(centeroids)

plt.scatter(x[:,0],x[:,1],c=clusters,cmap="viridis",marker="o",edgecolors="black")
plt.scatter(centeroids[:,0],centeroids[:,1],marker="*",c="red",s=200)
plt.show()

