from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris=load_iris()
x=iris.data
y=iris.target

k=KMeans(n_clusters=3,random_state=45)
k.fit(x)

cluster_labels=k.labels_
print(cluster_labels)
centeroids=k.cluster_centers_
print(centeroids)

plt.scatter(x[:,0],x[:,1],marker="o",edgecolors="black",c=cluster_labels)
plt.scatter(centeroids[:,0],centeroids[:,1],marker="*",cmap="viridis",c="red")
plt.show()

