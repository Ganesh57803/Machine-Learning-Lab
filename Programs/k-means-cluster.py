import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

# Load the Iris dataset
iris_dataset = datasets.load_iris()

# Create a DataFrame for features with descriptive column names
iris_features = pd.DataFrame(iris_dataset.data)
iris_features.columns = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']

# Create a DataFrame for target labels (real clusters)
iris_labels = pd.DataFrame(iris_dataset.target)
iris_labels.columns = ['Cluster_Label']

# Apply K-Means algorithm with 3 clusters
kmeans_model = KMeans(n_clusters=3)
kmeans_model.fit(iris_features)

# Define colormap for plotting
colormap = np.array(['red', 'lime', 'black'])

# Plot the real clusters (based on true labels)
plt.figure(figsize=(14, 14))

plt.subplot(2, 2, 1)
plt.scatter(iris_features.Petal_Length, iris_features.Petal_Width, 
            c=colormap[iris_labels.Cluster_Label], s=40)
plt.title('True Clusters (Based on Actual Labels)')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')

# Plot the K-Means clustering results
plt.subplot(2, 2, 2)
plt.scatter(iris_features.Petal_Length, iris_features.Petal_Width, 
            c=colormap[kmeans_model.labels_], s=40)
plt.title('K-Means Clustering Results')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')

# Display the plots
plt.show()
