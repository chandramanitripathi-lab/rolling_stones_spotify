# rolling_stones_spotify
missing value, matplotlip, chartbar,
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
 file_path = 'rolling_stones_spotify.csv'
data = pd.read_csv(file_path)
print(data.head())
print(data.info())
data = data.drop_duplicates()
print("\nMissing Values:")
print(data.isnull().sum())
avg_popularity_per_album = data.groupby('album')['popularity'].mean()
top_albums = avg_popularity_per_album.nlargest(2)
print("\nTop 2 Albums Based on Popular Songs:")
print(top_albums)
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
top_albums.plot(kind='bar', xlabel='Album', ylabel='Average Popularity', title='Top 2 Albums')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 6))
sns.histplot(data['danceability'], bins=20, kde=True)
plt.xlabel('Danceability')
plt.ylabel('Frequency')
plt.title('Distribution of Danceability')
plt.show()
numeric_data = data.select_dtypes(include=['number'])
correlation_matrix = numeric_data.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()
eda_data = data[['danceability','acousticness','energy','instrumentalness','liveness','loudness','speechiness','tempo','valence','popularity','duration_ms']]

sns.pairplot(eda_data)
plt.show()
data['release_date'] = pd.to_datetime(data['release_date'])
data['release_year'] = data['release_date'].dt.year

popularity_over_time = data.groupby('release_year')['popularity'].mean()

popularity_over_time.plot(
    kind='line',
    xlabel='Release Year',
    ylabel='Average Popularity',
    title='Popularity Over Time'
)

plt.show()
selected_features = data[['acousticness', 'danceability']]
scaler = StandardScaler()
X = scaler.fit_transform(selected_features)
k_values = range(1, 10) 
wcss_values = []
max_iters = 100
for k in k_values:
    np.random.seed(0)
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    for iteration in range(max_iters):
        distances = np.sqrt(np.sum((X[:, np.newaxis] - centroids) ** 2, axis=2))
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    

    wcss = np.sum(np.min(distances, axis=1))
    wcss_values.append(wcss)

plt.figure(figsize=(8, 6))
plt.plot(k_values, wcss_values, marker='o', linestyle='-', color='b')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.grid(True)
plt.show()
optimal_k = np.argmin(np.diff(wcss_values)) + 2  
k = optimal_k
np.random.seed(0)
centroids = X[np.random.choice(X.shape[0], k, replace=False)]
for iteration in range(max_iters):
    distances = np.sqrt(np.sum((X[:, np.newaxis] - centroids) ** 2, axis=2))
    labels = np.argmin(distances, axis=1)
    new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
    if np.all(centroids == new_centroids):
        break
    centroids = new_centroids

cluster_counts = np.bincount(labels)
for i in range(k):
    print(f"Cluster {i+1}")
    print(f"Centroid Used: {centroids[i]}, No. of Records: {cluster_counts[i]}")
    data['cluster'] = labels
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x='acousticness',
    y='danceability',
    hue='cluster',
    data=data,
    palette='Set1'
)

plt.xlabel('Acousticness')
plt.ylabel('Danceability')
plt.title('Clustering of Songs based on Acousticness and Danceability')

plt.show()
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(X)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=reduced_features[:, 0], y=reduced_features[:, 1], hue=labels, palette='Set1')
plt.xlabel('acousticness')
plt.ylabel('danceability')
plt.title('PCA Visualization of Song Clusters')
plt.show()
