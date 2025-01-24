import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('cleaned_data_video_game_reviews.csv')

features = df[['User Rating', 'Price', 'Game Length (Hours)']]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.savefig('elbow_method.png')
plt.close()

kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)

plt.scatter(df['User Rating'], df['Price'], c=df['Cluster'], cmap='viridis')
plt.xlabel('User Rating')
plt.ylabel('Price')
plt.title('Clustering Game Berdasarkan Rating dan Harga')
plt.savefig('clustering_results.png')
plt.close()

df.to_csv('clustered_data.csv', index=False)

cluster_profile = df.groupby('Cluster').agg({
    'User Rating': 'mean',
    'Price': 'mean',
    'Game Length (Hours)': 'median',
    'Genre': lambda x: x.mode()[0]
}).reset_index()

print("\nCluster Profiling:")
print(cluster_profile)

# Visualisasi Distribusi Genre per Cluster
plt.figure(figsize=(12,6))
sns.countplot(data=df, x='Cluster', hue='Genre')
plt.title('Genre Distribution Across Clusters')
plt.savefig('genre_cluster_distribution.png', bbox_inches='tight')
plt.close()