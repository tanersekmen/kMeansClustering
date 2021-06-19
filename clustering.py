import matplotlib.pyplot as plt
import numpy as np
from numpy.random import normal
import pandas as pd
import seaborn as sns


from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

#read the data 
def importData():
    df = pd.read_csv('credit_card.csv')
    # preparation of data
    df.drop(['CUST_ID'] , axis = 1, inplace=True)
    df.dropna(subset=['CREDIT_LIMIT'], inplace = True)
    df['MINIMUM_PAYMENTS'].fillna(df['MINIMUM_PAYMENTS'].median(), inplace=True)
    return df

# X_red that I used and reduction dimension part
def xRed(df):
    pca = PCA(n_components=0.95)
    X_red = pca.fit_transform(df)
    return X_red


# Choose the k value with elbow method
def elbowMethod(X_red):
    kmeans_models = [KMeans(n_clusters=k, random_state=23).fit(X_red) for k in range (1, 10)]
    innertia = [model.inertia_ for model in kmeans_models]
    # visualizaiton of elbow method
    plt.plot(range(1, 10), innertia)
    plt.title('Elbow method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.show()
    return kmeans_models

# silhouette graph according to k values when the data get clustering.
def silhouetteGraph(X_red, kmeans_models):
    silhoutte_scores = [silhouette_score(X_red, model.labels_) for model in kmeans_models[1:4]]
    plt.plot(range(2,5), silhoutte_scores, "bo-")
    plt.xticks([2, 3, 4])
    plt.title('Silhoutte scores vs Number of clusters')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhoutte score')
    plt.show()

# labeling part for all cluster id
def clusterId(X_red, df):
    kmeans = KMeans(n_clusters=2, random_state=23).fit(X_red)
    df['cluster_id'] = kmeans.labels_
    return df

# visualization of distribution of clustering
def visualize(df):
    plt.figure(figsize=(10,6))
    sns.scatterplot(data=df, x='CREDIT_LIMIT', y='PURCHASES', hue='cluster_id')
    plt.title('Distribution of clusters based on Credit limit and total purchases')
    plt.show()



def main():
    df = importData()
    X_red = xRed(df)
    kmeans_models = elbowMethod(X_red)
    silhouetteGraph(X_red, kmeans_models)
    df = clusterId(X_red, df)
    visualize(df)




if __name__ == '__main__':
    main()