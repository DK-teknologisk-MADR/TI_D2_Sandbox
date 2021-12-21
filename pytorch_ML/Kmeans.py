from sklearn.cluster import KMeans
import numpy as np






class Kmeans_Cities():
    def __init__(self,data):
        self.data = data
        self.cluster_model = None
        self.labels = None
    def fit(self,nr):
        self.cluster_model = KMeans(n_clusters = nr,verbose=1)
        self.cluster_model.fit(X=self.data)
        clusters = self.cluster_model.cluster_centers_
        city_indices = []
        cities = np.zeros((nr, self.data.shape[1]))
        for label in np.arange(nr):
            indices_with_label = np.arange(self.data)[self.cluster_model.labels_ == label]
            data_in_cluster = self.data[indices_with_label]
            min_dist_index_in_cluster_data,pt = self.get_nearest_points(clusters[label],data_in_cluster)
            min_dist_index = indices_with_label[min_dist_index_in_cluster_data]
            city_indices.append(int(min_dist_index))
            cities[label] = pt
        self.labels = self.cluster_model.labels_
        return city_indices,cities

    def get_nearest_points(self,pt : np.ndarray,data : np.ndarray = None,nr = 1 ):
        if data is None:
            data = self.data
        dists = self.get_distances(data_1=data,data_2=pt)
        min_dist_indices = np.argsort(dists)[:nr]
        pts = data[min_dist_indices][:nr]
        return min_dist_indices,pts


    def get_distances(self,data_1,data_2):
        return ((data_1-data_2)**2).sum(axis=1)

