import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import imshow
import matplotlib.image as mpimg
matplotlib.use('tkAgg')

class KCenterMetric():
    def __init__(self,data : np.array):
        if isinstance(data,np.ndarray):
            self.data = data
        if isinstance(data,torch.Tensor):
            self.data = data.numpy()
        self.labels = None
    def fit(self,k):
        done = False
        in_game = np.ones(len(self.data))
        center = self.data.mean(axis=0)
        pt = center
        self.labels = np.zeros(len(self.data),dtype=np.int)
        results = np.zeros((len(self.data),2))
        results[:,0] = np.arange(len(self.data))
        results[:,1] = self.get_distances(data_1=self.data,data_2 = center)
        city_indices = []
        cities = np.zeros((k,self.data.shape[1]))

        for i in range(k):
            results = results[np.argsort(results[:,1],)[::-1]]
            new_city_index = int(round(results[0,0]))
            cities[i] = self.data[new_city_index]
            print("city chosen is ",cities[i])
            city_indices.append(int(round(new_city_index)))
            print("city index chosen is",int(round(new_city_index)))
            results = results[1:,:]
            data_of_indices = self.data[results[:,0].astype('int')]
            results[:,1] = np.minimum(results[:,1],self.get_distances(data_of_indices, cities[i]))
        print(cities)
        for i,pt in enumerate(self.data):
            self.labels[i] = self.get_nearest_points(pt,cities,1)[0][0]
        return city_indices,cities

    def get_nearest_points(self,pt : np.ndarray,data : np.ndarray = None,nr = 1 ):
        if data is None:
            data = self.data
        dists = self.get_distances(data_1=data,data_2=pt)
        min_dist_indices = np.argsort(dists)[:nr]
        pts = data[min_dist_indices][:nr]
        return min_dist_indices,pts

    def get_farthest_point(self, pt: np.ndarray, data: np.ndarray = None ):
        if data is None:
            data = self.data
        dists = self.get_distances(data_1=data, data_2=pt)
        max_dist_index = np.argmax(dists)
        pt = data[max_dist_index]
        return max_dist_index, pt

    def get_distances(self,data_1,data_2):
        return ((data_1-data_2)**2).sum(axis=1)


