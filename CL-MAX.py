from functools import lru_cache
import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from iteration_utilities import deepflatten
import json
from Modified_MAFIA import *
import os


class CL_MAX():
    def __init__(self, min_support, num_of_clusters, rounding_threshold, batch_size=100):
        self.min_support = min_support
        self.num_of_clusters = num_of_clusters
        self.threshold = rounding_threshold
        self.seen_dict = {}
        self.path = ''
        self.batch_size = batch_size

    def find_path(self):
        self.path = os.getcwd()+'/Datasets'
        return self.path

    def read_dataset_from_file(self, name_of_dataset, delimiter):
        df = []
        maximum = 0
        with open(self.path+'/Actual Datasets/'+name_of_dataset+'.csv') as file:
            for line in file:
                transaction = [int(item)
                               for item in line.replace('\n', '').split(delimiter)]
                df.append(transaction)
                maximum = np.max([maximum, np.max(transaction)])
        self.maximum = maximum
        self.transactions = df

    def load_one_hot_dataset(self, name_of_dataset):
        df = pd.read_csv(self.path+'/One-hot/'+name_of_dataset+'01.csv')
        self.dataset = df

    def remove_non_frequent_single_items(self):
        count = np.zeros(self.maximum+1)
        for transaction in self.transactions:
            for item in transaction:
                count[item] += 1

        for i in range(len(count)):
            if count[i] < min_support:
                self.dataset[i] = 0
        temp = [[count[i], i] for i in range(self.maximum)]
        self.sorted_items = sorted(temp)

    def cluster_transactions(self):
        kmeans = MiniBatchKMeans(n_clusters=self.num_of_clusters,
                                 batch_size=self.batch_size, max_iter=20).fit(self.dataset)
        labels = kmeans.labels_
        cnt_labels = np.zeros(self.num_of_clusters)
        clusters = np.zeros((self.num_of_clusters, self.dataset.shape[1]))
        for i in range(len(labels)):
            clusters[labels[i]] += np.array(self.dataset.iloc[i])
            cnt_labels[labels[i]] += 1
        for i in range(len(clusters)):
            clusters[i] /= cnt_labels[i]
        for i in range(len(clusters)):
            clusters[i][clusters[i] >= self.threshold] = 1
            clusters[i][clusters[i] < self.threshold] = 0
        clusters = np.array(clusters, dtype=np.int)
        self.clusters = clusters
        return clusters

    def convert_clusters_to_itemset(self):
        items = []
        for cluster in self.clusters:
            s = []
            for i in range(len(cluster)):
                if cluster[i] == 1:
                    s.append(i)
            if len(s) > 0:
                items.append(s)
        return items

    def order_correction(self, possible_candidates):
        temp_candidates = []
        for candidate in possible_candidates:
            temp_can = []
            for i in range(self.maximum):
                if self.sorted_items[i][1] in candidate:
                    temp_can.append(self.sorted_items[i][1])
            temp_candidates.append(temp_can)
        return temp_candidates

    def _CL_MAX(self):
        clusters = self.cluster_transactions()
        possible_candidates = self.convert_clusters_to_itemset()
        possible_candidates = self.order_correction(possible_candidates)
        for item in possible_candidates:
            str_item = json.dumps(item)
            self.seen_dict[str_item] = 0
        min_support_count = self.min_support*len(self.transactions)
        MFIs = mafiaAlgorithm(self.transactions, min_support_count,
                              possible_candidates, self.seen_dict)
        return MFIs


if __name__ == '__main__':
    name_of_dataset = 'chess'
    min_support = 0.9
    num_cluster = 10
    rounding_threshold = 0.9

    cl_max = CL_MAX(min_support, num_cluster, rounding_threshold)
    path = cl_max.find_path()
    cl_max.read_dataset_from_file(name_of_dataset, ',')
    cl_max.load_one_hot_dataset(name_of_dataset)
    cl_max.remove_non_frequent_single_items()
    MFIs = cl_max._CL_MAX()
    print(MFIs)
