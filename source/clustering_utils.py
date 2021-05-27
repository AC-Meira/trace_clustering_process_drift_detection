# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 18:48:33 2021

@author: T-Gamer
"""
import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.stats import skew
# import model_utils


def isin2D(sub_arrays, full_array):
    out = np.zeros((sub_arrays.shape[0],len(full_array)),dtype=bool)
    sidx = full_array.argsort()
    idx = np.searchsorted(full_array, sub_arrays, sorter=sidx)
    idx[idx==len(full_array)] = 0
    idx0 = sidx[idx]
    np.put_along_axis(out, idx0, full_array[idx0] == sub_arrays, axis=1)
    return out



class Monic:
    """
        Implementation of the cluster tracking methods from MONIC framework.
        Based on paper:
        SPILIOPOULOU, M. & NTOUTSI, I. & THEODORIDIS, Y. & SCHULT, R.; 
        The MONIC framework for cluster transition detection (2006)

        Attributes:
            clustering_i (list of np.array): List of clusters at timestep i
            clustering_j (list of np.array): List of clusters at timestep j
            overlap_matrix (dict): Dictionary of overlap between all clusters
            max_overlaps (dict): Dictionary of the max overlaps for all clusters
                in clustering_i
            t (int, optional): Current timestep
            age (func, optional): Age function with parameters (i, t)
            
    """

    def __init__(
        self, data_points_i, labels_i, data_points_j, labels_j, t=1, age=None, threshold_match=0.5, threshold_split=0.25
    ):
            
        # print('data_points_i:', len(data_points_i))
        # print('labels_i:', labels_i)
        # print('data_points_j:', len(data_points_j))
        # print('labels_j:', labels_j)
        
        self.overlap_matrix = {}
        self.max_overlaps = {}
        self.matches = {}
        self.survivors = {}
        self.splits = {}
        self.absorptions = {}
        self.t = t
        
        if age is None:
            self.age = self.no_age
        else:
            self.age = age
        
        self.clustering_i = self.get_clusters(data_points_i, labels_i)
        self.clustering_j = self.get_clusters(data_points_j, labels_j)

        self.overlap_matrix = self.cluster_overlap_matrix()
        print('self.overlap_matrix: ',self.overlap_matrix)

        for i in self.overlap_matrix:
            self.max_overlaps[i] = self.keys_values_with_max_value(self.overlap_matrix[i])
        print('self.max_overlaps: ',self.max_overlaps)
        
        for i in self.overlap_matrix:
            self.matches[i] = self.get_cluster_match(i, threshold_match)
        print('self.matches: ',self.matches)
        
        for i in self.overlap_matrix:
            self.survivors[i] = self.cluster_survived(i, threshold_match)
        print('self.survivors: ',self.survivors)
        
        for i in self.overlap_matrix:
            self.splits[i] = self.cluster_split(i, threshold_split, threshold_match)
            self.absorptions[i] = self.cluster_absorbed(i, threshold_match)            
        print('self.splits: ',self.splits)
        print('self.absorptions: ',self.absorptions)
        
       
    def no_age(self, i, t):
        """
        Aging function that controls the window/weight associated with
        data point i at timestep t
        
        Args:
            i (int): Position of data point in the sequence 
            t (int): Current timestep 
        
        Returns:
            (int) Age of data point i at timestep t
        """
        return 1


    def get_clusters(self, data_points, labels):
        """
        Given an array of data points and its cluster assignments,
        returns the clusters as sets of points.   

        Args:
            data_points (np.array): Array of data points.
            labels (list): The respective cluster assigned to each point.

        Returns:
            (list) Clusters represented by the set of points assigned to them
        """
        resp = []
        for label in np.unique(labels):
            resp.append(data_points[labels == label])

        return resp

    
    def keys_values_with_max_value(self, dict):
        """
        Returns a (key, value) pair of the key with the max value in the dict.
        Source: https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
        """
        return [(key, val) for m in [max(dict.values())] 
                for key, val in dict.items() if val == m]


    def cluster_overlap(self, cluster_X, cluster_Y):
        """
        Calculates the overlap between two clusters, i.e. how much one 
        matches another

        Args:
            cluster_X (np.array): List of data points in the cluster
            cluster_Y (np.array): List of data points in the cluster
        
        Returns:
            (float) The overlap between the two clusters (0-1)
        """
        overlap_sum = 0
        X_sum = 0
        
        # cluster_X = cluster_X.to_numpy()
        # cluster_Y = cluster_Y.to_numpy()
        
        for i in range(len(cluster_X)):
            # print('cluster_X[i]: ', round(cluster_X.iloc[i],2))
            # print('cluster_Y: ', round(cluster_Y,2))
            # print('count: ', (np.isin(cluster_Y, cluster_X.iloc[i])))
            
            # if np.isin(round(cluster_Y,2), round(cluster_X.iloc[i],2) ).all():
            #     overlap_sum += self.age(i, self.t)
            # X_sum += self.age(i, self.t)
            overlap_sum += (1 - distance.cdist(cluster_Y, [cluster_X.iloc[i]], metric = "cosine").mean())
            X_sum += 1

        # print('overlap_sum/X_sum: ',overlap_sum/X_sum)
        return overlap_sum/X_sum

    def cluster_overlap_matrix(self, pandas=False):
        """
        Calculates the overlap between all clusters in two clustering situations and
        returns the overlap matrix

        Args:
            pandas (bool, optional): Whether to return as a pandas DataFrame or dict 
        
        Returns:
            (pd.DataFrame) DataFrame of the overlap between all clusters
            OR
            (dict) Dicionary of the overlap between all clusters

        """
        resp = {}
        for i in range(len(self.clustering_i)):
            resp[i] = {}
            for j in range(len(self.clustering_j)):
                resp[i][j] = self.cluster_overlap(self.clustering_i[i], self.clustering_j[j])
        if pandas:
            return pd.DataFrame.from_dict(resp).transpose()
        else: 
            return resp
        
    # ---------------------
    # External Transitions
    # ---------------------

    def get_cluster_match(self, X, threshold_match=0.5):
        """
        Returns the best match of cluster_X between all clusters in clustering_i that 
        overlap above the threshold (defaults to 0.5)

        Args:
            X (int): Index of cluster_X in clustering_i
            threshold_match (float, optional): Threshold of cluster overlap [0.5, 1]
        
        Returns:
            (int) Index of the cluster that best matches cluster_X_i in clustering_j
                OR
            (None) if there is none cluster that matches the criteria 

        """

        if self.max_overlaps[X][0][1] > threshold_match:
            return self.max_overlaps[X][0][0]
        else:
            return None


    def cluster_survived(self, X, threshold_match=0.5):
        """
        Returns whether the cluster_X at timestep i has survived in clustering at timestep j. 
        The condition is that there is a best match of cluster_X_i between all clusters in clustering_i that 
        overlap above the threshold (defaults to 0.5) and that cluster is not the best match of 
        any other cluster of clustering_i.

        Args:
            X (int): Index of cluster_X
            threshold_match (float, optional): Threshold of cluster overlap [0.5, 1]
            
        Returns:
            (None) if the cluster did not survived
                OR
            (int) The index of the cluster that best matches cluster_X at i in clustering_j if it survived

        """
        if self.matches[X] is not None:
            for i in self.matches:
                if i != X and self.matches[i] == self.matches[X]:
                    return None
            return self.matches[X]
        else:
            return None


    def cluster_split(self, X, threshold_split=0.25, threshold_match=0.5):
        """
        Returns whether the cluster_X at timestep i has been split in clustering at timestep j. 
        The condition is that the cluster must not have survived and the cluster must be a match of
        a subset of clusters that, individually, overlap cluster_X above 'threshold_split' and joined 
        together match cluster_X above 'threshold_match'. 

        Args:
            X (int): Index of cluster_X
            threshold_split (float, optional): Threshold of cluster overlap to consider split [0, threshold_match]
            threshold_match (float, optional): Threshold of cluster overlap to consider match [0.5, 1]
            
        Returns:
            (None) if the cluster not split and/or did not survived
                OR
            (int) The index of the cluster that best matches cluster_X at i in clustering_j if it survived
        """
        assert threshold_split < threshold_match, "'Threshold_match' must be greater than 'threshold_split'"
        
        if self.survivors[X] is not None:
            return None
        else:
            overlaps = self.overlap_matrix[X]

            sum_overlap = 0
            split_candidates_i = []
            for i in overlaps:
                if overlaps[i] > threshold_split: 
                    split_candidates_i.append(i)
                    sum_overlap += overlaps[i]
            
            if sum_overlap > threshold_match:
                return split_candidates_i
            else:
                return None
    
    
    def cluster_absorbed(self, X, threshold_match=0.5):
        """
        Returns whether the cluster_X at timestep i has been absorbed in clustering at timestep j. 
        The condition is that the cluster at timestep j that best matches X, also is the best match
        of other cluster at timestamp i. 

        Args:
            X (int): Index of cluster_X
            threshold_match (float, optional): Threshold of cluster overlap to consider match [0.5, 1]
        
        Returns:
            (None) if the cluster was not absorbed
                OR
            (int) The index of the clusters that were absorbed by the best match of cluster X at i
        """

        if self.matches[X] is None:
            return None
        else:
            matches = []
            for i in self.matches:
                if i != X and self.matches[i] == self.matches[X]:
                    matches.append(i)
                
            if len(matches) > 0:
                return matches
            else:
                return None


    # ---------------------
    # Internal Transitions
    # ---------------------

    def size_transition(self, X, Y):
        """
        Calculates the difference in size (considering age) between two clusters

        Args:
            self.clustering_i[X] (np.array): List of data points in the cluster X of 
                clustering at timestep i
            self.clustering_j[Y] (np.array): List of data points in the cluster Y of 
                clustering at timestep j
            t (int, optional): Current timestep
            age (func, optional): Age function with parameters (i, t)
        
        Returns:
            (float) Difference in size of two clusters
        """
        sum_X = 0
        sum_Y = 0
        
        for i in range(len(self.clustering_i[X])):
            sum_X = self.age(i, self.t)

        for i in range(len(self.clustering_j[Y])):
            sum_Y = self.age(i, self.t)

        return sum_Y - sum_X
    
    
    def compactness_transition(self, X, Y):
        """
        Calculates the difference of compactness between two clusters

        Args:
            X (int): Index of cluster in clustering_i
            Y (int): Index of cluster in clustering_j
        
        Returns:
            (float) Difference in size of two clusters
        """
        #X_pdist = distance.pdist(self.clustering_i[X])
        #Y_pdist = distance.pdist(clustering_i_Y)

        X_std = self.clustering_i[X].std()
        Y_std = self.clustering_j[Y].std()

        return Y_std - X_std


    def center_location_transition(self, X, Y):
        """
        Calculates the difference of the center location between two clusters

        Args:
            X (int): Index of cluster in clustering_i
            Y (int): Index of cluster in clustering_j
        
        Returns:
            (float) Distance of the center location of two clusters
        """
        X_mean = self.clustering_i[X].mean(axis=0) 
        Y_mean = self.clustering_j[Y].mean(axis=0)

        return distance.euclidean(Y_mean, X_mean)


    def skewness_transition(self, X, Y):
        """
        Calculates the difference of skewness between two clusters

        Args:
            X (int): Index of cluster in clustering_i
            Y (int): Index of cluster in clustering_j
        
        Returns:
            (float) Difference of skewness of two clusters
        """
        X_skew = skew(self.clustering_i[X], axis=None)
        Y_skew = skew(self.clustering_j[Y], axis=None)

        return Y_skew - X_skew
    
    
    
    
#####################################################################################    

class RankedPoints:
    
    def __init__(self, points, clusterer, metric='euclidean', selection_method='centroid'):
        """ Rank points in a cluster based on their distance to the cluster centroid/medoid
        
        Parameters
        ----------
        
        points : array of shape (n_samples, n_features), and must be the same data passed into
                 HDBSCAN
        
        clusterer : Instance of HDBSCAN that has been fit to data
        
        metric: string or callable, optional (default='euclidean')
            The metric to use when calculating distance between points in a cluster and 
            the cluster centroid/medoid. If metric is a string or callable, it must be one of
            the options allowed by scipy.spatial.distance.cdist for its metric parameter.
        
        selection_method: string, optional (default='centroid')
            Method to use to find the weighted cluster center. Allowed options are 'centroid' 
            and 'medoid'.
        
        """
        self.clusterer = clusterer
        self.metric = metric
        
        allowed_methods = ['centroid', 'medoid']
        if selection_method not in allowed_methods:
            raise ValueError(f'Selection method must be one of {allowed_methods}')
        
        if selection_method == 'centroid' and metric != 'euclidean':
            raise ValueError(f'Metric must be euclidian when using selection_method centroid. '
                             f'Current metric is {metric}')
        
        self.selection_method = selection_method
        
        self._embedding_cols = [str(i) for i in range(points.shape[1])]
        self.embedding_df = pd.DataFrame(points, columns=self._embedding_cols)
        self.embedding_df['cluster'] = clusterer.labels_
    
    def calculate_all_distances_to_center(self):
        """For each cluster calculate the distance from each point to the centroid/medoid"""
        all_distances = pd.DataFrame()
        for label in np.unique(self.embedding_df['cluster']):           
            distance_df = self.calculate_distances_for_cluster(label)
            all_distances = pd.concat([all_distances, distance_df])
        
        self.embedding_df = self.embedding_df.merge(all_distances, left_index=True, right_index=True)
    
    def calculate_distances_for_cluster(self, cluster_id):
        """For a given cluster_id calculate the distance from each point to the centroid/medoid.
        
        Parameters
        ----------

        cluster_id : int
            The id of the cluster to compute the distances for. If the cluster id is -1 which
            corresponds to the noise point cluster, then this will return a distance of NaN.

        Returns
        -------

        df : A pandas DataFrame containing the distances from each point to the cluster centroid/medoid.
             The index of the dataframe corresponds to the index in the original data. 

        """
        cluster_of_interest = self.embedding_df[self.embedding_df['cluster'] == cluster_id].copy()
        
        if cluster_of_interest.empty:
            raise ValueError(f'Cluster id {cluster_id} not found')
        
        # Don't calculate distances for the noise cluster
        if cluster_id == -1:
            return pd.DataFrame(np.nan, columns=['dist_to_rep_point'], index=cluster_of_interest.index)
        
        if self.selection_method == 'centroid':
            rep_point = self.clusterer.weighted_cluster_centroid(cluster_id)
        if self.selection_method == 'medoid':
            rep_point = self.clusterer.weighted_cluster_medoid(cluster_id)
        
        dists = cdist(rep_point.reshape((1,len(self._embedding_cols))), cluster_of_interest[self._embedding_cols].values, metric=self.metric)
        return pd.DataFrame(dists[0], columns=['dist_to_rep_point'], index=cluster_of_interest.index)
    
    
    def rank_cluster_points_by_distance(self, cluster_id):
        """For a given cluster return a pandas dataframe of points ranked 
           by distance to the cluster centroid/medoid
        """
        cluster_of_interest = self.embedding_df[self.embedding_df['cluster'] == cluster_id].copy()
        
        if cluster_of_interest.empty:
            raise ValueError(f'Cluster id {cluster_id} not found')
            
        if 'dist_to_rep_point' not in self.embedding_df.columns:
            distance_df = self.calculate_distances_for_cluster(cluster_id)
            cluster_of_interest = cluster_of_interest.merge(distance_df, left_index=True, right_index=True)
        
        cluster_of_interest.sort_values('dist_to_rep_point', inplace=True)
        return cluster_of_interest
    
    def get_closest_samples_for_cluster(self, cluster_id, n_samples=5):
        """Get the N closest points to the cluster centroid/medoid"""
        return self.rank_cluster_points_by_distance(cluster_id).head(n_samples)
    
    def get_furthest_samples_for_cluster(self, cluster_id, n_samples=5):
        """Get the N points furthest away from the cluster centroid/medoid"""
        return self.rank_cluster_points_by_distance(cluster_id).tail(n_samples)