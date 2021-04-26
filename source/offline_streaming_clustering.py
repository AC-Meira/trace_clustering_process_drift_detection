from sklearn.metrics import silhouette_score, davies_bouldin_score, mean_squared_error
from sklearn.metrics import homogeneity_score, completeness_score, adjusted_rand_score
from sklearn.metrics import fowlkes_mallows_score, v_measure_score, adjusted_mutual_info_score
from sklearn.metrics import calinski_harabasz_score
from scipy.spatial import distance
from scipy.stats import skew
from scipy.optimize import linear_sum_assignment
import pandas as pd
import numpy as np
import math
from tqdm import tqdm_notebook
from copy import deepcopy
from sklearn.base import clone as sk_clone
import hdbscan

def get_validation_indexes(X, y_pred):
    """
        Returns clustering validation indexes (Silhouette and DBi) 
        based on input X and groups y_pred.
    """
    
    try:
        sc = silhouette_score(X, y_pred)
    except ValueError:
        sc = float("nan")
    
    try:
        dbs = davies_bouldin_score(X, y_pred)
    except ValueError:
        dbs = float("nan")
        
    return {
        "Silhouette": sc,
        "DBi": dbs
    }


def get_density_metrics(X, y_pred):
    """
        Returns clustering validation indexes (Silhouette and DBi) 
        based on input X and groups y_pred.
    """
    
    try:
        ch = calinski_harabasz_score(X, y_pred)
    except ValueError:
        ch = float("nan")
        
        
    #Compute the density based cluster validity index for the clustering specified by labels 
    # and for each cluster in labels.
    try:
        vi, vi_per_cluster = hdbscan.validity.validity_index(X.values, y_pred, per_cluster_scores=True)
        vi_mean = vi_per_cluster.mean()
        vi_std = vi_per_cluster.std()
        
    except ValueError:
        vi = float("nan")
        vi_mean = float("nan")
        vi_std = float("nan")
        # vi_per_cluster = list(["nan"])
    
        
    return {
        "calinski_harabasz_score": ch
        ,"validity_index": vi
        ,"validity_index_mean": vi_mean
        ,"validity_index_std": vi_std
        # ,"validity_index_per_cluster_list": list(vi_per_cluster)
    }


def get_inter_dist_metrics(centroids):
    """
        Returns clustering validation indexes (Silhouette and DBi) 
        based on input X and groups y_pred.
    """
    # Inter-clusters distance
    try:
        inter_dist_euclidean = distance.pdist(centroids, metric = 'euclidean')
        inter_dist_euclidean_mean = inter_dist_euclidean.mean()
        inter_dist_euclidean_std = inter_dist_euclidean.std()
        inter_dist_euclidean_max = inter_dist_euclidean.max()
    except ValueError:
        inter_dist_euclidean_mean = float("nan")
        inter_dist_euclidean_std = float("nan")
        inter_dist_euclidean_max = float("nan")
        
    try:
        inter_dist_correlation = distance.pdist(centroids, metric = 'correlation')
        inter_dist_correlation_mean = inter_dist_correlation.mean()
        inter_dist_correlation_std = inter_dist_correlation.std()
        inter_dist_correlation_max = inter_dist_correlation.max()
    except ValueError:
        inter_dist_correlation_mean = float("nan")
        inter_dist_correlation_std = float("nan")
        inter_dist_correlation_max = float("nan")
        
    try:
        inter_dist_hamming = distance.pdist(centroids, metric = 'hamming')
        inter_dist_hamming_mean = inter_dist_hamming.mean()
        inter_dist_hamming_std = inter_dist_hamming.std()
        inter_dist_hamming_max = inter_dist_hamming.max()
    except ValueError:
        inter_dist_hamming_mean = float("nan")
        inter_dist_hamming_std = float("nan")
        inter_dist_hamming_max = float("nan")
        
    try:
        inter_dist_jaccard = distance.pdist(centroids, metric = 'jaccard')
        inter_dist_jaccard_mean = inter_dist_jaccard.mean()
        inter_dist_jaccard_std = inter_dist_jaccard.std()
        inter_dist_jaccard_max = inter_dist_jaccard.max()
    except ValueError:
        inter_dist_jaccard_mean = float("nan")
        inter_dist_jaccard_std = float("nan")
        inter_dist_jaccard_max = float("nan")
        
    try:
        inter_dist_cosine = distance.pdist(centroids, metric = 'cosine')
        inter_dist_cosine_mean = inter_dist_cosine.mean()
        inter_dist_cosine_std = inter_dist_cosine.std()
        inter_dist_cosinen_max = inter_dist_cosine.max()
    except ValueError:
        inter_dist_cosine_mean = float("nan")
        inter_dist_cosine_std = float("nan")
        inter_dist_cosinen_max = float("nan")
        
    try:
        inter_dist_cityblock = distance.pdist(centroids, metric = 'cityblock')
        inter_dist_cityblock_mean = inter_dist_cityblock.mean()
        inter_dist_cityblock_std = inter_dist_cityblock.std()
        inter_dist_cityblock_max = inter_dist_cityblock.max()
    except ValueError:
        inter_dist_cityblock_mean = float("nan")
        inter_dist_cityblock_std = float("nan")
        inter_dist_cityblock_max = float("nan")
        
        
    return {
        "inter_dist_euclidean_mean": inter_dist_euclidean_mean
        ,"inter_dist_euclidean_std": inter_dist_euclidean_std
        ,"inter_dist_euclidean_max": inter_dist_euclidean_max
        
        ,"inter_dist_correlation_mean": inter_dist_correlation_mean
        ,"inter_dist_correlation_std": inter_dist_correlation_std
        ,"inter_dist_correlation_max": inter_dist_correlation_max
        
        ,"inter_dist_hamming_mean": inter_dist_hamming_mean
        ,"inter_dist_hamming_std": inter_dist_hamming_std
        ,"inter_dist_hamming_max": inter_dist_hamming_max
        
        ,"inter_dist_jaccard_mean": inter_dist_jaccard_mean
        ,"inter_dist_jaccard_std": inter_dist_jaccard_std
        ,"inter_dist_jaccard_max": inter_dist_jaccard_max
        
        ,"inter_dist_cosine_mean": inter_dist_cosine_mean
        ,"inter_dist_cosine_std": inter_dist_cosine_std
        ,"inter_dist_cosinen_max": inter_dist_cosinen_max
        
        ,"inter_dist_cityblock_mean": inter_dist_cityblock_mean
        ,"inter_dist_cityblock_std": inter_dist_cityblock_std
        ,"inter_dist_cityblock_max": inter_dist_cityblock_max
    }


def get_centroids_metrics(X, y_pred, centroids):
    """
        Calculate trace clustering features (Radius, intra-cluster 
        distance, skewness and standard deviation for each cluster)
        based on centroids and traces. 

        Parameters:
        -----------
            X (pd.DataFrame): Dataset of traces in vector space representation
            y_pred (np.array): Attribution of each trace to one cluster
            centroids (np.array): Centroids of each cluster
    """
    r = {
        "radius_list": [],
        "dist_intra_cluster_list": [],
        "skewness_list": [],
        "cluster_std_list": [],
        "wcss_list": []
    }

    # Calculate features for each cluster
    for j in range(len(centroids)):
        # Set of traces in X that belong to cluster 'j'
        X_in_cluster = X[y_pred == j]

        # Calculate radius as maximum distance of a point to centroid
        try:
            radius = distance.cdist(X_in_cluster, [centroids[j]]).max()
            r["radius_list"].append(radius)
        except ValueError:
            r["radius_list"].append(0)
            
        # Average intra-cluster distances
        try:
            dist_intra = distance.pdist(X_in_cluster).mean()
            r["dist_intra_cluster_list"].append(dist_intra)
        except ValueError:
            r["dist_intra_cluster_list"].append(0)

        # Std of cluster
        try:
            c_std = X_in_cluster.std().values
            r["cluster_std_list"].append(c_std)
        except ValueError:
            r["cluster_std_list"].append(0)
        
        # Skewness of cluster
        try:
            skewness = skew(X_in_cluster, axis=None)
            r["skewness_list"].append(skewness)
        except ValueError:
            r["skewness_list"].append(0)
            
        
        # Calculate within-cluster sum-of-squares
        try:
            wcss = sum(abs(distance.cdist(X_in_cluster, [centroids[j]]))**2)
            r["wcss_list"].append(wcss)
        except ValueError:
            r["wcss_list"].append(0)
            
            
    # Get mean and std for all lists
    try:
        r["radius_mean"] = np.mean(r["radius_list"])
        r["radius_std"] = np.std(r["radius_list"])
        r["radius_sum"] = np.sum(r["radius_list"])
    except ValueError:
        r["radius_mean"] = float("nan")
        r["radius_std"] = float("nan")
        r["radius_sum"] = float("nan")
    
    try:
        r["dist_intra_cluster_mean"] = np.mean(r["dist_intra_cluster_list"])
        r["dist_intra_cluster_std"] = np.std(r["dist_intra_cluster_list"])
        r["dist_intra_cluster_sum"] = np.sum(r["dist_intra_cluster_list"])
    except ValueError:
        r["dist_intra_cluster_mean"] = float("nan")
        r["dist_intra_cluster_std"] = float("nan")
        r["dist_intra_cluster_sum"] = float("nan")
    
    try:
        r["cluster_std_mean"] = np.mean(r["cluster_std_list"])
        r["cluster_std_std"] = np.std(r["cluster_std_list"])
        r["cluster_std_sum"] = np.sum(r["cluster_std_list"])
    except ValueError:
        r["cluster_std_mean"] = float("nan")
        r["cluster_std_std"] = float("nan")
        r["cluster_std_sum"] = float("nan")
      
    try:     
        r["skewness_mean"] = np.mean(r["skewness_list"])
        r["skewness_std"] = np.std(r["skewness_list"])
        r["skewness_sum"] = np.sum(r["skewness_list"])
    except ValueError:
        r["skewness_mean"] = float("nan")
        r["skewness_std"] = float("nan")
        r["skewness_sum"] = float("nan")
    
    try:
        r["wcss_mean"] = np.mean(r["wcss_list"])
        r["wcss_std"] = np.std(r["wcss_list"])
        r["wcss_sum"] = np.sum(r["wcss_list"])
    except ValueError:
        r["wcss_mean"] = float("nan")
        r["wcss_std"] = float("nan")
        r["wcss_sum"] = float("nan")
            
    return r


def get_mean_squared_error(clustering_i, clustering_j):
    """
        Calculate features based on the mean squared error between
        clustering_j and clustering_i.

        Parameters:
        -----------
            clustering_i (np.array): Set of centroids at i
            clustering_j (np.array): Set of centroids at the following index
    """
    
    # try:
    #     mse = np.mean((clustering_i - clustering_j) ** 2, axis=0)

    #     return {
    #         "total_MSE": np.sum(mse),
    #         "avg_MSE": np.mean(mse),
    #         "count_non_zero_MSE": np.count_nonzero(mse)        
    #     }
    # except:
    #     return {}
    
    r = {}
    
    # Mean Squared Error (MSE)
    try:
        mse = np.mean((clustering_i - clustering_j) ** 2, axis=0)
        r["total_MSE"] = np.sum(mse)
        r["avg_MSE"] = np.mean(mse)
        r["count_non_zero_MSE"] = np.count_nonzero(mse)
    except ValueError:
        r["total_MSE"] = float("nan")
        r["avg_MSE"] = float("nan")
        r["count_non_zero_MSE"] = float("nan")
        
        
    # Root Mean Squared Error (RMSE)
    try:
        rmse = mse**(1/2)
        r["total_RMSE"] = np.sum(rmse)
        r["avg_RMSE"] = np.mean(rmse)
        r["count_non_zero_RMSE"] = np.count_nonzero(rmse)
    except ValueError:
        r["total_RMSE"] = float("nan")
        r["avg_RMSE"] = float("nan")
        r["count_non_zero_RMSE"] = float("nan")
        
        
    # Mean Absolute Error (MAE)
    try:
        mae = np.mean(abs(clustering_i - clustering_j), axis=0)
        r["total_MAE"] = np.sum(mae)
        r["avg_MAE"] = np.mean(mae)
        r["count_non_zero_MAE"] = np.count_nonzero(mae)
    except ValueError:
        r["total_MAE"] = float("nan")
        r["avg_MAE"] = float("nan")
        r["count_non_zero_MAE"] = float("nan")
    
    return r
    
    
# def get_ground_truth_metrics(y_true_i, y_true_j):
#     """
#         Calculate features based on the mean squared error between
#         clustering_j and clustering_i.

#         Parameters:
#         -----------
#             clustering_i (np.array): Set of centroids at i
#             clustering_j (np.array): Set of centroids at the following index
#     """
        
#     try:
#         ar = adjusted_rand_score(y_true_i, y_true_j)
#     except ValueError:
#         ar = float("nan")
        
#     try:
#         ami = adjusted_mutual_info_score(y_true_i, y_true_j)
#     except ValueError:
#         ami = float("nan")
        
#     try:
#         hs = homogeneity_score(y_true_i, y_true_j)
#     except ValueError:
#         hs = float("nan")
        
#     try:
#         cs = completeness_score(y_true_i, y_true_j)
#     except ValueError:
#         cs = float("nan")
        
#     try:
#         vm = v_measure_score(y_true_i, y_true_j)
#     except ValueError:
#         vm = float("nan")
         
#     try:
#         fm = fowlkes_mallows_score(y_true_i, y_true_j)
#     except ValueError:
#         fm = float("nan")
        
  
#     return {
#         "adjusted_rand_score": ar,
#         "adjusted_mutual_info_score": ami,
#         "homogeneity_score": hs,
#         "completeness_score": cs,
#         "v_measure_score": vm,
#         "fowlkes_mallows_score": fm,
#     } 
    
    
    
def compare_clusterings(interation_1, interation_2):
    """
        Compare two clusterings in consecutive windows and return
        features tracking the evolution of clustering.
        
        Parameters:
        -----------
            resp_1 (dict): Information about clustering at i
            resp_2 (dict): Information about clustering at i + 1
    """
    # Copy resps
    resp_1 = interation_1.copy()
    resp_2 = interation_2.copy()
    
    # print('resp_1["i"]: ', resp_1["i"])
    # print('resp_2["i"]: ', resp_2["i"])
    
    # print('resp_1["centroids"]: ', resp_1["centroids"].shape)
    # print('resp_2["centroids"]: ', resp_2["centroids"].shape)
    
    ### Match centroids from both windows
    if len(resp_1["centroids"]) > len(resp_2["centroids"]):
        # Get distances
        dist_centroids = distance.cdist(resp_1["centroids"], resp_2["centroids"])
        
        # Get first matches
        resp_1_clusters_ind, resp_2_clusters_ind = linear_sum_assignment(dist_centroids)
        # print("resp_1_clusters_ind: ", resp_1_clusters_ind)
        # print("resp_2_clusters_ind: ", resp_2_clusters_ind)
        
        # Get resp_1 clusters remaining
        remaining = sorted(set(range(0, len(resp_1["centroids"]))) - set(resp_1_clusters_ind))
        # print("resp_1_clusters_ind_remaining: ", remaining)
        
        # For the remainings, get resp_2 nearest cluster
        remaining_match = [np.argmin(item) for item in dist_centroids[remaining]]
        # print("remaining_match: ", remaining_match)
        
        # Append remaining and remaining_match
        resp_1_clusters_ind = np.append(resp_1_clusters_ind, remaining)
        resp_2_clusters_ind = np.append(resp_2_clusters_ind, remaining_match)
        resp_1["centroids"] = resp_1["centroids"][resp_1_clusters_ind]
        resp_2["centroids"] = resp_2["centroids"][resp_2_clusters_ind]
        # print("resp_1_clusters_ind: ", resp_1_clusters_ind)
        # print("resp_2_clusters_ind: ", resp_2_clusters_ind)
            
    elif len(resp_1["centroids"]) < len(resp_2["centroids"]):
        # Get distances
        dist_centroids = distance.cdist(resp_2["centroids"], resp_1["centroids"])
        
        # Get first matches
        resp_2_clusters_ind, resp_1_clusters_ind = linear_sum_assignment(dist_centroids)
        # print("resp_1_clusters_ind: ", resp_1_clusters_ind)
        # print("resp_2_clusters_ind: ", resp_2_clusters_ind)
        
        # Get resp_2 clusters remaining
        remaining = sorted(set(range(0, len(resp_2["centroids"]))) - set(resp_2_clusters_ind))
        # print("resp_1_clusters_ind_remaining: ", remaining)
        
        # For the remainings, get resp_1 nearest cluster
        remaining_match = [np.argmin(item) for item in dist_centroids[remaining]]
        # print("remaining_match: ", remaining_match)
        
        # Append remaining and remaining_match
        resp_1_clusters_ind = np.append(resp_1_clusters_ind, remaining_match)
        resp_2_clusters_ind = np.append(resp_2_clusters_ind, remaining)
        resp_1["centroids"] = resp_1["centroids"][resp_1_clusters_ind]
        resp_2["centroids"] = resp_2["centroids"][resp_2_clusters_ind]
        # print("resp_1_clusters_ind: ", resp_1_clusters_ind)
        # print("resp_2_clusters_ind: ", resp_2_clusters_ind)
      
    else:
        # Get distances
        dist_centroids = distance.cdist(resp_1["centroids"], resp_2["centroids"])
        
        # Get first matches
        resp_1_clusters_ind, resp_2_clusters_ind = linear_sum_assignment(dist_centroids)
        # print("resp_1_clusters_ind: ", resp_1_clusters_ind)
        # print("resp_2_clusters_ind: ", resp_2_clusters_ind)
        
        resp_1["centroids"] = resp_1["centroids"][resp_1_clusters_ind]
        resp_2["centroids"] = resp_2["centroids"][resp_2_clusters_ind]
        # print("resp_1_clusters_ind: ", resp_1_clusters_ind)
        # print("resp_2_clusters_ind: ", resp_2_clusters_ind)
        
        
    # print("resp_1_clusters_ind: ", resp_1_clusters_ind)
    # print("resp_2_clusters_ind: ", resp_2_clusters_ind)
    # print('resp_1["centroids"]: ', resp_1["centroids"].shape)
    # print('resp_2["centroids"]: ', resp_2["centroids"].shape)
    
    r = {}

    # If there is no centroid in the two clusterings, return empty
    if len(resp_1["centroids"]) == 0 or len(resp_2["centroids"]) == 0:
        return r

    for key in resp_1:
        try:
            if key not in ["i","y_pred"]:
                key_ = key.replace("_list", "")
 
                # ---------------
                # diff_centroids
                # ---------------
                if key == "centroids":
                    # print('centroid key: ', key)
                    # print('resp_1[key]: ', resp_1[key])
                    # print('resp_2[key]: ', resp_2[key])
                    
                    # # Calculates the minimum average distance between centroids
                    r["mean_diff_" + key_] = min(
                        distance.cdist(resp_1[key], resp_2[key]).min(axis=0).mean(),
                        distance.cdist(resp_2[key], resp_1[key]).min(axis=0).mean()
                    )
                    
                    # Calculates the minimum std distance between centroids
                    r["std_diff_" + key_] = min(
                        distance.cdist(resp_1[key], resp_2[key]).min(axis=0).std(),
                        distance.cdist(resp_2[key], resp_1[key]).min(axis=0).std()
                    )

                    # -----------------------------
                    # Add Mean Squared Error features to the return
                    # -----------------------------
                    r.update(
                        get_mean_squared_error(resp_1[key], resp_2[key])
                    )

                # -----------------------------
                # Metrics based on ground truth
                # -----------------------------
                # elif key == "y_pred":
                #     r.update(
                #         get_ground_truth_metrics(resp_1[key], resp_2[key])
                #     )
                    
                # -------------------------
                # diff_volume
                # diff_radius
                # diff_dist_intra_cluster 
                # diff_skewness
                # diff_cluster_std
                # diff_wcss
                # -------------------------
                # For list of individual features per cluster, calculates the 
                # average squared difference between them
                # elif isinstance(resp_1[key], list):
                elif (type(resp_1[key]) == list):  
                    # print('list key: ', key)
                    # print('resp_1[key]: ', resp_1[key])
                    # print('resp_2[key]: ', resp_2[key])
                    
                    # Use the previous centroids Match for reorganize lists
                    resp_1[key] = np.array(resp_1[key])[resp_1_clusters_ind]
                    resp_2[key] = np.array(resp_2[key])[resp_2_clusters_ind]
                    # print('resp_1[key]: ', resp_1[key])
                    # print('resp_2[key]: ', resp_2[key])
                    
                    try:
                        r["mean_diff_" + key_] = (
                            (np.array(resp_2[key]) - np.array(resp_1[key])) ** 2
                            # (resp_2[key] - resp_1[key]) ** 2
                        ).mean()
                        r["std_diff_" + key_] = (
                            (np.array(resp_2[key]) - np.array(resp_1[key])) ** 2
                        ).std()
                    except:
                        r["mean_diff_" + key_] = float("nan")
                        r["std_diff_" + key_] = float("nan")

                else:
                    # -----------------
                    # diff_DBi
                    # diff_Silhouette
                    # diff_k
                    # etc
                    # -----------------
                    # For numeric features, calculate the difference
                    # print('else key: ', key)
                    try:
                        r["diff_" + key_] = resp_2[key] - resp_1[key]
                    except:
                        r["diff_" + key_] = float("nan")
            else:
                r["i"] = resp_2["i"]

        except Exception as e:
            raise e
            
    # print("######################################################")
    return r

def run_offline_clustering_window(
        tokens, representation_function,
    model, window#, df
    , sliding_window=False, sliding_step=5
):
    """
        Runs the trace clustering approach based on moving trace windows

        Parameters:
        -----------
                  model (sklearn): Scikit-learn clustering model
                     window (int): Size of the trace window to consider when clustering
                df (pd.DataFrame): Dataset with traces in vector space representation
      sliding_window(bool, False): Whether to use a sliding window or not
            sliding_step(int, 5): Size of the step in the case of sliding window

        Returns:
        --------
            all_metrics (pd.DataFrame): DataFrame with the results of execution, features 
                extracted from trace clustering and resulting centroids
    """
    resp = []

    if sliding_window:
        loop = range(0, len(tokens) - window + 1, sliding_step)
    else:
        loop = range(0, len(tokens), window)

    X_full = representation_function(tokens)

    for i in loop:
        # Selects traces inside the window
        # X = df.loc[i : i + window - 1]
        X_tokens = tokens.loc[i : i + window - 1]
        
        # Transform traces
        X = representation_function(X_tokens)
        
        # Reindex for all activities
        X = X.reindex(columns = X_full.columns, fill_value=0, copy=True)

       
        # Fit and predict model to the current window
        model_clone = sk_clone(model)
        y_pred = model_clone.fit_predict(X)
        
        # Remove traces outliers
        n_outliers = y_pred[np.where(y_pred < 0)].size
        X = X.loc[np.where(y_pred >= 0)]
        y_pred = y_pred[np.where(y_pred >= 0)]
        
        ### Rename centroids based on distance from origin (zero point)
        # Centroids
        centers =  (
            pd.DataFrame(X)
            .groupby(y_pred)
            .mean()
            # .drop(-1, errors="ignore")
            .values
        )
        
        # Lookup table to order clusters labels the same way
        idx = np.argsort(centers.sum(axis=1))
        lut = np.zeros_like(idx)
        lut[idx] = np.arange(len(idx))
        try:
            y_pred = lut[y_pred]
        except:
            pass
    
        # Recalculate centroids with new ordering
        centers =  (
            pd.DataFrame(X)
            .groupby(y_pred)
            .mean()
            # .drop(-1, errors="ignore")
            .values
        )
        ### End rename
        
        
        # Start dictionary to be filled with the results
        r = {"i": i, "k": len(np.unique(y_pred))}
        
        # Add labels
        r["y_pred"] = y_pred
        
        # Add centroids to results
        r["centroids"] = centers
        
        # Add info about traces
        r["n_variants"] = len(X_tokens.unique())
        r["n_representation_distinct"] = len(np.unique(X, axis=0))
        r["n_outliers"] = n_outliers
        
        # Count traces per clusters
        values, counts = np.unique(y_pred, return_counts=True)
        r["volume_list"] = list(counts)


        # ----------------------------
        # Calculate validation indexes
        # ----------------------------
        # if max(counts) >= 2 and len(values) > 1:
        r.update(get_validation_indexes(X, y_pred))
        
        
        # ----------------------------
        # Calculate density metrics
        # ----------------------------
        r.update(get_density_metrics(X, y_pred))
        
        
        # ----------------------------
        # Add Inter-clusters distance
        # ----------------------------
        r.update(get_inter_dist_metrics(r["centroids"]))


        # ----------------------------
        # Add centroids features
        # ----------------------------
        r.update(get_centroids_metrics(X, y_pred, r["centroids"]))
        
        
        # ----------------------------
        # If hdbscan and have min span tree, calculate relative_validity (DBCV simplified)
        # ----------------------------
        # if "HDBSCAN" and "gen_min_span_tree=True" in str(model):
        #     try:
        #         r["relative_validity"] = model_clone.relative_validity_
        #     except:
        #         r["relative_validity"] = float("nan")
            
            # try:
            #     r["DBCV_variant"] = DBCV_variant(model_clone)
            # except:
            #     r["DBCV_variant"] = float("nan")
            

            
        # Add current iteration to full response
        resp.append(r)

    # Turn into dataframe
    run_df = pd.DataFrame(resp).set_index("i")
    run_df.fillna(0, inplace=True)

    # Expand values for individual clusters
    # for col in [
    #     "radius_list",
    #     "dist_intra_cluster_list",
    #     "skewness_list",
    #     "cluster_std_list",
    #     "volume_list",
    # ]:
    # for col in run_df.loc[:, run_df.columns.str.endswith('list')].columns:
    #     # min_individuals = run_df[col].apply(len).max()

    #     try:
    #         # Create averages
    #         if col != "volume_list":
    #             run_df["avg_" + col.replace("_list", "")] = run_df[col].apply(lambda x: np.mean(x))
    #             run_df["std_" + col.replace("_list", "")] = run_df[col].apply(lambda x: np.std(x))
                
    #     except Exception as e:
    #         print(e)
    #         pass

    # Calculate time-dependent features
    measures = [compare_clusterings(resp[i], resp[i + 1]) for i in range(len(resp) - 1)]
    measures_df = pd.DataFrame(measures).set_index("i")
    measures_df.fillna(0, inplace=True)

    # Merge results
    all_metrics = run_df.join(measures_df)
    all_metrics.index += all_metrics.index[1]

    return all_metrics