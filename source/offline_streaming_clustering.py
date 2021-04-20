from sklearn.metrics import silhouette_score, davies_bouldin_score, mean_squared_error
from sklearn.metrics import homogeneity_score, completeness_score, adjusted_rand_score
from sklearn.metrics import fowlkes_mallows_score, v_measure_score, adjusted_mutual_info_score
from sklearn.metrics import calinski_harabasz_score
from scipy.spatial import distance
from scipy.stats import skew
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
    except ValueError:
        inter_dist_euclidean_mean = float("nan")
        inter_dist_euclidean_std = float("nan")
        
    try:
        inter_dist_correlation = distance.pdist(centroids, metric = 'correlation')
        inter_dist_correlation_mean = inter_dist_correlation.mean()
        inter_dist_correlation_std = inter_dist_correlation.std()
    except ValueError:
        inter_dist_correlation_mean = float("nan")
        inter_dist_correlation_std = float("nan")
        
    try:
        inter_dist_hamming = distance.pdist(centroids, metric = 'hamming')
        inter_dist_hamming_mean = inter_dist_hamming.mean()
        inter_dist_hamming_std = inter_dist_hamming.std()
    except ValueError:
        inter_dist_hamming_mean = float("nan")
        inter_dist_hamming_std = float("nan")
        
    # try:
    #     inter_dist_jaccard = distance.pdist(centroids, metric = 'jaccard')
    #     inter_dist_jaccard_mean = inter_dist_jaccard.mean()
    #     inter_dist_jaccard_std = inter_dist_jaccard.std()
    # except ValueError:
    #     inter_dist_jaccard_mean = float("nan")
    #     inter_dist_jaccard_std = float("nan")
        
    try:
        inter_dist_cosine = distance.pdist(centroids, metric = 'cosine')
        inter_dist_cosine_mean = inter_dist_cosine.mean()
        inter_dist_cosine_std = inter_dist_cosine.std()
    except ValueError:
        inter_dist_cosine_mean = float("nan")
        inter_dist_cosine_std = float("nan")
        
    # try:
    #     inter_dist_cityblock = distance.pdist(centroids, metric = 'cityblock')
    #     inter_dist_cityblock_mean = inter_dist_cityblock.mean()
    #     inter_dist_cityblock_std = inter_dist_cityblock.std()
    # except ValueError:
    #     inter_dist_cityblock_mean = float("nan")
    #     inter_dist_cityblock_std = float("nan")
        
        
    return {
        "inter_dist_euclidean_mean": inter_dist_euclidean_mean
        ,"inter_dist_euclidean_std": inter_dist_euclidean_std
        
        ,"inter_dist_correlation_mean": inter_dist_correlation_mean
        ,"inter_dist_correlation_std": inter_dist_correlation_std
        
        ,"inter_dist_hamming_mean": inter_dist_hamming_mean
        ,"inter_dist_hamming_std": inter_dist_hamming_std
        
        # ,"inter_dist_jaccard_mean": inter_dist_jaccard_mean
        # ,"inter_dist_jaccard_std": inter_dist_jaccard_std
        
        ,"inter_dist_cosine_mean": inter_dist_cosine_mean
        ,"inter_dist_cosine_std": inter_dist_cosine_std
        
        # ,"inter_dist_cityblock_mean": inter_dist_cityblock_mean
        # ,"inter_dist_cityblock_std": inter_dist_cityblock_std
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
        vi_per_cluster = list(["nan"])
    
        
    return {
        "calinski_harabasz_score": ch
        ,"validity_index": vi
        ,"validity_index_mean": vi_mean
        ,"validity_index_std": vi_std
        ,"validity_index_per_cluster": list(vi_per_cluster)
    }


def get_label_metrics(y_true_i, y_true_j):
    """
        Calculate features based on the mean squared error between
        clustering_j and clustering_i.

        Parameters:
        -----------
            clustering_i (np.array): Set of centroids at i
            clustering_j (np.array): Set of centroids at the following index
    """
        
    try:
        ar = adjusted_rand_score(y_true_i, y_true_j)
    except ValueError:
        ar = float("nan")
        
    try:
        ami = adjusted_mutual_info_score(y_true_i, y_true_j)
    except ValueError:
        ami = float("nan")
        
    try:
        hs = homogeneity_score(y_true_i, y_true_j)
    except ValueError:
        hs = float("nan")
        
    try:
        cs = completeness_score(y_true_i, y_true_j)
    except ValueError:
        cs = float("nan")
        
    try:
        vm = v_measure_score(y_true_i, y_true_j)
    except ValueError:
        vm = float("nan")
         
    try:
        fm = fowlkes_mallows_score(y_true_i, y_true_j)
    except ValueError:
        fm = float("nan")
        
  
    return {
        "adjusted_rand_score": ar,
        "adjusted_mutual_info_score": ami,
        "homogeneity_score": hs,
        "completeness_score": cs,
        "v_measure_score": vm,
        "fowlkes_mallows_score": fm,
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
        "cluster_std_list": []
    }

    # Calculate features for each cluster
    for j in range(len(centroids)):
        # Set of traces in X that belong to cluster 'j'
        X_in_cluster = X[y_pred == j]

        try:
            # Calculate radius as maximum distance of a point to centroid
            r["radius_list"].append(distance.cdist(X_in_cluster, [centroids[j]]).max())
        except ValueError:
            r["radius_list"].append(0)

        # Average intra-cluster distances
        dist_intra = distance.pdist(X_in_cluster).mean()
        r["dist_intra_cluster_list"].append(dist_intra)

        # Std of cluster
        c_std = X_in_cluster.std()
        r["cluster_std_list"].append(c_std)
        
        # Skewness of cluster
        skewness = skew(X_in_cluster, axis=None)
        r["skewness_list"].append(skewness)

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
        
    try:
        mse = np.mean((clustering_i - clustering_j) ** 2, axis=0)

        return {
            "total_MSE": np.sum(mse),
            "avg_MSE": np.mean(mse),
            "count_non_zero_MSE": np.count_nonzero(mse)        
        }
    except:
        return {}
    
    
def compare_clusterings(resp_1, resp_2):
    """
        Compare two clusterings in consecutive windows and return
        features tracking the evolution of clustering.
        
        Parameters:
        -----------
            resp_1 (dict): Information about clustering at i
            resp_2 (dict): Information about clustering at i + 1
    """
    
    r = {}

    # If there is no centroid in the two clusterings, return empty
    if len(resp_1["centroids"]) == 0 or len(resp_2["centroids"]) == 0:
        return r

    for key in resp_1:
        try:
            if key != "i":
                key_ = key.replace("_list", "")
                
                # ---------------
                # diff_labels
                # ---------------
                if key == "y_pred":
                    r.update(
                        get_label_metrics(resp_1[key], resp_2[key])
                    )
            
                # ---------------
                # diff_centroids
                # ---------------
                elif key == "centroids":
                    # Calculates the minimum average distance between centroids  

                    r["diff_" + key_] = min(
                        distance.cdist(resp_1[key], resp_2[key]).min(axis=0).mean(),
                        distance.cdist(resp_2[key], resp_1[key]).min(axis=0).mean(),
                    )
                    
                    r["std_diff_" + key_] = min(
                        distance.cdist(resp_1[key], resp_2[key]).min(axis=0).std(),
                        distance.cdist(resp_2[key], resp_1[key]).min(axis=0).std(),
                    )

                    # Add Mean Squared Error features to the return
                    r.update(
                        get_mean_squared_error(resp_1[key], resp_2[key])
                    )

                # -------------------------
                # diff_radius
                # diff_skewness
                # diff_dist_intra_cluster
                # diff_cluster_std
                # -------------------------
                # For list of individual features per cluster, calculates the 
                # average squared difference between them
                elif isinstance(resp_1[key], list):                    
                    try:
                        r["diff_" + key_] = (
                            (np.array(resp_2[key]) - np.array(resp_1[key])) ** 2
                        ).mean()
                    except:
                        r["diff_" + key_] = float("nan")

                else:
                # -----------------
                # diff_DBi
                # diff_Silhouette
                # diff_k
                # etc
                # -----------------
                # For numeric features, calculate the 
                    try:
                        r["diff_" + key_] = resp_2[key] - resp_1[key]
                    except:
                        r["diff_" + key_] = float("nan")
            else:
                r["i"] = resp_2["i"]

        except Exception as e:
            raise e

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
        
        
        # Centroids
        centers =  (
            pd.DataFrame(X)
            .groupby(y_pred)
            .mean()
            .drop(-1, errors="ignore")
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
            .drop(-1, errors="ignore")
            .values
        )
        
        # Start dictionary to be filled with the results
        r = {"i": i, "k": len(np.unique(y_pred[y_pred > 0]))}
        
        # Add labels
        r["y_pred"] = y_pred
        
        # Add info about traces
        r["n_variants"] = len(X_tokens.unique())
        r["n_representation_distinct"] = len(np.unique(X, axis=0))

        # Count traces per clusters
        values, counts = np.unique(y_pred, return_counts=True)

        # ----------------------------
        # Calculate validation indexes
        # ----------------------------
        # if max(counts) >= 2 and len(values) > 1:
        r.update(get_validation_indexes(X, y_pred))
        
        
        # ----------------------------
        # Calculate density metrics
        # ----------------------------
        r.update(get_density_metrics(X, y_pred))
        
        
        # Add centroids to results
        r["centroids"] = centers
        
        # Add Inter-clusters distance
        r.update(get_inter_dist_metrics(r["centroids"]))

        # Add features to results
        r["volume_list"] = counts
        r.update(get_centroids_metrics(X, y_pred, r["centroids"]))
        
        #If hdbscan and have min span tree, calculate relative_validity (DBCV simplified)
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
    for col in [
        "radius_list",
        "dist_intra_cluster_list",
        "skewness_list",
        "cluster_std_list",
        "volume_list",
    ]:
        # min_individuals = run_df[col].apply(len).max()

        try:
            # Create averages
            if col != "volume_list":
                run_df["avg_" + col.replace("_list", "")] = run_df[col].apply(lambda x: np.mean(x))
                run_df["std_" + col.replace("_list", "")] = run_df[col].apply(lambda x: np.std(x))
                
        except Exception as e:
            print(e)
            pass

    # Calculate time-dependent features
    measures = [compare_clusterings(resp[i], resp[i + 1]) for i in range(len(resp) - 1)]
    measures_df = pd.DataFrame(measures).set_index("i")
    measures_df.fillna(0, inplace=True)

    # Merge results
    all_metrics = run_df.join(measures_df)
    all_metrics.index += all_metrics.index[1]

    return all_metrics