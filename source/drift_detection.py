import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import scipy.stats

def get_metrics(drifts: list, not_drifts: list, resp: list, window_size=0, log_size=0, margin_error=0, verbose=False) -> dict:
    """
    Given the drifts predicted and the ground truth, calculates binary classification metrics 
    (Precision, Recall, F1 Score) and delay of detection. We consider a drift to be detected
    correctly when it has been found within 2 * window_size after its true index. 

    Parameters:
    ------------
        drifts (list): List with the index of the detected drifts
        resp (list): List with the index of the ground truth drifts
        window_size (int): Size of the trace clustering window used to consider the delay.
        verbose (bool): Whether to print results

    Returns:
    ---------
        dict: dictionary with the value of each metric
    """
    
    # # Create list with drift detection prediction
    # df_drifts_pred = pd.DataFrame(drifts, columns=['init'])
    # df_drifts_pred['end'] = df_drifts_pred['init'] + window_size-1
    # df_drifts_pred["y_pred"] = 1
    # # df_drifts_pred.set_index("i", inplace=True)
    
    # # Create list with not drift detection prediction
    # df_not_drifts_pred = pd.DataFrame(not_drifts, columns=['init'])
    # df_not_drifts_pred['end'] = df_not_drifts_pred['init'] + window_size-1
    # df_not_drifts_pred["y_pred"] = 0
    # # df_not_drifts_pred.set_index("i", inplace=True)
    
    # # Concatenate predictions, sort them and Initiate y_true
    # test_drifts = pd.concat([df_drifts_pred, df_not_drifts_pred], axis=0).sort_values(by='init', ascending=True).reset_index(drop=True)
    # test_drifts['y_true'] = 0
    # print(test_drifts)
    
    # # Create list with ground truth drifts ranges
    # ground_truth_ranges = pd.DataFrame(resp, columns=['init'])
    # ground_truth_ranges['end'] = ground_truth_ranges['init'] + margin_error*window_size-1
    # print(ground_truth_ranges)
    
    # # Add ground_truth 
    # for i, test_drift in test_drifts.iterrows():
    #     for j, ground_truth in ground_truth_ranges.iterrows():
    #         # print("i:",i)
    #         # print("j:",j)
    #         # print("test_drift:",test_drift)
    #         # print("ground_truth:",ground_truth)
            
    #         if (
    #             (((test_drift['init']>=ground_truth['init']) & (test_drift['init']<=ground_truth['end']))
    #             | ((test_drift['end']>=ground_truth['init']) & (test_drift['end']<=ground_truth['end'])))
    #             ):
    #             print("if")
    #             test_drift['y_true'] = 1
    #             pass
    # print(test_drifts)      

    
    # Initialize metrics with 0
    precision = 0
    recall = 0
    specificity = 0
    precision_negative = 0
    tp = 0
    tn = 0
    delay = 0
    avg_delay = 0
    resp_ = resp.copy()
    resp_2 = resp.copy()
    predicted_true = [0 for x in resp_]
    
    # Transforms the window_size into a vector with the size of the drifts found
    if isinstance(window_size, int):
        window_size_drifts = np.repeat(window_size, len(drifts))
    
    # Iterates over all drifts found and to all ground truths drifts
    for i in range(len(drifts)):    
        for j in range(len(resp_)):            
            # check if the drift found is within 2 * window_size after its true index
            # or if is within 0,8 window_size before its true index (test had at least 20% drift)
            if (-0.8 * window_size_drifts[i] < drifts[i] - resp_[j] <= 2 * window_size_drifts[i]):
                if verbose:
                    print((drifts[i], drifts[i] + window_size_drifts[i], resp_[j]))
                
                # drift found correctly
                delay += drifts[i] - resp_[j]
                tp += 1
                resp_[j] = np.inf
                predicted_true[j] = 1
                break
            
    # Transforms the window_size into a vector with the size of the drifts found
    if isinstance(window_size, int):
        window_size_not_drift = np.repeat(window_size, len(not_drifts))
    # Iterates over all not drifts found and to all ground truths drifts
    for i in range(len(not_drifts)):    
        for j in range(len(resp_2)):            
            # check if the drift found is within 2 * window_size after its true index
            # or if is within 0,8 window_size before its true index (test had at least 20% drift)
            if not (-0.8 * window_size_not_drift[i] < not_drifts[i] - resp_2[j] <= 2 * window_size_not_drift[i]):
                if verbose:
                    print((not_drifts[i], not_drifts[i] + window_size_not_drift[i], resp_2[j]))
                
                # not drift found correctly
                tn += 1
                break
    
    # Get metrics
    if len(drifts) > 0:
        precision = tp/len(drifts)   

    if len(resp_) > 0:
        recall = tp/len(resp_)
        
    try:
        f1 = scipy.stats.hmean([precision, recall])
    except ValueError:
        f1 = 0.0
        
    if len(not_drifts) > 0:
        specificity = tn/(tn+len(drifts)-tp)
        precision_negative = tn/(tn+len(resp_)-tp)
    
    # Get delay
    if tp > 0:
        avg_delay = (delay/tp)/window_size_drifts[0]
    
    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Specificity": specificity,
        "Precision_negative": precision_negative,
        "Delay": avg_delay,
        "Correct_Predictions": predicted_true,
        "Support_correct": sum(predicted_true),
        "Support": len(drifts)+len(not_drifts),
        "Mean_test_per_drift": (len(drifts)+len(not_drifts))/len(resp),
        # "Not_Drifts_Found": not_drifts,
        "Drifts_Found": drifts,
        "Resp": resp
    }

# def get_metrics(drifts: list, not_drifts: list, resp: list, window_size=100, verbose=False) -> dict:
#     """
#     Given the drifts predicted and the ground truth, calculates binary classification metrics 
#     (Precision, Recall, F1 Score) and delay of detection. We consider a drift to be detected
#     correctly when it has been found within 2 * window_size after its true index. 

#     Parameters:
#     ------------
#         drifts (list): List with the index of the detected drifts
#         resp (list): List with the index of the ground truth drifts
#         window_size (int): Size of the trace clustering window used to consider the delay.
#         verbose (bool): Whether to print results

#     Returns:
#     ---------
#         dict: dictionary with the value of each metric
#     """
#     # Initialize metrics with 0
#     precision = 0
#     specificity = 0
#     recall = 0
#     tp = 0
#     delay = 0
#     avg_delay = 0
#     resp_ = resp.copy()
#     predicted = [0 for x in resp_]
    
#     # Transforms the window_size into a vector with the size of the drifts found
#     if isinstance(window_size, int):
#         window_size = np.repeat(window_size, len(drifts))
    
#     # Iterates over all drifts found and to all ground truths drifts
#     for i in range(len(drifts)):    
#         for j in range(len(resp_)):            
#             # check if the drift found is within 2 * window_size after its true index
#             if 0 <= drifts[i] - resp_[j] <= 2 * window_size[i]:
#                 if verbose:
#                     print((drifts[i], drifts[i] + window_size[i], resp_[j]))
                
#                 # drift found correctly
#                 delay += drifts[i] - resp_[j]
#                 tp += 1
#                 resp_[j] = np.inf
#                 predicted[j] = 1
#                 break
    
#     if len(drifts) > 0:
#         precision = tp/len(drifts)

#     if len(resp_) > 0:
#         recall = tp/len(resp_)
        
#     try:
#         f1 = scipy.stats.hmean([precision, recall])
#     except ValueError:
#         f1 = 0.0
        
#     if tp > 0:
#         avg_delay = (delay/tp)/window_size[0]
    
#     return {
#         "Precision": precision,
#         "Recall": recall,
#         "F1": f1,
#         "Delay": avg_delay,
#         "Correct_Predictions": predicted,
#         "Support_correct": sum(predicted),
#         "Support": len(drifts)+len(not_drifts),
#         "Not_Drifts_Found": not_drifts,
#         "Drifts_Found": drifts,
#         "Resp": resp,
#     }


def detect_concept_drift(
    df, var_ref, rolling_window=3, std_tolerance=3, min_tol=0.0025, verbose=False
):
    """
        Performs the drift detection algorithm. Based on the features measured from 
        tracking the evolution of the trace clustering over the windows, estimate 
        tolerance boundaries to detect a drift when the feature value measured at the
        current index lies outside of the boundaries. 

        Parameters:
        ------------
            df (pd.DataFrame): DataFrame of features from tracking the trace clusterings
            var_ref (str): Name of the feature to apply the algorithm. It has to be present
                in the df.columns
            rolling_window (int): The number of rolling windows to consider when estimating
                the tolerance boundaries and detecting the drifts. It smooths the analyzed 
                feature to reduce false positive detections due to noise. 
            std_tolerance (int): Number of times the rolling standard deviation used to 
                the tolerance boundaries. A higher value provides higher tolerance and 
                lower sensitivity to drifts.
            min_tol (float [0.0 - 1.0]): Number of times the rolling average to calculate a
                minimum tolerance boundaries. Useful for not detecting false positives in more 
                stable feature values when the tolerance could be too little.
            verbose (bool): Whether to print the index of drifts as they are found
        
        Returns:
        ---------
            list: List of index of the detected drifts
            dict: Dictionary with the rolling average, lowers and uppers boundaries 
                to assist plotting
    """
    # Initialize variables
    window_buffer = []
    drifts = []
    not_drifts = []
    mean = None
    std = None

    # Lists to keep the rolling average and the lower and upper boundaries to 
    # be returned at the end of the execution of this method to support the plots
    lowers = []
    uppers = []
    means = df[var_ref].rolling(window=rolling_window).mean().values.tolist()
    

    # Iterates over the values
    for i, row in df.iterrows():

        # If the rolling window is of the desired size
        if len(window_buffer) < rolling_window:
            window_buffer.append(row[var_ref])
            lowers.append(np.nan)
            uppers.append(np.nan)

        else:
            if mean is not None:
                # To avoid errors in multiplication with 0
                if mean == 0:
                    mean == 1

                # Calculates tolerance boundaries considering the rolling mean and std
                expected_lower = min(
                    mean - (std_tolerance * std),
                    (1 - min_tol) * mean
                )
                expected_upper = max(
                    mean + (std_tolerance * std),
                    (1 + min_tol) * mean
                )

                # Adds into the list to return in the end
                lowers.append(expected_lower)
                uppers.append(expected_upper)

                # Checks whether the current value lies outside the tolerance boundaries
                if expected_lower > row[var_ref] or row[var_ref] > expected_upper:
                    if verbose:
                        print(i, expected_lower, expected_upper, row[var_ref])
                    drifts.append(i)

                    window_buffer = []
                else:
                    not_drifts.append(i)
            else:
                lowers.append(np.nan)
                uppers.append(np.nan)

            if len(window_buffer) > 0:
                window_buffer.pop(0)

            window_buffer.append(row[var_ref])
            
            # if mean is not None:
            #     if expected_lower > np.mean(window_buffer) or np.mean(window_buffer) > expected_upper:
            #         if verbose:
            #             print(i, expected_lower, expected_upper, np.mean(window_buffer), row[var_ref])
            #         drifts.append(i)
            #         window_buffer = []
                    
            if i in drifts:
                mean = None
                std = None
            else:
                mean = np.mean(window_buffer)
                std = np.std(window_buffer)

    return drifts, not_drifts, {"lowers": lowers, "uppers": uppers, "means": means}