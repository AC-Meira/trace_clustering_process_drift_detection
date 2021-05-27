import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import scipy.stats
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import ruptures as rpt


# def get_change_points():
    

def get_metrics(drifts: list, not_drifts: list, resp: list, window_size=0, sliding_step=0, log_size=0, margin_error=0, verbose=False) -> dict:
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
    # Initialize metrics with 0
    precision = 0
    recall = 0
    f1 = 0
    specificity = 0
    precision_negative = 0
    tp = 0
    delay = 0
    avg_delay = 0
    predicted_true = [0 for x in resp]
    
    # Create list with all windows
    if sliding_step > 0:
        windows = pd.DataFrame(range(0, log_size+1, sliding_step), columns=['init'])
    else:
        windows = pd.DataFrame(range(0, log_size+1, window_size), columns=['init'])
    # windows = pd.DataFrame()
    # windows['init'] = run_df.index
    windows['end'] = windows['init']+window_size-1
              
    # windows['init'] = windows['end'] - window_size
    
    # Add drift detections and not drift detections as y_pred. If didn't test a window fill with -1
    windows['y_pred'] = [1 if window in drifts else 0 if window in not_drifts else -1 
                          for window in windows['init']]
        
    # Add ground truth as y_true
    windows['y_true'] = 0
    windows['y_true_margin_error'] = 0
    windows['ground_truth_init'] = -1
    
    for i, window in windows.iterrows():
        for ground_truth in resp:
            
            if  ((ground_truth-sliding_step <= window['init']) & (window['end'] <= ground_truth+((1+margin_error)*(window_size-sliding_step)))):
                window['y_true_margin_error'] = 1
                window['ground_truth_init'] = ground_truth-sliding_step
                if (window['init'] < ground_truth < window['end']):
                    window['y_true'] = 1
                pass
     
    # print(windows) 
    # Consider drift detection inside margin of error
    windows['y_true'] = np.where(((windows['y_pred']==1) & (windows['y_true_margin_error']==1)), 1, windows['y_true'])
    windows.drop(columns='y_true_margin_error', inplace=True)
    # print(windows) 
    
    # Remove windows that was ground truth but was detected after and inside margin of error
    windows = windows[~((windows['y_true']==1) 
                      & ((windows['y_true'].shift(-1)==1) 
                          |(windows['y_true'].shift(-2)==1)))
                      ]
    # print(windows) 
    
    # Remove windows that was not tested and was not ground truth as well
    windows = windows[~((windows['y_true']==0) & (windows['y_pred']==-1))]
    # print(windows) 
    
    # Replace y_pred value in windows not tested (marked as -1) to 0 
    windows['y_pred'] = windows['y_pred'].replace(-1,0)
    # print(windows) 
    
    # Get metrics in confusion matrix
    try:
        # precision = precision_score(windows['y_true'], windows['y_pred'], labels=[0,1])
        # recall = recall_score(windows['y_true'], windows['y_pred'], labels=[0,1])
        # f1 = f1_score(windows['y_true'], windows['y_pred'], labels=[0,1])
        tn, fp, fn, tp = confusion_matrix(windows['y_true'], windows['y_pred'], labels=[0,1]).ravel()
        precision = tp/(tp+fp) if (tp+fp)>0 else 0
        recall = tp/(tp+fn) if (tp+fn)>0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall)>0 else 0
        specificity = tn/(tn+fp) if (tn+fp)>0 else 0
        precision_negative = tn/(tn+fn) if (tn+fn)>0 else 0
    except ValueError:
        pass


    # Get delay, average delay and list of correct predictions
    if tp > 0:
        delay = sum([row[1] - row[0]
                    for row in windows[['ground_truth_init','init', 'y_true', 'y_pred']].values
                    if row[2]==1 if row[3]==1
                ])
        avg_delay = (delay/tp)/window_size
        predicted_true = list(windows[(windows['y_true']==1) & (windows['y_pred']==1)]['init'])
    

    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Specificity": specificity,
        "Precision_negative": precision_negative,
        "Delay": avg_delay,
        "Correct_Predictions": predicted_true,
        "Support_correct": len(predicted_true),
        "tests": len(drifts)+len(not_drifts),
        "Mean_test_per_drift": (len(drifts)+len(not_drifts))/len(resp),
        # # "Not_Drifts_Found": not_drifts,
        "Drifts_Found": drifts,
        "Resp": resp
    }
    

#################################################################################################### 
    # # Initialize metrics with 0
    # precision = 0
    # recall = 0
    # specificity = 0
    # precision_negative = 0
    # tp = 0
    # tn = 0
    # delay = 0
    # avg_delay = 0
    # resp_ = resp.copy()
    # resp_2 = resp.copy()
    # predicted_true = [0 for x in resp_]
    
    # # Transforms the window_size into a vector with the size of the drifts found
    # if isinstance(window_size, int):
    #     window_size_drifts = np.repeat(window_size, len(drifts))
    
    # # Iterates over all drifts found and to all ground truths drifts
    # for i in range(len(drifts)):    
    #     for j in range(len(resp_)):            
    #         # check if the drift found is within 2 * window_size after its true index
    #         # or if is within 0.8 window_size before its true index (test had at least 20% drift)
    #         # if (-0.8 * window_size_drifts[i] < drifts[i] - resp_[j] <= 2 * window_size_drifts[i]):
    #         if (0 <= drifts[i] - resp_[j] <= 2 * window_size_drifts[i]):
    #             if verbose:
    #                 print((drifts[i], drifts[i] + window_size_drifts[i], resp_[j]))
                
    #             # drift found correctly
    #             delay += drifts[i] - resp_[j]
    #             tp += 1
    #             resp_[j] = np.inf
    #             predicted_true[j] = 1
    #             break
          
            
    # # Transforms the window_size into a vector with the size of the drifts found
    # if isinstance(window_size, int):
    #     window_size_not_drift = np.repeat(window_size, len(not_drifts))
    # # Iterates over all not drifts found and to all ground truths drifts
    # for i in range(len(not_drifts)):    
    #     for j in range(len(resp_2)):            
    #         # check if the drift found is within 2 * window_size after its true index
    #         # or if is within 0,8 window_size before its true index (test had at least 20% drift)
    #         # if not (-0.8 * window_size_not_drift[i] <= not_drifts[i] - resp_2[j] <= 2 * window_size_not_drift[i]):
    #         if not (0 <= not_drifts[i] - resp_2[j] <= 2 * window_size_not_drift[i]):
    #             if verbose:
    #                 print((not_drifts[i], not_drifts[i] + window_size_not_drift[i], resp_2[j]))
                
    #             # not drift found correctly
    #             tn += 1
    #             break
    
    # # Get metrics
    # if len(drifts) > 0:
    #     precision = tp/len(drifts)   

    # if len(resp_) > 0:
    #     recall = tp/len(resp_)
        
    # try:
    #     f1 = scipy.stats.hmean([precision, recall])
    # except ValueError:
    #     f1 = 0.0
        
    # if len(not_drifts) > 0:
    #     specificity = tn/(tn+len(drifts)-tp)
    #     precision_negative = tn/(tn+len(resp_)-tp)
    
    # # Get delay
    # if tp > 0:
    #     avg_delay = (delay/tp)/window_size_drifts[0]
        

    # return {
    #     "Precision": precision,
    #     "Recall": recall,
    #     "F1": f1,
    #     "Specificity": specificity,
    #     "Precision_negative": precision_negative,
    #     "Delay": avg_delay,
    #     "Correct_Predictions": predicted_true,
    #     "Support_correct": sum(predicted_true),
    #     "Support": len(drifts)+len(not_drifts),
    #     "Mean_test_per_drift": (len(drifts)+len(not_drifts))/len(resp),
    #     # "Not_Drifts_Found": not_drifts,
    #     "Drifts_Found": drifts,
    #     "Resp": resp
    # }


#################################################################################################### 
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

# def detect_concept_drift(
#     df, var_ref, rolling_window=3, std_tolerance=3, min_tol=0.0025, verbose=False
# ):
#     """
#         Performs the drift detection algorithm. Based on the features measured from 
#         tracking the evolution of the trace clustering over the windows, estimate 
#         tolerance boundaries to detect a drift when the feature value measured at the
#         current index lies outside of the boundaries. 

#         Parameters:
#         ------------
#             df (pd.DataFrame): DataFrame of features from tracking the trace clusterings
#             var_ref (str): Name of the feature to apply the algorithm. It has to be present
#                 in the df.columns
#             rolling_window (int): The number of rolling windows to consider when estimating
#                 the tolerance boundaries and detecting the drifts. It smooths the analyzed 
#                 feature to reduce false positive detections due to noise. 
#             std_tolerance (int): Number of times the rolling standard deviation used to 
#                 the tolerance boundaries. A higher value provides higher tolerance and 
#                 lower sensitivity to drifts.
#             min_tol (float [0.0 - 1.0]): Number of times the rolling average to calculate a
#                 minimum tolerance boundaries. Useful for not detecting false positives in more 
#                 stable feature values when the tolerance could be too little.
#             verbose (bool): Whether to print the index of drifts as they are found
        
#         Returns:
#         ---------
#             list: List of index of the detected drifts
#             dict: Dictionary with the rolling average, lowers and uppers boundaries 
#                 to assist plotting
#     """
#     # Initialize variables
#     window_buffer = []
#     drifts = []
#     not_drifts = []
#     mean = None
#     std = None

#     # Lists to keep the rolling average and the lower and upper boundaries to 
#     # be returned at the end of the execution of this method to support the plots
    
#     lowers = []
#     uppers = []
#     means = df[var_ref].rolling(window=rolling_window).mean().values.tolist()
    

#     # Iterates over the values
#     for i, row in df.iterrows():

#         # If the rolling window is of the desired size
#         if len(window_buffer) < rolling_window:
#             window_buffer.append(row[var_ref])
#             lowers.append(np.nan)
#             uppers.append(np.nan)

#         else:
#             if mean is not None:
#                 # To avoid errors in multiplication with 0
#                 if mean == 0:
#                     mean == 1

#                 # Calculates tolerance boundaries considering the rolling mean and std
#                 expected_lower = min(
#                     mean - (std_tolerance * std),
#                     (1 - min_tol) * mean
#                 )
#                 expected_upper = max(
#                     mean + (std_tolerance * std),
#                     (1 + min_tol) * mean
#                 )

#                 # Adds into the list to return in the end
#                 lowers.append(expected_lower)
#                 uppers.append(expected_upper)

#                 # Checks whether the current value lies outside the tolerance boundaries
                
#                 if expected_lower > row[var_ref] or row[var_ref] > expected_upper:
#                     if verbose:
#                         print(i, expected_lower, expected_upper, row[var_ref])
#                     drifts.append(i)

#                     window_buffer = []
#                 else:
#                     not_drifts.append(i)
#             else:
#                 lowers.append(np.nan)
#                 uppers.append(np.nan)

#             if len(window_buffer) > 0:
#                 window_buffer.pop(0)

#             window_buffer.append(row[var_ref])
            
#             # if mean is not None:
#             #     if expected_lower > np.mean(window_buffer) or np.mean(window_buffer) > expected_upper:
#             #         if verbose:
#             #             print(i, expected_lower, expected_upper, np.mean(window_buffer), row[var_ref])
#             #         drifts.append(i)
#             #         window_buffer = []
                    
#             if i in drifts:
#                 mean = None
#                 std = None
#             else:
#                 mean = np.mean(window_buffer)
#                 std = np.std(window_buffer)

#     return drifts, not_drifts, {"lowers": lowers, "uppers": uppers, "means": means}