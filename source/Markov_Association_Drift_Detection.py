# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 00:56:39 2021

@author: AC-Meira
"""
import pandas as pd
import numpy as np
from source import log_representation as lr
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


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
                if (window['init'] <= ground_truth <= window['end']):
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
        "Support_correct": len(predicted_true),
        "tests": len(drifts)+len(not_drifts),
        "Mean_test_per_drift": (len(drifts)+len(not_drifts))/len(resp),
        # # "Not_Drifts_Found": not_drifts,
        "Drifts_Found": drifts,
        "Resp": resp
    }



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






def run_window_measures(
        sequences=[]
        , window_size=200
        , sliding_step=0
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
    # Create final dictionary
    result = []
    
    # Check if sequences is not null
    if len(sequences) > 0:
        pass
    else:
        return []

    # Setting looping to simulate streaming of data (online)
    if sliding_step>0:
        loop = range(0, len(sequences) - window_size + 1, sliding_step)
    else:
        loop = range(0, len(sequences), window_size)
        
    representation_past = pd.DataFrame()
    
    measures = ['markov_support','markov_confidence','markov_leverage'#, 'markov_conviction'
        , 'Causality', 'Parallel', 'Choice', 'Direct_succession', 'Involved_loop', 'Self_loop'
        #, 'assrules_support', 'assrules_confidence', 'assrules_leverage', 'assrules_conviction'
          ] 

    # Loop
    for i in loop:
        # print("Window from ", i, " to ", i+window_size)
        
        # Start dictionary with measures from each loop
        dict_temp = {"i": i}
        
        # Get sequences in loop window
        sequences_window = sequences[i:i+window_size]
        
        # Add info about sequences in dict
        dict_temp["n_variants"] = len(sequences_window.unique())
        
        # --------------------------------------------#
        # Get Markov Transition Matrix Representation #
        # --------------------------------------------#
        representation_curr = lr.get_assrules_markov_alpha_representation(sequences_window)
        
        # Get difference from past representation (if it exists)
        if representation_past.shape[0] > 0:
            difference_trans = representation_curr.subtract(representation_past, fill_value=0)
            dict_temp.update((abs(difference_trans)+1).mean())
            # dict_temp["count_nonzero_trans_prob"] = np.count_nonzero(difference)
        representation_past = representation_curr
        
    
        # ----------------------------
        # Make combinations of measures
        # ----------------------------
        if i > 0:
            df_temp = pd.DataFrame([dict_temp])[measures]#.astype(float)
            # df_temp = (df_temp-df_temp.min())/(df_temp.max()-df_temp.min())
            # df_temp = (df_temp-df_temp.mean())/df_temp.std()
            dict_temp["diff_total"] = df_temp.values.mean()
            
            # dict_temp["conviction_ass_markov"] = max(dict_temp['conviction_trans_prob'], dict_temp['conviction_ass_rules'])
            # dict_temp["confidence_ass_markov"] = max(dict_temp['confidence_trans_prob'], dict_temp['confidence_ass_rules'])
            # dict_temp["support_ass_markov"] = max(dict_temp['confidence_trans_prob'], dict_temp['support_ass_rules'])
        # else:
        #     dict_temp["conviction_ass_markov"] = 0
        #     dict_temp["confidence_ass_markov"] = 0
        # ----------------------------
        # Calculate validation indexes
        # ----------------------------
        # if max(counts) >= 2 and len(values) > 1:
        # r.update(get_validation_indexes(X, y_pred))
        
        # Add current iteration to final result
        result.append(dict_temp)

    # Turn into dataframe
    result = pd.DataFrame(result).set_index("i")
    # result.fillna(0, inplace=True)

    # # Calculate time-dependent features
    # measures = [compare_clusterings(resp[i], resp[i + 1]) for i in range(len(resp) - 1)]
    # measures_df = pd.DataFrame(measures).set_index("i")
    # measures_df.fillna(0, inplace=True)

    # # Merge results
    # all_metrics = run_df.join(measures_df)
    # # all_metrics.index += all_metrics.index[1]

    return result