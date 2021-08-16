# import sys
# sys.path.append('D:\OneDrive\_Cloud Disk\Projetos de programação\Python Projects\Git\GitHub')
# from ERMiner import sequential_pattern_mining
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
import itertools
from functools import reduce

def get_traces_as_tokens(traces_df, col_ref="Activity"):
    """
        Groups activities executions into traces
        as a string of tokens
        
        Ex:
             Activity |  Timestamp
            -----------------------
            START     | 2019-09-01
            A         | 2019-09-01
            B         | 2019-09-01
            C         | 2019-09-01
            END-A     | 2019-09-01
            
       into: "START A B C END-A"  
    """
    return traces_df.groupby("Trace_order")[col_ref].apply(
        lambda x: " ".join(x.values)
    )

def get_count_representation(tokens, binary=True, tfidf=False, ngram_range=(1, 1)):
    """
        Generic method to represent traces as vectors by counting
        activities or transitions.

        Parameters:
        ------------
            tokens (pd.Series): Trace represented as tokens (series of strings)
            binary (bool): Count binary or frequency
            tfidf (bool): Use tf-idf to normalize frequency
            ngram_range (tuple): Range of ngrams to obtain representation
    """
    
    if tfidf:
        cv = TfidfVectorizer(
            norm = None,
            smooth_idf = False,
            tokenizer=str.split, 
            lowercase=False,
            use_idf=True,
            ngram_range=ngram_range,
            min_df=0,
            max_df=1.0
        )
    else:
        cv = CountVectorizer(
            tokenizer=str.split, 
            lowercase=False,
            ngram_range=ngram_range,
            min_df=0,
            max_df=1.0,
            binary=binary
        )
    
    cv_result = cv.fit_transform(tokens)
    
    return pd.DataFrame(
        cv_result.todense(), 
        columns=cv.get_feature_names()
    )

# # # # # # # #
# Transitions #
# # # # # # # #
def get_binary_transitions_representation(tokens):    
    """
        Binary Transistions representation of traces
        (1 or 0 if the transition occur in traces)
    """
    return get_count_representation(tokens, True, False, (2,2))

def get_frequency_transitions_representation(tokens):    
    """
        Frequency Transistions representation of traces
        (# of occurences of a transition on the trace)
    """
    return get_count_representation(tokens, False, False, (2,2))

def get_tfidf_transitions_representation(tokens):    
    """
        TF-IDF Transistions representation of traces
        (frequency of the transition occur in traces 
        weighted by inverse document frequency)
    """
    return get_count_representation(tokens, False, True, (2,2))


# # # # # # 
# Activity #
# # # # # # 
def get_binary_representation(tokens):    
    """
        Binary representation of traces
        (1 or 0 if an activity occur in traces)
    """
    return get_count_representation(tokens, True, False, (1,1))

def get_frequency_representation(tokens):    
    """
        Frequency representation of traces
        (# of times an activity occur in traces)
    """
    return get_count_representation(tokens, False, False, (1,1))

def get_tfidf_representation(tokens):    
    """
        TF-IDF representation of traces
        (frequency of the occurence of activity in traces 
        weighted by inverse document frequency)
    """
    return get_count_representation(tokens, False, True, (1,1))


# # # # # # # # # # # # # #
# Activity and Transition #
# # # # # # # # # # # # # #
def get_activity_transitions_frequency_representation(tokens):    
    """
        Binary representation of traces
        (1 or 0 if an activity occur in traces)
    """
    return pd.concat([get_frequency_transitions_representation(tokens), get_frequency_representation(tokens)],axis=1)

def get_activity_transitions_binary_representation(tokens):    
    """
        Frequency representation of traces
        (# of times an activity occur in traces)
    """
    
    return pd.concat([get_binary_transitions_representation(tokens), get_binary_representation(tokens)],axis=1)



# # # # # # # # # # #
# Association Rules #
# # # # # # # # # # #
def get_association_rules_representation(tokens): 
    """
        Get association rules for all activities
        (support, confidence, lift, leverage and conviction)
    """
    
    # Prepare datafre - Get tokens as dummies
    tokens_dummies = get_count_representation(tokens, True, False, (1,1))
    tokens_dummies['end_node'] = 1

    # Get frequent itens
    frequent_itemsets = apriori(tokens_dummies, min_support=1e-27, max_len=2, use_colnames=True, verbose=0)

    # Get association rules
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=1e-27)
    
    # Get extras estatistics
    # rules['freq'] = rules['support']*
    # df.to_numpy().sum()
    
    # Prepare final dataframe
    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x))).astype("unicode")
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x))).astype("unicode")
    rules['conviction'] = rules['conviction'].replace([np.inf], 100)
    # frozensets = ["antecedents", "consequents"]
    # rules[frozensets] = [list(x) for x in frozensets]
    rules = rules.sort_values(by=['antecedents', 'consequents'], ascending=[True, True]).reset_index(drop=True).set_index(['antecedents','consequents'])
    
    return rules


# # # # # # # # # # #
# Transition Matrix #
# # # # # # # # # # #
def get_transition_matrix_representation(tokens): 
    """
        Get the probability of each transition
    """
    # Prepare dataframe
    transitions = pd.DataFrame(tokens.str.split()).explode('Activity').rename(columns={"Activity": "antecedents"})
    transitions["consequents"] = transitions.groupby('Trace_order').shift(-1).fillna('end_node')
    
    # Get estatistics
    Matrix = pd.crosstab(transitions["antecedents"], transitions["consequents"], normalize='all').stack().reset_index().rename(columns={0: "support"})
    # Percentual_Transition_Matrix = pd.crosstab(transitions["antecedents"], transitions["consequents"], normalize='all').stack().reset_index().rename(columns={0: "perc"})
    # Matrix = pd.crosstab(transitions["antecedents"], transitions["consequents"], normalize='index').stack().reset_index().rename(columns={0: "support"})
    Matrix = pd.merge(Matrix, Matrix.groupby("antecedents", as_index=False)["support"].sum().rename(columns={'support':'antecedents support'}), on=["antecedents"], how='inner')
    Matrix = pd.merge(Matrix, Matrix.groupby("consequents", as_index=False)["support"].sum().rename(columns={'support':'consequents support'}), on=["consequents"], how='inner')
    Matrix['confidence'] = np.float64(Matrix['support']/Matrix['antecedents support'])
    Matrix['lift'] = np.float64(Matrix['confidence']/Matrix['consequents support'])
    Matrix['leverage'] = np.float64(Matrix['support'] - (Matrix['antecedents support']*Matrix['consequents support']))
    Matrix['conviction'] = np.float64((1 - Matrix['consequents support'])/(1 - Matrix['confidence']))
    
    
    # Prepare final dataframe
    Matrix['conviction'] = Matrix['conviction'].replace([np.inf], 100)
    # dfs_to_merge = [Frequency_Transition_Matrix, Percentual_Transition_Matrix, Probability_Transition_Matrix]
    # Transition_Matrix_Final = reduce(lambda  left,right: pd.merge(left, right, on=['antecedents', 'consequents'], how='outer'), dfs_to_merge).fillna(0)
    Matrix = Matrix.sort_values(by=['antecedents', 'consequents'], ascending=[True, True]).reset_index(drop=True).set_index(['antecedents','consequents'])

    return Matrix
    

# # # # # # # # # # #
# Sequential Rules  #
# # # # # # # # # # #
# def get_sequential_rules_representation(tokens):   
#     spm = sequential_pattern_mining.ERMiner(minsup=0.00001, minconf=0.00001, single_consequent=False)

#     sequential_rules = spm.fit(tokens)
    
#     return rules
    

# # # # # # # # # #
# Extra Functions #
# # # # # # # # # #
def reinverse_tokens(tokens, inv_aliases, ret_string=True):
    """
        Invert aliases back to full activities names
    """
    r = []
    
    if isinstance(tokens, str):
        t = tokens.split()
    else:
        t = tokens
    
    for token in t:
        if token in inv_aliases:
            r.append(inv_aliases[token])
        else:
            r.append(token)
    
    if ret_string:
        return " ".join(r)
    
    return r


