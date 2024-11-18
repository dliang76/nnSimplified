import pandas as pd
import numpy as np
from typing import Union, Dict, List, Tuple
import torch
from statistics import mean
import scipy.stats

def get_confusion_matrix(truth: list, predictions: list) -> pd.DataFrame:
    ''' Calculate confusion matrix given truth and predictions

        Author(s): dliang1122@gmail.com

        Args:
        ----------
        truth (list): list of true values
        predictions (list): list of predicted values

        Return:
        ---------
        pandas dataframe: confusion matrix in tabular form

    '''
    
    df = pd.DataFrame({'label':truth, 'prediction':predictions})
    
    confusion_matrix = df.groupby(['label', 'prediction'])\
                         .agg(count = ('prediction', len))\
                         .reset_index()\
                         .pivot_table(index = 'label',
                                      columns = 'prediction', 
                                      values = 'count',
                                      sort = True)

    ### fill in missing rows or columns
    # find all possible values
    if isinstance(truth, torch.Tensor):
        truth_set = set(truth.unique().numpy())
    else:
        truth_set = set(truth)

    if isinstance(predictions, torch.Tensor):
        predictions_set = set(predictions.unique().numpy())
    else:
        predictions_set = set(predictions)

    all_possible_values = truth_set | predictions_set
    # add missing columns/rows to confusion matrix
    confusion_matrix = confusion_matrix.reindex(list(confusion_matrix.columns) + list(all_possible_values - set(confusion_matrix.columns)), axis = 1)\
                                      .reindex(list(confusion_matrix.index) + list(all_possible_values - set(confusion_matrix.index)), axis = 0).fillna(0)
    # sort col and row names
    confusion_matrix = confusion_matrix.loc[sorted(all_possible_values), sorted(all_possible_values)]
    
    confusion_matrix.index.name = None
    confusion_matrix.columns.name = None

    return confusion_matrix


def get_binary_cm_elements(cm: np.ndarray) -> Dict[str, float]:
    ''' simple function to extract elements of confusion matrix with format array([tn, fp],[fn, tp])

        Author(s): dliang1122@gmail.com

        Arg
        -----
        cm (np.ndarray): confusion matrix in numpy array form

        Return
        -----
        dict
    '''

    # true positive
    tp = cm[1, 1]
    # true negative
    tn = cm[0, 0]
    # false positive
    fp = cm[0, 1]
    # false negative
    fn = cm[1, 0]

    # predicted positive
    pp = tp + fp
    # predicted negative
    pn = fn + tn
    # positive
    p = tp + fn
    # negative
    n = tn + fp

    return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'pp': pp, 'pn': pn, 'p': p, 'n': n}


def calc_metric_binary(confusion_matrix: Union[np.ndarray, pd.DataFrame],
                       target_class: Union[str, int] = 1,
                       metric_name: str = 'all') -> Union[float, Dict[str, float]]:
    ''' function for calculating metrics for binary classifier from confusion matrix

        Author(s): dliang1122@gmail.com

        Arg
        -----
        confusion_matrix (np.ndarray or pd.DataFrame): confusion matrix
        target_class (str or int): specify positive class. For confusion matrix innp array format, this is the position of the positive class.
        metric_name (str): metric to return. If 'all', return all metrics in a dict format

        Return
        -----
        float, dict
    '''
    if isinstance(confusion_matrix, pd.DataFrame):
        # convert pandas to numpy to speed things up
        cm = confusion_matrix.values

        if confusion_matrix.columns[1] != target_class:
            # reverse column order and index order so that the array has the format array([tn, fp],[fn, tp])
            cm = cm[[1,0]][:,[1,0]]

    elif isinstance(confusion_matrix, np.ndarray):
        # avoid modifying the original
        cm = confusion_matrix.copy()

        if target_class == 0:
            # reverse column order and index order so that the array has the format array([tn, fp],[fn, tp])
            cm = cm[[1,0]][:,[1,0]]

    else:
        raise ValueError('Unrecognized confusion matrix format! Must be either a pandas dataframe or a 2D numpy array!')

    # get total counts
    total_count = cm.sum()

    # extract elements of confusion matrix
    cm_elements = get_binary_cm_elements(cm = cm)

    #### calculate precision, recall and F1
    # get true positive, true negative, predicted positive, positive
    tp, pp, p = cm_elements['tp'], cm_elements['pp'], cm_elements['p']

    if tp != 0:
        # normal calculation performance metrics
        precision = tp/pp
        recall = tp/p
        f1 = (2 * recall * precision/(recall + precision))
        note = ''

    # the following are edge cases
    elif p == 0 :
        note = 'No Positives Exist!'

        if pp == 0:
            # every thing is correctly predicted as negative so all metric is 1
            # https://github.com/dice-group/gerbil/wiki/Precision,-Recall-and-F1-measure
            precision, recall, f1 = 1., 1., 1.
        else:
            precision, recall, f1 = 0., 0., 0.
    else:
        if pp == 0:
            note = 'No Positives Predicted!'
            precision, recall, f1 = 1., 0., 0.   # for precision continuity, set precision to 1 if no predicted positive
        else:
            note = 'Reverse Predictions Detected'
            precision, recall, f1 = 0., 0., 0.

    # calculate accuracy
    tn = cm_elements['tn']
    accuracy = (tp + tn) / total_count

    # false positive rate
    fp, n = cm_elements['fp'], cm_elements['n']
    fpr = fp / n

    # get cohen's kappa
    kappa, ci = calc_cohens_kappa(cm)

    # get IAM
    iam = calc_IAM(cm)

    metrics = {
                'Accuracy': accuracy,
                'F1': f1,
                'Precision': precision,
                'Recall': recall,
                'FPR': fpr,
                'PSI': calc_psi(cm),
                'CohensKappa': kappa,
                'CohensKappa_0.95CI_L': ci[0],
                'CohensKappa_0.95CI_U': ci[1],
                'IAM': iam,
                'Note': note
              }

    if metric_name == 'all':
        return metrics
    else:
        return metrics[metric_name]


def calc_metric_multiclass(confusion_matrix: Union[np.ndarray, pd.DataFrame],
                           metric_name: str = 'all') -> Union[float, Dict[str, float]]:
    ''' Function for calculating metrics for multi-class classification from confusion matrix

        Author(s): dliang1122@gmail.com

        Arg
        -----
        confusion_matrix (np.ndarray or pd.DataFrame): confusion matrix
        metric_name (str): metric to return. If 'all', return all metrics in a dict format

        Return
        -----
        float, dict
    '''
    if isinstance(confusion_matrix, pd.DataFrame):
        # convert to numpy to speed things up
        cm = confusion_matrix.values

    elif isinstance(confusion_matrix, np.ndarray):
        # avoid modifying the original
        cm = confusion_matrix.copy()

    else:
        raise ValueError('Unrecognized confusion matrix format! Must be either a pandas dataframe or a 2D numpy array!')

    # get total counts
    total_count = cm.sum()

    # get # of True positive (TPs), Predicted Positive (PPs), Positive (Ps) for each class
    TPs = np.diag(cm)
    PPs = cm.sum(axis=0)
    Ps = cm.sum(axis=1)

    ####  get precision and recall for each class
    with np.errstate(divide='ignore', invalid='ignore'):
        # ignore divide by 0 warning; will correct these edge cases later
        precisions = TPs/PPs
        recalls = TPs/Ps

    # correct values for edge cases
    cond1 = (PPs == 0) & (Ps == 0)
    cond2 = (PPs != 0) & (Ps == 0)
    cond3 = (PPs == 0) & (Ps != 0)

    precisions = np.where(cond3, 0 , np.where(cond1, 1 , precisions))
    recalls = np.where(cond2, 0 , np.where(cond1, 1 , recalls))

    # calcualte f1 values
    with np.errstate(divide='ignore', invalid='ignore'):
        f1s = np.where((precisions + recalls) > 0, ((2 * precisions * recalls)/(precisions + recalls)), 0)

    # get cohen's kappa
    kappa, ci = calc_cohens_kappa(cm)

    # get IAM
    iam = calc_IAM(cm)

    # use weighted metrics
    metrics = {
                'Accuracy': TPs.sum()/total_count,
                'MacroF1': f1s.mean(),
                'MacroPrecision': precisions.mean(),
                'MacroRecall': recalls.mean(),
                'WeightedF1': (f1s * Ps).sum()/Ps.sum(),
                'WeightedPrecision': (precisions * Ps).sum()/Ps.sum(),
                'WeightedRecall': (recalls * Ps).sum()/Ps.sum(),
                'r^2': calc_r_squared(cm),
                'PSI': calc_psi(cm),
                'CohensKappa': kappa,
                'CohensKappa_0.95CI_L': ci[0],
                'CohensKappa_0.95CI_U': ci[1],
                'IAM': iam
              }

    if metric_name == 'all':
        return metrics
    else:
        return metrics[metric_name]


def calc_cohens_kappa(confusion_matrix: Union[np.ndarray, pd.DataFrame]) -> Tuple[float, List[float]]:
    '''
        Author(s): kumarsajal49@gmail.com
    '''

    """calculate Cohen's Kappa"""
    if isinstance(confusion_matrix, pd.DataFrame):
        # convert to numpy to speed things up
        cm = confusion_matrix.values

    elif isinstance(confusion_matrix, np.ndarray):
        # avoid modifying the original
        cm = confusion_matrix.copy()

    else:
        raise ValueError('Unrecognized confusion matrix format! Must be either a pandas dataframe or a 2D numpy array!')

    # find row-sums
    row_sums = cm.sum(axis=1)

    # find column-sums
    col_sums = cm.sum(axis=0)

    # find total sum of cm
    total_sum = row_sums.sum()

    # proportion of agreements
    pr_agr = 0
    for i in range(len(cm)):
        pr_agr += (cm[i, i] / total_sum)

    # calculate expected values
    pr_exp = 0
    for i in range(len(cm)):
        pr_exp += (row_sums[i] / total_sum) * (col_sums[i] / total_sum)

    # calculate cohen's kappa
    k = (pr_agr - pr_exp) / (1 - pr_exp)

    # find 95% CI
    # we assume that both the number of agreements and disagreements are large (k is approximately standard normal)
    # calculate standard error
    sek = np.sqrt((pr_agr * (1 - pr_agr))/(total_sum * np.square(1 - pr_exp)))

    # CI
    ci = [k - (1.96*sek), k + (1.96*sek)]

    return k, ci


def calc_IAM(confusion_matrix):
    """
    Author(s): kumarsajal49@gmail.com

    calculate Imbalance Accuracy Metric
    Mortaz, E. (2020). Imbalance accuracy metric for model selection in multi-class imbalance
    classification problems. Knowl. Based Syst., 210, 106490.
    """

    if isinstance(confusion_matrix, pd.DataFrame):
        # convert to numpy to speed things up
        cm = confusion_matrix.values

    elif isinstance(confusion_matrix, np.ndarray):
        # avoid modifying the original
        cm = confusion_matrix.copy()

    else:
        raise ValueError('Unrecognized confusion matrix format! Must be either a pandas dataframe or a 2D numpy array!')

    # find row-sums
    row_sums = np.array(cm.sum(axis=1))

    # find column-sums
    col_sums = np.array(cm.sum(axis=0))

    # calculate IAM
    coef = 1 / cm.shape[0]
    iam = 0
    for i in range(cm.shape[0]):
        # calculate off diagonal row/column sums
        off_diag_row_sum = row_sums[i] - cm[i, i]
        off_diag_col_sum = col_sums[i] - cm[i, i]

        # calculate iam
        numerator = (cm[i, i] - max(off_diag_row_sum, off_diag_col_sum))
        denominator = max(row_sums[i], col_sums[i])
        iam += ((numerator / denominator) * (coef))

    return iam


def calc_r_squared(confusion_matrix: Union[np.ndarray, pd.DataFrame]) -> float:
    ''' Calculting r^2 from confusion matrix; class labels are mapped to int values (e.g. 0, 1, 2) based on their positions

        Author(s): dliang1122@gmail.com
    '''

    if isinstance(confusion_matrix, pd.DataFrame):
        # convert to numpy to speed things up
        cm = confusion_matrix.values

    elif isinstance(confusion_matrix, np.ndarray):
        # avoid modifying the original
        cm = confusion_matrix.copy()

    else:
        raise ValueError('Unrecognized confusion matrix format! Must be either a pandas dataframe or a 2D numpy array!')

    # get actual values, predicted values and weights (counts) from confusion matrix
    actual, predicted = np.where(cm != 0)
    weights = cm[actual, predicted]

    actual_mean = (actual * weights).sum() / weights.sum()
    TSS = (np.square(actual - actual_mean) * weights).sum()
    RSS = (np.square(actual - predicted) * weights).sum()

    return 1 - RSS/TSS


def calc_psi(confusion_matrix: Union[np.ndarray, pd.DataFrame]):
    ''' Calculting PSI from confusion matrix; class labels are mapped to int values (e.g. 0, 1, 2) based on their positions

        Author(s): dliang1122@gmail.com
    '''

    if isinstance(confusion_matrix, pd.DataFrame):
        # convert to numpy to speed things up
        cm = confusion_matrix.values

    elif isinstance(confusion_matrix, np.ndarray):
        # avoid modifying the original
        cm = confusion_matrix.copy()

    else:
        raise ValueError('Unrecognized confusion matrix format! Must be either a pandas dataframe or a 2D numpy array!')

    # get truth bins
    actual = cm.sum(axis = 1)
    actual_percent = actual/actual.sum()

    # get prediction bins
    predicted = cm.sum(axis = 0)
    predicted_percent = predicted/predicted.sum()

    # calculate index for each bin and PSI
    with np.errstate(divide='ignore', invalid='ignore'):
        index = (actual_percent - predicted_percent) * np.log(actual_percent/predicted_percent)
    psi = index.sum()

    return psi


def get_singleClass_performance(confusion_matrix: pd.DataFrame):
    ''' function for calculating singel class performance from a multiclass confusion matrix
    
        Author(s): dliang1122@gmail.com
    '''
    # convert to np array to speed things up
    cm = confusion_matrix.values

    performance = {}
    for i, c in enumerate(confusion_matrix.columns):
        # true positive
        tp = cm[i, i]
        # true negative
        tn = np.delete(np.delete(cm, i, axis = 0), i, axis = 1).sum()
        # false positive
        fp = np.delete(cm[:, i], i).sum()
        # false negative
        fn = np.delete(cm[i, :], i).sum()

        # construct binary confusion matrix
        binary_cm = np.array([[tn, fp],[fn,tp]])

        performance[c] = calc_metric_binary(confusion_matrix = binary_cm, target_class = 1)

    return performance


def get_optimum_binary_prob_cutoff(truth: List[int], pred_prob: List[float], target_metric: str, prob_step_size: float = 0.01):
    ''' Function for search the optimal probability cutoff for binary classifiers

        Author(s): dliang1122@gmail.com

        Args:
        ---------
        truth (list of int): true values (0 or 1)
        pred_prob (list of float): predicted probability-like values (between 0 and 1)
        target_metric (str): name of the metric used for optimization
        prob_step_size (float): step size (granularity) used for probability cutoff 
    '''
    # get metrics for each prob threshold
    cutoff_metrics = pd.DataFrame()

    # loop through probability cutoff and collect metrics
    for i in range(int(1/prob_step_size)):
        prob_cutoff = i * prob_step_size
        pred = (pred_prob >= prob_cutoff).int()
        cm = get_confusion_matrix(truth, pred)
        cutoff_metric = pd.DataFrame({k:[v] for k,v in calc_metric_binary(cm).items()})
        cutoff_metric['cutoff'] = prob_cutoff
        cutoff_metrics = cutoff_metrics.append(cutoff_metric)

    cutoff_metrics.reset_index(drop = True, inplace = True)

    # find optimum cutoff that maximize the target matric
    optimum_cutoff = cutoff_metrics.loc[cutoff_metrics[target_metric].idxmax(),'cutoff']

    return optimum_cutoff


def baseline_metrics(pred, y, label, train_mode = 'multiclass'):
    '''
        Author(s): sam88ir@gmail.com, kumarsajal49@gmail.com
    '''
    pred_set = set(pred[label])
    y_set = set(y[label])
    all_possible_values = pred_set | y_set
    if "ERROR" in all_possible_values:
        all_possible_values.remove("ERROR")

    pred = pred[label].value_counts()
    y = y[label].value_counts()
    for i in all_possible_values:
        if i not in pred.index:
            pred.append(pd.Series([0], index=[i]))
        if i not in y.index:
            y.append(pd.Series([0], index=[i]))

    count_pred  = [i for i in pred[all_possible_values]]
    count_y     = [i for i in y[all_possible_values]]
    freq_pred = [c/sum(count_pred) for c in count_pred]
    freq_y    = [c/sum(count_y) for c in count_y]

    confusion_matrix = pd.DataFrame(columns = [i for i in range(len(freq_pred))],index = [i for i in range(len(freq_y))])

    for i,iv in enumerate(freq_y):
        for c,cv in enumerate(freq_pred):

            confusion_matrix.iat[i,c] = iv*cv

    confusion_matrix = confusion_matrix.astype(float)
    if train_mode == 'binary':
        metrics_dict = calc_metric_binary(confusion_matrix = confusion_matrix, 
                                                   target_class = 1,
                                                   metric_name = 'all')
    else:
        metrics_dict = calc_metric_multiclass(confusion_matrix = confusion_matrix, metric_name = 'all')

    return metrics_dict


def calc_pvalue(X_test, y_test, model, data, label, data_test = None ,nsample = 1000, samplesize = 10000, target_metric = "WeightedF1", population_metric_value = None,train_mode = "multiclass"):
    '''
        Author(s): sam88ir@gmail.com
    '''
    m = len(y_test)
    target_metric_list = []

    pred_raw = model(X_test)

    prediction = pd.DataFrame()
    prediction["pred"] = torch.max(pred_raw, dim = 1).indices
    prediction["truth"] = y_test

    for i in range(nsample):
        
        randlist = pd.DataFrame(index=np.random.randint(m, size=samplesize))
        sample = prediction.merge(randlist, left_index=True, right_index=True, how='right')
        confusion_matrix = get_confusion_matrix(sample["truth"], sample["pred"])
        if train_mode == 'binary':
            metrics_dict = calc_metric_binary(confusion_matrix = confusion_matrix, 
                                                   target_class = 1,
                                                   metric_name = 'all')
        else:
            metrics_dict = calc_metric_multiclass(confusion_matrix = confusion_matrix, metric_name = 'all')

        target_metric_list.append(metrics_dict[target_metric])

    mean_metric_value = mean(target_metric_list)

    if not population_metric_value:
        if data_test is not None:
            population_metrics = baseline_metrics(pred = data, y = data_test,
                                                          label = label, train_mode = train_mode)
            population_metric_value = population_metrics[target_metric]
        else:
            raise ValueError("Either evaluation set or population metric should be available to calculate p value")

    zscore = (population_metric_value-mean_metric_value)/((mean_metric_value*(1-mean_metric_value)/nsample)**0.5)
    p_value = scipy.stats.norm.sf(abs(zscore))

    return p_value
