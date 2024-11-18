import pandas as pd
import numpy as np
from typing import Union, Dict, List, Tuple
import torch
import torchmetrics
from ...dataloading.dataloader import DataLoader

# all_supported_metrics = [m for m in torchmetrics.__dict__['__all__'] if 'functional' not in m]
supported_regression_metrics = torchmetrics.regression.__dict__['__all__']
supported_classification_metrics = torchmetrics.classification.__dict__['__all__']
all_supported_metrics = supported_regression_metrics + supported_classification_metrics

metrics_higher_better_regression = ['CosineSimilarity', 'ExplainedVariance', 'R2Score', 'SpearmanCorrCoef', 
                                    'ConcordanceCorrCoef','KendallRankCorrCoef', 'PearsonCorrCoef']
metrics_higher_better_classification = ['BinaryAccuracy', 'MulticlassAccuracy', 'MultilabelAccuracy','BinaryCohenKappa',
                                        'MulticlassCohenKappa','Dice','MulticlassExactMatch','MultilabelExactMatch',
                                        'BinaryF1Score','BinaryFBetaScore','MulticlassF1Score','MulticlassFBetaScore',
                                        'MultilabelF1Score','MultilabelFBetaScore','BinaryJaccardIndex','MulticlassJaccardIndex',
                                        'MultilabelJaccardIndex','BinaryMatthewsCorrCoef','MulticlassMatthewsCorrCoef',
                                        'MultilabelMatthewsCorrCoef','BinaryPrecision','BinaryRecall','MulticlassPrecision',
                                        'MulticlassRecall','MultilabelPrecision','MultilabelRecall','MultilabelRankingAveragePrecision',
                                        'Accuracy','AUROC','BinaryAUROC','MulticlassAUROC','MultilabelAUROC','AveragePrecision',
                                        'BinaryAveragePrecision','MulticlassAveragePrecision','MultilabelAveragePrecision',
                                        'CohenKappa','F1Score','FBetaScore','ExactMatch','JaccardIndex','MatthewsCorrCoef',
                                        'Precision','Recall','BinaryRecallAtFixedPrecision','MulticlassRecallAtFixedPrecision',
                                        'MultilabelRecallAtFixedPrecision','ROC','BinaryROC','MulticlassROC','MultilabelROC',
                                        'BinarySpecificity','MulticlassSpecificity','MultilabelSpecificity','Specificity',
                                        'BinarySpecificityAtSensitivity','MulticlassSpecificityAtSensitivity',
                                        'MultilabelSpecificityAtSensitivity','BinaryPrecisionAtFixedRecall',
                                        'MulticlassPrecisionAtFixedRecall','MultilabelPrecisionAtFixedRecall',
                                        'PrecisionAtFixedRecall','RecallAtFixedPrecision']
metrics_lower_better_regression = ['KLDivergence', 'LogCoshError','MeanSquaredLogError','MeanAbsoluteError',
                                   'MeanAbsolutePercentageError','MinkowskiDistance','MeanSquaredError',
                                   'RelativeSquaredError','SymmetricMeanAbsolutePercentageError','WeightedMeanAbsolutePercentageError']

metrics_lower_better_classification = ['BinaryCalibrationError', 'MulticlassCalibrationError', 'BinaryFairness',
                                       'BinaryGroupStatRates', 'BinaryHammingDistance','MulticlassHammingDistance',
                                       'MultilabelHammingDistance','BinaryHingeLoss','MulticlassHingeLoss',
                                       'MultilabelCoverageError','MultilabelRankingLoss','CalibrationError',
                                       'HammingDistance','HingeLoss','TweedieDevianceScore']
metrics_non_directional = ['BinaryConfusionMatrix','ConfusionMatrix', 'MulticlassConfusionMatrix',
                           'MultilabelConfusionMatrix','PrecisionRecallCurve',
                           'BinaryPrecisionRecallCurve',
                           'MulticlassPrecisionRecallCurve',
                           'MultilabelPrecisionRecallCurve',
                           'BinaryStatScores',
                           'MulticlassStatScores',
                           'MultilabelStatScores',
                           'StatScores']

def _initialize_metrics(metrics_name: str, **kwargs) -> torchmetrics:
    '''wrapper function for configuring a torchmetrics metrics module
    
       Authors(s): dliang1122@gmail.com

       init arg
       ----
       metrics_name (str): metrics name
       kwargs (kwargs): use kwargs to specify metrics parameters
                        e.g get_metrics(metrics_name = 'R2Score', num_outputs = 1, adjusted = 0 ....)

       Return
       ---------
       torchmetrics
    '''
    if metrics_name not in all_supported_metrics:
        raise ValueError(f'Metrics not supported. Supported metrics: {all_supported_metrics}')

    return eval(f'torchmetrics.{metrics_name}')(**kwargs)


def _metrics_collection(*args: str, device: str = 'cpu', **kwargs: dict) -> torchmetrics.MetricCollection:
    '''wrapper function for combining torchmetrics modules into a colleciton for making multi-metrics calculation
    
        Authors(s): dliang1122@gmail.com

       init arg
       ----
       arg (str): metrics name(s); specifying metrics this way will enforce default settings for metrics
                  e.g metrics_collection('MeanAbsoluteError', 'MeanSquaredError')
       kwargs (keyword args: dict): use metrics name as argument and dictionary for metrics parameters
                      e.g. metrics_collection(R2Score = {'num_outputs': 1, 'adjusted': 0}, 
                                              MeanSquaredError = {'squared': False})

       Return
       ---------
       torchmetrics.MetricCollection
    '''
    metrics_list = []

    for name in args:
        metrics = _initialize_metrics(metrics_name = name).to(device)
        metrics_list.append(metrics)

    for name, settings in kwargs.items():
        metrics = _initialize_metrics(metrics_name = name, **settings).to(device)
        metrics_list.append(metrics)

    return torchmetrics.MetricCollection(metrics_list)


def calc_metrics(model: torch.nn.Module,
                 dataloader: DataLoader,
                 metrics: Dict[str, dict]| List[str],
                 device: str = 'cpu'):
    '''function for calculating metrics; allow multiple metrics'''

    # force list if string
    if isinstance(metrics, str):
        metrics = [metrics]
    
    # initiate metrics
    if isinstance(metrics, list):
        metrics = _metrics_collection(*metrics, device = device)
    
    elif isinstance(metrics, dict):
        metrics = _metrics_collection(**metrics, device = device)

    # reset dataloader batch iteration
    dataloader.reset()

    with torch.no_grad(): # turn off gradient
        # set model to evaluation mode to avoid 
        model.eval()

        for y, X in dataloader:
            y = y.to(device)

            if isinstance(X, (list, tuple)):
                X = tuple(x.to(device) for x in X)
            else:
                X.to(device)

            pred = model(X)

            metrics(pred, y)

    # aggregate result from all batches
    result = {k:v.item() for k,v in metrics.compute().items()}

    # clear accumulated result for next calculation
    metrics.reset()

    return result