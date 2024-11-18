from typing import List, Union, Dict
import torch
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from ...preprocessing import transform
from ...metrics import calc_metric_binary
from ..helper import convert_cm_counts_to_cm, agg_cm_counts
from ...dataloading.dataloader import DataLoader, DistributedDataLoader
from .baseTrainer import nnTrainer
from ..metrics.regression import get_metrics, metrics_collection

# class nnRegression(nnTrainer):
#     '''Class for setting up a regression trainer for neural net'''
#     def __init__(self, 
#                  model, 
#                  loss_func: Union[str, Dict[str, dict]] = 'MSELoss',
#                  optimizer: Union[str, Dict[str, dict]] = "AdamW",
#                  device: Union[str, torch.device] = 'cpu'):
#         super().__init__(model = model, 
#                          loss_func = loss_func, 
#                          optimizer = optimizer,
#                          device = device)


#     def _calc_epoch_metrics(self, dataloader: Union[DataLoader, DistributedDataLoader], metrics: Dict[str:dict]]| List[str], loss_runtime_arg = {}):
#         '''metrics calculation for binary classifier
#         '''

#         # initiate metrics
#         if isinstance(metrics, str):
#             metrics = get_metrics(metrics)

#         elif isinstance(metrics, list):
#             metrics = metrics_collection(*metrics)
        
#         elif isinstance(metrics, dict):
#             metrics = metrics_collection(**metrics)

#         # reset dataloader batch iteration
#         dataloader.reset()

#         for y, X in dataloader:
#             pred = self.model(X)

#             metrics(pred, y)

#         # aggregate result from all batches
#         result = metrics.compute()

#         # clear accumulated result for next epoch
#         metrics.reset()


#         # track total loss and data weight
#         total_loss = 0
#         total_data_weight = 0

#         # reset dataloader batch iteration
#         dataloader.reset()

#         # loop through batches
#         for y, X in dataloader:

#             # get score, batch_loss etc.
#             score, batch_loss, data_weight_sum = self._calc_batch_loss(y = y, X = X, loss_runtime_arg = loss_runtime_arg, train_mode = False)

#             # accumulate loss and weight sum
#             total_data_weight += data_weight_sum
#             total_loss += batch_loss

#             score = self.score

#             # convert model output score to probabilities
#             probs = self._convert_score_to_proba(score) # predicted probabilities

#             # accumulate confusion matrix counts (truth - prediction counts) for different probability cutoff; for calculating ROC, ROC_AUC and optimal cutoff
#             # e.g. {(0,0) : 10, (0,1): 20, (1,0): 5, (1,1): 30}
#             y = y.int()
#             for k,v in cutoff_cm_dict.items():
#                 encoded_pred = self._convert_proba_to_pred(probability = probs, binary_cutoff = k) # get encoded probabilities; 0 or 1.
#                 cutoff_cm_dict[k] = agg_cm_counts(truth = y,
#                                                   prediction = encoded_pred,
#                                                   prev_cm_count_dict = v)

#         # record test loss
#         metrics = {'test_loss': total_loss / total_data_weight}

#         # collect confusion matrix and metrics for each cutoff
#         cm_dict, cutoff_metrics_dict= {}, {}

#         for k, v in cutoff_cm_dict.items():
#             # convert truth-prediction pair counts to confusion matrix
#             cm_dict[k] = convert_cm_counts_to_cm(cm_count_dict = v, n_classes = len(self._config['label_encoder'].classes_))

#             cutoff_metrics_dict[k] = calc_metric_binary(cm_dict[k]) # get all metrics from confusion matrix

#         # convert cutoff metrics dict to pandas with columns = metrics, index = cutoffs
#         additional_metrics_df = pd.DataFrame(cutoff_metrics_dict).T

#         if self._config['cutoff_tuning']:
#             if self._config['cutoff_tuning_metric'] in ['Accuracy', 'F1', 'Precision', 'Recall', 'PSI', 'CohensKappa']:
#                 self._config['binary_cutoff'] = additional_metrics_df[self._config['cutoff_tuning_metric']].astype(float).idxmax()
#             elif self._config['cutoff_tuning_metric'] in ['FPR']:
#                 self._config['binary_cutoff'] = additional_metrics_df[self._config['cutoff_tuning_metric']].astype(float).idxmin()

#         metrics['prob_cutoff'] = self._config['binary_cutoff'] # store prob cutoff used
#         metrics.update(cutoff_metrics_dict[self._config['binary_cutoff']]) # store additonal metrics

#         # ROC metrics
#         metrics['ROC_AUC'] = np.trapz(additional_metrics_df['Recall'][::-1], additional_metrics_df['FPR'][::-1]) # store ROC AUC; area estimated using trapezoidal rule.
#         metrics['ROC_DATA'] = additional_metrics_df.loc[:,['FPR', 'Recall']] # store ROC data; TPR (Recall) vs FPR
#         metrics['ROC_DATA'].plot('FPR', 'Recall', legend = False, xlabel = 'False Positive Rate', ylabel = 'True Positive Rate', title = 'ROC')
#         plt.show()

#         # store confusion matrix; convert encoded labels back to original labels
#         cm = cm_dict[self._config['binary_cutoff']]
#         cm.columns = self._config['label_encoder'].inverse_transform(cm.columns)
#         cm.index = self._config['label_encoder'].inverse_transform(cm.index)
#         metrics['confusion_matrix']= cm

#         return metrics







# class mlpRegression(nnTrainer):
#     '''Class for generating binary classifier using multi-layer perceptron model
#        loss function is set to be focal loss

#        Authors(s): dliang1122@gmail.com

#        init arg
#        ----
#        label_col (str): name of the label column
#        feature_cols (list): list of feature columns
#        pos_class (list): positive class assignment
#        gamma (float): focusing parameter for Focal Loss; higher -> more weight on hard-to-classify class. If 0, binary crossentropy loss.
#        class_weights (list, torch.FloatTensor): class weights. e.g [1,2] for putting 2x weight on the postive class
#        mlp_hidden_layers (list): hidden layer structure in list form. E.g [256, 128, 64] for 3 hidden layers with 256, 128 and 64 nodes
#        mlp_config (dict): additional configuration for multilayer perceptrons. See modun.nn.model.mlp for the options
#        optimizer (dict): optimizer setting
#        best_metric (str): metric used for deciding which epoch contains the best model; determines which epoch to save as best_epoch
#        cutoff_tuning (bool): whether to tune probability cutoff for predicting positive or negative class; probability cutoff = 0.5 if not using tuning.
#        cutoff_tuning_metric (str): metric used for tuning probability cutoff; if not set, use best_metric
#        device (str or torch.device): computation device (e.g. cpu or cuda)
#     '''
#     def __init__(self,
#                  label_col: str,
#                  feature_cols: List[str],
#                  pos_class: str,
#                  gamma: float = 2.,
#                  class_weights: Union[float, List[float], torch.FloatTensor] = None,
#                  mlp_config: dict = {'hidden_layer_sizes': [],
#                                      'weight_init': "kaiming_normal",
#                                      'activation': "ReLU",
#                                      'batch_normalization': True,
#                                      'batchnorm_momentum': 0.1,
#                                      'dropout_rate': 0.5},
#                  optimizer: Union[str, Dict[str, dict]] = {"AdamW": {'lr': 0.01, 'weight_decay': 0.01}},
#                  best_metric: str = 'ROC_AUC',
#                  cutoff_tuning: bool = True,
#                  cutoff_tuning_metric: str = 'F1',
#                  device: Union[str, torch.device] = 'cpu'):

#         if not pos_class:
#             raise RuntimeError('Must have designated pos_class (positive class) for binary classification!')
#         else:
#             label_values = [pos_class]

#         ### construct trainer
#         super(mlpRegression, self).__init__(label_col = label_col,
#                                         feature_cols = feature_cols,
#                                         label_values = label_values,
#                                         gamma = gamma,
#                                         class_weights = class_weights,
#                                         mlp_config = mlp_config,
#                                         optimizer = optimizer,
#                                         best_metric = best_metric,
#                                         device = device)
#         # whether to tune cutoff
#         self._config['cutoff_tuning'] = cutoff_tuning

#         # initialize probability cutoff (for determining postive and negative class).
#         self._config['binary_cutoff'] = 0.5

#         if self._config['cutoff_tuning']:

#             if cutoff_tuning_metric:
#                 if cutoff_tuning_metric in self.supported_cutoff_metric:
#                     self._config['cutoff_tuning_metric'] = cutoff_tuning_metric
#                 else:
#                     raise ValueError(f'The specified cutoff_tuning_metric ({cutoff_tuning_metric}) is not supported. Supported options: {self.supported_cutoff_metric}')
#             else:
#                 # if cutoff_tuning_metric is not specified, use best_metric for cutoff tuning
#                 if self.best_metric != 'ROC_AUC':
#                     self._config['cutoff_tuning_metric'] = self.best_metric
#                 else:
#                     raise RuntimeError('cutoff_tuning_metric is not specified!')

#         else:
#             # set cutoff_tuning_metric to None if not tuning cutoff
#             self._config['cutoff_tuning_metric'] = None

#     @property
#     def supported_loss_func(self):
#         return {k:v for k,v in super(mlpRegression, self).supported_loss_func.items() if k == 'FocalLossBinary'}

#     @property
#     def supported_best_metric(self):
#         return ['Accuracy', 'F1', 'Precision', 'Recall', 'FPR', 'PSI', 'CohensKappa', 'ROC_AUC']

#     @property
#     def supported_cutoff_metric(self):
#         return [m for m in self.supported_best_metric if m != 'ROC_AUC'] # AUC is independent of prob cutoff and cannot be used for cutoff tuning

#     def _configure_minibatch_transformation(self):
#         # encode label -> convert to torch tensors
#         label_encoding = transform.encode_label(label_col = self._config['label_col'],
#                                                 ordered_labels = None,
#                                                 target_class = self._config['label_values'][0])

#         self._config['label_encoder'] = label_encoding.le

#         tensors_conversion = transform.pd_to_torch(train_type = 'binary',
#                                                    feature_cols = self._config['feature_cols'],
#                                                    label_col = self._config['label_col'])

#         transformations = [label_encoding, tensors_conversion]

#         return transformations

#     def _configure_loss_func(self, gamma = 2, class_weights = None):
#         class_weights = torch.FloatTensor(class_weights) if class_weights else None

#         loss_func = {'FocalLossBinary': {'gamma': gamma, 'weight': class_weights}}

#         return loss_func

#     def _convert_score_to_proba(self, score: torch.Tensor):
#         '''convert model output to probability'''

#         return torch.sigmoid(score)[:,0]

#     def _convert_proba_to_pred(self, probability: torch.Tensor, binary_cutoff: float = 0.5):
#         '''convert predicted probability to predictions (encoded class labels; e.g. 0, 1)'''

#         return (probability >= binary_cutoff).int()

#     def predict(self, X: torch.Tensor):
#         ''' predict classes (encoded)'''

#         probs = self.predict_proba(X)
#         encoded_pred = self._convert_proba_to_pred(probability = probs, binary_cutoff = self._config['binary_cutoff'])

#         return encoded_pred

#     def _calc_metrics(self, dataloader: Union[DataLoader, DistributedDataLoader], loss_runtime_arg = {}):
#         '''metrics calculation for binary classifier
#         '''
#         # track total loss and data weight
#         total_loss = 0
#         total_data_weight = 0

#         # reset dataloader batch iteration
#         dataloader.reset()

#         # tracking confusion matrix for different probability cutoff (0 to 1)
#         cutoff_cm_dict = {(k/100):{} for k in range(100)}

#         # loop through batches
#         for y, X in dataloader:

#             # get score, batch_loss etc.
#             score, batch_loss, data_weight_sum = self._calc_batch_loss(y = y, X = X, loss_runtime_arg = loss_runtime_arg, train_mode = False)

#             # accumulate loss and weight sum
#             total_data_weight += data_weight_sum
#             total_loss += batch_loss

#             # convert model output score to probabilities
#             probs = self._convert_score_to_proba(score) # predicted probabilities

#             # accumulate confusion matrix counts (truth - prediction counts) for different probability cutoff; for calculating ROC, ROC_AUC and optimal cutoff
#             # e.g. {(0,0) : 10, (0,1): 20, (1,0): 5, (1,1): 30}
#             y = y.int()
#             for k,v in cutoff_cm_dict.items():
#                 encoded_pred = self._convert_proba_to_pred(probability = probs, binary_cutoff = k) # get encoded probabilities; 0 or 1.
#                 cutoff_cm_dict[k] = agg_cm_counts(truth = y,
#                                                   prediction = encoded_pred,
#                                                   prev_cm_count_dict = v)

#         # record test loss
#         metrics = {'test_loss': total_loss / total_data_weight}

#         # collect confusion matrix and metrics for each cutoff
#         cm_dict, cutoff_metrics_dict= {}, {}

#         for k, v in cutoff_cm_dict.items():
#             # convert truth-prediction pair counts to confusion matrix
#             cm_dict[k] = convert_cm_counts_to_cm(cm_count_dict = v, n_classes = len(self._config['label_encoder'].classes_))

#             cutoff_metrics_dict[k] = calc_metric_binary(cm_dict[k]) # get all metrics from confusion matrix

#         # convert cutoff metrics dict to pandas with columns = metrics, index = cutoffs
#         additional_metrics_df = pd.DataFrame(cutoff_metrics_dict).T

#         if self._config['cutoff_tuning']:
#             if self._config['cutoff_tuning_metric'] in ['Accuracy', 'F1', 'Precision', 'Recall', 'PSI', 'CohensKappa']:
#                 self._config['binary_cutoff'] = additional_metrics_df[self._config['cutoff_tuning_metric']].astype(float).idxmax()
#             elif self._config['cutoff_tuning_metric'] in ['FPR']:
#                 self._config['binary_cutoff'] = additional_metrics_df[self._config['cutoff_tuning_metric']].astype(float).idxmin()

#         metrics['prob_cutoff'] = self._config['binary_cutoff'] # store prob cutoff used
#         metrics.update(cutoff_metrics_dict[self._config['binary_cutoff']]) # store additonal metrics

#         # ROC metrics
#         metrics['ROC_AUC'] = np.trapz(additional_metrics_df['Recall'][::-1], additional_metrics_df['FPR'][::-1]) # store ROC AUC; area estimated using trapezoidal rule.
#         metrics['ROC_DATA'] = additional_metrics_df.loc[:,['FPR', 'Recall']] # store ROC data; TPR (Recall) vs FPR
#         metrics['ROC_DATA'].plot('FPR', 'Recall', legend = False, xlabel = 'False Positive Rate', ylabel = 'True Positive Rate', title = 'ROC')
#         plt.show()

#         # store confusion matrix; convert encoded labels back to original labels
#         cm = cm_dict[self._config['binary_cutoff']]
#         cm.columns = self._config['label_encoder'].inverse_transform(cm.columns)
#         cm.index = self._config['label_encoder'].inverse_transform(cm.index)
#         metrics['confusion_matrix']= cm

#         return metrics