import pandas as pd
from typing import List
import matplotlib.pyplot as plt
from copy import deepcopy
from ...utils import save_pickle, load_pickle
from matplotlib.ticker import MaxNLocator

def unravel_nested_dict(d):
    result = dict()

    for k,v in d.items():
        if isinstance(v, dict):
            # if nested, unravel
            k = k if isinstance(k, tuple) else (k,)

            for k1, v1 in unravel_nested_dict(v).items():
                k1 = k1 if isinstance(k1, tuple) else (k1,)

                result[k + k1] = v1
        else:
            # if not nested
            result[k] = v if isinstance(v, (tuple, list)) else [v]

    return result

class trainingReport(object):
    def __init__(self):

        #initialize report list
        self._report_list = []

    @property
    def report_list(self):
        return self._report_list

    def add_entry(self, report_dict: dict):
        report_dict = deepcopy(report_dict)
        self._report_list.append(report_dict)

    def update_entry(self, index:int, report_dict: dict):
        if index not in list(range(len(self.report_list))):
            raise ValueError(f"No report found for index = {index}!")
        else:
            self._report_list[index] = deepcopy(report_dict)

    def pop_entry(self, index: int):
        if index not in list(range(len(self.report_list))):
            raise ValueError(f"No report found for index = {index}!")
        else:
            self._report_list.pop(index)

    def to_df(self,
              metrics: List[str] | str | None = None,
              dataset_types: List[str] = ['train', 'test']):
        '''convert report to pandas data frame'''

        if self._report_list:
            result = dict()
            for i in self._report_list:
                i = unravel_nested_dict(i) # unravel nested dictionary

                for k,v in i.items():
                    if k in result:
                        result[k].extend(v)
                    else:
                        result[k] = v.copy()

            index_cols = [c for c in result.keys() if not isinstance(c, (tuple))]

            df = pd.DataFrame(result).set_index(index_cols)
            df.columns = pd.MultiIndex.from_tuples(df.columns)

            if metrics is not None:
                if not isinstance(metrics, (list, tuple)):
                    metrics = [metrics]

                if not isinstance(dataset_types, (list, tuple)):
                    dataset_types = [dataset_types]

                columns = [c for c in df.columns if c[0] in dataset_types and c[1] in metrics]
                df = df[columns]
        else:
            df = None

        return df

    def plot(self,
             metrics: List[str] | str = None,
             dataset_types: List[str] = ['train', 'test'],
             marker: str = None,
             linestyle: str = '-',
             size: List[float] = None,
             annotate_min_metrics = [],
             annotate_max_metrics = [],
             yscale: str = 'linear',
             xscale: str = 'linear'):

        data = self.to_df(metrics = metrics, dataset_types = dataset_types) # gather metrics into a dataframe

        plt.figure(figsize = size) # set figure size

        x_val = data.index.get_level_values('epoch')

        # loop through dataset types (e.g. train, test) and plot
        for d in dataset_types:
            df = data.get(d) # get metrics for the dataset type

            if df is not None:
                plt.plot(x_val[0], df.values.min(), marker = 'None', linestyle = 'None', label=fr'$\bf {{{d}}}$') # use a dummy plot to generate legend groups

                for m in df.columns:
                    p = plt.plot(x_val, df[m], marker = marker, linestyle = linestyle, label = m)

                    if m in annotate_min_metrics:
                        # find min and max points
                        y_min_idx = df[m].idxmin()
                        y_min = df[m].loc[y_min_idx]
                        x_min = y_min_idx[0]

                        plt.plot(x_min, y_min, marker = '.', color = p[0].get_color(), label = '_')
                        plt.annotate(text = str(round(y_min, 5)),
                                     xy = (x_min, y_min),
                                     xytext= (x_min, y_min + 0.1 * (max(p[0].get_ydata()) - y_min)),
                                     arrowprops=dict(arrowstyle = '->'), color = p[0].get_color())

                    if m in annotate_max_metrics:
                        y_max_idx = df[m].idxmax()
                        y_max = df[m].loc[y_max_idx]
                        x_max = y_max_idx[0]

                        plt.plot(x_max, y_max, marker = '.', color = p[0].get_color(), label = '_')
                        plt.annotate(text = str(round(y_max,5)),
                                    xy = (x_max, y_max),
                                    xytext = (x_max, y_max + 0.1 * (y_max - min(p[0].get_ydata()))),
                                    arrowprops=dict(arrowstyle = '->'), color = p[0].get_color())

        plt.legend() # turn on legend
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True)) # force integer x-axis ticks
        plt.yscale(yscale), plt.xscale(xscale) # scale x and y
        plt.xlabel('Epoch'), plt.ylabel('Metrics / Loss') # label axis
        plt.grid(which = 'both', axis = 'both', linestyle = '--') # turn grid on

    def __getitem__(self, index):
        '''Enables selecting a report by index'''
        return self._report_list[index]

    def __iter__(self):
        '''make object iterable'''
        for report in self._report_list:
            yield report

    def reset(self):
        self._report_list = []

    def save(self, file_path: str):
        save_pickle(obj = self.report_list, path = file_path)

    def load(self, file_path: str):
        self._report_list = load_pickle(path = file_path)

    def __repr__(self):
        repr = str(self.__class__) + '\n'
        for r in self:
            repr += f"Epoch {r['epoch']} (lr={r['initial_lr']}):"
            repr += "\n  Train:"
            for m,v in r['train'].items():
                repr += f'\n     {m}: {v}'

            if 'test' in r:
                repr += "\n  Test:"
                for m,v in r['test'].items():
                    repr += f'\n     {m}: {v}'

            repr += '\n\n'

        return repr