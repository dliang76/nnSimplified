import boto3
from datetime import datetime
# import pyspark
# import pyspark.sql.functions as F
import pandas as pd
from hashlib import sha256
import random

def timeit(method):
    ''' python decorator for measuring execution time
        
        Authors(s): dliang1122@gmail.com
    '''
    def wrapper(*args, **kwarg):
        if kwarg.get('measure_time'):
            ts = datetime.now()
            result = method(*args, **kwarg)
            te = datetime.now()
            print(f'Execution time: {te - ts}')
        else:
            result = method(*args, **kwarg)
        return result

    return wrapper
    
def convert_list_to_level_dict(lst):
    ''' convert list of items to dict using recursion
        eg. [lvl1, lvl2, lvl3, lvl4] -> {lvl1:{lvl2:{lvl3:[lvl4]}}}

        Authors(s): dliang1122@gmail.com
    '''
    if len(lst) == 1:
        return [lst[0]]
    
    return {lst[0]:convert_list_to_level_dict(lst[1:])}

def dict_merge(d1, d2):
    ''' merge to dictionaries of form {lvl1:{lvl2:{lvl3:{..{lvln:[values]}}}}}
    
        Authors(s): dliang1122@gmail.com
    '''
    d1 = d1.copy()
    d2 = d2.copy()
    
    '''return new merged dict of dicts'''
    if type(d1) == list and type(d2) == list:
        return sorted(set(d1 + d2))
    elif type(d1) == set and type(d2) == set:
        return sorted(d1 | d2)
    elif type(d1) != dict and type(d2) != dict:
        return [d1, d2]
    
    for k, v in d1.items():
        if k in d2:
            d2[k] = dict_merge(v, d2[k])
    d3 = d1
    d3.update(d2)
    return d3


def hash_sort(lst, seed: int = None):
    ''' sort list using hashed element values
    
        Authors(s): dliang1122@gmail.com
    '''
    if seed is None:
        seed = random.randint(0, 1e10)

    return sorted(lst, key = lambda x: sha256((str(x) + str(seed)).encode()).hexdigest())


def ray_fix(func):
    ''' use this function (either as a decorator or wrapper) to fix the issue with ray remote functions in spark environment.
        
        Authors(s): dliang1122@gmail.com

        e.g
            @ray.remote
            @ray_fix
            def func():
                ...
    '''
    func.__module__ = "pyjava_auto_generate__exec__"
    return func


''''PySpark functions'''
# def sharding(data, inputCol, shardCol, n_shard):
#     """simple function for create shards for spark dataframe

#     Authors(s): dliang1122@gmail.com

#     Args
#     ----------
#     data (spark dataframe): data we want to shard
#     inputCol (string): name of the column that we want to use for sharding
#     shardCol (string): name of the output column that contains shard info
#     n_shard (string): number of shards

#     Returns
#     -------
#     spark dataframe: original data + shard info
#     """
#     return data.withColumn(shardCol,
#                            (F.conv(F.substring(F.sha2(F.col(inputCol).cast('string'), 256), 1, 13),16,10) % n_shard).cast('int'))


# def hashing(data, inputCol, hashedCol, hash_key, keep_inputCol = True):
#     """simple function for hashing column in spark dataframe

#     Authors(s): dliang1122@gmail.com

#     Args
#     ----------
#     data (spark or pandas dataframe): data we want to shard
#     inputCol (string): name of the column that we want to hash
#     hashedCol (string): name of the output column that contains hashed data

#     Returns
#     -------
#     spark dataframe: original data + hashed data column
#     """
#     if isinstance(data, pd.DataFrame):
#         result = data.copy(deep = True)
#         result[hashedCol] = [sha256((str(v)+str(hash_key)).encode()).hexdigest() for v in result[inputCol]]

#         if not keep_inputCol:
#             result = result.drop(inputCol, axis = 1)

#     elif isinstance(data, pyspark.sql.dataframe.DataFrame):
#         result = data.withColumn(hashedCol, F.sha2(F.concat(F.col(inputCol).cast('string'),
#                                                             F.lit(str(hash_key))),
#                                                    256))

#         if not keep_inputCol:
#             result = result.drop(inputCol)
#     else:
#         raise ValueError('Data type not supported!')

#     return result
