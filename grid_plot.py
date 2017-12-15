import time
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D, Dropout, BatchNormalization, GlobalMaxPool2D

resolution = 28
classes = 10

ALL_MODELS = ['logistic regression', 
'multi layer perceptron', 
'1 conv layer 16 filters 3x3',
'1 conv layer 16 filters 3x3 with maxpool',
'2 conv layer 16 and 32 filters 3x3 with maxpool',
'2 conv layer 64 filters 3x3 with maxpool',
'2 blocks of 2 conv layer 16 and 32 filters 3x3 with maxpool',
'2 blocks of 2 conv layer 16 and 32 filters 3x3 with maxpool and dropout',
'2 blocks of 2 conv layer 16 and 32 filters 3x3 with maxpool with dropout with batch normalization',      
'2 blocks of 2 conv layer 16 and 32 filters 3x3 with maxpool, dense layer with dropout with batch normalization',
'2 blocks of 2 conv layer 16 and 32 filters 3x3 with maxpool, global average pooling layer with dropout with batch normalization'
]

def plot_loss_grid(models = ALL_MODELS, save_path='../resources/cached_model_grid_scores.csv'):
    grid = pd.read_csv(save_path)   
    grid = grid[grid['model_names'].isin(models)]

    f, (ax1, ax2) = plt.subplots(2, figsize=(12,12));
    grid_loss = grid[(grid['score'] == 'loss') 
                     & (grid['data_fold'] != 'overfit')]
    sns.swarmplot(data=grid_loss, 
                        y='variable', x='value', hue='model_names', ax=ax1);
    ax1.set(xlabel='losses', ylabel='');
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='lower right', borderaxespad=0.);
    grid_overfit= grid[(grid['score'] == 'loss')
                       & (grid['data_fold'] == 'overfit')]
    sns.swarmplot(data=grid_overfit, 
                        y='variable', x='value', hue='model_names', ax=ax2);
    ax2.set(xlabel='losses', ylabel='');
    ax2.legend_.remove();
    plt.show();
    
def plot_acc_grid(models = ALL_MODELS, save_path='../resources/cached_model_grid_scores.csv'):
    grid = pd.read_csv(save_path)   
    grid = grid[grid['model_names'].isin(models)]

    f, (ax1, ax2) = plt.subplots(2, figsize=(12,12));
    grid_acc = grid[(grid['score'] == 'acc') 
                    & (grid['data_fold'] != 'overfit')]
    sns.swarmplot(data=grid_acc, 
                  y='variable', x='value', hue='model_names', ax=ax1);
    ax1.set(xlabel='scores', ylabel='');
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='lower right', borderaxespad=0.);
    grid_overfit= grid[(grid['score'] == 'acc')
                       & (grid['data_fold'] == 'overfit')]
    sns.swarmplot(data=grid_overfit, 
                        y='variable', x='value', hue='model_names', ax=ax2);
    ax2.set(xlabel='scores', ylabel='');
    ax2.legend_.remove();
    plt.show();
    
def plot_complexity(models=ALL_MODELS, save_path='../resources/cached_model_grid_scores.csv'):
    grid = pd.read_csv(save_path)    
    grid = grid[grid['model_names'].isin(models)]

    plt.figure(figsize=(12,12));
    sns.lmplot(data=grid, x='time_to_train', y='params', 
               hue='model_names', fit_reg=False, legend=False);
    plt.legend(bbox_to_anchor=(1.05, 1), loc='lower right', borderaxespad=0.);
    plt.show();
    
def score_model_grid(models, names, train_times, X_train, Y_train, X_test, Y_test):
    ids, train_loss, test_loss, train_acc, test_acc, total_params = [], [], [], [], [], []
    for i, model in enumerate(models):
        tr_l, tr_ac = model.evaluate(X_train, Y_train) 
        ts_l, ts_ac = model.evaluate(X_test, Y_test)
        tot_prm = _get_params_nr(model)
        ids.append(i)
        train_loss.append(tr_l)
        test_loss.append(ts_l)
        train_acc.append(tr_ac)
        test_acc.append(ts_ac)
        total_params.append(tot_prm)
    df = pd.DataFrame({'acc_train':train_acc,
                       'acc_test':test_acc,
                       'loss_train':train_loss,
                       'loss_test':test_loss,
                       'params':total_params,
                       'time_to_train':train_times,
                       'model_names':names})
    
    df['acc_overfit'] = df['acc_train'] - df['acc_test']
    df['loss_overfit'] = df['loss_test'] - df['loss_train']
    
    df = pd.melt(df, id_vars=['model_names','time_to_train','params'])
    
    def split_variable(x):
        s, d = x.split('_')
        return pd.Series({'data_fold':s, 'score': d})
    df[['score', 'data_fold']] = df['variable'].apply(lambda x: split_variable(x))
    
    return df

def _get_params_nr(model):
    params = 0
    for layer in model.layers:
        params += layer.count_params()
    return params