import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import datetime
import random
from itertools import product
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split, PredefinedSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, cohen_kappa_score
from scipy.stats import uniform, loguniform, randint

from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Dsiplaying only tensorflow error messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or '2'
import tensorflow as tf
# ann
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import L1, L2, L1L2
from tensorflow.keras.callbacks import EarlyStopping


# baselines 
from sklearn.dummy import DummyClassifier

from .utils import display_status_progress
from .constants import OUTPUT_PATH


nb_unit_grid = [16, 32, 64, 128, 256]
n_hidden_layer_grid = [1,2,3]
batch_size_grid = [16,32,64]
l1_grid = [0, 0.1,0.01,0.001]
l2_grid = l1_grid.copy()
dropout_interval = [0,0.5]
    
# Baseline models
def initiate_ELO_baseline(X_train):
    if "ELO_diff_before" in X_train.columns:
        baseline_pipe = make_pipeline(make_column_transformer((StandardScaler(), ["ELO_diff_before"]), remainder='drop'),LogisticRegression())
    else :
        baseline_pipe = DummyClassifier()
    return baseline_pipe

# ANN functions
def initialize_ann_model(X, n_hidden_layer, nb_unit_grid ,dropout_rate, l1, l2 ,acti_reg, kern_reg, bias_reg):
    # Regularization 
    reg = L1L2(l1 = l1, l2 = l2)
    
    # Architechure
    model = Sequential([Dense(int(random.choice(nb_unit_grid)), input_dim=X.shape[1])])
    if random.choice([True, False]):
        model.add(Dropout(dropout_rate))
    for i in range(n_hidden_layer):
        model.add(Dense(int(random.choice(nb_unit_grid)), activation='relu', activity_regularizer = reg if acti_reg else None, bias_regularizer = reg if bias_reg else None, kernel_regularizer= reg if kern_reg else None))
        model.add(Dropout(dropout_rate))         
    model.add(Dense(1, activation='sigmoid'))
    
    # Optimization
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return model

def random_search_ann(X_train, y_train, X_val, y_val,log_file_str, nb_unit_grid, n_hidden_layer_grid ,batch_size_grid, l1_grid, l2_grid, dropout_interval, n_iter=1000, verbose=True, return_model:bool=False):

    best_score = 0
    
    for i in range(n_iter) :
        
        batch_size = random.choice(batch_size_grid)
        l1 = float(np.round(l1_grid[np.random.randint(len(l1_grid))] * random.choice(range(10)),4)) 
        l2 = float(np.round(l2_grid[np.random.randint(len(l2_grid))] * random.choice(range(10)),4))
        dropout = np.round(np.random.uniform(dropout_interval[0], dropout_interval[1]),2)
        n_hidden_layer = random.choice(n_hidden_layer_grid)
        
        
        set_up_dict={"batch_size": batch_size}
        model = initialize_ann_model(X_train, n_hidden_layer, nb_unit_grid, dropout_rate= dropout, l1= l1, l2= l2, acti_reg=random.choice([True,False]), bias_reg=random.choice([True,False]), kern_reg=random.choice([True,False]))
        es = EarlyStopping(patience=256, restore_best_weights=True, verbose=False)
        history = model.fit(
            X_train,
            y_train,
            validation_split = 0.25,
            #validation_data=[X_val,y_val],
            epochs=16000,
            batch_size=batch_size,
            callbacks=[es],
            verbose=0,
        )
        exp = log_ann_training(history, X_train, y_train, X_val, y_val, log_file_str=log_file_str, set_up_dict= set_up_dict, return_experiment=True)

        if exp['eval_accuracy'] > best_score :
            best_model = model
            best_score = exp['eval_accuracy']    
            best_arch = exp['architecture']

        if verbose :
            display_status_progress(i+1, n_iter, f'Searching ANN - best accuracy so far : {np.round(best_score,3)} - ')
    
    if return_model :
        return best_model, best_arch

def summarize_log_infos(history, X_train, y_train, X_test, y_test, set_up_dict, id):
    
    model = history.model
    
    n_epochs = np.argmin(history.history["val_loss"])
    last_epoch_accuracy_ratio = history.history["val_accuracy"][-1] / history.history["accuracy"][-1]
    last_epoch_loss_ratio = history.history["val_loss"][-1] / history.history["loss"][-1]
    
    train_metrics = model.evaluate(X_train, y_train,verbose = False)
    eval_metrics = model.evaluate(X_test, y_test, verbose = False)

    architecture = []
    for layer in model.layers :
        layer_type = layer.name[:layer.name.find('_')] if '_' in layer.name else layer.name
        if layer_type=='dense':
            architecture.append({"type" : layer_type, 
                                "size":layer.bias.shape[0],
                                "activity_regularizer" : {} if layer.activity_regularizer == None else {key : np.round(float(layer.activity_regularizer.__dict__[key]),4) for key in layer.activity_regularizer.__dict__.keys()},
                                "kernel_regularizer" : {} if layer.kernel_regularizer == None else {key : np.round(float(layer.kernel_regularizer.__dict__[key]),4) for key in layer.kernel_regularizer.__dict__.keys()},
                                "bias_regularizer" : {} if layer.bias_regularizer == None else {key : np.round(float(layer.bias_regularizer.__dict__[key]),4) for key in layer.bias_regularizer.__dict__.keys()},
                                })
        elif layer_type == 'dropout':
            architecture.append({"type" : layer_type, 
                                "droupout_rate": layer.rate})
    exp = {"index" : id, 
           "date" : datetime.date.today(), 
           "eval_loss" : eval_metrics[0], 
           "eval_accuracy" : eval_metrics[1],
           "train_loss" : train_metrics[0], 
           "train_accuracy":train_metrics[1],
           "n_epochs":n_epochs,
           "last_epoch_loss_ratio": last_epoch_loss_ratio,
           "last_epoch_accuracy_ratio": last_epoch_accuracy_ratio,
           "architecture": architecture }
    exp.update(set_up_dict)
    return exp

def log_ann_training(history, X_train, y_train, X_test, y_test, log_file_str, set_up_dict, return_experiment=False):
    if os.path.exists(log_file_str):
        log_df = pd.read_csv(log_file_str, index_col='index')
        id = log_df.index.max()+1
        log_df = log_df.reset_index()
        first = False
    else :
        id = 1
        first = True

    exp = summarize_log_infos(history, X_train, y_train, X_test, y_test, set_up_dict, id)

    exp_df = pd.DataFrame([exp])
    if first :
        log_df = exp_df.set_index('index')
    else :    
        log_df = pd.concat([log_df, exp_df]).set_index('index')
    log_df.to_csv(log_file_str)
    if return_experiment:
        return exp

# Model comparison
def initiate_model_list(X_train,n_class:int):

    clas_models_names = ["Logistic Regression",
                    "k - Nearest Neighbors Classifier",                
                    "Random Forest",
                    "Ada Boost",
                    "XGB Classifier",
                    'Support Vector Classifier - Linear kernel',
                    'Support Vector Classifier - Polynomial kernel',
                    'Support Vector Classifier - RBF kernel',
                    "Partial Least Squares - Discriminant Analysis",                                     
                    "ANN",
                    "Baseline home",
                    "Baseline ELO"
    ]

    clas_models = [LogisticRegression(max_iter=1000),
                KNeighborsClassifier(),
                RandomForestClassifier(),
                AdaBoostClassifier(),
                XGBClassifier(),
                SVC(kernel = "linear"),
                SVC(kernel = "poly"),
                SVC(kernel = "rbf"),
                PLSRegression(),  
                'initiate_ann_',
                DummyClassifier(),
                initiate_ELO_baseline(X_train)
    ]
    
    clas_hp = [{'penalty':['l1','l2','elasticnet',None], 'C':loguniform(1,1000), 'solver':['saga']},
                {'n_neighbors':randint(2,60)},
                {'n_estimators':randint(10,500), 'criterion':['gini', 'entropy', 'log_loss'], 'max_depth': randint(1,20), 'min_samples_leaf' : randint(1,40)},
                {'n_estimators':randint(10,500), 'learning_rate': loguniform(1e-3,10)},
                {'eta':uniform(0,1), 'gamma':loguniform(1e-4,1000), 'max_depth': randint(1,20), 'lambda':loguniform(1e-2,10)},
                {'kernel':['linear'], 'C':loguniform(1,100)},
                {'kernel':['poly'], 'C':loguniform(1,100), 'degree':randint(2,3), 'gamma':loguniform(1e-4, 1)},
                {'kernel':['rbf'], 'C':loguniform(1,100), 'gamma':loguniform(1e-4, 1)},
                {'n_components':randint(1,8)},  
                {},
                {},
                {},
                {}
    ]
    
    return clas_models_names,clas_models, clas_hp
     
def compare_models_w_hpo(X_train, y_train, X_val, y_val, X_test, y_test, cv:int = 0, n_iter_ml = 100, n_iter_ann = 10, verbose = True):

    different_test_scores = []
    best_HP = []
    asc = False
    kpi = 'Accuracy'
    

    n_class = len(np.unique(y_train))
    f1s = []
    roc_aucs = []
    cks= []
    clas_models_names, clas_models, clas_hp = initiate_model_list(X_train,n_class)
    for model_name, model,hp in zip(clas_models_names, clas_models,clas_hp):
        if verbose :
            print(f"Testing {model_name}", end='\r')
        if model_name[:3] == 'ANN':
            #y_train = to_categorical(y_train)

            model, _ = random_search_ann(
                X_train = X_train,
                y_train = y_train,
                X_val = X_val,
                y_val = y_val,
                log_file_str= os.path.join(OUTPUT_PATH,'ann_test.csv'),
                nb_unit_grid=nb_unit_grid,
                n_hidden_layer_grid=n_hidden_layer_grid,
                batch_size_grid=batch_size_grid,
                l1_grid=l1_grid,
                l2_grid=l2_grid,
                dropout_interval=dropout_interval,
                n_iter=n_iter_ann,
                verbose = True,
                return_model=True
            )   
            temp_y_pred = model.predict(X_test)   
            #temp_y_pred = np.argmax(temp_y_pred,axis = 1) 
                    
        else :
            X_train_tmp = pd.concat([X_train, X_val])
            y_train_tmp = pd.concat([y_train, y_val])
            split = [-1] * X_train.shape[0] + [0] *X_val.shape[0]
            ps = PredefinedSplit(test_fold=split)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                hrs = RandomizedSearchCV(model, hp, cv = 5 if cv>0 else ps,n_jobs=-1, n_iter = n_iter_ml,refit=True).fit(X_train_tmp, y_train_tmp)
            best_HP.append(hrs.best_params_)
            temp_y_pred = hrs.predict(X_test)
        
        roc_aucs.append(roc_auc_score(y_test, temp_y_pred))
        temp_y_pred = np.array([ 1 if pred > 0.5 else 0 for pred in temp_y_pred])      
        acc = accuracy_score(y_test,temp_y_pred)
        cks.append(cohen_kappa_score(y_test, temp_y_pred))
        different_test_scores.append(acc) 
        if verbose :
            print(f"Tested {model_name} - reached accuracy of {np.round(acc,4)} over the test set", end = '\n', flush = True)
        f1s.append(f1_score(temp_y_pred,y_test))
        comparing_models = pd.DataFrame(list(zip(clas_models_names, best_HP,different_test_scores, f1s, roc_aucs, cks)),
                                                        columns =['Model', 'Best_hyperparameters',kpi, 'F1-score', 'ROC-AUC', 'Cohen-Kappa'])

    return comparing_models.set_index('Model').sort_values(by = kpi,ascending = asc)