import tkinter as tk
from tkinter import filedialog
import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from statistics import mean
import datetime
from sklearn.feature_selection import RFE, RFECV
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
import random

def preprocessing_db(df, MAX_bonds, INIT_POS):
    df = df.copy(deep = True)
    ## Case of joining sections
    df['Section2_O'] = df.iloc[:, INIT_POS:INIT_POS + MAX_bonds].sum(axis=1)
    df['Section2_M'] = df.iloc[:, INIT_POS + MAX_bonds:INIT_POS + 2*MAX_bonds].sum(axis=1)
    df['Section3_O'] = df.iloc[:, INIT_POS + 2*MAX_bonds:INIT_POS + pow(MAX_bonds+1, 2) + 2*MAX_bonds].sum(axis=1)
    df['Section3_M'] = df.iloc[:, INIT_POS + pow(MAX_bonds+1, 2) + 2*MAX_bonds:INIT_POS + 2*pow(MAX_bonds+1, 2) + 2*MAX_bonds].sum(axis=1)
    df['Section4_O'] = df.iloc[:, INIT_POS + 2*pow(MAX_bonds+1, 2) + 2*MAX_bonds: INIT_POS + pow(MAX_bonds+1,4) + 2*pow(MAX_bonds+1, 2) + 2*MAX_bonds].sum(axis=1)
    df['Section4_M'] = df.iloc[:, INIT_POS + pow(MAX_bonds+1,4) + 2*pow(MAX_bonds+1, 2) + 2*MAX_bonds: INIT_POS + 2*pow(MAX_bonds+1,4) + 2*pow(MAX_bonds+1, 2) + 2*MAX_bonds].sum(axis=1)
    df['Section4_OM'] = df.iloc[:, INIT_POS + 2*pow(MAX_bonds+1,4) + 2*pow(MAX_bonds+1, 2) + 2*MAX_bonds:INIT_POS + 3*pow(MAX_bonds+1,4) + 2*pow(MAX_bonds+1, 2) + 2*MAX_bonds].sum(axis=1)

    ## drop all except the ones created
    X_clean = df.drop(df.iloc[:, INIT_POS:INIT_POS + 3*pow(MAX_bonds+1,4) + 2*pow(MAX_bonds+1, 2) + 2*MAX_bonds],axis = 1)
    X_clean.columns = X_clean.columns.astype(str)
    X_clean = X_clean.loc[:,~X_clean.T.duplicated(keep='first')]
    ## drop 0s
    X_clean.drop([col for col, val in X_clean.sum().iteritems() if val == 0], axis=1, inplace=True)
    return X_clean

def train_model(train_samples, ground_truth, ratio, is_joined, article_name, has_paper_params, seeds):
    models = []
    bias = []
    std_scaler = []
    dropped_cols = []
    mean_scaler = []
    std_var_scaler = []
    results_model = []
    paper_params = ""
    if has_paper_params:
        paper_params = 'eq_params_'
    current_date = str(datetime.datetime.now().date())
    if 'anantha' in article_name:
        columns_results = ['train_RMSE', 'train_MAE', 'train_r2_score', 'validation_RMSE', 'validation_MAE', 'validation_r2_score']

    else:
        columns_results = ['train_CCC','train_RMSE', 'train_MAE', 'train_r2_score', 'validation_CCC','validation_RMSE', 'validation_MAE', 'validation_r2_score']

    # save the model to disk
    if is_joined:
        addon_name = "joined_NF_"
    else:
        addon_name = "NoZeros_NF_"

    for i in seeds: ## 10 different models created using a set of different samples
        X_train, X_val, y_train, y_val = train_test_split(train_samples, ground_truth, test_size=ratio, random_state=i)
        ## remove 0s and keep NF values individually
        X_train = X_train.copy(deep = True)
        X_val = X_val.copy(deep = True)
        zero_cols = [col for col, val in X_train.sum().iteritems() if val == 0]
        X_train.drop(zero_cols, axis=1, inplace=True)
        X_train.columns = range(X_train.columns.size)
        ## LinearRegression
        model = LinearRegression()
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        model.fit(X_train_scaled, y_train)
        X_val.drop(zero_cols, axis=1, inplace=True)
        X_val.columns = range(X_val.columns.size)
        X_val_scaled = scaler.transform(X_val)
        ## model results
        predictions_train = model.predict(X_train_scaled)
        predictions_val = model.predict(X_val_scaled)
        if article_name == "anantha_eq1":
            train_rmse, train_mae, train_r2 = compute_metrics(y_train, predictions_train, model, X_train_scaled,article_name)
            train_results = list([train_rmse, train_mae, train_r2])
            val_rmse, val_mae, val_r2 = compute_metrics(y_val, predictions_val, model, X_val_scaled,article_name)
            val_results = list([val_rmse, val_mae, val_r2])
        else:
            train_ccc, train_rmse, train_mae, train_r2 = compute_metrics(y_train, predictions_train, model, X_train_scaled,article_name)
            train_results = list([train_ccc, train_rmse, train_mae, train_r2])
            val_ccc, val_rmse, val_mae, val_r2 = compute_metrics(y_val, predictions_val, model, X_val_scaled,article_name)
            val_results = list([val_ccc, val_rmse, val_mae, val_r2])
        

        filename = paper_params + addon_name + f"{article_name}_model_{i}_LinearRegression_{str(datetime.datetime.now().date())}.sav"
        pickle.dump(model, open(filename, 'wb'))
        models.append(model)
        bias.append(model.intercept_)
        std_scaler.append(scaler)
        dropped_cols.append(zero_cols)
        mean_scaler. append(scaler.mean_)
        std_var_scaler.append(scaler.scale_)
        results_model.append(train_results + val_results)
    ## save Train and Validation results
    arr = np.asarray(results_model)
    results_filename = paper_params + addon_name + f'{article_name}_train_val_results_{current_date}.csv'
    pd.DataFrame(arr).to_csv(results_filename, index_label = "Index", header  = columns_results)    

    ## save mean and std var for model in Golang
    arr = np.asarray(mean_scaler)
    results_filename = paper_params + addon_name + f'{article_name}_train_val_mean_scaler_{current_date}.csv'
    pd.DataFrame(arr).to_csv(results_filename, index_label = "Index", header  = "")

    ## save mean and std var for model in Golang
    arr = np.asarray(std_var_scaler)
    results_filename = paper_params + addon_name + f'{article_name}_train_val_std_var_scaler_{current_date}.csv'
    pd.DataFrame(arr).to_csv(results_filename, index_label = "Index", header  = "")
    return models, bias, std_scaler, dropped_cols

def calculate_ccc(y, y_predicted):
    mean_y_predicted = mean(y_predicted)
    mean_y = mean(y)
    values_y = y.values
    numerator = 2*(np.dot(values_y.T - mean_y, y_predicted.T - mean_y_predicted))
    denominator = np.sum(pow(y - mean_y, 2)) + np.sum(pow(y_predicted - mean_y_predicted, 2)) + len(y)*pow(mean_y - mean_y_predicted,2)
    return numerator/denominator

def compute_metrics(y, y_predicted, model, x, paper_analysis):
    if paper_analysis =="anantha_eq1":
        y_predicted_boolean = [1.0 if single_prediction > 0.5 else 0.0 for single_prediction in list(y_predicted)]
        return [mean_squared_error(y, y_predicted_boolean, squared=False), mean_absolute_error(y, y_predicted_boolean), model.score(x, y)]
    else:
        return [calculate_ccc(y, y_predicted), mean_squared_error(y, y_predicted, squared=False), mean_absolute_error(y, y_predicted), model.score(x, y)]

def fit_models(X_test, y_test, models, bias, scaler, dropped_colums, is_joined, article_eq_name, has_paper_params):
    all_models_coef = []
    results_test = []
    remaining_cols_ = []
    current_date = str(datetime.datetime.now().date())
    paper_params = ""
    if has_paper_params:
        paper_params = 'eq_params_'
    if is_joined:
        addon_name = "joined_NF_"
    else:
        addon_name = "NoZeros_NF_"

    coef_filename = paper_params + f'{article_eq_name}_{addon_name}models_coef_{current_date}.csv'
    results_filename = paper_params + f'{article_eq_name}_{addon_name}results_{current_date}.csv'
    cols_filename = paper_params + f'{article_eq_name}_{addon_name}rem_cols_{current_date}.csv'
    bias_filename = paper_params + f'{article_eq_name}_{addon_name}bias_{current_date}.csv'

    if 'anantha' in article_eq_name:
        header  = ['test_RMSE','test_MAE','test_r2']
    else:
        header  = ['test_CCC', 'test_RMSE','test_MAE','test_r2']
    for i in range(len(bias)):
        X_test_df = pd.DataFrame(X_test).copy(deep = True)
        X_test_df.drop(dropped_colums[i], axis=1, inplace=True)
        remaining_cols = list(X_test_df.columns)
        X_test_df.columns = range(X_test_df.columns.size)
        X_test_removed = X_test_df.to_numpy()
        X_test_removed = scaler[i].transform(X_test_removed)
        coefs_model = list(models[i].coef_)
        y_predicted = bias[i] + X_test_removed.dot(coefs_model)
        all_models_coef.append(coefs_model)
        remaining_cols_.append(remaining_cols)
        results_test.append(compute_metrics(y_test, y_predicted, models[i], X_test_removed, paper_analysis=article_eq_name))

    arr = np.asanyarray(bias)
    pd.DataFrame(arr).to_csv(bias_filename, index_label = "Index", header  = ['bias'])    
    arr = np.asarray(results_test)
    pd.DataFrame(arr).to_csv(results_filename, index_label = "Index", header  = header)    
    arr = np.asarray(all_models_coef)
    pd.DataFrame(arr).to_csv(coef_filename, index_label = "Index")
    ## save remaining columns
    arr = np.asarray(remaining_cols_)
    pd.DataFrame(arr).to_csv(cols_filename, index_label = "Index")

def rfe_analysis(X_train_val, y, paper_analysis):
    X_train = X_train_val.copy(deep = True)
    zero_cols = [col for col, val in X_train.sum().iteritems() if val == 0]
    X_train.drop(zero_cols, axis=1, inplace=True)
    # step-1: create a cross-validation scheme
    if paper_analysis == "papa_eq3":
        folds = KFold(n_splits = 2, shuffle = True, random_state = 42)
    else:
        folds = KFold(n_splits = 2, shuffle = True, random_state = 42)

    # step-2: specify range of hyperparameters to tune
    hyper_params = [{'n_features_to_select': list(range(1, len(X_train.columns.values)))}]

    lm = LinearRegression()
    lm.fit(X_train, y)
    rfe = RFE(lm)             

    # 3.2 call GridSearchCV()
    scoring_keys =['r2'] 
    model_cv = GridSearchCV(estimator = rfe, 
                            param_grid = hyper_params, 
                            scoring= scoring_keys, 
                            cv = folds, 
                            verbose = 1,
                            return_train_score=True,
                            refit=False)      

    # fit the model
    model_cv.fit(X_train, y)   
    # cv results
    cv_results = pd.DataFrame(model_cv.cv_results_)
    return cv_results, zero_cols, X_train

def optimal_RFE(num_feat, X_train, Y_train, X_test, Y_test, name_eq):
        X_train_ = X_train.copy(deep=True)
        X_test = X_test.copy(deep=True)
        lm = LinearRegression()
        lm.fit(X_train_, Y_train)

        rfe = RFE(lm, n_features_to_select=num_feat)             
        rfe = rfe.fit(X_train_, Y_train)
        index_features_selected = [1 if elem_rank == 1 else 0 for elem_rank in rfe.ranking_]
        id_cols = []
        for idx in range(0, len(index_features_selected)):
            if index_features_selected[idx] == 1:
                if X_train_.columns[idx] > 6:
                    id_cols.append(X_train_.columns[idx] + 7)
                else:
                    id_cols.append(X_train_.columns[idx])
            else:
                id_cols.append(0)

        # predict Y' for test samples
        y_pred = rfe.predict(X_test)
        r2_test = r2_score(Y_test, y_pred)

        results_test = [calculate_ccc(Y_test, y_pred), mean_squared_error(Y_test, y_pred, squared=False), mean_absolute_error(Y_test, y_pred), r2_test]
        ## save LR coefs + intercept
        coefs_TiO2_optimal_filename = f"{name_eq}_rfe_coefs_optimal_features{num_feat}.txt"
        coefs_TiO2 = list(rfe.estimator_.coef_)
        coefs_TiO2.insert(0, rfe.estimator_.intercept_)
        coefs_TiO2 = np.array(coefs_TiO2)
        np.savetxt(coefs_TiO2_optimal_filename, coefs_TiO2)
        #Save results
        results_filename = f"{name_eq}_rfe_results_num_feat_{num_feat}.txt"
        arr = np.array(results_test)
        np.savetxt(results_filename, arr)

        ## save idx_cols
        feat_cols_papa_filename = f"{name_eq}_rfe_idx_features_num_feat_{num_feat}.txt"
        idx_rfe_feat = np.array(id_cols)
        np.savetxt(feat_cols_papa_filename, idx_rfe_feat)

def main():
    ## read csv file
    tk.Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    filename = tk.filedialog.askopenfilename() # show an "Open" dialog box and return the path to the selected file
    print(filename)
    df = pd.read_csv(filename, sep=";", header = None)
    df = df.T
    paper_analysis = input("paper? ")
    ratio_train_val = 0.85
    ratio_val_adjusted = 0.15 / ratio_train_val
    is_CV = False
    seeds = random.choices(range(0,100), k=15)
    if is_CV:
        if paper_analysis == "papa":
            eq = input("eq?")
            if eq =="2":
                ## Select only TiO2
                DB_TiO2 = df[df[0].str.contains(r'TiO2(?!$)')]
                Y_TiO2 = DB_TiO2[6].astype(float)
                removed_col = [0, 6, 7, 8, 9, 10] ## Label, Y, and 4 first val of NF
                X_TiO2 = DB_TiO2.drop(DB_TiO2.columns[removed_col], axis=1).astype(float) ## dropped: label component, Size Comp(dup in NF), Y, AtomicNum(only eq2,3)
                ## X_TiO2.columns = range(X_TiO2.columns.size)
                MAX_bonds = int(DB_TiO2.iloc[0,8])
                ## train model
                X_train_val, X_test, y_train_val, y_test = train_test_split(X_TiO2, Y_TiO2, test_size=0.15)
                cv_results_rfe, zero_cols, X_train_val = rfe_analysis(X_train_val, y_train_val, "papa_eq2")
                filename = paper_analysis + f"eq_{eq}_RFE_analysis_LR_{str(datetime.datetime.now().date())}.csv"
                cv_results_rfe.to_csv(filename)
                X_test.drop(zero_cols, axis=1, inplace=True)

                # final model
                n_features_optimal = 21
                optimal_RFE(n_features_optimal, X_train_val, y_train_val, X_test, y_test, name_eq="papa_eq2")

            elif eq =="3": # ZnO
                DB_ZnO = df[df[0].str.contains(r'ZnO(?!$)')]
                Y_ZnO = DB_ZnO[6].astype(float)
                removed_col = [0, 6, 7, 8, 9, 10] ## Label, Y, and 4 first val of NF
                X_ZnO = DB_ZnO.drop(DB_ZnO.columns[removed_col], axis=1).astype(float) 
                X_ZnO.columns = range(X_ZnO.columns.size)
                MAX_bonds = int(DB_ZnO.iloc[0,8])
                #### RFE BLOCK
                X_train_val, X_test, y_train_val, y_test = train_test_split(X_ZnO, Y_ZnO, test_size=0.15)
                cv_results_rfe, zero_cols, X_train_val = rfe_analysis(X_train_val, y_train_val, "papa_eq3")
                filename = paper_analysis + f"eq_{eq}_RFE_analysis_LR_{str(datetime.datetime.now().date())}.csv"
                cv_results_rfe.to_csv(filename)
                X_test.drop(zero_cols, axis=1, inplace=True)

                # final model
                n_features_optimal = 112
                optimal_RFE(n_features_optimal, X_train_val, y_train_val, X_test, y_test, name_eq="papa_eq3")
            elif eq=="1": # TiO2 + ZnO

                Y_TiO2_ZnO = df[6].astype(float)
                removed_col = [0, 6, 7, 8, 9, 10] ## Label, Y, and 4 first val of NF
                X_TiO2_ZnO = df.drop(df.columns[removed_col], axis=1).astype(float) 
                X_TiO2_ZnO.columns = range(X_TiO2_ZnO.columns.size)
                MAX_bonds = int(df.iloc[0,8])
                #### RFE BLOCK
                X_train_val, X_test, y_train_val, y_test = train_test_split(X_TiO2_ZnO, Y_TiO2_ZnO, test_size=0.15)
                cv_results_rfe, zero_cols, X_train_val = rfe_analysis(X_train_val, y_train_val, "papa_eq1")
                filename = paper_analysis + f"eq_{eq}_RFE_analysis_LR_{str(datetime.datetime.now().date())}.csv"
                cv_results_rfe.to_csv(filename)
                X_test.drop(zero_cols, axis=1, inplace=True)

                # final model
                n_features_optimal = 163
                optimal_RFE(n_features_optimal, X_train_val, y_train_val, X_test, y_test, name_eq="papa_eq1")
        elif paper_analysis =="anantha":
            eq =1
            tk.Tk().withdraw() 
            filename = tk.filedialog.askopenfilename()
            print(filename)
            df2 = pd.read_csv(filename, sep=";", header = None)
            df2 = df2.T
            full_df = pd.concat([df, df2])
            Y_categorical = full_df[23].astype("string")
            Y = [float(u in 'Toxic') for u in Y_categorical]
            removed_col = [0, 9, 10, 23, 24, 25] ## Label, CellLine, CellType, and Y
            X = full_df.drop(full_df.columns[removed_col], axis=1).astype(float) 
            X.columns = range(X.columns.size)
            MAX_bonds = int(full_df.iloc[0,25])
            #### RFE BLOCK
            X_train_val, X_test, y_train_val, y_test = train_test_split(X, Y, test_size=0.15)
            cv_results_rfe, zero_cols, X_train_val = rfe_analysis(X_train_val, y_train_val, "anantha_eq1")
            filename = paper_analysis + f"eq_{eq}_RFE_analysis_LR_{str(datetime.datetime.now().date())}.csv"
            cv_results_rfe.to_csv(filename)
            X_test.drop(zero_cols, axis=1, inplace=True)
            # final model
            n_features_optimal = 163
            optimal_RFE(n_features_optimal, X_train_val, y_train_val, X_test, y_test, name_eq="papa_eq1")
    else:       
        if paper_analysis == "papa":
            ## Select only TiO2
            DB_TiO2 = df[df[0].str.contains(r'TiO2(?!$)')]
            Y_TiO2 = DB_TiO2[6].astype(float)
            removed_col = [0, 6, 7, 8, 9, 10] ## Label, Y, and 4 first val of NF
            X_TiO2 = DB_TiO2.drop(DB_TiO2.columns[removed_col], axis=1).astype(float) ## dropped: label component, Size Comp(dup in NF), Y, AtomicNum(only eq2,3)
            X_TiO2.columns = range(X_TiO2.columns.size)
            MAX_bonds = int(DB_TiO2.iloc[0,8])
            is_loaded_model = False

            ## Joined Sections
            X_clean = preprocessing_db(X_TiO2, MAX_bonds, INIT_POS = 7)
            X_train_val, X_test, y_train_val, y_test = train_test_split(X_clean, Y_TiO2, test_size=0.15)
            models, bias, scaler, dropped_cols = train_model(X_train_val, y_train_val, ratio_val_adjusted, True, 'papas_eq2', has_paper_params=False, seeds=seeds) 
            fit_models(X_test, y_test, models, bias, scaler, dropped_cols, is_joined=True, article_eq_name='papas_eq2', has_paper_params=False)

            ## selected params in paper + Joined
            removed_col = [0, 2, 3, 5, 6, 7, 8, 9, 10] ## Label, X1, X5, Y, and 4 first val of NF
            X_TiO2 = DB_TiO2.drop(DB_TiO2.columns[removed_col], axis=1).astype(float)
            X_TiO2.columns = range(X_TiO2.columns.size)
            X_clean = preprocessing_db(X_TiO2, MAX_bonds, INIT_POS = 4)
            X_train_val, X_test, y_train_val, y_test = train_test_split(X_clean, Y_TiO2, test_size=0.15)
            models, bias, scaler, dropped_cols = train_model(X_train_val, y_train_val, ratio_val_adjusted, True, 'papas_eq2', has_paper_params=True, seeds=seeds)
            fit_models(X_test, y_test, models, bias, scaler, dropped_cols, is_joined=True, article_eq_name='papas_eq2', has_paper_params=True)

            ## remove 0s and keep NF values individually
            X_train_val, X_test, y_train_val, y_test = train_test_split(X_TiO2, Y_TiO2, test_size=0.15)
            models, bias, scaler, dropped_cols = train_model(X_train_val, y_train_val, ratio_val_adjusted, False, 'papas_eq2', has_paper_params=False, seeds=seeds)
            fit_models(X_test, y_test, models, bias, scaler, dropped_cols, is_joined=False, article_eq_name='papas_eq2', has_paper_params=False)

            ## selected params in paper + NoZeros
            removed_col = [0, 2, 3, 5, 6, 7, 8, 9, 10] ## Label, Y, and 4 first val of NF
            X_TiO2 = DB_TiO2.drop(DB_TiO2.columns[removed_col], axis=1).astype(float)
            X_TiO2.columns = range(X_TiO2.columns.size)
            X_train_val, X_test, y_train_val, y_test = train_test_split(X_TiO2, Y_TiO2, test_size=0.15)
            models, bias, scaler, dropped_cols = train_model(X_train_val, y_train_val, ratio_val_adjusted, False, 'papas_eq2', has_paper_params=True, seeds=seeds)
            fit_models(X_test, y_test, models, bias, scaler, dropped_cols, is_joined=False, article_eq_name='papas_eq2', has_paper_params=True)

            if is_loaded_model: 
                loaded_model = pickle.load(open('.\\joined_NF_papas_eq2_model_3_LinearRegression.sav', 'rb'))
                result = loaded_model.score(X_test, y_test)
                
                coefs_loaded_model = list(loaded_model.coef_)
                y_predicted = loaded_model.intercept_ + X_test.dot(coefs_loaded_model)
                compute_metrics(y_test, y_predicted, loaded_model, X_test, True)
            DB_ZnO = df[df[0].str.contains(r'ZnO(?!$)')]
            Y_ZnO = DB_ZnO[6].astype(float)
            removed_col = [0, 6, 7, 8, 9, 10] ## Label, Y, and 4 first val of NF
            X_ZnO = DB_ZnO.drop(DB_ZnO.columns[removed_col], axis=1).astype(float) 
            X_ZnO.columns = range(X_ZnO.columns.size)
            MAX_bonds = int(DB_ZnO.iloc[0,8])

            ##joined models
            X_clean = preprocessing_db(X_ZnO, MAX_bonds, INIT_POS=7)
            X_train_val, X_test, y_train_val, y_test = train_test_split(X_clean, Y_ZnO, test_size=0.15)
            models, bias, scaler, dropped_cols = train_model(X_train_val, y_train_val, ratio_val_adjusted, True, 'papas_eq3', has_paper_params=False, seeds=seeds) 
            fit_models(X_test, y_test, models, bias, scaler, dropped_cols, is_joined=True, article_eq_name='papas_eq3', has_paper_params=False)

            ## selected params in paper + Joined
            removed_col = [0, 1, 5, 6, 7, 8, 9, 10] ## Label, X0, X5, Y, and 4 first val of NF
            X_ZnO = DB_ZnO.drop(DB_ZnO.columns[removed_col], axis=1).astype(float)
            X_ZnO.columns = range(X_ZnO.columns.size)
            X_clean = preprocessing_db(X_ZnO, MAX_bonds, INIT_POS = 5)
            X_train_val, X_test, y_train_val, y_test = train_test_split(X_clean, Y_ZnO, test_size=0.15)
            models, bias, scaler, dropped_cols = train_model(X_train_val, y_train_val, ratio_val_adjusted, True, 'papas_eq3', has_paper_params=True, seeds=seeds)
            fit_models(X_test, y_test, models, bias, scaler, dropped_cols, is_joined=True, article_eq_name='papas_eq3', has_paper_params=True)

            ## remove 0s and keep NF values individually
            X_train_val, X_test, y_train_val, y_test = train_test_split(X_ZnO, Y_ZnO, test_size=0.15)
            models, bias, scaler, dropped_cols = train_model(X_train_val, y_train_val, ratio_val_adjusted, False, 'papas_eq3', has_paper_params=False, seeds=seeds) 
            fit_models(X_test, y_test, models, bias, scaler, dropped_cols, is_joined=False, article_eq_name='papas_eq3', has_paper_params=False)

            ## selected params in paper + NoZeros
            removed_col = [0, 1, 5, 6, 7, 8, 9, 10] ## Label, X0, X5, Y, and 4 first val of NF
            X_ZnO = DB_ZnO.drop(DB_ZnO.columns[removed_col], axis=1).astype(float)
            X_ZnO.columns = range(X_ZnO.columns.size)
            X_train_val, X_test, y_train_val, y_test = train_test_split(X_ZnO, Y_ZnO, test_size=0.15)
            models, bias, scaler, dropped_cols = train_model(X_train_val, y_train_val, ratio_val_adjusted, False, 'papas_eq3', has_paper_params=True, seeds=seeds)
            fit_models(X_test, y_test, models, bias, scaler, dropped_cols, is_joined=False, article_eq_name='papas_eq3', has_paper_params=True)
            
            ## No filter based on name tag
            Y_TiO2_ZnO = df[6].astype(float)
            removed_col = [0, 6, 7, 8, 9, 10] ## Label, Y, and 4 first val of NF
            X_TiO2_ZnO = df.drop(df.columns[removed_col], axis=1).astype(float) 
            X_TiO2_ZnO.columns = range(X_TiO2_ZnO.columns.size)
            MAX_bonds = int(df.iloc[0,8])

            ##joined models
            X_clean = preprocessing_db(X_TiO2_ZnO, MAX_bonds, INIT_POS=7)
            X_train_val, X_test, y_train_val, y_test = train_test_split(X_clean, Y_TiO2_ZnO, test_size=0.15)
            models, bias, scaler, dropped_cols = train_model(X_train_val, y_train_val,ratio_val_adjusted, True, 'papas_eq1', has_paper_params=False, seeds=seeds) 
            fit_models(X_test, y_test, models, bias, scaler, dropped_cols, is_joined=True, article_eq_name='papas_eq1', has_paper_params=False)

            ## selected params in paper + Joined
            removed_col = [0, 2, 5, 6, 7, 8, 9, 10] ## Label, X1, X5, Y, and 4 first val of NF
            X_TiO2_ZnO = df.drop(df.columns[removed_col], axis=1).astype(float)
            X_TiO2_ZnO.columns = range(X_TiO2_ZnO.columns.size)
            X_clean = preprocessing_db(X_TiO2_ZnO, MAX_bonds, INIT_POS = 5)
            X_train_val, X_test, y_train_val, y_test = train_test_split(X_clean, Y_TiO2_ZnO, test_size=0.15)
            models, bias, scaler, dropped_cols = train_model(X_train_val, y_train_val, ratio_val_adjusted, True, 'papas_eq1', has_paper_params=True, seeds=seeds)
            fit_models(X_test, y_test, models, bias, scaler, dropped_cols, is_joined=True, article_eq_name='papas_eq1', has_paper_params=True)

            ## remove 0s and keep NF values individually
            zero_cols = [col for col, val in X_TiO2_ZnO.sum().iteritems() if val == 0]
            X_TiO2_ZnO.drop(zero_cols, axis=1, inplace=True)
            X_TiO2_ZnO.columns = range(X_TiO2_ZnO.columns.size)
            X_train_val, X_test, y_train_val, y_test = train_test_split(X_TiO2_ZnO, Y_TiO2_ZnO, test_size=0.15)
            models, bias, scaler, dropped_cols = train_model(X_train_val, y_train_val, ratio_val_adjusted, False, 'papas_eq1', has_paper_params=False, seeds=seeds) 
            fit_models(X_test, y_test, models, bias, scaler, dropped_cols, is_joined=False, article_eq_name='papas_eq1', has_paper_params=False)
            
            ## selected params in paper + NoZeros
            removed_col = [0, 2, 5, 6, 7, 8, 9, 10] ## Label, X1, X5, Y, and 4 first val of NF
            X_TiO2_ZnO = df.drop(df.columns[removed_col], axis=1).astype(float)
            X_TiO2_ZnO.columns = range(X_TiO2_ZnO.columns.size)
            X_train_val, X_test, y_train_val, y_test = train_test_split(X_TiO2_ZnO, Y_TiO2_ZnO, test_size=0.15)
            models, bias, scaler, dropped_cols = train_model(X_train_val, y_train_val, ratio_val_adjusted, False, 'papas_eq1', has_paper_params=True, seeds=seeds)
            fit_models(X_test, y_test, models, bias, scaler, dropped_cols, is_joined=False, article_eq_name='papas_eq1', has_paper_params=True)

        if paper_analysis == "gajevicz":
            ## No filter based on name tag
            Y = df[4].astype(float)
            removed_col = [0, 4, 5, 6, 7] ## Label, Y, and 3 first val of NF
            X = df.drop(df.columns[removed_col], axis=1).astype(float) ## dropped: label component, Size Comp(dup in NF), Y, AtomicNum(only eq2,3)
            X.columns = range(X.columns.size)
            MAX_bonds = int(df.iloc[0,6])
            paper_eq_name = 'gajewicz_eq4'
            X_train_val, X_test, y_train_val, y_test = train_test_split(X, Y, test_size=0.15)
            models, bias, scaler, dropped_cols = train_model(X_train_val, y_train_val, ratio_val_adjusted, False, paper_eq_name, seeds=seeds) 
            fit_models(X_test, y_test, models, bias, scaler, dropped_cols, is_joined=False, article_eq_name=paper_eq_name)

        if paper_analysis == "anantha":
            ## No filter based on name tag
            ## load another file as df2 so that it can be appended to df
            ## read csv file
            tk.Tk().withdraw() 
            filename = tk.filedialog.askopenfilename()
            print(filename)
            df2 = pd.read_csv(filename, sep=";", header = None)
            df2 = df2.T
            full_df = pd.concat([df, df2])
            Y_categorical = full_df[23].astype("string")
            Y = [float(u in 'Toxic') for u in Y_categorical]
            removed_col = [0, 5, 7, 8, 9, 10, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24, 25] ## Label, CellLine, CellType, and Y
            X = full_df.drop(full_df.columns[removed_col], axis=1).astype(float) 
            X.columns = range(X.columns.size)
            MAX_bonds = int(full_df.iloc[0,25])
            paper_eq_name = 'anantha_eq1'
            X_train_val, X_test, y_train_val, y_test = train_test_split(X, Y, test_size=0.15)
            models, bias, scaler, dropped_cols = train_model(X_train_val, y_train_val, ratio_val_adjusted, False, paper_eq_name, has_paper_params=True, seeds=seeds) 
            fit_models(X_test, y_test, models, bias, scaler, dropped_cols, is_joined=False, article_eq_name=paper_eq_name, has_paper_params=True)

if __name__ == "__main__":
    main()
