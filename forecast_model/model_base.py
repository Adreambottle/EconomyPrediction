#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression, Ridge, ElasticNet
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsRegressor as Knn

# from extra_models.CNNClassifier import CNNClassifier
# from extra_models.DQN import DQN
# from extra_models.ESN import ESNClassifier
import warnings

warnings.filterwarnings('ignore')
"""
基础模型库
该系统如需要添加新的建模方法，请在这里添加，模型要求如下：
    1：模型封装成类
    2：模型需要有fit方法可以调用xx.fit(x_train,y_train)训练
    3：模型需要有predict方法可以调用xx.predict(x_test)预测
    4：需添加相应的para_dict及模型主要参数，方便相应的模型参数寻优
"""
class_model_list = ["svm", "xgb", "gbdt", "rfc", "lr", "dtc", "knnc", "bagc", "etc", "adac", 'cnn', 'esn', 'dqn']
regression_model_list = ["knn", "rf", "ada", "gbrt", "svr", "lasso", "decision tree", "linear", "ridge", "enet"]
extra_model_list = ['cnn', 'esn', 'dqn']

"""
"svm"   sklearn.svm.SVC
"xgb"   xgboost.XGBClassifier
"gbdt"  sklearn.ensemble.GradientBoostingClassifier
"rfc"   sklearn.ensemble.RandomForestClassifier
"lr"    sklearn.linear_model.LinearRegression
"dtc"   sklearn.tree.DecisionTreeClassifier
"knnc"  sklearn.neighbors.KNeighborsClassifier
"bagc"  sklearn.ensemble.BaggingClassifier
"etc"   sklearn.ensemble.ExtraTreesClassifier
"adac"  sklearn.ensemble.AdaBoostClassifier

# Extra_model_list
'cnn'   extra_models.CNNClassifier.CNNClassifier
'esn'   extra_models.ESN.ESNClassifier
'dqn'   extra_models.DQN.DQN

# Regressor
"knn"     sklearn.neighbors.KNeighborsRegressor as Knn
"rf"      sklearn.ensemble.RandomForestClassifier
"ada"     sklearn.ensemble.AdaBoostRegressor
"gbrt"    sklearn.ensemble.GradientBoostingRegressor
"svr"     sklearn.svm.SVR
"lasso"   sklearn.linear_model.Lasso
"decision tree"       sklearn.tree.DecisionTreeRegressor
"linear"  sklearn.linear_model.LinearRegression
"ridge"   sklearn.linear_model.Ridge
"enet"    sklearn.linear_model.ElasticNet
"""



def get_para(model_type, feature_num):
    # 定义搜寻范围
    svr_para_dict = [
        {"C": [0.1, 1, 10, 100], "kernel": ['linear'], "epsilon": [0.01, 0.05, 0.1, 0.5, 1]},
        {"C": [0.1, 1, 10, 100], "kernel": ['poly'], "degree": [2, 3, 4],
         "gamma": [1 / feature_num, 1e-2, 1e-3, 10], "epsilon": [0.01, 0.05, 0.1, 0.5, 1]},
        {"C": [0.1, 1, 10, 100], "kernel": ['rbf', 'sigmoid'], "gamma": [1 / feature_num, 1e-2, 1e-3, 10],
         "epsilon": [0.01, 0.05, 0.1, 0.5, 1]}
    ]
    svc_para_dict = [
        {"C": [0.1, 1, 10, 100], "kernel": ['linear']},
        {"C": [0.1, 1, 10, 100], "kernel": ['poly'], "degree": [2, 3, 4],
         "gamma": [1 / feature_num, 1e-2, 1e-3, 10]},
        {"C": [0.1, 1, 10, 100], "kernel": ['rbf', 'sigmoid'], "gamma": [1 / feature_num, 1e-2, 1e-3, 10]}
    ]
    xgb_c_para_dict = [
        {"": [0.01, 0.1, 1], "min_child_weight": [0.1, 0.5, 1, 2, 5], "max_depth": [3, 4, 5, 6],
         "n_estimators": [20, 40, 60, 80, 100, 120, 140]}
    ]
    gbdt_c_para_dict = [
        {"learning_rate": [0.01, 0.1, 1], "n_estimators": [20, 40, 60, 80, 100, 120, 140], "max_depth": [2, 3, 4],
         "min_samples_split": [2, 3, 4, 5], "min_samples_leaf": [1, 2, 3], "random_state": [42]}
    ]
    gbdt_r_para_dict = [
        {"loss": ['ls', 'lad', 'huber', 'quantile'], "learning_rate": [0.01, 0.1, 1],
         "n_estimators": [20, 40, 60, 80, 100, 120, 140], "max_depth": [2, 3, 4],
         "min_samples_split": [2, 3, 4, 5], "min_samples_leaf": [1, 2, 3],"random_state":[42]}
    ]
    rf_c_para_dictt = [
        {"n_estimators": [5, 10, 15, 20], "max_features": ['sqrt', None], "min_samples_split": [2, 3, 4],
         "min_samples_leaf": [1, 2, 3], "random_state": [42],"bootstrap":[False]}
    ]
    rf_r_para_dictt = [
        {"criterion": ["mse", "mae"], "n_estimators": [5, 10, 15, 20], "max_features": ['sqrt', None],
         "min_samples_split": [2, 3, 4], "min_samples_leaf": [1, 2, 3], "random_state": [42],
         "bootstrap":[False]}
    ]

    logis_para_dict = [{"penalty": ["l1", "l2"], "C": [0.1, 0.5, 1.0, 1.5, 2.0],"random_state": [42]}]
    dt_c_pata_dict = [
        {"criterion": ["gini", "entropy"], "min_samples_split": [2, 3, 4, 5], "min_samples_leaf": [1, 2, 3, 4],
         "random_state": [42]}
    ]
    dt_r_pata_dict = [
        {"criterion": ["mse", "mae"], "min_samples_split": [2, 3, 4, 5], "min_samples_leaf": [1, 2, 3, 4],
         "random_state": [42]}
    ]
    knn_c_para_dict = [{"n_neighbors": [3, 4, 5, 6], "p": [1, 2], "weights": ["uniform", "distance"]}]
    knn_r_para_dict = [{"n_neighbors": [3, 4, 5, 6, 7, 8], "p": [1, 2], "weights": ["uniform", "distance"]}]
    bag_para_dict = [{"n_estimators": [5, 8, 10, 15], "random_state": [42],"bootstrap":[False]}]
    etc_para_dict = [
        {"criterion": ["gini", "entropy"], "min_samples_split": [2, 3, 4, 5], "min_samples_leaf": [1, 2, 3, 4],
         "random_state": [42]}]
    ada_c_para_dict = [{"n_estimators": [10, 20, 30, 40, 50, 60, 70, 80], "learning_rate": [0.1, 0.5, 1.0, 1.5, 2.0],
                        "random_state": [42]}]
    ada_r_para_dict = [{"n_estimators": [10, 20, 30, 40, 50, 60, 70, 80], "learning_rate": [0.1, 0.5, 1.0, 1.5, 2.0],
                        "loss": ['linear', 'square', 'exponential'], "random_state": [42]}]
    lasso_para_dict = [{"alpha": [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]}]
    linear_para_dict = [{"normalize": [True, False]}]
    ridge_para_dict = [{"alpha": [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5],"random_state": [42]}]
    enet_para_dict = [{"l1_ratio": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
    dqn_c_para_dict = [{"training_steps": [10000, 20000], "memory_size": [1000, 2000], "batch_size": [16, 32, 64],
                        "target_update": [200], "epsilon_decay": [1 / 2000, 1 / 1000]}]
    cnn_c_para_dict = [{"verbose": [True, False], "augment": [False, True]}]
    esn_c_para_dict = [{"n_reservoir": [500, 1000, 2000], "sparsity": [0.3, 0.5, 0.7], "spectral_radius": [0.6, 0.8],
                        "noise": [0.005, 0.007]}]

    if model_type == 'svm':
        return svc_para_dict
    elif model_type == 'xgb':
        return xgb_c_para_dict
    elif model_type == 'gbdt':
        return gbdt_c_para_dict
    elif model_type == 'rfc':
        return rf_c_para_dictt
    elif model_type == 'lr':
        return logis_para_dict
    elif model_type == 'dtc':
        return dt_c_pata_dict
    elif model_type == 'knnc':
        return knn_c_para_dict
    elif model_type == 'bagc':
        return bag_para_dict
    elif model_type == 'etc':
        return etc_para_dict
    elif model_type == 'adac':
        return ada_c_para_dict
    elif model_type == 'sqn':
        return dqn_c_para_dict
    elif model_type == 'cnn':
        return cnn_c_para_dict
    elif model_type == 'esn':
        return esn_c_para_dict

    # ##############回归###################

    elif model_type == 'knn':
        return knn_r_para_dict
    elif model_type == 'rf':
        return rf_r_para_dictt
    elif model_type == 'ada':
        return ada_r_para_dict
    elif model_type == 'gbrt':
        return gbdt_r_para_dict
    elif model_type == 'svr':
        return svr_para_dict
    elif model_type == 'lasso':
        return lasso_para_dict
    elif model_type == 'decision tree':
        return dt_r_pata_dict
    elif model_type == 'linear':
        return linear_para_dict
    elif model_type == 'ridge':
        return ridge_para_dict
    elif model_type == 'enet':
        return enet_para_dict
    else:
        raise ValueError("分类模型选择错误，请在可选的分类列表中选择分类模型")

    pass


def get_base_model(model_type, para=None):
    # #############分类模型#####################
    if model_type == 'svm':
        if para is None:
            model = SVC()
        else:
            model = SVC(**para)
    elif model_type == 'xgb':
        if para is None:
            model = xgb.XGBClassifier()
        else:
            model = xgb.XGBClassifier(**para)
    elif model_type == 'gbdt':
        if para is None:
            model = GradientBoostingClassifier(random_state=42)
        else:
            model = GradientBoostingClassifier(**para)
    elif model_type == 'rfc':
        if para is None:
            model = RandomForestClassifier(bootstrap=False,random_state=42)
        else:
            model = RandomForestClassifier(**para)
    elif model_type == 'lr':
        if para is None:
            model = LogisticRegression(random_state=42)
        else:
            model = LogisticRegression(**para)
    elif model_type == 'dtc':
        if para is None:
            model = DecisionTreeClassifier(random_state=42)
        else:
            model = DecisionTreeClassifier(**para)
    elif model_type == 'knnc':
        if para is None:
            model = KNeighborsClassifier()
        else:
            model = KNeighborsClassifier(**para)
    elif model_type == 'bagc':
        if para is None:
            model = BaggingClassifier(KNeighborsClassifier(), random_state=42,bootstrap=False)
        else:
            model = BaggingClassifier(KNeighborsClassifier(), **para)
    elif model_type == 'etc':
        if para is None:
            model = ExtraTreesClassifier(random_state=42)
        else:
            model = ExtraTreesClassifier(**para)
    elif model_type == 'adac':
        if para is None:
            model = AdaBoostClassifier(random_state=42)
        else:
            model = AdaBoostClassifier(**para)
    # elif model_type == 'dqn':
    #     if para is None:
    #         model = DQN()
    #     else:
    #         model = DQN(**para)
    # elif model_type == 'cnn':
    #     if para is None:
    #         model = CNNClassifier()
    #     else:
    #         model = CNNClassifier(**para)
    # elif model_type == 'esn':
    #     if para is None:
    #         model = ESNClassifier()
    #     else:
    #         model = ESNClassifier(**para)

    # ##############回归模型#######################
    elif model_type == 'knn':
        if para is None:
            model = Knn()
        else:
            model = Knn(**para)
    elif model_type == 'rf':
        if para is None:
            model = RandomForestRegressor(random_state=42,bootstrap=False)
        else:
            model = RandomForestRegressor(**para)
    elif model_type == 'ada':
        if para is None:
            model = AdaBoostRegressor(random_state=42)
        else:
            model = AdaBoostRegressor(**para)
    elif model_type == 'gbrt':
        if para is None:
            model = GradientBoostingRegressor(random_state=42)
        else:
            model = GradientBoostingRegressor(**para)
    elif model_type == 'svr':
        if para is None:
            model = SVR()
        else:
            model = SVR(**para)
    elif model_type == 'lasso':
        if para is None:
            model = Lasso()
        else:
            model = Lasso(**para)
    elif model_type == 'decision tree':
        if para is None:
            model = DecisionTreeRegressor(random_state=42)
        else:
            model = DecisionTreeRegressor(**para)
    elif model_type == 'linear':
        if para is None:
            model = LinearRegression()
        else:
            model = LinearRegression(**para)
    elif model_type == 'ridge':
        if para is None:
            model = Ridge(random_state=42)
        else:
            model = Ridge(**para)
    elif model_type == 'enet':
        if para is None:
            model = ElasticNet()
        else:
            model = ElasticNet(**para)
    else:
        raise ValueError("分类模型选择错误，请在可选的分类列表中选择分类模型")
    return model
    pass


def weight_balance_para(model_name, weight_scale=None):
    if model_name in ["svm", "rfc", "lr", 'dtc', 'etc']:
        para_dict = {"class_weight": "balanced"}
        pass
    elif model_name in ["xgb"]:
        assert weight_scale is not None
        para_dict = {"scale_pos_weight": weight_scale}
        pass
    else:
        para_dict = {}
    return para_dict
    pass
