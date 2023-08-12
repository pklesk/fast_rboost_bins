"""
Auxiliary script with command-line user interface for conducting experiments on classifiers related to this project: 
``FastRealBoostBins`` (two variants) against other ensemble classifiers from ``sklearn.ensemble``.

By executing 

.. code-block:: console
    python main_experimenter.py -h 

one obtains help on script arguments.  
"""

import numpy as np
from frbb import FastRealBoostBins
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from keras.datasets import cifar10, mnist
import time
import re
from itertools import product, compress
from matplotlib import pyplot as plt
from utils import cpu_and_system_props, gpu_props, dict_to_str
import pickle
from datetime import date
import sys
import argparse
import colorama

__author__ = "Przemysław Klęsk"
__email__ = "pklesk@zut.edu.pl"

np.set_printoptions(linewidth=512)
np.set_printoptions(threshold=np.inf)    

# folders
FOLDER_EXTRAS = "../extras/"
FOLDER_DATA_RAW = "../data_raw/"
FOLDER_DATA = "../data/"

# main settings and constants
DATA_KIND = "random" # choices: {"real", "random"}
REAL_DATA_DEFS = [
    ("fddb-patches", "read_data_fddb_patches", "FDDB-PATCHES (3NPI)"),
    ("cifar-10", "read_data_cifar_10", "CIFAR-10 (AIRPLANE)"),
    ("mnist-b", "read_data_mnist_b", "MNIST-B (DIGIT 0)"),
    ("fddb-hfs-100", "read_data_fddb_haar_npi_100", "FDDB-HFs (100NPI)"),
    ("fddb-hfs-300", "read_data_fddb_haar_npi_300", "FDDB-HFs (300NPI)"),    
    ("hagrid-hfs-10", "read_data_hagrid_haar_npi_10", "HaGRID-HFs (PALM, 10NPI)"),    
    ("hagrid-hfs-30", "read_data_hagrid_haar_npi_30", "HaGRID-HFs (PALM, 30NPI)")    
    ]
REAL_DATA_FLAGS = [True, False, False, False, False, False, False]
CLFS_DEFS = [
        (AdaBoostClassifier, {"algorithm": "SAMME.R"}, {"color": "black"}),
        (GradientBoostingClassifier, {"max_depth": 1}, {"color": "green"}),
        (HistGradientBoostingClassifier, {"max_depth": 1, "early_stopping": False}, {"color": "orange"}),
        (FastRealBoostBins, {"fit_mode": "numba_jit", "decision_function_mode": "numba_jit"}, {"color": "blue"}),
        (FastRealBoostBins, {"fit_mode": "numba_cuda", "decision_function_mode": "numba_cuda"}, {"color": "red"})        
        ]
CLFS_FLAGS = [True, False, False, True, True]
RANDOM_DTYPE = np.int8 # choices: {np.int8, np.uint8, ..., np.int64, np.uint64} or {np.float32, np.float64}
NMM_MAGN_ORDERS = [(3, 4, 4)] # only in case of data kind "random"
TS = [16, 32, 64]
BS = [8]
SEED = 0
PLOTS = False
PLOTS_ARG_NAME = "T"
PLOTS_VALUES_NAMES = ["acc_test", "acc_train", "time_fit", "time_predict_train", "time_predict_test"]
EPS = 1e-9

# plot settings
PLOT_FONTSIZE_SUPTITLE = 13
PLOT_FONTSIZE_TITLE = 9.5
PLOT_FONTSIZE_AXES = 12.5
PLOT_FONTSIZE_LEGEND = 9.5
PLOT_FIGSIZE = (10, 6.5)
PLOT_MARKERSIZE = 4
PLOT_GRID_COLOR = (0.4, 0.4, 0.4) 
PLOT_GRID_DASHES = (4.0, 4.0)
PLOT_LEGEND_LOC = "best"
PLOT_LEGEND_HANDLELENGTH = 4
PLOT_LEGEND_LABELSPACING = 0.1

def clean_name(name):
    name = name.replace("\n", "")
    name = name.replace("\t", "")
    name = re.sub(" +", " ", name)
    return name

def hash_function(s):
    h = 0
    for c in s:
        h *= 31 
        h += ord(c)
    return h

def experiment_hash_str(kind="real", params=None, digits=10):
    return kind + "_" + str((hash_function(str(params)) & ((1 << 32) - 1)) % 10**digits).rjust(digits, "0") + "_" + date.today().strftime("%Y%m%d")

def list_to_str(l):
    list_str = ""
    for i, elem in enumerate(l):
        list_str += "[" if i == 0 else " "  
        list_str += str(elem) + (",\n" if i < len(l) - 1 else "]")
    return list_str 

def unpickle_objects(fname):
    print(f"UNPICKLE OBJECTS... [from file: {fname}]")
    t1 = time.time()
    try:    
        f = open(fname, "rb")
        some_list = pickle.load(f)
        f.close()
    except IOError:
        sys.exit("[error occurred when trying to open or read the file]")
    t2 = time.time()
    print(f"UNPICKLE OBJECTS DONE. [time: {t2 - t1} s]")
    return some_list

def read_data_fddb_patches(): 
    fname = "fddb_patches/fddb_patches_32x32_NPI_3_SEED_0.bin"
    [X_train, y_train, X_test, y_test] = unpickle_objects(FOLDER_DATA_RAW + fname)
    n = np.product(X_train.shape[1:])
    X_train = np.reshape(X_train, (X_train.shape[0], n))
    X_test = np.reshape(X_test, (X_test.shape[0], n))         
    return X_train, y_train, X_test, y_test

def read_data_cifar_10():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    class_index = 0 # 'airplane'
    indexes = y_train == class_index
    y_train[indexes] = 1
    y_train[~indexes] = -1
    y_train = y_train.flatten()
    indexes = y_test == class_index
    y_test[indexes] = 1
    y_test[~indexes] = -1
    y_test = y_test.flatten()   
    n = X_train.shape[1] * X_train.shape[2] * X_train.shape[3]
    X_train = np.reshape(X_train, (X_train.shape[0], n))
    X_test = np.reshape(X_test, (X_test.shape[0], n))         
    return X_train, y_train, X_test, y_test

def read_data_mnist_b(seed=0):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    indexes = y_train == 0
    y_train[indexes] = 1
    y_train[~indexes] = -1
    y_train = y_train.flatten()
    indexes = y_test == 0
    y_test[indexes] = 1
    y_test[~indexes] = -1
    y_test = y_test.flatten()   
    n = X_train.shape[1] * X_train.shape[2]
    X_train = np.reshape(X_train, (X_train.shape[0], n))
    X_test = np.reshape(X_test, (X_test.shape[0], n))
    np.random.seed(seed)
    noise_train = np.random.randint(0, 256, X_train.shape, dtype=np.uint16)
    X_train = ((X_train.astype(np.uint16) + noise_train) // 2).astype(np.uint8)
    noise_test = np.random.randint(0, 256, X_test.shape, dtype=np.uint16)
    X_test = ((X_test.astype(np.uint16) + noise_test) // 2).astype(np.uint8)             
    return X_train, y_train, X_test, y_test                    
    
def read_data_fddb_haar_npi_100(): 
    fname = "data_face_n_18225_S_5_P_5_NPI_100_SEED_0.bin"
    [X_train, y_train, X_test, y_test] = unpickle_objects(FOLDER_DATA + fname)
    return X_train, y_train, X_test, y_test

def read_data_fddb_haar_npi_300(): 
    fname = "data_face_n_18225_S_5_P_5_NPI_300_SEED_0.bin"
    [X_train, y_train, X_test, y_test] = unpickle_objects(FOLDER_DATA + fname)
    return X_train, y_train, X_test, y_test

def read_data_hagrid_haar_npi_10(): 
    fname = "data_hand_n_18225_S_5_P_5_NPI_10_SEED_0.bin"
    [X_train, y_train, X_test, y_test] = unpickle_objects(FOLDER_DATA + fname)
    return X_train, y_train, X_test, y_test

def read_data_hagrid_haar_npi_30(): 
    fname = "data_hand_n_18225_S_5_P_5_NPI_30_SEED_0.bin"
    [X_train, y_train, X_test, y_test] = unpickle_objects(FOLDER_DATA + fname)
    return X_train, y_train, X_test, y_test    
    
def experimenter_for_random_data(dtype=RANDOM_DTYPE, nmm_magn_orders=NMM_MAGN_ORDERS,
                                 clfs_defs=CLFS_DEFS, clfs_flags=CLFS_FLAGS, 
                                 Ts=TS, Bs=BS, seed=0, 
                                 plots=PLOTS, plots_arg_name=PLOTS_ARG_NAME, plots_values_names=PLOTS_VALUES_NAMES,
                                 cpu_props=None, gpu_props=None):
    print("EXPERIMENTER FOR RANDOM DATA...")
    hash_str = experiment_hash_str("random", locals())
    print(f"[experiment hash string: {hash_str}]")    
    print(f"[clfs definitions:]")
    for clf_id, (clf_class, clf_consts, _) in enumerate(clfs_defs):
        print(f"[clf def {clf_id} (active: {clfs_flags[clf_id]}): {clf_class.__name__}({clf_consts})]")
    print(f"[other settings -> dtype: {dtype.__name__}, nmm_magn_orders: {nmm_magn_orders}, Ts: {Ts}, Bs: {Bs}, seed: {seed}]")
    clfs_defs = list(compress(clfs_defs, clfs_flags))        
    cpu_gpu_info = f"[cpu: {cpu_props['cpu_name']}, gpu: {gpu_props['name']}]".upper()
    t1 = time.time()
    np.random.seed(seed)
    max_value = (np.iinfo(dtype).max // 2) if np.issubdtype(dtype, np.integer) else 1.0                
    n_experiments = len(nmm_magn_orders) * len(Ts) * len(Bs)
    experiments_descr = np.empty(n_experiments, dtype=object)    
    results_descr = np.empty((n_experiments, len(clfs_defs)), dtype=object)
    clfs_names = np.empty((n_experiments, len(clfs_defs)), dtype=object)    
    for experiment_id, (nmmo, T, B) in enumerate(product(nmm_magn_orders, Ts, Bs)):
        print("=" * 196)
        print(f"[experiment: {experiment_id + 1}/{n_experiments}, params: {(nmmo, T, B)}]")              
        n = int(10**nmmo[0])
        m_train = int(10**nmmo[1])
        m_test = int(10**nmmo[2])               
        X_train = (max_value * np.random.rand(m_train, n)).astype(dtype)
        y_train = (np.random.randint(0, 2, size=m_train) * 2 - 1).astype(np.int8)
        X_test = (max_value * np.random.rand(m_test, n)).astype(dtype)
        y_test = (np.random.randint(0, 2, size=m_test) * 2 - 1).astype(np.int8)        
        experiments_descr[experiment_id] = {"T": T, "B": B, "n": n, "m_train": m_train, "m_test": m_test}
        print(f"[description: {experiments_descr[experiment_id]}]")
        print("=" * 196)
        results_arr = []
        clfs_now = []
        for clf_id, (clf_class, clf_consts, _) in enumerate(clfs_defs):
            clf = clf_class(**clf_consts)
            params = clf.get_params()
            if isinstance(clf, AdaBoostClassifier):
                params["n_estimators"] = T
            if isinstance(clf, GradientBoostingClassifier):
                params["n_estimators"] = T                
            if isinstance(clf, HistGradientBoostingClassifier):
                params["max_iter"] = T
                params["max_bins"] = B
            if isinstance(clf, FastRealBoostBins):
                params["T"] = T
                params["B"] = B                                
            clf.set_params(**params)
            clfs_now.append(clf)
            clfs_names[experiment_id, clf_id] = clean_name(str(clf))            
        
        for clf_id, clf in enumerate(clfs_now):
            print(f"[clf: {clfs_names[experiment_id, clf_id]}]")
            print(f"[fit...]")
            t1_fit = time.time()
            clf.fit(X_train, y_train)
            t2_fit = time.time()            
            time_fit = t2_fit - t1_fit
            print(f"[fit done; time: {time_fit} s]")
            print(f"[predict train...]")
            t1_predict_train = time.time()
            predictions_train = clf.predict(X_train)
            t2_predict_train = time.time()        
            time_predict_train = t2_predict_train - t1_predict_train
            acc_train = np.mean(predictions_train == y_train)
            print(f"[predict train done; time: {time_predict_train} s, acc: {acc_train}]")
            print(f"[predict test...]")
            t1_predict_test = time.time()
            predictions_test = clf.predict(X_test)
            t2_predict_test = time.time()        
            time_predict_test = t2_predict_test - t1_predict_test
            acc_test = np.mean(predictions_test == y_test)
            print(f"[predict test done; time: {time_predict_test} s, acc: {acc_test}]")
            results_arr.append([time_fit, time_predict_train, time_predict_test, acc_train, acc_test])            
            results_descr[experiment_id, clf_id] = {"time_fit": [time_fit, None], "time_predict_train": [time_predict_train, None], "time_predict_test": [time_predict_test, None],
                                                     "acc_train" : [acc_train, None], "acc_test" : [acc_test, None]}
            print("-" * 196)
        results_arr = np.array(results_arr)
        results_arr[results_arr == 0.0] = EPS
        results_ratios = np.max(results_arr, axis=0) / results_arr  
        print(f"[results summary for experiment {experiment_id + 1}/{n_experiments} ({experiments_descr[experiment_id]}):]")
        for clf_id in range(len(clfs_defs)):
            results_descr[experiment_id, clf_id]["time_fit"][1] = results_ratios[clf_id, 0]
            results_descr[experiment_id, clf_id]["time_predict_train"][1] = results_ratios[clf_id, 1]
            results_descr[experiment_id, clf_id]["time_predict_test"][1] = results_ratios[clf_id, 2]
            results_descr[experiment_id, clf_id]["acc_train"][1] = results_ratios[clf_id, 3]
            results_descr[experiment_id, clf_id]["acc_test"][1] = results_ratios[clf_id, 4]
            print(f"[{clf_id}: {clfs_names[experiment_id, clf_id]}]")
            print(f"[{clf_id}: {results_descr[experiment_id, clf_id]}]")                            
    t2 = time.time()
    if plots and plots_arg_name and plots_values_names:
        print("[about to generate wanted plots]")
    print(f"EXPERIMENT FOR RANDOM DATA DONE. [time: {t2 - t1} s, hash string: {hash_str}]")
    if plots and plots_arg_name and plots_values_names:
        value_names_mapper = {"time_fit": "FIT TIME [s]", "time_predict_train": "PREDICT TIME (TRAIN) [s]", "time_predict_test": "PREDICT TIME (TEST) [s]",
                              "acc_train": "ACC (TRAIN)", "acc_test": "ACC (TEST)"}        
        nonargs = {}
        nonargs_experiment_ids = {}
        for experiment_id in range(n_experiments):
            descr = experiments_descr[experiment_id].copy()
            descr.pop(plots_arg_name)
            key = str(descr)
            if key not in nonargs: 
                nonargs[key] = descr
                nonargs_experiment_ids[key] = []
            nonargs_experiment_ids[key].append(experiment_id)
        for key, vn in product(nonargs.keys(), plots_values_names):
            plt.figure(1, figsize=PLOT_FIGSIZE)
            y_min = np.inf
            y_max = -np.inf
            for clf_id, (clf_class, clf_consts, clf_style) in enumerate(clfs_defs): 
                xs = []
                ys = []
                for experiment_id in nonargs_experiment_ids[key]:
                    ed = experiments_descr[experiment_id]
                    rd = results_descr[experiment_id, clf_id] 
                    xs.append(ed[plots_arg_name])
                    ys.append(rd[vn][0])                    
                label = f"{clf_class.__name__}({clf_consts})"                
                plt.plot(xs, ys, label=label, **{**clf_style, **{"marker": "o", "markersize": PLOT_MARKERSIZE}})
                y_min = min(y_min, min(ys))
                y_max = max(y_max, max(ys))                    
            plt.xticks(xs)    
            plt.suptitle(f"{value_names_mapper[vn]}\n{key}", fontsize=PLOT_FONTSIZE_SUPTITLE)
            plt.title(f"\n{cpu_gpu_info}", fontsize=PLOT_FONTSIZE_TITLE)
            plt.xlabel(plots_arg_name.upper(), fontsize=PLOT_FONTSIZE_AXES)            
            plt.ylabel(value_names_mapper[vn], fontsize=PLOT_FONTSIZE_AXES)
            plt.yscale("log")            
            plt.legend(loc=PLOT_LEGEND_LOC, prop={"size": PLOT_FONTSIZE_LEGEND}, handlelength=PLOT_LEGEND_HANDLELENGTH, labelspacing=PLOT_LEGEND_LABELSPACING)        
            plt.grid(color=PLOT_GRID_COLOR, zorder=0, dashes=PLOT_GRID_DASHES)
            plt.tight_layout()
            fake_data_name = f"{nonargs[key]['n']}-{nonargs[key]['m_train']}-{nonargs[key]['m_test']}"
            fname = "fig_experiment_" + hash_str + "_" + fake_data_name + "_" + vn
            plt.savefig(FOLDER_EXTRAS +  fname + ".eps")
            plt.savefig(FOLDER_EXTRAS +  fname + ".pdf")
            plt.close()
            
def experimenter_for_real_data(real_data_defs=REAL_DATA_DEFS, real_data_flags=REAL_DATA_FLAGS,
                               clfs_defs=CLFS_DEFS, clfs_flags=CLFS_FLAGS,
                               Ts=TS, Bs=BS, seed=SEED, 
                               plots=PLOTS, plots_arg_name=PLOTS_ARG_NAME, plots_values_names=PLOTS_VALUES_NAMES,
                               cpu_props=None, gpu_props=None):
    print("EXPERIMENTER FOR REAL DATA...")    
    hash_str = experiment_hash_str("real", locals())
    print(f"[experiment hash string: {hash_str}]")
    print(f"[data definitions:]")
    for data_id, (data_name_short, data_reader, data_name_full) in enumerate(real_data_defs):
        print(f"[data def {data_id} (active: {real_data_flags[data_id]}): (name short: '{data_name_short}', reading function: {data_reader}(), name full: '{data_name_full})']")        
    print(f"[clfs definitions:]")
    for clf_id, (clf_class, clf_consts, _) in enumerate(clfs_defs):
        print(f"[clf def {clf_id} (active: {clfs_flags[clf_id]}): {clf_class.__name__}({clf_consts})]")
    print(f"[other settings -> Ts: {Ts}, Bs: {Bs}, seed: {seed}]")        
    real_data_defs = list(compress(real_data_defs, real_data_flags))        
    clfs_defs = list(compress(clfs_defs, clfs_flags))    
    cpu_gpu_info = f"[cpu: {cpu_props['cpu_name']}, gpu: {gpu_props['name']}]".upper()
    t1 = time.time()
    np.random.seed(seed)
    n_experiments = len(real_data_defs) * len(Ts) * len(Bs)
    experiments_descr = np.empty(n_experiments, dtype=object)    
    results_descr = np.empty((n_experiments, len(clfs_defs)), dtype=object)
    clfs_names = np.empty((n_experiments, len(clfs_defs)), dtype=object)
    datas = [(data_name_full, globals()[reader_name]()) for data_name_short, reader_name, data_name_full in real_data_defs]    
    data_names_mapper = {data_name_full: data_name_short for data_name_short, _, data_name_full in real_data_defs}   
    for experiment_id, ((data_name, data), T, B) in enumerate(product(datas, Ts, Bs)):
        print("=" * 196)
        print(f"[experiment: {experiment_id + 1}/{n_experiments}, params: {(data_name, T, B)}]")
        X_train, y_train, X_test, y_test = data
        m_train, n = X_train.shape
        m_test = X_test.shape[0]
        experiments_descr[experiment_id] = {"T": T, "B": B, "data_name": data_name, "n": n, "m_train": m_train, "m_test": m_test}
        print(f"[description: {experiments_descr[experiment_id]}]")
        print("=" * 196)
        results_arr = []
        clfs_now = []
        for clf_id, (clf_class, clf_consts, _) in enumerate(clfs_defs):
            clf = clf_class(**clf_consts)
            params = clf.get_params()
            if isinstance(clf, AdaBoostClassifier):
                params["n_estimators"] = T
            if isinstance(clf, GradientBoostingClassifier):
                params["n_estimators"] = T                
            if isinstance(clf, HistGradientBoostingClassifier):
                params["max_iter"] = T
                params["max_bins"] = B
            if isinstance(clf, FastRealBoostBins):
                params["T"] = T
                params["B"] = B                                
            clf.set_params(**params)
            clfs_now.append(clf)
            clfs_names[experiment_id, clf_id] = clean_name(str(clf))            
        
        for clf_id, clf in enumerate(clfs_now):
            print(f"[clf: {clfs_names[experiment_id, clf_id]}]")
            print(f"[fit...]")
            t1_fit = time.time()
            clf.fit(X_train, y_train)
            t2_fit = time.time()            
            time_fit = t2_fit - t1_fit
            print(f"[fit done; time: {time_fit} s]")
            print(f"[predict train...]")
            t1_predict_train = time.time()
            predictions_train = clf.predict(X_train)
            t2_predict_train = time.time()        
            time_predict_train = t2_predict_train - t1_predict_train
            acc_train = np.mean(predictions_train == y_train)
            print(f"[predict train done; time: {time_predict_train} s, acc: {acc_train}]")
            print(f"[predict test...]")
            t1_predict_test = time.time()
            predictions_test = clf.predict(X_test)
            t2_predict_test = time.time()        
            time_predict_test = t2_predict_test - t1_predict_test
            acc_test = np.mean(predictions_test == y_test)
            print(f"[predict test done; time: {time_predict_test} s, acc: {acc_test}]")
            results_arr.append([time_fit, time_predict_train, time_predict_test, acc_train, acc_test])            
            results_descr[experiment_id, clf_id] = {"time_fit": [time_fit, None], "time_predict_train": [time_predict_train, None], "time_predict_test": [time_predict_test, None],
                                                     "acc_train" : [acc_train, None], "acc_test" : [acc_test, None]}
            print("-" * 196)
        results_arr = np.array(results_arr)
        results_arr[results_arr == 0.0] = EPS
        results_ratios = np.max(results_arr, axis=0) / results_arr  
        print(f"[results summary for experiment {experiment_id + 1}/{n_experiments} ({experiments_descr[experiment_id]}):]")
        for clf_id in range(len(clfs_defs)):
            results_descr[experiment_id, clf_id]["time_fit"][1] = results_ratios[clf_id, 0]
            results_descr[experiment_id, clf_id]["time_predict_train"][1] = results_ratios[clf_id, 1]
            results_descr[experiment_id, clf_id]["time_predict_test"][1] = results_ratios[clf_id, 2]
            results_descr[experiment_id, clf_id]["acc_train"][1] = results_ratios[clf_id, 3]
            results_descr[experiment_id, clf_id]["acc_test"][1] = results_ratios[clf_id, 4]
            print(f"[{clf_id}: {clfs_names[experiment_id, clf_id]}]")
            print(f"[{clf_id}: {results_descr[experiment_id, clf_id]}]")                            
    t2 = time.time()
    if plots and plots_arg_name and plots_values_names:
        print("[about to generate wanted plots]")
    print(f"EXPERIMENT FOR REAL DATA DONE. [time: {t2 - t1} s, hash string: {hash_str}]")
    if plots and plots_arg_name and plots_values_names:
        value_names_mapper = {"time_fit": "FIT TIME [s]", "time_predict_train": "PREDICT TIME (TRAIN) [s]", "time_predict_test": "PREDICT TIME (TEST) [s]", 
                              "acc_train": "ACC (TRAIN)", "acc_test": "ACC (TEST)"}        
        nonargs = {}
        nonargs_experiment_ids = {}
        for experiment_id in range(n_experiments):
            descr = experiments_descr[experiment_id].copy()
            descr.pop(plots_arg_name)
            key = str(descr)
            if key not in nonargs: 
                nonargs[key] = descr
                nonargs_experiment_ids[key] = []
            nonargs_experiment_ids[key].append(experiment_id)
        for key, vn in product(nonargs.keys(), plots_values_names):
            plt.figure(1, figsize=PLOT_FIGSIZE)
            y_min = np.inf
            y_max = -np.inf
            for clf_id, (clf_class, clf_consts, clf_style) in enumerate(clfs_defs): 
                xs = []
                ys = []
                for experiment_id in nonargs_experiment_ids[key]:
                    ed = experiments_descr[experiment_id]
                    rd = results_descr[experiment_id, clf_id] 
                    xs.append(ed[plots_arg_name])
                    ys.append(rd[vn][0])                    
                label = f"{clf_class.__name__}({clf_consts})"                
                plt.plot(xs, ys, label=label, **{**clf_style, **{"marker": "o", "markersize": PLOT_MARKERSIZE}})
                y_min = min(y_min, min(ys))
                y_max = max(y_max, max(ys))                    
            plt.xticks(xs)    
            plt.suptitle(f"{value_names_mapper[vn]}\n{key}", fontsize=PLOT_FONTSIZE_SUPTITLE)
            plt.title(f"\n{cpu_gpu_info}", fontsize=PLOT_FONTSIZE_TITLE)
            plt.xlabel(plots_arg_name.upper(), fontsize=PLOT_FONTSIZE_AXES)            
            plt.ylabel(value_names_mapper[vn], fontsize=PLOT_FONTSIZE_AXES)
            plt.yscale("log")               
            plt.legend(loc=PLOT_LEGEND_LOC, prop={"size": PLOT_FONTSIZE_LEGEND}, handlelength=PLOT_LEGEND_HANDLELENGTH, labelspacing=PLOT_LEGEND_LABELSPACING)        
            plt.grid(color=PLOT_GRID_COLOR, zorder=0, dashes=PLOT_GRID_DASHES)
            plt.tight_layout()
            fname = "fig_experiment_" + hash_str + "_" + data_names_mapper[nonargs[key]["data_name"]] + "_" + vn
            plt.savefig(FOLDER_EXTRAS +  fname + ".eps")
            plt.savefig(FOLDER_EXTRAS +  fname + ".pdf")
            plt.close()       
            
def str_to_dtype(s):
    return eval(s)

def str_to_tuple(s):
    return eval(s)

def str_to_bool(s):
    return eval(s)
            
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--CLFS_FLAGS", type=str_to_bool, default=CLFS_FLAGS, nargs="+", 
                        help=f"boolean flags (list) specifying which classifiers from the predefined set will participate in experiments (default: {CLFS_FLAGS}) (attention: type them using spaces as separators)")    
    parser.add_argument("-dk", "--DATA_KIND", type=str, default=DATA_KIND, choices=["real", "random"], help=f"kind of data on which to experiment (default: {DATA_KIND})")
    parser.add_argument("-rdf", "--REAL_DATA_FLAGS", type=str_to_bool, default=REAL_DATA_FLAGS, nargs="+", 
                        help=f"boolean flags (list) specifying which data sets from the predefined set will participate in experiments on real data (default: {REAL_DATA_FLAGS}) (attention: type them using spaces as separators)")
    parser.add_argument("-rd", "--RANDOM_DTYPE", type=str_to_dtype, default=RANDOM_DTYPE, choices=[np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64, np.float32, np.float64], 
                        help=f"dtype of input numpy arrays for experiments on random data (default: {RANDOM_DTYPE}) (attention: please type it as e.g. 'np.uint8', 'np.float32', etc.)")
    parser.add_argument("-nmm", "--NMM_MAGN_ORDERS", type=str_to_tuple, default=NMM_MAGN_ORDERS, nargs="+", 
                        help=f"list of tuples represented as strings defining orders of magnitude for random input arrays in experiments on random data (default: {NMM_MAGN_ORDERS}) (attention: type them using spaces as separators and with each tuple in quotation marks)")
    parser.add_argument("-ts", "--TS", type=int, default=TS, nargs="+", 
                        help=f"ensemble sizes (list) to impose on each type of classifier in experiments (default: {TS}) (attention: type them using spaces as separators)")
    parser.add_argument("-bs", "--BS", type=int, default=BS, nargs="+", 
                        help=f"bins counts (list) to impose on each type of classifier in experiments (default: {BS}) (attention: type them using spaces as separators)")
    parser.add_argument("-s", "--SEED", type=int, default=SEED, help=f"randomization seed, (default: {SEED})")    
    parser.add_argument("-p", "--PLOTS", type=str_to_bool, default=PLOTS, help=f"boolean flag indicating if plots should be generated after experiments (default: {PLOTS})")
    parser.add_argument("-pan", "--PLOTS_ARG_NAME", type=str, default=PLOTS_ARG_NAME, choices=["T", "B", "n", "m_train", "m_test"], 
                        help=f"name of argument quantity to be placed on horizontal axis in plots (default: {PLOTS_ARG_NAME})")
    parser.add_argument("-pvn", "--PLOTS_VALUES_NAMES", type=str, default=PLOTS_VALUES_NAMES, choices=PLOTS_VALUES_NAMES, nargs="+", 
                        help=f"names of value quantities to be placed on vertical axis in plots (default: {PLOTS_VALUES_NAMES}) (attention: type them using spaces as separators)")
    args = parser.parse_args()
    return vars(args)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    colorama.init()     
    print(colorama.Fore.LIGHTYELLOW_EX + "\"FAST-REAL-BOOST-BINS\": AN ENSEMBLE CLASSIFIER FOR FAST PREDICTIONS IMPLEMENTED IN PYTHON VIA NUMBA.JIT AND NUMBA.CUDA. [main_experimenter]" + colorama.Style.RESET_ALL, flush=True)
    print(colorama.Fore.YELLOW + "[for help use -h or --help switch]")        
    print(f"CLASSIFIERS DEFINITIONS:\n{list_to_str(CLFS_DEFS)}")
    print(f"REAL DATA DEFINITIONS:\n{list_to_str(REAL_DATA_DEFS)}")
    print(f"RANDOM DATA SIZES DEFINABLE VIA -nmm (--NMM_MAGN_ORDERS) OPTION." + colorama.Style.RESET_ALL)
    n_clfs_flags = len(CLFS_FLAGS)
    n_real_data_flags = len(REAL_DATA_FLAGS)
    args = parse_args()
    globals().update(args)                                                                      
    print(f"MAIN-EXPERIMENTER STARTING...")    
    cpu_props = cpu_and_system_props()
    gpu_props = gpu_props()
    print(f"CPU AND SYSTEM PROPS: {cpu_props}")
    print(f"GPU PROPS: {gpu_props}")
    print(f"ARGUMENTS:\n{dict_to_str(args)}") 
    data_kind = DATA_KIND
    clfs_defs = CLFS_DEFS
    clfs_flags = CLFS_FLAGS
    Ts = TS
    Bs = BS
    seed = SEED
    plots = PLOTS
    plots_arg_name = PLOTS_ARG_NAME
    plots_values_names = PLOTS_VALUES_NAMES
    if len(CLFS_FLAGS) != n_clfs_flags:
            print(f"[exactly {n_clfs_flags} classifiers flags must be specified via -cf (--CLFS_FLAGS) option]") 
    if data_kind == "real":
        if len(REAL_DATA_FLAGS) != n_real_data_flags:
            print(f"[exactly {n_real_data_flags} real data flags must be specified via -rdf (--REAL_DATA_FLAGS) option]")
        else:
            if np.any(REAL_DATA_FLAGS):   
                real_data_flags = REAL_DATA_FLAGS
                real_data_defs = REAL_DATA_DEFS        
                experimenter_for_real_data(real_data_defs=real_data_defs, real_data_flags=real_data_flags,
                                           clfs_defs=clfs_defs, clfs_flags=clfs_flags, 
                                           Ts=Ts, Bs=Bs, seed=seed, 
                                           plots=plots, plots_arg_name=plots_arg_name, plots_values_names=plots_values_names,
                                           cpu_props=cpu_props, gpu_props=gpu_props)
            else:
                print("[no real data selected as active for experiments, see -rdf (--REAL_DATA_FLAGS) option]")
    elif data_kind == "random":
        if NMM_MAGN_ORDERS:
            random_dtype = RANDOM_DTYPE
            experimenter_for_random_data(dtype=random_dtype, nmm_magn_orders=NMM_MAGN_ORDERS,
                                         clfs_defs=clfs_defs, clfs_flags=clfs_flags,                                
                                         Ts=Ts, Bs=Bs, seed=seed, 
                                         plots=plots, plots_arg_name=plots_arg_name, plots_values_names=plots_values_names,
                                         cpu_props=cpu_props, gpu_props=gpu_props)
        else:
            print("[no magnitude orders for sizes of random data selected, see -nmm (--NMM_MAGN_ORDERS)]")                  
    print("MAIN-EXPERIMENTER DONE.")