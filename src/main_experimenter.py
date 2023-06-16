import numpy as np
from frbb import FastRealBoostBins
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from keras.datasets import cifar10
import csv
import time
import re
from itertools import product
from matplotlib import pyplot as plt
from utils import cpu_and_system_props, gpu_props

np.set_printoptions(linewidth=512)
np.set_printoptions(threshold=np.inf)    

# folders
FOLDER_EXTRAS = "../extras/"
FOLDER_DATA_RAW = "../data_raw/"
FOLDER_DATA_RAW_LEUKEMIA = FOLDER_DATA_RAW + "leukemia/"
FOLDER_DATA_RAW_SPAMBASE = FOLDER_DATA_RAW + "spambase/"

# constants
NMM_MAGN_ORDERS_DEFAULT = [(4, 3, 4)]
TS_DEFAULT = [8, 16, 32, 64, 128]
BS_DEFAULT = [8]
CLF_DEFS_DEFAULT = [
        (AdaBoostClassifier, {"algorithm": "SAMME.R"}, {"color": "black"}),
        (FastRealBoostBins, {"fit_mode": "numba_jit", "decision_function_mode": "numba_jit"}, {"color": "blue"}),
        (FastRealBoostBins, {"fit_mode": "numba_cuda", "decision_function_mode": "numba_cuda"}, {"color": "red"})        
        ]
EPS = 1e-7

# plot settings
PLOT_FONTSIZE_SUPTITLE = 12
PLOT_FONTSIZE_TITLE = 8
PLOT_FONTSIZE_AXES = 11
PLOT_FONTSIZE_LEGEND = 8
PLOT_FIGSIZE = (8, 4.5)
PLOT_MARKERSIZE = 4
PLOT_GRID_COLOR = (0.4, 0.4, 0.4) 
PLOT_GRID_DASHES = (4.0, 4.0)
PLOT_LEGEND_LOC = "bottom_right"
PLOT_LEGEND_HANDLELENGTH = 4

def clean_name(name):
    name = name.replace("\n", "")
    name = name.replace("\t", "")
    name = re.sub(" +", " ", name)
    return name

def data_leukemia(filepath):
    data_as_list = []
    row_y = []    
    with open(filepath) as csv_file:
        read_csv = csv.reader(csv_file, delimiter=',')
        for i, row in enumerate(read_csv):
            if i == 0:
                row_y = row
            else:            
                data_as_list.append(row)
    for i in range(len(row_y)):
        row_y[i] = '-1' if row_y[i] == 'ALL' else '1'
    data_as_list.append(row_y)
    Xy = np.array(data_as_list).astype(np.float32)
    Xy = Xy.T
    X = Xy[:, :-1]
    y = Xy[:, -1].astype(int)      
    return X, y

def data_spambase(filepath):
    data_as_list = []
    with open(filepath) as csv_file:
        read_csv = csv.reader(csv_file, delimiter=',')
        for row in read_csv:
            row[-1] = '1' if row[-1] == '0' else '-1'
            data_as_list.append(row)
    Xy = np.array(data_as_list).astype(np.float32)
    X = Xy[:, :-1]
    y = Xy[:, -1].astype(int)      
    return X, y

def experimenter_leukemia_data():
    print("EXPERIMENT LEUKEMIA DATA...")
    t1 = time.time()
    T = 64
    B = 32
    seed = 0
    np.random.seed(seed)
    X, y = data_leukemia(FOLDER_DATA_RAW_LEUKEMIA + "leukemia_big.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=seed)
    print(f"[X.shape: {X.shape}]")
    print(f"[X_train.shape: {X_train.shape}], X_test.shape: {X_test.shape}")

    clfs = [
        FastRealBoostBins(T=T, B=B, fit_mode="numba_cuda", decision_function_mode="numba_cuda", verbose=False),      
        AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1, criterion="entropy"), n_estimators=T, algorithm="SAMME.R", random_state=seed)
        ]
    for clf in clfs:
        print(f"[clf: {clf}]")
        print(f"[fit...]")
        t1_fit = time.time()
        clf.fit(X_train, y_train)
        t2_fit = time.time()
        print(f"[fit done; time: {t2_fit - t1_fit} s]")
        print(f"[predict (on train data)...]")
        t1_predict_train = time.time()
        predictions_train = clf.predict(X_train)
        t2_predict_train = time.time()                
        acc_train = np.mean(predictions_train == y_train)
        print(f"[predict (on train data) done; time: {t2_predict_train - t1_predict_train} s, acc: {acc_train}]")        
        print(f"[predict (on test data)...]")
        t1_predict_test = time.time()
        predictions_test = clf.predict(X_test)
        t2_predict_test = time.time()                
        acc_test = np.mean(predictions_test == y_test)
        print(f"[predict (on test data) done; time: {t2_predict_test - t1_predict_test} s, acc: {acc_test}]")        
        print("-" * 196)
    t2 = time.time()             
    print(f"EXPERIMENT LEUKEMIA DATA DONE. [time: {t2 - t1} s]")
    
def experimenter_spambase_data():
    print("EXPERIMENT SPAMBASE DATA...")
    t1 = time.time()
    T = 256
    B = 8
    seed = 0
    np.random.seed(seed)
    X, y = data_spambase(FOLDER_DATA_RAW_SPAMBASE + "spambase.data")
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=seed)
    print(f"[X.shape: {X.shape}]")
    print(f"[X_train.shape: {X_train.shape}], X_test.shape: {X_test.shape}")

    clfs = [
        FastRealBoostBins(T=T, B=B, fit_mode="numba_cuda", decision_function_mode="numba_cuda", verbose=False),      
        AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1, criterion="entropy"), n_estimators=T, algorithm="SAMME.R", random_state=seed)
        ]
    for clf in clfs:
        print(f"[clf: {clf}]")
        print(f"[fit...]")
        t1_fit = time.time()
        clf.fit(X_train, y_train)
        t2_fit = time.time()
        print(f"[fit done; time: {t2_fit - t1_fit} s]")
        print(f"[predict (on train data)...]")
        t1_predict_train = time.time()
        predictions_train = clf.predict(X_train)
        t2_predict_train = time.time()                
        acc_train = np.mean(predictions_train == y_train)
        print(f"[predict (on train data) done; time: {t2_predict_train - t1_predict_train} s, acc: {acc_train}]")        
        print(f"[predict (on test data)...]")
        t1_predict_test = time.time()
        predictions_test = clf.predict(X_test)
        t2_predict_test = time.time()                
        acc_test = np.mean(predictions_test == y_test)
        print(f"[predict (on test data) done; time: {t2_predict_test - t1_predict_test} s, acc: {acc_test}]")        
        print("-" * 196)
    t2 = time.time()             
    print(f"EXPERIMENT SPAMBASE DATA DONE. [time: {t2 - t1} s]")    
    
def experimenter_cifar10_data():
    print("EXPERIMENT CIFAR-10 DATA...", flush=True)
    t1 = time.time()
    T = 16
    B = 8
    seed = 0
    np.random.seed(seed)
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    indexes = y_train == 0
    y_train[indexes] = 1
    y_train[~indexes] = -1
    y_train = y_train.flatten()
    indexes = y_test == 0
    y_test[indexes] = 1
    y_test[~indexes] = -1
    y_test = y_test.flatten()
    n = X_train.shape[1] * X_train.shape[2] * X_train.shape[3]
    X_train = np.reshape(X_train, (X_train.shape[0], n))
    X_test = np.reshape(X_test, (X_test.shape[0], n))    
    print(f"[X_train.shape: {X_train.shape}], X_test.shape: {X_test.shape}")

    clfs = [
        FastRealBoostBins(T=T, B=B, fit_mode="numba_cuda", decision_function_mode="numba_cuda", verbose=True),      
        AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1, criterion="entropy"), n_estimators=T, algorithm="SAMME.R", random_state=seed)
        ]
    for clf in clfs:
        print(f"[clf: {clf}]")
        print(f"[fit...]")
        t1_fit = time.time()
        clf.fit(X_train, y_train)
        t2_fit = time.time()
        print(f"[fit done; time: {t2_fit - t1_fit} s]")
        print(f"[predict (on train data)...]")
        t1_predict_train = time.time()
        predictions_train = clf.predict(X_train)
        t2_predict_train = time.time()                
        acc_train = np.mean(predictions_train == y_train)
        print(f"[predict (on train data) done; time: {t2_predict_train - t1_predict_train} s, acc: {acc_train}]")        
        print(f"[predict (on test data)...]")
        t1_predict_test = time.time()
        predictions_test = clf.predict(X_test)
        t2_predict_test = time.time()                
        acc_test = np.mean(predictions_test == y_test)
        print(f"[predict (on test data) done; time: {t2_predict_test - t1_predict_test} s, acc: {acc_test}]")        
        print("----------------------------------------------------------------------------------------------------------------------------------------------------------------")
    t2 = time.time()             
    print(f"EXPERIMENT CIFAR-10 DATA DONE. [time: {t2 - t1} s]")

def experimenter_random_data(dtype=np.int8, nmm_magn_orders=NMM_MAGN_ORDERS_DEFAULT, Ts=TS_DEFAULT, Bs=BS_DEFAULT, clf_defs=CLF_DEFS_DEFAULT, seed=0, 
                             plots=True, plots_arg_name=None, plots_values_names=[]):
    print("EXPERIMENTER RANDOM DATA...")
    print(f"[settings -> dtype: {dtype.__name__}, nmm_magn_orders: {nmm_magn_orders}, Ts: {Ts}, Bs: {Bs}, no. of clf_defs: {len(clf_defs)}, seed: {seed}]")
    print(f"[clfs definitions:]")
    for clf_id, (clf_class, clf_consts, _) in enumerate(clf_defs):
        print(f"[def {clf_id}: {clf_class.__name__}({clf_consts})]")    
    cpu_gpu_info = f"[cpu: {cpu_and_system_props()['cpu_name']}, gpu: {gpu_props()['name']}]".upper()
    t1 = time.time()
    np.random.seed(seed)
    max_value = (np.iinfo(dtype).max // 2) if np.issubdtype(dtype, np.integer) else 1.0                
    n_experiments = len(Ts) * len(Bs) * len(nmm_magn_orders)
    experiments_descr = np.empty(n_experiments, dtype=object)    
    results_descr = np.empty((n_experiments, len(clf_defs)), dtype=object)
    clfs_names = np.empty((n_experiments, len(clf_defs)), dtype=object)    
    for experiment_id, (nmmo, T, B) in enumerate(product(nmm_magn_orders, Ts, Bs)):
        print("=" * 196)
        print(f"[experiment: {experiment_id + 1}/{n_experiments}, params: {(nmmo, T, B)}]")              
        n = int(10**nmmo[0])
        m_train = int(10**nmmo[1])
        m_test = int(10**nmmo[2])               
        X_train = (max_value * np.random.rand(m_train, n)).astype(dtype)
        y_train = np.random.randint(0, 2, size=m_train) * 2 - 1
        X_test = (max_value * np.random.rand(m_test, n)).astype(dtype)
        y_test = np.random.randint(0, 2, size=m_test) * 2 - 1        
        experiments_descr[experiment_id] = {"T": T, "B": B, "n": n, "m_train": m_train, "m_test": m_test}
        print(f"[description: {experiments_descr[experiment_id]}]")
        print("=" * 196)
        results_arr = []
        clfs_now = []
        for clf_id, (clf_class, clf_consts, _) in enumerate(clf_defs):
            clf = clf_class(**clf_consts)
            params = clf.get_params()
            if isinstance(clf, FastRealBoostBins):
                params["T"] = T
                params["B"] = B
            if isinstance(clf, AdaBoostClassifier):
                params["n_estimators"] = T
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
        for clf_id in range(len(clf_defs)):
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
    print(f"EXPERIMENT RANDOM DATA DONE. [time: {t2 - t1} s]")
    if plots and plots_arg_name and plots_values_names:
        value_name_mapper = {"time_fit": "FIT TIME [s]", "time_predict_train": "PREDICT TIME (TRAIN DATA) [s]", "time_predict_test": "PREDICT TIME (TEST DATA) [s]",
                             "acc_train": "ACC (TRAIN DATA)", "acc_test": "ACC (TEST DATA)"}        
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
            for clf_id, (clf_class, clf_consts, clf_style) in enumerate(clf_defs): 
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
            plt.suptitle(f"{value_name_mapper[vn]} {key.upper()}", fontsize=PLOT_FONTSIZE_SUPTITLE)
            plt.title(f"\n{cpu_gpu_info}", fontsize=PLOT_FONTSIZE_TITLE)
            plt.xlabel(plots_arg_name.upper(), fontsize=PLOT_FONTSIZE_AXES)            
            plt.ylabel(value_name_mapper[vn], fontsize=PLOT_FONTSIZE_AXES)
            #y_range = y_max - y_min
            #plt.ylim([y_min - 0.1 * y_range, y_max + 0.1 * y_range])
            plt.yscale("log")            
            plt.legend(loc=PLOT_LEGEND_LOC, prop={"size": PLOT_FONTSIZE_LEGEND}, handlelength=PLOT_LEGEND_HANDLELENGTH)        
            plt.grid(color=PLOT_GRID_COLOR, zorder=0, dashes=PLOT_GRID_DASHES)                     
            plt.show()                                                        

if __name__ == "__main__":
    print("DEMONSTRATION OF \"FAST REAL BOOST WITH BINS\" ALGORITHM IMPLEMENTED VIA NUMBA.JIT AND NUMBA.CUDA.")
    print(f"CPU AND SYSTEM PROPS: {cpu_and_system_props()}")
    print(f"GPU PROPS: {gpu_props()}")    
    print("MAIN (EXPERIMENTER) STARTING...")
    experimenter_random_data(dtype=np.int16, nmm_magn_orders=NMM_MAGN_ORDERS_DEFAULT, Ts=TS_DEFAULT, Bs=BS_DEFAULT, clf_defs=CLF_DEFS_DEFAULT, seed=0, 
                             plots=True, plots_arg_name="T", plots_values_names=["time_fit", "time_predict_test"])    
    print("MAIN (EXPERIMENTER) DONE.")