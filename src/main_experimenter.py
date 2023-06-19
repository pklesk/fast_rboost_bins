import numpy as np
from frbb import FastRealBoostBins
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from keras.datasets import cifar10, mnist
import time
import re
from itertools import product, compress
from matplotlib import pyplot as plt
from utils import cpu_and_system_props, gpu_props
import pickle

np.set_printoptions(linewidth=512)
np.set_printoptions(threshold=np.inf)    

# folders
FOLDER_EXTRAS = "../extras/"
FOLDER_DATA_RAW = "../data_raw/"

# constants
DATA_KIND_DEFAULT = "real" # possible values: "real" or "random"
REAL_DATA_DEFS_DEFAULT = [
    ("fddb-patches", "read_data_fddb_patches", "FDDB-PATCHES (3NPI)"),
    ("cifar10", "read_data_cifar10", "CIFAR-10 (TRUCK)"),
    ("mnist", "read_data_mnist", "MNIST (DIGIT 0)")
    ]
REAL_DATA_FLAGS_DEFAULT = [True, False, False]
CLFS_DEFS_DEFAULT = [
        (AdaBoostClassifier, {"algorithm": "SAMME.R"}, {"color": "black"}),
        (GradientBoostingClassifier, {"max_depth": 1}, {"color": "green"}),
        (HistGradientBoostingClassifier, {"max_depth": 1, "early_stopping": False}, {"color": "orange"}),
        (FastRealBoostBins, {"fit_mode": "numba_jit", "decision_function_mode": "numba_jit"}, {"color": "blue"}),
        (FastRealBoostBins, {"fit_mode": "numba_cuda", "decision_function_mode": "numba_cuda"}, {"color": "red"})        
        ]
CLFS_FLAGS_DEFAULT = [True, True, True, True, True]
NMM_MAGN_ORDERS_DEFAULT = [(3, 4, 4)]
TS_DEFAULT = [8, 16, 32, 64, 128, 256]
BS_DEFAULT = [8]
SEED_DEFAULT = 0
EPS = 1e-9

# plot settings
PLOT_FONTSIZE_SUPTITLE = 11
PLOT_FONTSIZE_TITLE = 8
PLOT_FONTSIZE_AXES = 10
PLOT_FONTSIZE_LEGEND = 7
PLOT_FIGSIZE = (10, 6.5)
PLOT_MARKERSIZE = 4
PLOT_GRID_COLOR = (0.4, 0.4, 0.4) 
PLOT_GRID_DASHES = (4.0, 4.0)
PLOT_LEGEND_LOC = "upper left" #"lower right"
PLOT_LEGEND_HANDLELENGTH = 4

def clean_name(name):
    name = name.replace("\n", "")
    name = name.replace("\t", "")
    name = re.sub(" +", " ", name)
    return name

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

def read_data_cifar10(class_index=9): # default class: 9 ('truck')
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
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

def read_data_mnist(seed=0):
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
                    
    
def experimenter_random_data(clfs_defs=CLFS_DEFS_DEFAULT, clfs_flags=CLFS_FLAGS_DEFAULT,
                             dtype=np.int8, nmm_magn_orders=NMM_MAGN_ORDERS_DEFAULT, Ts=TS_DEFAULT, Bs=BS_DEFAULT, seed=0, 
                             plots=True, plots_arg_name=None, plots_values_names=[]):
    print("EXPERIMENTER RANDOM DATA...")    
    print(f"[clfs definitions:]")
    for clf_id, (clf_class, clf_consts, _) in enumerate(clfs_defs):
        print(f"[clf def {clf_id} (active: {clfs_flags[clf_id]}): {clf_class.__name__}({clf_consts})]")
    print(f"[other settings -> dtype: {dtype.__name__}, nmm_magn_orders: {nmm_magn_orders}, Ts: {Ts}, Bs: {Bs}, seed: {seed}]")    
    cpu_gpu_info = f"[cpu: {cpu_and_system_props()['cpu_name']}, gpu: {gpu_props()['name']}]".upper()
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
        y_train = np.random.randint(0, 2, size=m_train) * 2 - 1
        X_test = (max_value * np.random.rand(m_test, n)).astype(dtype)
        y_test = np.random.randint(0, 2, size=m_test) * 2 - 1        
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
            plt.suptitle(f"{value_name_mapper[vn]} {key.upper()}", fontsize=PLOT_FONTSIZE_SUPTITLE)
            plt.title(f"\n{cpu_gpu_info}", fontsize=PLOT_FONTSIZE_TITLE)
            plt.xlabel(plots_arg_name.upper(), fontsize=PLOT_FONTSIZE_AXES)            
            plt.ylabel(value_name_mapper[vn], fontsize=PLOT_FONTSIZE_AXES)
            plt.yscale("log")
            plt.legend(loc=PLOT_LEGEND_LOC, prop={"size": PLOT_FONTSIZE_LEGEND}, handlelength=PLOT_LEGEND_HANDLELENGTH)        
            plt.grid(color=PLOT_GRID_COLOR, zorder=0, dashes=PLOT_GRID_DASHES)                     
            plt.show()
            
def experimenter_real_data(real_data_defs=REAL_DATA_DEFS_DEFAULT, real_data_flags=REAL_DATA_FLAGS_DEFAULT,
                           clfs_defs=CLFS_DEFS_DEFAULT, clfs_flags=CLFS_FLAGS_DEFAULT,
                           Ts=TS_DEFAULT, Bs=BS_DEFAULT, seed=0, 
                           plots=True, plots_arg_name=None, plots_values_names=[]):
    print("EXPERIMENTER REAL DATA...")    
    print(f"[data definitions:]")
    for data_id, (data_name_short, data_reader, data_name_full) in enumerate(real_data_defs):
        print(f"[data def {data_id} (active: {real_data_flags[data_id]}): (name short: '{data_name_short}', reading function: {data_reader}(), name full: '{data_name_full})']")        
    print(f"[clfs definitions:]")
    for clf_id, (clf_class, clf_consts, _) in enumerate(clfs_defs):
        print(f"[clf def {clf_id} (active: {clfs_flags[clf_id]}): {clf_class.__name__}({clf_consts})]")
    print(f"[other settings -> Ts: {Ts}, Bs: {Bs}, seed: {seed}]")        
    real_data_defs = list(compress(real_data_defs, real_data_flags))        
    clfs_defs = list(compress(clfs_defs, clfs_flags))    
    cpu_gpu_info = f"[cpu: {cpu_and_system_props()['cpu_name']}, gpu: {gpu_props()['name']}]".upper()
    t1 = time.time()
    np.random.seed(seed)
    n_experiments = len(real_data_defs) * len(Ts) * len(Bs)
    experiments_descr = np.empty(n_experiments, dtype=object)    
    results_descr = np.empty((n_experiments, len(clfs_defs)), dtype=object)
    clfs_names = np.empty((n_experiments, len(clfs_defs)), dtype=object)
    datas = [(data_name_full, globals()[reader_name]()) for _, reader_name, data_name_full in real_data_defs]   
    for experiment_id, ((data_name, data), T, B) in enumerate(product(datas, Ts, Bs)):
        print("=" * 196)
        print(f"[experiment: {experiment_id + 1}/{n_experiments}, params: {(data_name, T, B)}]")
        X_train, y_train, X_test, y_test = data
        m_train, n = X_train.shape
        m_test = X_test.shape[0]
        experiments_descr[experiment_id] = {"T": T, "B": B, "data_name:": data_name, "n": n, "m_train": m_train, "m_test": m_test}
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
            plt.suptitle(f"{value_name_mapper[vn]} {key.upper()}", fontsize=PLOT_FONTSIZE_SUPTITLE)
            plt.title(f"\n{cpu_gpu_info}", fontsize=PLOT_FONTSIZE_TITLE)
            plt.xlabel(plots_arg_name.upper(), fontsize=PLOT_FONTSIZE_AXES)            
            plt.ylabel(value_name_mapper[vn], fontsize=PLOT_FONTSIZE_AXES)
            plt.yscale("log")               
            plt.legend(loc=PLOT_LEGEND_LOC, prop={"size": PLOT_FONTSIZE_LEGEND}, handlelength=PLOT_LEGEND_HANDLELENGTH)        
            plt.grid(color=PLOT_GRID_COLOR, zorder=0, dashes=PLOT_GRID_DASHES)
            plt.tight_layout()                     
            plt.show()                                                                    

if __name__ == "__main__":
    print("DEMONSTRATION OF \"FAST REAL BOOST WITH BINS\" ALGORITHM IMPLEMENTED VIA NUMBA.JIT AND NUMBA.CUDA.")
    print(f"CPU AND SYSTEM PROPS: {cpu_and_system_props()}")
    print(f"GPU PROPS: {gpu_props()}")        
    print(f"MAIN (EXPERIMENTER) STARTING...]")
    data_kind = DATA_KIND_DEFAULT
    clfs_defs = CLFS_DEFS_DEFAULT
    clfs_flags = CLFS_FLAGS_DEFAULT
    seed = SEED_DEFAULT
    if data_kind == "real":        
        real_data_flags = REAL_DATA_FLAGS_DEFAULT
        real_data_defs = REAL_DATA_DEFS_DEFAULT        
        print(f"[data kind: {data_kind}, real data flags: {real_data_flags}, clfs flags: {clfs_flags}, seed: {seed}]")        
        experimenter_real_data(real_data_defs=real_data_defs, real_data_flags=real_data_flags,
                               clfs_defs=clfs_defs, clfs_flags=clfs_flags, 
                               Ts=TS_DEFAULT, Bs=BS_DEFAULT, seed=seed, 
                               plots=True, plots_arg_name="T", plots_values_names=["acc_test", "acc_train", "time_fit", "time_predict_test"])
    elif data_kind == "random":
        print(f"[data kind: {data_kind}, real data flags: {real_data_flags}, clfs flags: {clfs_flags}, seed: {seed}]")
        experimenter_random_data(clfs_defs=clfs_defs, clfs_flags=clfs_flags,
                                 dtype=np.int16, nmm_magn_orders=NMM_MAGN_ORDERS_DEFAULT, Ts=TS_DEFAULT, Bs=BS_DEFAULT, seed=seed, 
                                 plots=True, plots_arg_name="T", plots_values_names=["time_fit", "time_predict_test"])            
    print("MAIN (EXPERIMENTER) DONE.")