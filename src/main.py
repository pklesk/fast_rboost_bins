import sys
import numpy as np
from frbb import FastRealBoostBins
from numba import cuda
import cv2
import haar
import datagenerator
import time
import pickle
from joblib import Parallel, delayed
from functools import reduce
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt

__version__ = "0.8.0"
__author__ = "Przemysław Klęsk"
__email__ = "pklesk@zut.edu.pl"


# main settings
KIND = "hand"
S = 5 # parameter "scales" to generete Haar-like features
P = 5 # parameter "positions" to generete Haar-like features
NPI = 20 # "negatives per image" - no. of negatives (negative windows) to sample per image (image real or generated synthetically) 
T = 2048 # size of ensemble in FastRealBoostBins (equivalently, no. of boosting rounds when fitting)
B = 8 # no. of bins
SEED = 0 # randomization seed
DEMO_HAAR_FEATURES_ALL = False
DEMO_HAAR_FEATURES_SELECTED = False
REGENERATE_DATA = True
FIT_OR_REFIT_MODEL = True
MEASURE_ACCS_OF_MODEL = True
DEMO_DETECT_IN_VIDEO = False

# cv2 camera settings
CV2_VIDEO_CAPTURE_CAMERA_INDEX = 0
CV2_VIDEO_CAPTURE_IS_IT_MSWINDOWS = False

# detection procedure settings
DETECTION_SCALES = 10
DETECTION_WINDOW_HEIGHT_MIN = 64
DETECTION_WINDOW_WIDTH_MIN = 64
DETECTION_WINDOW_GROWTH = 1.2
DETECTION_WINDOW_JUMP = 0.05
DETECTION_THRESHOLD = 7.0 
DETECTION_POSTPROCESS = "avg" # possible values: None, "nms", "avg"

# folders
FOLDER_DATA = "../data/"
FOLDER_CLFS = "../models/"
FOLDER_EXTRAS = "../extras/"

def gpu_props():
    gpu = cuda.get_current_device()
    props = {}
    props["name"] = gpu.name.decode("ASCII")
    props["max_threads_per_block"] = gpu.MAX_THREADS_PER_BLOCK
    props["max_block_dim_x"] = gpu.MAX_BLOCK_DIM_X
    props["max_block_dim_y"] = gpu.MAX_BLOCK_DIM_Y
    props["max_block_dim_z"] = gpu.MAX_BLOCK_DIM_Z
    props["max_grid_dim_x"] = gpu.MAX_GRID_DIM_X
    props["max_grid_dim_y"] = gpu.MAX_GRID_DIM_Y
    props["max_grid_dim_z"] = gpu.MAX_GRID_DIM_Z    
    props["max_shared_memory_per_block"] = gpu.MAX_SHARED_MEMORY_PER_BLOCK
    props["async_engine_count"] = gpu.ASYNC_ENGINE_COUNT
    props["can_map_host_memory"] = gpu.CAN_MAP_HOST_MEMORY
    props["multiprocessor_count"] = gpu.MULTIPROCESSOR_COUNT
    props["warp_size"] = gpu.WARP_SIZE
    props["unified_addressing"] = gpu.UNIFIED_ADDRESSING
    props["pci_bus_id"] = gpu.PCI_BUS_ID
    props["pci_device_id"] = gpu.PCI_DEVICE_ID
    props["compute_capability"] = gpu.compute_capability            
    CC_CORES_PER_SM_DICT = {
        (2,0) : 32,
        (2,1) : 48,
        (3,0) : 192,
        (3,5) : 192,
        (3,7) : 192,
        (5,0) : 128,
        (5,2) : 128,
        (6,0) : 64,
        (6,1) : 128,
        (7,0) : 64,
        (7,5) : 64,
        (8,0) : 64,
        (8,6) : 128
        }
    props["cores_per_SM"] = CC_CORES_PER_SM_DICT.get(gpu.compute_capability)
    props["cores_total"] = props["cores_per_SM"] * gpu.MULTIPROCESSOR_COUNT
    return props   
                            
def pickle_objects(fname, some_list):
    print(f"PICKLE OBJECTS... [to file: {fname}]")
    t1 = time.time()
    try:
        f = open(fname, "wb+")
        pickle.dump(some_list, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
    except IOError:
        sys.exit("[error occurred when trying to open or pickle the file]")
    t2 = time.time()
    print(f"PICKLE OBJECTS DONE. [time: {t2 - t1} s]")

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

def measure_accs_of_model(clf, X_train, y_train, X_test, y_test):
    print(f"MEASURE ACCS OF MODEL...")
    t1_accs = time.time()    
    t1 = time.time()
    acc_train = clf.score(X_train, y_train)
    t2 = time.time()
    print(f"[train acc: {acc_train}; data shape: {X_train.shape}, time: {t2 - t1} s]")
    ind_pos = y_train == 1
    X_train_pos = X_train[ind_pos]
    y_train_pos = y_train[ind_pos]    
    t1 = time.time()
    sens_train = clf.score(X_train_pos, y_train_pos)
    t2 = time.time()
    print(f"[train sensitivity: {sens_train}; data shape: {X_train_pos.shape}, time: {t2 - t1} s]")
    ind_neg = y_train == -1
    X_train_neg = X_train[ind_neg]
    y_train_neg = y_train[ind_neg]    
    t1 = time.time()
    far_train = 1.0 - clf.score(X_train_neg, y_train_neg)
    t2 = time.time()
    print(f"[train far: {far_train}; data shape: {X_train_neg.shape}, time: {t2 - t1} s]")
    t1 = time.time()
    acc_test = clf.score(X_test, y_test)
    t2 = time.time()
    print(f"[test acc: {acc_test}; data shape: {X_test.shape}, time: {t2 - t1} s]")
    ind_pos = y_test == 1
    X_test_pos = X_test[ind_pos]
    y_test_pos = y_test[ind_pos]    
    t1 = time.time()
    sens_test = clf.score(X_test_pos, y_test_pos)
    t2 = time.time()
    print(f"[test sensitivity: {sens_test}; data shape: {X_test_pos.shape}, time: {t2 - t1} s]")
    ind_neg = y_test == -1
    X_test_neg = X_test[ind_neg]
    y_test_neg = y_test[ind_neg]    
    t1 = time.time()
    far_test = 1.0 - clf.score(X_test_neg, y_test_neg)
    t2 = time.time()
    print(f"[test far: {far_test}; data shape: {X_test_neg.shape}, time: {t2 - t1} s]")
    t2_accs = time.time()
    print("MEASURE ACCS DONE. [time: " + str(t2_accs - t1_accs) + " s]")
    
def best_threshold_via_prec(roc, y_test):
    py = np.zeros(2)
    py[0] = np.mean(y_test == -1)
    py[1] = np.mean(y_test == 1)
    fprs, tprs, dts = roc
    sub = fprs > 0.0
    fprs_sub = fprs[sub]
    tprs_sub = tprs[sub]
    dts_sub = dts[sub]
    tp = tprs_sub * py[1]
    fp = fprs_sub * py[0]
    precs = tp / (tp + fp)
    best_index = np.argmax(precs)
    best_thr = dts_sub[best_index]
    best_prec = precs[best_index]    
    if best_index > 0:
        best_thr = 0.5 * (best_thr + dts_sub[best_index - 1]) 
        best_prec = 0.5 * (best_prec + precs[best_index - 1])
    print(f"[best_threshold_via_prec -> best_thr: {best_thr}, best_prec: {best_prec}; py: {py}, fprs_sub[best_index]: {fprs_sub[best_index]}, tprs_sub[best_index]: {tprs_sub[best_index]}]")
    return best_thr, best_prec

def draw_feature_at(i, j0, k0, shcoords_one_feature):
    i_copy = i.copy()
    j, k, h, w = shcoords_one_feature[0]
    cv2.rectangle(i_copy, (k0 + k, j0 + j), (k0 + k + w - 1, j0 + j + h - 1), (0, 0, 0), cv2.FILLED)
    j, k, h, w = shcoords_one_feature[1]
    cv2.rectangle(i_copy, (k0 + k, j0 + j), (k0 + k + w - 1, j0 + j + h - 1), (255, 255, 255), cv2.FILLED)
    return i_copy

def demo_haar_features(hinds, hcoords, n):
    print(f"DEMO OF HAAR FEATURES... [hcoords.shape: {hcoords.shape}]")
    i = cv2.imread(FOLDER_EXTRAS + "photo_for_face_features_demo.jpg")
    j0, k0 = 116, 100
    w = h = 221
    # i = cv2.imread(FOLDER_EXTRAS + "photo_for_hand_features_demo.jpg")
    # j0, k0 = 86, 52
    # w = h = 221
    i_resized = haar.resize_image(i)
    i_gray = cv2.cvtColor(i_resized, cv2.COLOR_BGR2GRAY)
    ii = haar.integral_image_numba_jit(i_gray)
    cv2.rectangle(i_resized, (k0, j0), (k0 + w - 1, j0 + h - 1), (0, 0, 255), 1)
    title = "DEMO OF FEATURES [press ESC to quit or any other key to continue]"    
    cv2.imshow(title, i_resized)
    cv2.waitKey()    
    for ord_ind, (ind, c) in enumerate(zip(hinds, hcoords)):
        hcoords_window = (np.array([h, w, h, w]) * c).astype(np.int16) 
        i_with_feature = draw_feature_at(i_resized, j0, k0, hcoords_window)
        i_temp = cv2.addWeighted(i_resized, 0.5, i_with_feature, 0.5, 0.0)
        print(f"feature index (ordinal): {ord_ind}")
        print(f"feature multi-index: {ind}")
        print(f"feature hcoords (cartesian):\n {c}")
        print(f"feature hcoords in window:\n {hcoords_window}")
        print(f"feature value: {haar_feature_numba_jit(ii, j0, k0, hcoords_window)}")        
        print("----------------------------------------------------------------")
        cv2.imshow(title, i_temp)
        key = cv2.waitKey()
        if key & 0xFF == 27: # esc key
            break
    cv2.destroyAllWindows()
    hcoords_window_subset = (np.array([h, w, h, w]) * hcoords).astype(np.int16) 
    t1 = time.time()
    features = haar.haar_features_one_window_numba_jit(ii, j0, k0, hcoords_window_subset, n, np.arange(n))
    t2 = time.time()
    print(f"[time of extraction of all {n} haar features for this window (in one call): {t2 - t1} s]")
    print(f"[features: {features}]") 
    print(f"DEMO OF HAAR FEATURES DONE.")

def prepare_detection_windows_and_scaled_haar_coords(image_height, image_width, hcoords, features_indexes):
    hcoords_subset = hcoords[features_indexes]
    windows = []
    shcoords_multiple_scales = []
    for s in range(DETECTION_SCALES):
        h = int(round(DETECTION_WINDOW_HEIGHT_MIN * DETECTION_WINDOW_GROWTH**s))
        w = int(round(DETECTION_WINDOW_WIDTH_MIN * DETECTION_WINDOW_GROWTH**s))        
        dj = int(round(h * DETECTION_WINDOW_JUMP))
        dk = int(round(w * DETECTION_WINDOW_JUMP))     
        j_start = ((image_height - h) % dj) // 2
        k_start = ((image_width - w) % dk) // 2
        shcoords_multiple_scales.append((np.array([h, w, h, w]) * hcoords_subset).astype(np.int16))
        for j0 in range(j_start, image_height - h + 1, dj):
            for k0 in range(k_start, image_width - w + 1, dk):
                windows.append(np.array([s, j0, k0, h, w], dtype=np.int16))
    windows = np.array(windows) 
    shcoords_multiple_scales = np.array(shcoords_multiple_scales) 
    return windows, shcoords_multiple_scales

def prepare_detection_windows(image_height, image_width):
    windows = []
    for s in range(DETECTION_SCALES):
        h = int(round(DETECTION_WINDOW_HEIGHT_MIN * DETECTION_WINDOW_GROWTH**s))
        w = int(round(DETECTION_WINDOW_WIDTH_MIN * DETECTION_WINDOW_GROWTH**s))        
        dj = int(round(h * DETECTION_WINDOW_JUMP))
        dk = int(round(w * DETECTION_WINDOW_JUMP))     
        j_start = ((image_height - h) % dj) // 2
        k_start = ((image_width - w) % dk) // 2
        for j0 in range(j_start, image_height - h + 1, dj):
            for k0 in range(k_start, image_width - w + 1, dk):
                windows.append(np.array([s, j0, k0, h, w], dtype=np.int16))
    windows = np.array(windows)  
    return windows

def prepare_scaled_haar_coords(hcoords, features_indexes):
    hcoords_subset = hcoords[features_indexes]
    shcoords_multiple_scales = []
    for s in range(DETECTION_SCALES):
        h = int(round(DETECTION_WINDOW_HEIGHT_MIN * DETECTION_WINDOW_GROWTH**s))
        w = int(round(DETECTION_WINDOW_WIDTH_MIN * DETECTION_WINDOW_GROWTH**s))        
        shcoords_multiple_scales.append((np.array([h, w, h, w]) * hcoords_subset).astype(np.int16))
    shcoords_multiple_scales = np.array(shcoords_multiple_scales) 
    return shcoords_multiple_scales

def detect_simple(i, clf, hcoords, n, features_indexes, threshold=0.0, windows=None, shcoords_multiple_scales=None, verbose=False):
    if verbose:
        print("[detect_simple...]")
    t1 = time.time()
    times = {}
    t1_preprocess = time.time()
    i_resized = haar.resize_image(i)
    i_gray = cv2.cvtColor(i_resized, cv2.COLOR_BGR2GRAY)
    i_h, i_w = i_gray.shape
    t2_preprocess = time.time()
    dt_preprocess = t2_preprocess - t1_preprocess
    times["preprocess"] = dt_preprocess
    if verbose:
        print(f"[detect_simple: preprocessing done; time: {dt_preprocess} s, i_gray.shape: {i_gray.shape}]")
    t1_ii = time.time()
    ii = haar.integral_image_numba_jit(i_gray)
    t2_ii = time.time()
    dt_ii = t2_ii - t1_ii
    times["ii"] = dt_ii
    if verbose:
        print(f"[detect_simple: integral_image_numba_jit done; time: {dt_ii} s]")
    if windows is None:
        t1_prepare = time.time()
        windows, shcoords_multiple_scales = prepare_detection_windows_and_scaled_haar_coords(i_h, i_w, hcoords, features_indexes)
        t2_prepare = time.time()
        dt_prepare = t2_prepare - t1_prepare
        times["prepare"] = dt_prepare    
        if verbose:
            print(f"[detect_simple: prepare_detection_windows_and_scaled_haar_coords done; time: {dt_prepare} s, windows to check: {windows.shape[0]}]")
    t1_haar = time.time()    
    X = haar.haar_features_multiple_windows_numba_jit(ii, windows, shcoords_multiple_scales, n, features_indexes)
    t2_haar = time.time()
    dt_haar = t2_haar - t1_haar
    times["haar"] = dt_haar
    if verbose:
        print(f"[detect_simple: haar_features_multiple_windows done; time: {dt_haar} s]")
    t1_frbb = time.time()    
    responses = clf.decision_function(X)
    t2_frbb = time.time()
    dt_frbb = t2_frbb - t1_frbb 
    times["frbb"] = dt_frbb 
    if verbose:
        print(f"[detect_simple: clf.decision_function done; time: {dt_frbb} s]")
    t1_ti = time.time()                
    detected = responses > threshold
    detections = windows[detected, 1:] # skipping scale index
    responses = responses[detected] 
    t2_ti = time.time()
    dt_ti = t2_ti - t1_ti
    times["ti"] = dt_ti
    if verbose:
        print(f"[detect_simple: finding detections (thresholding and indexing) done; time: {dt_ti} s]")        
    t2 = time.time()
    if verbose:
        print(f"[detect_simple done; time: {t2 - t1} s]")                    
    return detections, responses, times

def detect_parallel(i, clf, hcoords, n, features_indexes, threshold=0.0, windows=None, shcoords_multiple_scales=None, n_jobs=8, verbose=False):
    if verbose:
        print("[detect_parallel...]")
    t1 = time.time()
    times = {}
    t1_preprocess = time.time()
    i_resized = haar.resize_image(i)
    i_gray = cv2.cvtColor(i_resized, cv2.COLOR_BGR2GRAY)
    i_h, i_w = i_gray.shape
    t2_preprocess = time.time()
    dt_preprocess = t2_preprocess - t1_preprocess
    times["preprocess"] = dt_preprocess
    if verbose:
        print(f"[detect_parallel: preprocessing done; time: {dt_preprocess} s; i_gray.shape: {i_gray.shape}]")
    t1_ii = time.time()
    ii = haar.integral_image_numba_jit(i_gray)
    t2_ii = time.time()
    dt_ii = t2_ii - t1_ii
    times["ii"] = dt_ii
    if verbose:
        print(f"[detect_parallel: integral_image_numba_jit done; time: {dt_ii} s]")
    if windows is None:
        t1_prepare = time.time()
        windows, shcoords_multiple_scales = prepare_detection_windows_and_scaled_haar_coords(i_h, i_w, hcoords, features_indexes)
        t2_prepare = time.time()
        dt_prepare = t2_prepare - t1_prepare
        times["prepare"] = dt_prepare
        if verbose:
            print(f"[detect_parallel: prepare_detection_windows_and_scaled_haar_coords done; time: {dt_prepare} s; windows to check: {windows.shape[0]}]")    
    t1_parallel = time.time()
    m = windows.shape[0]
    n_calls = n_jobs * 1
    job_size = m // n_calls
    job_ranges = job_size * np.ones(n_calls, dtype=np.int32)        
    job_ranges[:m % n_calls] += 1
    job_ranges = np.r_[0, np.cumsum(job_ranges)]                  
    with Parallel(n_jobs=n_jobs, verbose=0) as parallel:      
        def worker(job_index):
            job_slice = slice(job_ranges[job_index], job_ranges[job_index + 1])
            job_windows = windows[job_slice]
            X = haar.haar_features_multiple_windows_numba_jit_tf(ii, job_windows, shcoords_multiple_scales, n, features_indexes)
            job_responses = clf.decision_function(X)         
            return job_responses 
        workers_results = parallel((delayed(worker)(job_index) for job_index in range(n_calls)))
        responses =  reduce(lambda a, b: np.r_[a, b], [jr for jr in workers_results])
        detected = responses > threshold
        detections = windows[detected, 1:] # skipping scale index
        responses = responses[detected] 
    t2_parallel = time.time()
    dt_parallel = t2_parallel - t1_parallel
    times["parallel"] = dt_parallel
    if verbose:
        print(f"[detect_parallel: all parallel jobs done (haar_features_multiple_windows_numba_jit_tf, clf.decision_function, finding detections (thresholding and indexing); time {dt_parallel} s]")    
    t2 = time.time()
    if verbose:
        print(f"[detect_parallel done; time: {t2 - t1} s]")                    
    return detections, responses, times

def detect_cuda(i, clf, hcoords, features_indexes, threshold=0.0, windows=None, shcoords_multiple_scales=None, 
                dev_windows=None, dev_shcoords_multiple_scales=None, dev_X_selected=None, dev_mins_selected=None, dev_maxes_selected=None, dev_logits=None, dev_responses=None, 
                verbose=False):
    if verbose:
        print("[detect_cuda...]")
    t1 = time.time()
    times = {}
    t1_preprocess = time.time()
    i_resized = haar.resize_image(i)
    i_gray = cv2.cvtColor(i_resized, cv2.COLOR_BGR2GRAY)
    i_h, i_w = i_gray.shape
    t2_preprocess = time.time()
    dt_preprocess = t2_preprocess - t1_preprocess
    times["preprocess"] = dt_preprocess
    if verbose:
        print(f"[detect_cuda: preprocessing done; time: {dt_preprocess} s; i_gray.shape: {i_gray.shape}]")
    t1_ii = time.time()
    ii = haar.integral_image_numba_jit(i_gray)
    t2_ii = time.time()
    dt_ii = t2_ii - t1_ii
    times["ii"] = dt_ii
    if verbose:
        print(f"[detect_cuda: integral_image_numba_jit done; time: {dt_ii} s]")
    if windows is None:
        t1_prepare = time.time()
        windows, shcoords_multiple_scales = prepare_detection_windows_and_scaled_haar_coords(i_h, i_w, hcoords, features_indexes)
        dev_windows = cuda.to_device(windows)
        dev_shcoords_multiple_scales = cuda.to_device(shcoords_multiple_scales)        
        t2_prepare = time.time()
        dt_prepare = t2_prepare - t1_prepare 
        times["prepare"] = dt_prepare
        if verbose:
            print(f"[detect_cuda: prepare_detection_windows_and_scaled_haar_coords done; time: {dt_prepare} s; windows to check: {windows.shape[0]}]")
    if dev_X_selected is None:
        dev_X_selected = cuda.device_array((m, T), dtype=np.int16)
    if dev_mins_selected is None:
        dev_mins_selected = cuda.to_device(clf.mins_selected_)
    if dev_maxes_selected is None:        
        dev_maxes_selected = cuda.to_device(clf.maxes_selected_)
    if dev_logits is None:
        dev_logits = cuda.to_device(clf.logits_)                
    if dev_responses is None:
        dev_responses = cuda.device_array(m, dtype=np.float32)
    t1_haar = time.time()
    tpb = cuda.get_current_device().MAX_THREADS_PER_BLOCK // 2
    bpg = windows.shape[0]
    dev_ii = cuda.to_device(ii)
    haar.haar_features_multiple_windows_numba_cuda[bpg, tpb](dev_ii, dev_windows, dev_shcoords_multiple_scales, dev_X_selected)
    cuda.synchronize()
    t2_haar = time.time()
    dt_haar = t2_haar - t1_haar
    times["haar"] = dt_haar
    if verbose:
        print(f"[detect_cuda: haar_features_multiple_windows_numba_cuda done; time: {dt_haar} s]")
    t1_frbb = time.time()    
    FastRealBoostBins.decision_function_numba_cuda_job[bpg, tpb](dev_X_selected, dev_mins_selected, dev_maxes_selected, dev_logits, dev_responses)
    responses = dev_responses.copy_to_host()
    cuda.synchronize()
    t2_frbb = time.time()
    dt_frbb = t2_frbb - t1_frbb
    times["frbb"] = dt_frbb  
    if verbose:
        print(f"[detect_cuda: FastRealBoostBins.decision_function_numba_cuda_job done; time: {dt_frbb} s]")
    t1_ti = time.time()                
    detected = responses > threshold
    detections = windows[detected, 1:] # skipping scale index
    responses = responses[detected] 
    t2_ti = time.time()
    dt_ti = t2_ti - t1_ti
    times["ti"] = dt_ti 
    if verbose:
        print(f"[detect_cuda: finding detections (thresholding and indexing) done; time: {dt_ti} s]")        
    t2 = time.time()
    if verbose:
        print(f"[detect_cuda done; time: {t2 - t1} s]")                 
    return detections, responses, times

def postprocess_nms(detections, responses, threshold=0.5):
    d = detections
    r = responses
    d_final = []
    r_final = []
    indexes = np.ones(len(detections), dtype=bool)
    arg_sort = np.argsort(-r, kind="stable")
    d = d[arg_sort]
    r = r[arg_sort]
    for i in range(len(detections) - 1):
        if indexes[i] == False:
            continue
        indexes[i] = False
        d_final.append(d[i])
        r_final.append(r[i])
        for j in range(i + 1, len(detections)):
            if indexes[j] == False:
                continue            
            if haar.iou2(d[i], d[j]) >= threshold:
                indexes[j] = False
    return d_final, r_final

def postprocess_avg(detections, responses, threshold=0.5):
    d = detections
    r = responses
    d_final = []
    r_final = []
    indexes = np.ones(len(detections), dtype=bool)
    arg_sort = np.argsort(-r, kind="stable")
    d = d[arg_sort]
    r = r[arg_sort]
    for i in range(len(detections) - 1):
        if indexes[i] == False:
            continue
        indexes[i] = False
        d_avg = []
        r_avg = []
        d_avg.append(d[i])
        r_avg.append(r[i])
        for j in range(i + 1, len(detections)):
            if indexes[j] == False:
                continue            
            if haar.iou2(d[i], d[j]) >= threshold:
                indexes[j] = False
                d_avg.append(d[j])
                r_avg.append(r[j])
        d_avg = (np.round(np.mean(np.array(d_avg), axis=0))).astype(np.int16)
        r_avg = np.mean(r_avg)
        d_final.append(d_avg)
        r_final.append(r_avg)
    return d_final, r_final

def demo_detect_in_video(clf, hcoords, threshold, computations="cuda", postprocess="avg", n_jobs=8, verbose_loop=True, verbose_detect=False):
    print("DEMO OF DETECT IN VIDEO...")
    gpu_name = gpu_props()["name"]
    features_indexes = clf.features_indexes_
    video = cv2.VideoCapture(CV2_VIDEO_CAPTURE_CAMERA_INDEX + (cv2.CAP_DSHOW if CV2_VIDEO_CAPTURE_IS_IT_MSWINDOWS else 0))
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    video.set(cv2.CAP_PROP_FPS, 30)    
    _, frame = video.read()
    frame_h, frame_w, _ = frame.shape
    resized_height = haar.HEIGHT    
    resized_width = int(np.round(frame.shape[1] / frame.shape[0] * resized_height))
    windows, shcoords_multiple_scales = prepare_detection_windows_and_scaled_haar_coords(resized_height, resized_width, hcoords, features_indexes)
    print(f"[frame shape: {frame.shape}]")
    print(f"[windows per frame: {windows.shape[0]}]")
    print(f"[terms per window: {clf.T_}]")
    print(f"[about to start a camera...]")
    h_scale = frame_h / resized_height
    w_scale = frame_w / resized_width 
    n_frames = 0
    ma_decay = 0.9
    fps_disp_ma = 0.0    
    fps_disp = 0.0
    fps_comps_ma = 0.0    
    fps_comps = 0.0   
    draw_thickness = 1 if postprocess == None else 2
    # device side arrays in case of cuda method
    dev_windows = None 
    dev_shcoords_multiple_scales = None    
    dev_X_selected = None     
    dev_mins_selected = None
    dev_maxes_selected = None
    dev_logits = None
    dev_responses = None
    if computations == "cuda":
        m, T = windows.shape[0], features_indexes.size
        dev_windows = cuda.to_device(windows)
        dev_shcoords_multiple_scales = cuda.to_device(shcoords_multiple_scales)        
        dev_X_selected = cuda.device_array((m, T), dtype=np.int16)
        dev_mins_selected = cuda.to_device(clf.mins_selected_)
        dev_maxes_selected = cuda.to_device(clf.maxes_selected_)
        dev_logits = cuda.to_device(clf.logits_)
        dev_responses = cuda.device_array(m, dtype=np.float32)
    tpf_prev = 0.0
    time_comps = 0.0
    time_comps_haar = 0.0
    time_comps_frbb = 0.0
    t1_loop = time.time()
    while(True):
        t1 = time.time()
        if verbose_loop:
            print(f"----------------------------------------------------------------")
            print(f"[frame: {n_frames}]")        
        t1_read = time.time()
        _, frame = video.read()
        t2_read = time.time()
        t1_flip = time.time()
        frame = cv2.flip(frame, 1)
        t2_flip = time.time()        
        t1_comps = time.time()
        if computations == "simple":
            detections, responses, times = detect_simple(frame, clf, hcoords, n, features_indexes, threshold, windows, shcoords_multiple_scales, verbose=verbose_detect)
        elif computations == "parallel":
            detections, responses, times = detect_parallel(frame, clf, hcoords, n, features_indexes, threshold, windows, shcoords_multiple_scales, n_jobs=n_jobs, verbose=verbose_detect)        
        elif computations == "cuda":
            detections, responses, times = detect_cuda(frame, clf, hcoords, features_indexes, threshold, windows, shcoords_multiple_scales, 
                                                       dev_windows, dev_shcoords_multiple_scales, dev_X_selected, dev_mins_selected, dev_maxes_selected, dev_logits, dev_responses, 
                                                       verbose=verbose_detect)                        
        t2_comps = time.time()
        t1_post = time.time()
        if postprocess is not None:
            if postprocess == "nms":
                detections, responses = postprocess_nms(detections, responses, 0.25)
            if postprocess == "avg":
                detections, responses = postprocess_avg(detections, responses, 0.25)
        t2_post = time.time()
        t1_other = time.time() 
        for index, (j0, k0, h, w) in enumerate(detections):
            js = int(np.round(j0 * h_scale))
            ks = int(np.round(k0 * w_scale))
            hs = int(np.round(h * h_scale))
            ws = int(np.round(w * w_scale))
            cv2.rectangle(frame, (ks, js), (ks + ws - 1, js + hs - 1), (0, 0, 255), draw_thickness)
            if postprocess:
                cv2.putText(frame, f"{responses[index]:.1f}", (k0, j0 + ws - 2), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), draw_thickness)
        normalizer_ma = 1.0 / (1.0 - ma_decay**(n_frames + 1))
        if n_frames > 0:
            fps_disp_ma = ma_decay * fps_disp_ma + (1.0 - ma_decay) * 1.0 / tpf_prev
            fps_disp = fps_disp_ma * normalizer_ma
        fps_comps_ma = ma_decay * fps_comps_ma + (1.0 - ma_decay) * 1.0 / (t2_comps - t1_comps)
        fps_comps = fps_comps_ma * normalizer_ma            
        time_comps += t2_comps - t1_comps
        text_shift = 16
        cv2.putText(frame, f"FRAME: {n_frames}", (0, 0 + text_shift), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)        
        cv2.putText(frame, f"WINDOWS PER FRAME: {windows.shape[0]}", (0, frame_h - 1 - 4 * text_shift), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)
        cv2.putText(frame, f"TERMS PER WINDOW: {clf.T_}", (0, frame_h - 1 - 3 * text_shift), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)
        cv2.putText(frame, f"GPU: {gpu_name}", (0, frame_h - 1 - 2 * text_shift), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)
        comps_details = ""
        if times["haar"] and times["frbb"]:
            comps_details += f"[HAAR: {times['haar'] * 1000:06.2f} ms"            
            comps_details += f", FRBB: {times['frbb'] * 1000:06.2f} ms]"
            time_comps_haar += times["haar"]
            time_comps_frbb += times["frbb"]
        cv2.putText(frame, f"FPS (COMPUTATIONS): {fps_comps:.2f} {comps_details}", (0, frame_h - 1 - 1 * text_shift), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)
        cv2.putText(frame, f"FPS (DISPLAY): {fps_disp:.2f}", (0, frame_h - 1), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)                    
        imshow_name = "DEMO: FAST REAL-BOOST WITH BINS WORKING ON HAAR-LIKE FEATURES [press ESC to quit]"             
        cv2.imshow(imshow_name, frame)
        cv2.namedWindow(imshow_name)
        if cv2.waitKey(1) & 0xFF == 27: # esc key
            break       
        t2_other = time.time()        
        n_frames += 1
        if verbose_loop:
            print(f"[read time: {t2_read - t1_read} s]")
            print(f"[flip time: {t2_flip - t1_flip} s]")
            print(f"[windows per frame: {windows.shape[0]}]")
            print(f"[terms per window: {clf.T_}]")            
            print(f"[computations time: {t2_comps - t1_comps} s]")
            print(f"[postprocess time: {t2_post - t1_post} s]")
            print(f"[other time: {t2_other - t1_other} s]")
            print(f"[fps (computations): {fps_comps:.2f}]")
            print(f"[fps (display): {fps_disp:.2f}]")
            print(f"[detections in this frame: {len(detections)}]")           
        t2 = time.time()
        tpf_prev = t2 - t1        
    t2_loop = time.time()
    cv2.destroyAllWindows()
    video.release()    
    avg_fps_comps = n_frames / time_comps
    avg_time_comps_haar = time_comps_haar / n_frames
    avg_time_comps_frbb = time_comps_frbb / n_frames
    avg_fps_disp = n_frames / (t2_loop - t1_loop)
    print(f"DEMO OF DETECT IN VIDEO DONE. [avg fps (computations): {avg_fps_comps:.2f}, avg time haar: {avg_time_comps_haar * 1000:.2f} ms, avg time frbb: {avg_time_comps_frbb * 1000:.2f} ms; avg fps (display): {avg_fps_disp:.2f}]")
        
def demo_detect_in_video_multiple_clfs(clfs, hcoords, thresholds, computations="cuda", postprocess="avg", n_jobs=8, verbose_loop=True, verbose_detect=False):
    print("DEMO OF DETECT IN VIDEO...")
    colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0)] # TODO longer palette
    gpu_name = gpu_props()["name"]
    video = cv2.VideoCapture(CV2_VIDEO_CAPTURE_CAMERA_INDEX + (cv2.CAP_DSHOW if CV2_VIDEO_CAPTURE_IS_IT_MSWINDOWS else 0))
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    video.set(cv2.CAP_PROP_FPS, 30)    
    _, frame = video.read()
    frame_h, frame_w, _ = frame.shape
    resized_height = haar.HEIGHT    
    resized_width = int(np.round(frame.shape[1] / frame.shape[0] * resized_height))
    
    windows = prepare_detection_windows(resized_height, resized_width)    
    shcoords_multiple_scales = []
    features_indexes = []
    for clf in clfs:
        features_indexes.append(clf.features_indexes_)
        shcoords_multiple_scales.append(prepare_scaled_haar_coords(hcoords, clf.features_indexes_))
            
    print(f"[frame shape: {frame.shape}]")
    print(f"[windows per frame: {windows.shape[0]}]")
    print(f"[terms per window: {clf.T_}]")
    print(f"[about to start a camera...]")
    h_scale = frame_h / resized_height
    w_scale = frame_w / resized_width 
    n_frames = 0
    ma_decay = 0.9
    fps_disp_ma = 0.0    
    fps_disp = 0.0
    fps_comps_ma = 0.0    
    fps_comps = 0.0   
    draw_thickness = 1 if postprocess == None else 2
    # device side arrays in case of cuda method
    dev_windows = None 
    dev_shcoords_multiple_scales = []    
    dev_X_selected = []
    dev_mins_selected = []
    dev_maxes_selected = []
    dev_logits = []
    dev_responses = []
    if computations == "cuda":
        dev_windows = cuda.to_device(windows)
        for i, clf in enumerate(clfs):
            m, T = windows.shape[0], features_indexes[i].size        
            dev_shcoords_multiple_scales.append(cuda.to_device(shcoords_multiple_scales[i]))        
            dev_X_selected.append(cuda.device_array((m, T), dtype=np.int16))
            dev_mins_selected.append(cuda.to_device(clf.mins_selected_))
            dev_maxes_selected.append(cuda.to_device(clf.maxes_selected_))
            dev_logits.append(cuda.to_device(clf.logits_))
            dev_responses.append(cuda.device_array(m, dtype=np.float32))
    tpf_prev = 0.0
    time_comps = 0.0
    time_comps_haar = 0.0
    time_comps_frbb = 0.0
    t1_loop = time.time()
    while(True):
        t1 = time.time()
        if verbose_loop:
            print(f"----------------------------------------------------------------")
            print(f"[frame: {n_frames}]")        
        t1_read = time.time()
        _, frame = video.read()
        t2_read = time.time()
        t1_flip = time.time()
        frame = cv2.flip(frame, 1)
        t2_flip = time.time()        
        tpf_comps = 0.0
        tpf_post = 0.0
        times_haar = 0.0
        times_frbb = 0.0
        for i in range(len(clfs)):
            t1_comps = time.time()        
            if computations == "simple":
                detections, responses, times = detect_simple(frame, clfs[i], hcoords, n, features_indexes[i], thresholds[i], windows, shcoords_multiple_scales[i], verbose=verbose_detect)
            elif computations == "parallel":
                detections, responses, times = detect_parallel(frame, clfs[i], hcoords, n, features_indexes[i], thresholds[i], windows, shcoords_multiple_scales[i], n_jobs=n_jobs, verbose=verbose_detect)        
            elif computations == "cuda":
                detections, responses, times = detect_cuda(frame, clfs[i], hcoords, features_indexes[i], thresholds[i], windows, shcoords_multiple_scales[i], 
                                                           dev_windows, dev_shcoords_multiple_scales[i], dev_X_selected[i], dev_mins_selected[i], dev_maxes_selected[i], dev_logits[i], dev_responses[i], 
                                                           verbose=verbose_detect)
            t2_comps = time.time()
            tpf_comps += t2_comps - t1_comps            
            t1_post = time.time()
            if postprocess is not None:
                if postprocess == "nms":
                    detections, responses = postprocess_nms(detections, responses, 0.25)
                if postprocess == "avg":
                    detections, responses = postprocess_avg(detections, responses, 0.25)
            t2_post = time.time()
            tpf_post += t2_post - t1_post
            if times["haar"] and times["frbb"]:
                times_haar += times["haar"]
                times_frbb += times["frbb"] 
            for index, (j0, k0, h, w) in enumerate(detections):
                js = int(np.round(j0 * h_scale))
                ks = int(np.round(k0 * w_scale))
                hs = int(np.round(h * h_scale))
                ws = int(np.round(w * w_scale))
                cv2.rectangle(frame, (ks, js), (ks + ws - 1, js + hs - 1), colors[i], draw_thickness)
                if postprocess:
                    cv2.putText(frame, f"{responses[index]:.1f}", (k0, j0 + ws - 2), cv2.FONT_HERSHEY_PLAIN, 1.0, colors[i], draw_thickness)            
        normalizer_ma = 1.0 / (1.0 - ma_decay**(n_frames + 1))
        if n_frames > 0:
            fps_disp_ma = ma_decay * fps_disp_ma + (1.0 - ma_decay) * 1.0 / tpf_prev
            fps_disp = fps_disp_ma * normalizer_ma
        fps_comps_ma = ma_decay * fps_comps_ma + (1.0 - ma_decay) * 1.0 / tpf_comps
        fps_comps = fps_comps_ma * normalizer_ma
        time_comps += tpf_comps
        text_shift = 16
        cv2.putText(frame, f"FRAME: {n_frames}", (0, 0 + text_shift), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)        
        cv2.putText(frame, f"WINDOWS PER FRAME: {windows.shape[0]}", (0, frame_h - 1 - 4 * text_shift), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)
        cv2.putText(frame, f"TERMS PER WINDOW FOR ALL CLFS: {[clf.T_ for clf in clfs]}", (0, frame_h - 1 - 3 * text_shift), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)
        cv2.putText(frame, f"GPU: {gpu_name}", (0, frame_h - 1 - 2 * text_shift), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)
        comps_details = ""
        if times_haar > 0.0 and times_frbb > 0.0:
            comps_details += f"[HAAR: {times_haar * 1000:06.2f} ms"            
            comps_details += f", FRBB: {times_frbb * 1000:06.2f} ms]"
            time_comps_haar += times_haar
            time_comps_frbb += times_frbb
        cv2.putText(frame, f"FPS (COMPUTATIONS): {fps_comps:.2f} {comps_details}", (0, frame_h - 1 - 1 * text_shift), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)
        cv2.putText(frame, f"FPS (DISPLAY): {fps_disp:.2f}", (0, frame_h - 1), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)                    
        imshow_name = "DEMO: FAST REAL-BOOST WITH BINS WORKING ON HAAR-LIKE FEATURES [press ESC to quit]"             
        cv2.imshow(imshow_name, frame)
        cv2.namedWindow(imshow_name)
        if cv2.waitKey(1) & 0xFF == 27: # esc key
            break
        n_frames += 1
        if verbose_loop:
            print(f"[read time: {t2_read - t1_read} s]")
            print(f"[flip time: {t2_flip - t1_flip} s]")
            print(f"[windows per frame: {windows.shape[0]}]")
            print(f"[terms per window for all clfs: {[clf.T_ for clf in clfs]}]")
            print(f"[computations time: {tpf_comps} s]")
            print(f"[postprocess time: {tpf_post} s]")
            print(f"[fps (computations): {fps_comps:.2f}]")
            print(f"[fps (display): {fps_disp:.2f}]")
            print(f"[detections in this frame: {len(detections)}]")           
        t2 = time.time()
        tpf_prev = t2 - t1        
    t2_loop = time.time()
    cv2.destroyAllWindows()
    video.release()    
    avg_fps_comps = n_frames / time_comps
    avg_time_comps_haar = time_comps_haar / n_frames
    avg_time_comps_frbb = time_comps_frbb / n_frames
    avg_fps_disp = n_frames / (t2_loop - t1_loop)
    print(f"DEMO OF DETECT IN VIDEO DONE. [avg fps (computations): {avg_fps_comps:.2f}, avg time haar: {avg_time_comps_haar * 1000:.2f} ms, avg time frbb: {avg_time_comps_frbb * 1000:.2f} ms; avg fps (display): {avg_fps_disp:.2f}]")

        
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":        
    print("DEMONSTRATION OF \"FAST REAL-BOOST WITH BINS\" ALGORITHM IMPLEMENTED VIA NUMBA.JIT AND NUMBA.CUDA.")

    n = haar.HAAR_TEMPLATES.shape[0] * S**2 * (2 * P - 1)**2    
    hinds = haar.haar_indexes(S, P)
    hcoords = haar.haar_coords(S, P, hinds)
    
    data_suffix = f"{KIND}_n_{n}_S_{S}_P_{P}_NPI_{NPI}_SEED_{SEED}" 
    DATA_NAME = f"data_{data_suffix}.bin"
    CLF_NAME = f"clf_frbb_{data_suffix}_T_{T}_B_{B}.bin"    
    print(f"DATA_NAME: {DATA_NAME}")
    print(f"CLF_NAME: {CLF_NAME}")
    print(f"GPU_PROPS: {gpu_props()}")    
    
    if DEMO_HAAR_FEATURES_ALL:
        demo_haar_features(hinds, hcoords, n)      
    
    if REGENERATE_DATA:
        if KIND == "face":
            X_train, y_train, X_test, y_test = datagenerator.fddb_data_to_haar(hcoords, n, NPI, seed=SEED, verbose=False)
        elif KIND == "hand":                        
            X_train, y_train, X_test, y_test = datagenerator.hagrid_data(hcoords, n, NPI, seed=SEED, verbose=False)
        pickle_objects(FOLDER_DATA + DATA_NAME, [X_train, y_train, X_test, y_test])
    
    if FIT_OR_REFIT_MODEL or MEASURE_ACCS_OF_MODEL:
        if not REGENERATE_DATA: 
            [X_train, y_train, X_test, y_test] = unpickle_objects(FOLDER_DATA + DATA_NAME)
        print(f"[X_train.shape: {X_train.shape} (positives: {np.sum(y_train == 1)}), X_test.shape: {X_test.shape} (positives: {np.sum(y_test == 1)})]")
    
    if FIT_OR_REFIT_MODEL: 
        clf = FastRealBoostBins(T=T, B=B, fit_mode="numba_cuda", decision_function_mode="numba_cuda", verbose=True, debug_verbose=False)
        clf.fit(X_train, y_train)
        pickle_objects(FOLDER_CLFS + CLF_NAME, [clf])
    
    if (MEASURE_ACCS_OF_MODEL or DEMO_DETECT_IN_VIDEO) and not FIT_OR_REFIT_MODEL:
        [clf] = unpickle_objects(FOLDER_CLFS + CLF_NAME)
    
    if DEMO_HAAR_FEATURES_SELECTED and clf is not None:
        selected = features_indexes_
        demo_haar_features(hinds[selected], hcoords[selected], selected.size)
        
    if MEASURE_ACCS_OF_MODEL:
        measure_accs_of_model(clf, X_train, y_train, X_test, y_test)        
    
    if DEMO_DETECT_IN_VIDEO:
        #demo_detect_in_video(clf, hcoords, threshold=DETECTION_THRESHOLD, computations="cuda", postprocess=DETECTION_POSTPROCESS, n_jobs=8, verbose_loop=True, verbose_detect=True)
        clfs_names = ["clf_frbb_face_n_18225_S_5_P_5_NPI_200_SEED_0_T_1024_B_8.bin", "clf_frbb_hand_n_18225_S_5_P_5_NPI_10_SEED_0_T_2048_B_8.bin"]
        clfs = [unpickle_objects(FOLDER_CLFS + clf_name)[0] for clf_name in clfs_names]
        thresholds = [7.0, 10.0]
        demo_detect_in_video_multiple_clfs(clfs, hcoords, thresholds, computations="cuda", postprocess=DETECTION_POSTPROCESS, n_jobs=8, verbose_loop=True, verbose_detect=False)

    print("ALL DONE.")
    
    
    
    
if __name__ == "__rocs__":        
    print("ROCS...")
    
    clfs_settings = [                     
                      {"KIND": "hand", "S": 5, "P": 5, "NPI": 10, "SEED": 0, "T": 2048, "B": 8}                  
                     ]
    
    for s in clfs_settings:
        KIND = s["KIND"]
        S = s["S"]
        P = s["P"]
        NPI = s["NPI"]        
        SEED = s["SEED"]
        T = s["T"]
        B = s["B"] 
        n = haar.HAAR_TEMPLATES.shape[0] * S**2 * (2 * P - 1)**2    
        hinds = haar.haar_indexes(S, P)
        hcoords = haar.haar_coords(S, P, hinds)            
        data_suffix = f"{KIND}_n_{n}_S_{S}_P_{P}_NPI_{NPI}_SEED_{SEED}"                                      
        DATA_NAME = "data_hand_n_18225_S_5_P_5_NPI_10_SEED_0.bin"
        [X_train, y_train, X_test, y_test] = unpickle_objects(FOLDER_DATA + DATA_NAME)        
        CLF_NAME = f"clf_frbb_{data_suffix}_T_{T}_B_{B}.bin"            
        print("---")
        print(f"DATA_NAME: {DATA_NAME}")
        print(f"CLF_NAME: {CLF_NAME}")                            
        [clf] = unpickle_objects(FOLDER_CLFS + CLF_NAME)                
        responses_test = clf.decision_function(X_test)
        roc = roc_curve(y_test, responses_test)
        best_thr, best_prec = best_threshold_via_prec(roc, y_test) 
        fars, sens, thrs = roc
        roc_arr = np.array([fars, sens, thrs]).T        
        plt.plot(fars, sens, label=CLF_NAME)
        print(f"[X_train.shape: {X_train.shape}, X_test.shape: {X_test.shape}]")        
        # with np.printoptions(threshold=np.inf):
        #     print(roc_arr)
    plt.xscale("log")
    plt.xlabel("FAR")
    plt.ylabel("SENSITIVITY")
    plt.legend(loc="lower right", fontsize=8)
    plt.show()