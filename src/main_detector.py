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
from utils import cpu_and_system_props, gpu_props
import argparse

__author__ = "Przemysław Klęsk"
__email__ = "pklesk@zut.edu.pl"


# main settings
KIND = "face"
S = 5 # parameter "scales" to generete Haar-like features
P = 5 # parameter "positions" to generete Haar-like features
NPI = 300 # "negatives per image" - no. of negatives (negative windows) to sample per image (image real or generated synthetically) 
T = 1024 # size of ensemble in FastRealBoostBins (equivalently, no. of boosting rounds when fitting)
B = 8 # no. of bins
SEED = 0 # randomization seed
DEMO_HAAR_FEATURES_ALL = False
DEMO_HAAR_FEATURES_SELECTED = False
REGENERATE_DATA = False
FIT_OR_REFIT_MODEL = False
MEASURE_ACCS_OF_MODEL = False
ADJUST_DECISION_THRESHOLD_OF_MODEL = False
DEMO_DETECT_IN_VIDEO = False
DEMO_DETECT_IN_VIDEO_COMPUTATIONS = "gpu_cuda" # choices: "cpu_simple", "cpu_parallel", "gpu_cuda"
DEMO_DETECT_IN_VIDEO_PARALLEL_JOBS = 8
DEMO_DETECT_IN_VIDEO_VERBOSE_LOOP = True
DEMO_DETECT_IN_VIDEO_VERBOSE_DETECT = False
DEMO_DETECT_IN_VIDEO_FRAMES = None # if not None but integer then detection is stopped after seeing given number of frames
DEMO_DETECT_IN_VIDEO_MULTIPLE_CLFS = True

# cv2 camera settings
CV2_VIDEO_CAPTURE_CAMERA_INDEX = 0
CV2_VIDEO_CAPTURE_IS_IT_MSWINDOWS = False

# detection procedure settings
DETECTION_SCALES = 9 # 9 (lighter), 12 (heavier)
DETECTION_WINDOW_HEIGHT_MIN = 96 # 96 (lighter), 64 (heavier) 
DETECTION_WINDOW_WIDTH_MIN = 96 # 96 (lighter), 64 (heavier)
DETECTION_WINDOW_GROWTH = 1.2
DETECTION_WINDOW_JUMP = 0.05
DETECTION_DECISION_THRESHOLD = None # can be set to None (then classfier's internal threshold is used)
DETECTION_POSTPROCESS = "avg" # choices: None, "avg", "nms"

# settings for detection with multiple classifiers (special option)
MC_CLFS_NAMES = ["clf_frbb_face_n_18225_S_5_P_5_NPI_300_SEED_0_T_1024_B_8.bin", "clf_frbb_hand_n_18225_S_5_P_5_NPI_30_SEED_0_T_1024_B_8.bin"]
MC_DECISION_THRESHOLDS = [None, None]

# folders
FOLDER_DATA = "../data/"
FOLDER_CLFS = "../models/"
FOLDER_EXTRAS = "../extras/"
  
                            
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
    
def best_decision_threshold_via_precision(roc, y_test, heuristic_coeff):
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
        best_thr = heuristic_coeff * best_thr + (1.0 - heuristic_coeff) * dts_sub[best_index - 1] 
        best_prec = heuristic_coeff * best_prec + (1.0 - heuristic_coeff) * precs[best_index - 1]
    print(f"[best_decision_threshold_via_precision (heuristic_coeff: {heuristic_coeff}) -> best_thr: {best_thr}, best_prec: {best_prec}; py: {py}, best_index on roc: {best_index}, fprs_sub[best_index]: {fprs_sub[best_index]}, tprs_sub[best_index]: {tprs_sub[best_index]}]")
    return best_thr, best_prec

def adjust_decision_threshold_via_precision(clf, X_test, y_test, heuristic_coeff=0.25):
        print("ADJUST DECISION THRESHOLD VIA PRECISION...")
        t1 = time.time()
        responses_test = clf.decision_function(X_test)
        roc = roc_curve(y_test, responses_test)
        best_thr, _ = best_decision_threshold_via_precision(roc, y_test, heuristic_coeff)
        print(f"[adjusting decision threshold within clf from {clf.decision_threshold_} to {best_thr}]")
        clf.decision_threshold_ = best_thr
        t2 = time.time()
        print(f"ADJUST DECISION THRESHOLD VIA PRECISION DONE. [time: {t2 - t1} s] ")
        

def draw_feature_at(i, j0, k0, shcoords_one_feature):
    i_copy = i.copy()
    j, k, h, w = shcoords_one_feature[0]
    cv2.rectangle(i_copy, (k0 + k, j0 + j), (k0 + k + w - 1, j0 + j + h - 1), (0, 0, 0), cv2.FILLED)
    j, k, h, w = shcoords_one_feature[1]
    cv2.rectangle(i_copy, (k0 + k, j0 + j), (k0 + k + w - 1, j0 + j + h - 1), (255, 255, 255), cv2.FILLED)
    return i_copy

def demo_haar_features(hinds, hcoords, n, selected_indexes=None):
    print(f"DEMO OF HAAR FEATURES... [hcoords.shape: {hcoords.shape}]")
    print("[with focus placed on display window press 'esc' to quit or any other key to continue]")
    if KIND == "face":
        i = cv2.imread(FOLDER_EXTRAS + "photo_for_face_features_demo.jpg")
        j0, k0 = 94, 88
        w = h = 251
    elif KIND == "hand": 
        i = cv2.imread(FOLDER_EXTRAS + "photo_for_hand_features_demo.jpg")
        j0, k0 = 172, 53
        w = h = 257
    i_resized = haar.resize_image(i)
    i_gray = cv2.cvtColor(i_resized, cv2.COLOR_BGR2GRAY)
    ii = haar.integral_image_numba_jit(i_gray)
    cv2.rectangle(i_resized, (k0, j0), (k0 + w - 1, j0 + h - 1), (0, 0, 255), 1)
    title = "['esc' to quit or other to continue]"    
    font_size = 1.0
    text_shift = int(font_size * 16)
    color_info = (255, 255, 255)
    cv2.putText(i_resized, f"DEMO OF FEATURES [KIND: {KIND.upper()}]", (0, 0 + 1 * text_shift), cv2.FONT_HERSHEY_PLAIN, font_size, color_info, 1)        
    cv2.imshow(title, i_resized)
    cv2.waitKey()    
    if selected_indexes is None:
        selected_indexes = np.arange(n)
    for ord_ind in selected_indexes:
        ind = hinds[ord_ind]
        c = hcoords[ord_ind]
        hcoords_window = (np.array([h, w, h, w]) * c).astype(np.int16) 
        i_with_feature = draw_feature_at(i_resized, j0, k0, hcoords_window)
        i_temp = cv2.addWeighted(i_resized, 0.35, i_with_feature, 0.65, 0.0)
        feature_value = haar.haar_feature_numba_jit(ii, j0, k0, hcoords_window)
        print(f"[feature index (ordinal): {ord_ind}]")
        print(f"[feature multi-index: {ind}]")
        print(f"[feature hcoords (cartesian):\n {c}]")
        print(f"[feature hcoords in window:\n {hcoords_window}]")
        print(f"[feature value: {feature_value}]")
        print("-" * 160)
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

def detect_cpu_simple(i, clf, hcoords, n, features_indexes, decision_threshold=None, windows=None, shcoords_multiple_scales=None, verbose=False):
    if verbose:
        print("[detect_cpu_simple...]")
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
        print(f"[detect_cpu_simple: preprocessing done; time: {dt_preprocess} s, i_gray.shape: {i_gray.shape}]")
    t1_ii = time.time()
    ii = haar.integral_image_numba_jit(i_gray)
    t2_ii = time.time()
    dt_ii = t2_ii - t1_ii
    times["ii"] = dt_ii
    if verbose:
        print(f"[detect_cpu_simple: integral_image_numba_jit done; time: {dt_ii} s]")
    if windows is None:
        t1_prepare = time.time()
        windows, shcoords_multiple_scales = prepare_detection_windows_and_scaled_haar_coords(i_h, i_w, hcoords, features_indexes)
        t2_prepare = time.time()
        dt_prepare = t2_prepare - t1_prepare
        times["prepare"] = dt_prepare    
        if verbose:
            print(f"[detect_cpu_simple: prepare_detection_windows_and_scaled_haar_coords done; time: {dt_prepare} s, windows to check: {windows.shape[0]}]")
    t1_haar = time.time()    
    X = haar.haar_features_multiple_windows_numba_jit(ii, windows, shcoords_multiple_scales, n, features_indexes)
    t2_haar = time.time()
    dt_haar = t2_haar - t1_haar
    times["haar"] = dt_haar
    if verbose:
        print(f"[detect_cpu_simple: haar_features_multiple_windows done; time: {dt_haar} s]")
    t1_frbb = time.time()    
    responses = clf._decision_function_numba_jit(X)
    t2_frbb = time.time()
    dt_frbb = t2_frbb - t1_frbb 
    times["frbb"] = dt_frbb 
    if verbose:
        print(f"[detect_cpu_simple: clf._decision_function done; time: {dt_frbb} s]")
    t1_ti = time.time()
    if decision_threshold is None:
        decision_threshold = clf.decision_threshold_             
    detected = responses > decision_threshold
    detections = windows[detected, 1:] # skipping scale index
    responses = responses[detected] 
    t2_ti = time.time()
    dt_ti = t2_ti - t1_ti
    times["ti"] = dt_ti
    if verbose:
        print(f"[detect_cpu_simple: finding detections (thresholding and indexing) done; time: {dt_ti} s]")        
    t2 = time.time()
    if verbose:
        print(f"[detect_cpu_simple done; time: {t2 - t1} s]")                    
    return detections, responses, times

def detect_cpu_parallel(i, clf, hcoords, n, features_indexes, decision_threshold=None, windows=None, shcoords_multiple_scales=None, n_jobs=8, verbose=False):
    if verbose:
        print("[detect_cpu_parallel...]")
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
        print(f"[detect_cpu_parallel: preprocessing done; time: {dt_preprocess} s; i_gray.shape: {i_gray.shape}]")
    t1_ii = time.time()
    ii = haar.integral_image_numba_jit(i_gray)
    t2_ii = time.time()
    dt_ii = t2_ii - t1_ii
    times["ii"] = dt_ii
    if verbose:
        print(f"[detect_cpu_parallel: integral_image_numba_jit done; time: {dt_ii} s]")
    if windows is None:
        t1_prepare = time.time()
        windows, shcoords_multiple_scales = prepare_detection_windows_and_scaled_haar_coords(i_h, i_w, hcoords, features_indexes)
        t2_prepare = time.time()
        dt_prepare = t2_prepare - t1_prepare
        times["prepare"] = dt_prepare
        if verbose:
            print(f"[detect_cpu_parallel: prepare_detection_windows_and_scaled_haar_coords done; time: {dt_prepare} s; windows to check: {windows.shape[0]}]")    
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
            job_responses = clf._decision_function_numba_jit(X)         
            return job_responses 
        workers_results = parallel((delayed(worker)(job_index) for job_index in range(n_calls)))
        responses =  reduce(lambda a, b: np.r_[a, b], [jr for jr in workers_results])
        if decision_threshold is None:
            decision_threshold = clf.decision_threshold_
        detected = responses > decision_threshold
        detections = windows[detected, 1:] # skipping scale index
        responses = responses[detected] 
    t2_parallel = time.time()
    dt_parallel = t2_parallel - t1_parallel
    times["parallel"] = dt_parallel
    if verbose:
        print(f"[detect_cpu_parallel: all parallel jobs done (haar_features_multiple_windows_numba_jit_tf, clf.decision_function, finding detections (thresholding and indexing); time {dt_parallel} s]")    
    t2 = time.time()
    if verbose:
        print(f"[detect_cpu_parallel done; time: {t2 - t1} s]")                    
    return detections, responses, times

def detect_gpu_cuda(i, clf, hcoords, features_indexes, decision_threshold=None, windows=None, shcoords_multiple_scales=None, 
                    dev_windows=None, dev_shcoords_multiple_scales=None, dev_X_selected=None, dev_mins_selected=None, dev_maxes_selected=None, dev_logits=None, dev_responses=None, 
                    verbose=False):
    if verbose:
        print("[detect_gpu_cuda...]")
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
        print(f"[detect_gpu_cuda: preprocessing done; time: {dt_preprocess} s; i_gray.shape: {i_gray.shape}]")
    t1_ii = time.time()
    ii = haar.integral_image_numba_jit(i_gray)
    t2_ii = time.time()
    dt_ii = t2_ii - t1_ii
    times["ii"] = dt_ii
    if verbose:
        print(f"[detect_gpu_cuda: integral_image_numba_jit done; time: {dt_ii} s]")
    if windows is None:
        t1_prepare = time.time()
        windows, shcoords_multiple_scales = prepare_detection_windows_and_scaled_haar_coords(i_h, i_w, hcoords, features_indexes)
        dev_windows = cuda.to_device(windows)
        dev_shcoords_multiple_scales = cuda.to_device(shcoords_multiple_scales)        
        t2_prepare = time.time()
        dt_prepare = t2_prepare - t1_prepare 
        times["prepare"] = dt_prepare
        if verbose:
            print(f"[detect_gpu_cuda: prepare_detection_windows_and_scaled_haar_coords done; time: {dt_prepare} s; windows to check: {windows.shape[0]}]")
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
        print(f"[detect_gpu_cuda: haar_features_multiple_windows_numba_cuda done; time: {dt_haar} s]")
    t1_frbb = time.time()    
    FastRealBoostBins._decision_function_numba_cuda_job_int16[bpg, tpb](dev_X_selected, dev_mins_selected, dev_maxes_selected, dev_logits, dev_responses)
    responses = dev_responses.copy_to_host()
    cuda.synchronize()
    t2_frbb = time.time()
    dt_frbb = t2_frbb - t1_frbb
    times["frbb"] = dt_frbb  
    if verbose:
        print(f"[detect_gpu_cuda: FastRealBoostBins._decision_function_numba_cuda_job_int16 done; time: {dt_frbb} s]")
    t1_ti = time.time()
    if decision_threshold is None:
        decision_threshold = clf.decision_threshold_       
    detected = responses > decision_threshold
    detections = windows[detected, 1:] # skipping scale index
    responses = responses[detected] 
    t2_ti = time.time()
    dt_ti = t2_ti - t1_ti
    times["ti"] = dt_ti 
    if verbose:
        print(f"[detect_gpu_cuda: finding detections (thresholding and indexing) done; time: {dt_ti} s]")        
    t2 = time.time()
    if verbose:
        print(f"[detect_cuda done; time: {t2 - t1} s]")                 
    return detections, responses, times

def detect_gpu_cuda_within_multiple_clfs(ii, clf, decision_threshold, windows, 
                                         dev_windows=None, dev_shcoords_multiple_scales=None, dev_X_selected=None, dev_mins_selected=None, dev_maxes_selected=None, dev_logits=None, dev_responses=None, 
                                         verbose=False):
    if verbose:
        print("[detect_gpu_cuda_within_multiple_clfs...]")
    t1 = time.time()
    times = {}
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
        print(f"[detect_gpu_cuda_within_multiple_clfs: haar_features_multiple_windows_numba_cuda done; time: {dt_haar} s]")
    t1_frbb = time.time()    
    FastRealBoostBins._decision_function_numba_cuda_job_int16[bpg, tpb](dev_X_selected, dev_mins_selected, dev_maxes_selected, dev_logits, dev_responses)
    responses = dev_responses.copy_to_host()
    cuda.synchronize()
    t2_frbb = time.time()
    dt_frbb = t2_frbb - t1_frbb
    times["frbb"] = dt_frbb  
    if verbose:
        print(f"[detect_gpu_cuda_within_multiple_clfs: FastRealBoostBins._decision_function_numba_cuda_job_int16 done; time: {dt_frbb} s]")
    t1_ti = time.time()
    if decision_threshold is None:
        decision_threshold = clf.decision_threshold_       
    detected = responses > decision_threshold
    detections = windows[detected, 1:] # skipping scale index
    responses = responses[detected] 
    t2_ti = time.time()
    dt_ti = t2_ti - t1_ti
    times["ti"] = dt_ti 
    if verbose:
        print(f"[detect_gpu_cuda_within_multiple_clfs: finding detections (thresholding and indexing) done; time: {dt_ti} s]")        
    t2 = time.time()
    if verbose:
        print(f"[detect_gpu_cuda_within_multiple_clfs done; time: {t2 - t1} s]")                 
    return detections, responses, times

def postprocess_nms(detections, responses, iou_threshold=0.25):
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
            if haar.iou2(d[i], d[j]) >= iou_threshold:
                indexes[j] = False
    return d_final, r_final

def postprocess_avg(detections, responses, iou_threshold=0.25):
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
            if haar.iou2(d[i], d[j]) >= iou_threshold:
                indexes[j] = False
                d_avg.append(d[j])
                r_avg.append(r[j])
        d_avg = (np.round(np.mean(np.array(d_avg), axis=0))).astype(np.int16)
        r_avg = np.mean(r_avg)
        d_final.append(d_avg)
        r_final.append(r_avg)
    return d_final, r_final

def short_cpu_name(cpu_name):
    cpu_name = cpu_name.replace("CPU", "")
    cpu_name = cpu_name.replace("cpu", "")
    cpu_name = cpu_name.replace("  ", " ")
    cpu_name = cpu_name.replace(" @ ", " ")
    cpu_name = cpu_name.replace("(R)", "")
    return cpu_name    

def color_by_detector_title(colors_detect, detector_title):
    mappings = {"face" : 0, "hand": 1}
    title = detector_title.lower()
    index = 0
    if title in mappings:
        index = mappings[title]
    return colors_detect[index]  

def demo_detect_in_video(clf, hcoords, decision_threshold, computations="gpu_cuda", postprocess="avg", n_jobs=8, detector_title=None, verbose_loop=True, verbose_detect=False):
    print(f"DEMO OF DETECT IN VIDEO... [computations: {computations}]")
    colors_detect = [(0, 255, 255), (255, 255, 0), (255, 0, 255), (255, 0, 0), (0, 0, 255), (255, 0, 0)]
    color_detect = color_by_detector_title(colors_detect, detector_title)
    color_info = (255, 255, 255)
    font_size = 1.0
    text_shift = int(font_size * 16)
    gpu_name = gpu_props()["name"]
    cpu_name = short_cpu_name(cpu_and_system_props()["cpu_name"])
    features_indexes = clf.features_selected_
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
    print(f"[terms per window: {clf.T}]")    
    decision_threshold_source = "internal (clf.decision_threshold_)" if decision_threshold is None else "external (decision_threshold argument)"
    if decision_threshold is None: 
        decision_threshold = clf.decision_threshold_
    print(f"[decision threshold: {decision_threshold}, source: {decision_threshold_source}]")    
    print(f"[about to start video camera...]")
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
    if computations == "gpu_cuda":
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
            print("-" * 160)
            print(f"[frame: {n_frames}]")        
        t1_read = time.time()
        _, frame = video.read()
        t2_read = time.time()
        t1_flip = time.time()
        frame = cv2.flip(frame, 1)
        t2_flip = time.time()
        if verbose_loop:
            print(f"[read time: {t2_read - t1_read} s]")
            print(f"[flip time: {t2_flip - t1_flip} s]")
            print(f"[windows per frame: {windows.shape[0]}]")
            print(f"[terms per window: {clf.T}]")                            
        t1_comps = time.time()
        if computations == "cpu_simple":
            detections, responses, times = detect_cpu_simple(frame, clf, hcoords, n, features_indexes, decision_threshold, windows, shcoords_multiple_scales, verbose=verbose_detect)
        elif computations == "cpu_parallel":
            detections, responses, times = detect_cpu_parallel(frame, clf, hcoords, n, features_indexes, decision_threshold, windows, shcoords_multiple_scales, n_jobs=n_jobs, verbose=verbose_detect)        
        elif computations == "gpu_cuda":
            detections, responses, times = detect_gpu_cuda(frame, clf, hcoords, features_indexes, decision_threshold, windows, shcoords_multiple_scales, 
                                                           dev_windows, dev_shcoords_multiple_scales, dev_X_selected, dev_mins_selected, dev_maxes_selected, dev_logits, dev_responses, 
                                                           verbose=verbose_detect)                        
        t2_comps = time.time()
        t1_post = time.time()
        if postprocess is not None:
            if postprocess == "nms":
                detections, responses = postprocess_nms(detections, responses)
            if postprocess == "avg":
                detections, responses = postprocess_avg(detections, responses)
        t2_post = time.time()
        t1_other = time.time() 
        for index, (j0, k0, h, w) in enumerate(detections):
            js = int(np.round(j0 * h_scale))
            ks = int(np.round(k0 * w_scale))
            hs = int(np.round(h * h_scale))
            ws = int(np.round(w * w_scale))
            cv2.rectangle(frame, (ks, js), (ks + ws - 1, js + hs - 1), color_detect, draw_thickness)
            if postprocess:
                cv2.putText(frame, f"{responses[index]:.1f}", (ks, js + ws - 2), cv2.FONT_HERSHEY_PLAIN, font_size, color_detect, draw_thickness)
        normalizer_ma = 1.0 / (1.0 - ma_decay**(n_frames + 1))
        if n_frames > 0:
            fps_disp_ma = ma_decay * fps_disp_ma + (1.0 - ma_decay) * 1.0 / tpf_prev
            fps_disp = fps_disp_ma * normalizer_ma
        fps_comps_ma = ma_decay * fps_comps_ma + (1.0 - ma_decay) * 1.0 / (t2_comps - t1_comps)
        fps_comps = fps_comps_ma * normalizer_ma            
        time_comps += t2_comps - t1_comps
        cv2.putText(frame, "DEMO: FAST REAL BOOST BINS VIA NUMBA (+HAAR FEATURES)", (0, 0 + 1 * text_shift), cv2.FONT_HERSHEY_PLAIN, font_size, color_info, 2)
        cv2.putText(frame, f"DETECTOR KIND: {detector_title}", (0, 0 + 2 * text_shift), cv2.FONT_HERSHEY_PLAIN, font_size, color_info, 1)
        cv2.putText(frame, f"FRAME: {n_frames}", (0, 0 + 3 * text_shift), cv2.FONT_HERSHEY_PLAIN, font_size, color_info, 1)
        cv2.putText(frame, f"WINDOWS PER FRAME: {windows.shape[0]}", (0, frame_h - 1 - 4 * text_shift), cv2.FONT_HERSHEY_PLAIN, font_size, color_info, 1)
        cv2.putText(frame, f"TERMS PER WINDOW: {clf.T}", (0, frame_h - 1 - 3 * text_shift), cv2.FONT_HERSHEY_PLAIN, font_size, color_info, 1)
        computations_str = f"COMPUTATIONS: {computations.upper()}"
        if computations == "gpu_cuda":
            computations_str += f" [GPU: {gpu_name.upper()}]"
        else:
            computations_str += f" [CPU: {cpu_name.upper()}]"            
        cv2.putText(frame, f"{computations_str}", (0, frame_h - 1 - 2 * text_shift), cv2.FONT_HERSHEY_PLAIN, font_size, color_info, 1)
        comps_details = ""
        if "haar" in times and "frbb" in times:
            comps_details += f"[HAAR: {times['haar'] * 1000:06.2f} ms"            
            comps_details += f", FRBB: {times['frbb'] * 1000:06.2f} ms]"
            time_comps_haar += times["haar"]
            time_comps_frbb += times["frbb"]
        cv2.putText(frame, f"FPS (COMPUTATIONS): {fps_comps:.2f} {comps_details}", (0, frame_h - 1 - 1 * text_shift), cv2.FONT_HERSHEY_PLAIN, font_size, color_info, 1)
        cv2.putText(frame, f"FPS (DISPLAY): {fps_disp:.2f}", (0, frame_h - 1), cv2.FONT_HERSHEY_PLAIN, font_size, color_info, 1)                    
        imshow_name = "FAST REAL BOOST BINS ['esc' to quit]"
        cv2.namedWindow(imshow_name)             
        cv2.imshow(imshow_name, frame)
        n_frames += 1        
        if cv2.waitKey(1) & 0xFF == 27 or (DEMO_DETECT_IN_VIDEO_FRAMES is not None and n_frames >= DEMO_DETECT_IN_VIDEO_FRAMES): # esc key
            break       
        t2_other = time.time()                
        if verbose_loop:
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
        
def demo_detect_in_video_multiple_clfs(clfs, hcoords, decision_thresholds, postprocess="avg", detector_title=None, verbose_loop=True, verbose_detect=False):
    print("DEMO OF DETECT IN VIDEO  (MULTIPLE CLFS)...")
    colors_detect = [(0, 255, 255), (255, 255, 0), (255, 0, 255), (255, 0, 0), (0, 0, 255), (255, 0, 0)]
    color_info = (255, 255, 255)
    font_size = 1.0
    text_shift = int(font_size * 16)        
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
        features_indexes.append(clf.features_selected_)
        shcoords_multiple_scales.append(prepare_scaled_haar_coords(hcoords, clf.features_selected_))            
    print(f"[frame shape: {frame.shape}]")
    print(f"[windows per frame: {windows.shape[0]}]")
    print(f"[terms per window per clfs: {[clf.T for clf in clfs]}]")        
    print(f"[decision thresholds within clfs (internal): {[clf.decision_threshold_ for clf in clfs]}]")    
    print(f"[decision thresholds (external overriding argument): {decision_thresholds}]")        
    print(f"[about to start video camera...]")
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
    dev_shcoords_multiple_scales = []    
    dev_X_selected = []
    dev_mins_selected = []
    dev_maxes_selected = []
    dev_logits = []
    dev_responses = []
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
            print("-" * 160)
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
        t1_preprocess = time.time()
        i_resized = haar.resize_image(frame)
        i_gray = cv2.cvtColor(i_resized, cv2.COLOR_BGR2GRAY)
        t2_preprocess = time.time()            
        t1_ii = time.time()
        ii = haar.integral_image_numba_jit(i_gray)
        t2_ii = time.time()                    
        if verbose_loop:
            print(f"[read time: {t2_read - t1_read} s]")
            print(f"[flip time: {t2_flip - t1_flip} s]")
            print(f"[preprocessing done; time: {t2_preprocess - t1_preprocess} s; i_gray.shape: {i_gray.shape}]")
            print(f"[integral image done; time: {t2_ii - t1_ii} s]")            
            print(f"[windows per frame: {windows.shape[0]}]")
            print(f"[terms per window per clfs: {[clf.T for clf in clfs]}]")        
        for i in range(len(clfs)):
            t1_comps = time.time()      
            detections, responses, times = detect_gpu_cuda_within_multiple_clfs(ii, clfs[i], decision_thresholds[i], windows, 
                                                                                dev_windows, dev_shcoords_multiple_scales[i], dev_X_selected[i], dev_mins_selected[i], dev_maxes_selected[i], dev_logits[i], dev_responses[i], 
                                                                                verbose=verbose_detect)
            t2_comps = time.time()
            tpf_comps += t2_comps - t1_comps            
            t1_post = time.time()
            if postprocess is not None:
                if postprocess == "nms":
                    detections, responses = postprocess_nms(detections, responses)
                if postprocess == "avg":
                    detections, responses = postprocess_avg(detections, responses)
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
                cv2.rectangle(frame, (ks, js), (ks + ws - 1, js + hs - 1), colors_detect[i], draw_thickness)
                if postprocess:
                    cv2.putText(frame, f"{responses[index]:.1f}", (ks, js + ws - 2), cv2.FONT_HERSHEY_PLAIN, font_size, colors_detect[i], draw_thickness)            
        normalizer_ma = 1.0 / (1.0 - ma_decay**(n_frames + 1))
        if n_frames > 0:
            fps_disp_ma = ma_decay * fps_disp_ma + (1.0 - ma_decay) * 1.0 / tpf_prev
            fps_disp = fps_disp_ma * normalizer_ma
        fps_comps_ma = ma_decay * fps_comps_ma + (1.0 - ma_decay) * 1.0 / tpf_comps
        fps_comps = fps_comps_ma * normalizer_ma
        time_comps += tpf_comps
        cv2.putText(frame, "DEMO: FAST REAL BOOST BINS VIA NUMBA (+HAAR FEATURES)", (0, 0 + 1 * text_shift), cv2.FONT_HERSHEY_PLAIN, font_size, color_info, 2)
        cv2.putText(frame, f"DETECTOR KIND: {detector_title}", (0, 0 + 2 * text_shift), cv2.FONT_HERSHEY_PLAIN, 1.0, color_info, 1)
        cv2.putText(frame, f"FRAME: {n_frames}", (0, 0 + 3 * text_shift), cv2.FONT_HERSHEY_PLAIN, font_size, color_info, 1)        
        cv2.putText(frame, f"WINDOWS PER FRAME: {windows.shape[0]}", (0, frame_h - 1 - 4 * text_shift), cv2.FONT_HERSHEY_PLAIN, font_size, color_info, 1)
        cv2.putText(frame, f"TERMS PER WINDOW PER CLFS: {[clf.T for clf in clfs]}", (0, frame_h - 1 - 3 * text_shift), cv2.FONT_HERSHEY_PLAIN, font_size, color_info, 1)
        computations_str = f"COMPUTATIONS: CUDA"
        computations_str += f" [GPU: {gpu_name.upper()}]"
        cv2.putText(frame, f"{computations_str}", (0, frame_h - 1 - 2 * text_shift), cv2.FONT_HERSHEY_PLAIN, font_size, color_info, 1)
        comps_details = ""
        if times_haar > 0.0 and times_frbb > 0.0:
            comps_details += f"[HAAR: {times_haar * 1000:06.2f} ms"            
            comps_details += f", FRBB: {times_frbb * 1000:06.2f} ms]"
            time_comps_haar += times_haar
            time_comps_frbb += times_frbb
        cv2.putText(frame, f"FPS (COMPUTATIONS): {fps_comps:.2f} {comps_details}", (0, frame_h - 1 - 1 * text_shift), cv2.FONT_HERSHEY_PLAIN, font_size, color_info, 1)
        cv2.putText(frame, f"FPS (DISPLAY): {fps_disp:.2f}", (0, frame_h - 1), cv2.FONT_HERSHEY_PLAIN, font_size, color_info, 1)                    
        imshow_name = "FAST REAL BOOST BINS ['esc' to quit]"
        cv2.namedWindow(imshow_name)                     
        cv2.imshow(imshow_name, frame)
        n_frames += 1        
        if cv2.waitKey(1) & 0xFF == 27 or (DEMO_DETECT_IN_VIDEO_FRAMES is not None and n_frames >= DEMO_DETECT_IN_VIDEO_FRAMES): # esc key
            break        
        if verbose_loop:
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
    print(f"DEMO OF DETECT IN VIDEO (MULTIPLE CLFS) DONE. [avg fps (computations): {avg_fps_comps:.2f}, avg time haar: {avg_time_comps_haar * 1000:.2f} ms, avg time frbb: {avg_time_comps_frbb * 1000:.2f} ms; avg fps (display): {avg_fps_disp:.2f}]")

def experiment_some_rocs():
    print("ROCS...")    
    data_name = "data_face_n_18225_S_5_P_5_NPI_10_SEED_0.bin"    
    print(f"[data_name: {data_name}]")
    [X_train, y_train, X_test, y_test] = unpickle_objects(FOLDER_DATA + data_name)    
    print(f"[X_train.shape: {X_train.shape} with {np.sum(y_train == 1)} positives, X_test.shape: {X_test.shape} with {np.sum(y_test == 1)} positives]")
    clfs_names = [
        "clf_frbb_face_n_18225_S_5_P_5_NPI_200_SEED_0_T_1024_B_8.bin",
        "clf_frbb_face_n_18225_S_5_P_5_NPI_200_SEED_0_T_2048_B_8.bin"                                      
        ]    
    for clf_name in clfs_names:                                                                          
        print("-" * 160)        
        print(f"[clf_name: {clf_name}]")                            
        [clf] = unpickle_objects(FOLDER_CLFS + clf_name)                
        responses_test = clf.decision_function(X_test)
        roc = roc_curve(y_test, responses_test)
        best_decision_threshold_via_precision(roc, y_test, heuristic_coeff=0.25) 
        fars, sens, _ = roc                
        plt.plot(fars, sens, label=clf_name)                
    plt.xscale("log")
    plt.xlabel("FAR")
    plt.ylabel("SENSITIVITY")
    plt.legend(loc="lower right", fontsize=8)
    plt.show()

def str_to_float_or_none(s):
    if s.lower() == "none":
        return None
    return float(s)

def str_to_int_or_none(s):
    if s.lower() == "none":
        return None
    return int(s)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--KIND", type=str, default=KIND, choices=["face", "hand"], help=f"detector kind (default: {KIND})")
    parser.add_argument("-s", "--S", type=int, default=S, help=f"'scales' parameter of Haar-like features (default: {S})")
    parser.add_argument("-p", "--P", type=int, default=P, help=f"'positions' parameter of Haar-like features (default: {P})")
    parser.add_argument("-npi", "--NPI", type=int, default=NPI, help=f"'negatives per image' parameter, used in procedures generating data sets from images (default: {NPI} with -k set to {KIND})")
    parser.add_argument("-t", "--T", type=int, default=T, help=f"number of boosting rounds, (default: {T})")
    parser.add_argument("-b", "--B", type=int, default=B, help=f"numbet of bins, (default: {B})")
    parser.add_argument("-seed", "--SEED", type=int, default=SEED, help=f"randomization seed, (default: {SEED})")
    parser.add_argument("-dhfsa", "--DEMO_HAAR_FEATURES_ALL", action="store_true", help="turn on demo of all Haar-like features")
    parser.add_argument("-dhfss", "--DEMO_HAAR_FEATURES_SELECTED", action="store_true", help="turn on demo of selected Haar-like features")
    parser.add_argument("-rd", "--REGENERATE_DATA", action="store_true", help="turn on data regeneration")
    parser.add_argument("-form", "--FIT_OR_REFIT_MODEL", action="store_true", help="fit new or refit an existing model")
    parser.add_argument("-maom", "--MEASURE_ACCS_OF_MODEL", action="store_true", help="measure accuracies of a model")
    parser.add_argument("-adtom", "--ADJUST_DECISION_THRESHOLD_OF_MODEL", action="store_true", help="adjust decision threshold of a model (based on ROC for testing data)")
    parser.add_argument("-ddiv", "--DEMO_DETECT_IN_VIDEO", action="store_true", help="turn on demo of detection in video")
    parser.add_argument("-ddivc", "--DEMO_DETECT_IN_VIDEO_COMPUTATIONS", type=str, choices=["gpu_cuda", "cpu_simple", "cpu_parallel"], default=DEMO_DETECT_IN_VIDEO_COMPUTATIONS, 
                        help=f"type of computations for demo of detection in video (default: {DEMO_DETECT_IN_VIDEO_COMPUTATIONS})")
    parser.add_argument("-ddivpj", "--DEMO_DETECT_IN_VIDEO_PARALLEL_JOBS", type=int, default=DEMO_DETECT_IN_VIDEO_PARALLEL_JOBS, 
                        help=f"number of parallel jobs (only in case of 'cpu_parallel' set for -ddivc) (default: {DEMO_DETECT_IN_VIDEO_PARALLEL_JOBS})")
    parser.add_argument("-ddivvl", "--DEMO_DETECT_IN_VIDEO_VERBOSE_LOOP", action="store_true", help="turn on verbosity for main loop of detection in video")
    parser.add_argument("-ddivvd", "--DEMO_DETECT_IN_VIDEO_VERBOSE_DETECT", action="store_true", help="turn on detailed verbosity for detection in video")
    parser.add_argument("-ddivf", "--DEMO_DETECT_IN_VIDEO_FRAMES", type=str_to_float_or_none, default=DEMO_DETECT_IN_VIDEO_FRAMES, help="limit overall detection in video to given number of frames")
    parser.add_argument("-ddivmc", "--DEMO_DETECT_IN_VIDEO_MULTIPLE_CLFS", action="store_true", help="turn on demo of detection in video with multiple classifiers (currently: face and hand detectors)")
    parser.add_argument("-cv2vcci", "--CV2_VIDEO_CAPTURE_CAMERA_INDEX", type=int, default=CV2_VIDEO_CAPTURE_CAMERA_INDEX, help=f"video camera index (default: {CV2_VIDEO_CAPTURE_CAMERA_INDEX})")
    parser.add_argument("-cv2iim", "--CV2_VIDEO_CAPTURE_IS_IT_MSWINDOWS", action="store_true", help="specify if OS is MS Windows (for cv2 and directx purposes)")
    parser.add_argument("-ds", "--DETECTION_SCALES", type=int, default=DETECTION_SCALES, help=f"number of detection scales (default: {DETECTION_SCALES})")
    parser.add_argument("-dwhm", "--DETECTION_WINDOW_HEIGHT_MIN", type=int, default=DETECTION_WINDOW_HEIGHT_MIN, help=f"minimum height of detection window (default: {DETECTION_WINDOW_HEIGHT_MIN})")
    parser.add_argument("-dwwm", "--DETECTION_WINDOW_WIDTH_MIN", type=int, default=DETECTION_WINDOW_WIDTH_MIN, help=f"minimum width of detection window (default: {DETECTION_WINDOW_WIDTH_MIN})")
    parser.add_argument("-dwg", "--DETECTION_WINDOW_GROWTH", type=int, default=DETECTION_WINDOW_GROWTH, help=f"growth factor of detection window (default: {DETECTION_WINDOW_GROWTH})")
    parser.add_argument("-dwj", "--DETECTION_WINDOW_JUMP", type=int, default=DETECTION_WINDOW_JUMP, help=f"relative jump of detection window (default: {DETECTION_WINDOW_JUMP})")    
    parser.add_argument("-ddt", "--DETECTION_DECISION_THRESHOLD", type=str_to_float_or_none, default=DETECTION_DECISION_THRESHOLD, 
                        help=f"decision threshold, can be set to None then classifier's internal threshold is used (default: {DETECTION_DECISION_THRESHOLD})")
    parser.add_argument("-dp", "--DETECTION_POSTPROCESS", choices=["None", "avg", "nms"], default=DETECTION_POSTPROCESS, help=f"type of detection postprocessing (default: {DETECTION_POSTPROCESS})")
    parser.add_argument("-mccn", "--MC_CLFS_NAMES", type=str, default=MC_CLFS_NAMES, nargs="+", 
                        help=f"classifiers names (list) for detection with multiple classifiers (default: {MC_CLFS_NAMES})")    
    parser.add_argument("-mcdt", "--MC_DECISION_THRESHOLDS", type=str_to_float_or_none, default=MC_DECISION_THRESHOLDS, nargs="+", 
                        help=f"decision thresholds (list) for detection with multiple classifiers, any can be set to None (default: {MC_DECISION_THRESHOLDS})")                                    
    args = parser.parse_args()
    if args.DETECTION_DECISION_THRESHOLD == "None":
        args.DETECTION_DECISION_THRESHOLD = None
    if args.DETECTION_POSTPROCESS == "None":
        args.DETECTION_POSTPROCESS = None
    globals().update(vars(args))    


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":        
    print("\"FAST-REAL-BOOST-BINS\": AN ENSEMBLE CLASSIFIER FOR FAST PREDICTIONS IMPLEMENTED IN PYTHON VIA NUMBA.JIT AND NUMBA.CUDA.", flush=True)
    parse_args()
    print("MAIN-DETECTOR STARTING...")    
    print(f"CPU AND SYSTEM PROPS: {cpu_and_system_props()}")
    print(f"GPU PROPS: {gpu_props()}")    

    n = haar.HAAR_TEMPLATES.shape[0] * S**2 * (2 * P - 1)**2    
    hinds = haar.haar_indexes(S, P)
    hcoords = haar.haar_coords(S, P, hinds)
    clf = None
    
    data_suffix = f"{KIND}_n_{n}_S_{S}_P_{P}_NPI_{NPI}_SEED_{SEED}" 
    DATA_NAME = f"data_{data_suffix}"
    CLF_NAME = f"clf_frbb_{data_suffix}_T_{T}_B_{B}"
    
    if DEMO_HAAR_FEATURES_ALL or REGENERATE_DATA or FIT_OR_REFIT_MODEL or DEMO_HAAR_FEATURES_SELECTED or MEASURE_ACCS_OF_MODEL or ADJUST_DECISION_THRESHOLD_OF_MODEL or DEMO_DETECT_IN_VIDEO:
        print(f"DATA NAME: {DATA_NAME}")
        print(f"CLF NAME: {CLF_NAME}")
    
    if DEMO_HAAR_FEATURES_ALL:
        demo_haar_features(hinds, hcoords, n)      
    
    if REGENERATE_DATA:
        if KIND == "face":
            X_train, y_train, X_test, y_test = datagenerator.fddb_data_to_haar(hcoords, n, NPI, seed=SEED, verbose=False)
        elif KIND == "hand":                        
            X_train, y_train, X_test, y_test = datagenerator.hagrid_data_to_haar(hcoords, n, NPI, seed=SEED, verbose=False)
        pickle_objects(FOLDER_DATA + DATA_NAME + ".bin", [X_train, y_train, X_test, y_test])    
    
    if FIT_OR_REFIT_MODEL or MEASURE_ACCS_OF_MODEL or ADJUST_DECISION_THRESHOLD_OF_MODEL:
        if not REGENERATE_DATA: 
            [X_train, y_train, X_test, y_test] = unpickle_objects(FOLDER_DATA + DATA_NAME + ".bin")
        print(f"[X_train.shape: {X_train.shape} with {np.sum(y_train == 1)} positives, X_test.shape: {X_test.shape} with {np.sum(y_test == 1)} positives]")
    
    if FIT_OR_REFIT_MODEL: 
        clf = FastRealBoostBins(T=T, B=B, fit_mode="numba_cuda", decision_function_mode="numba_cuda", verbose=True, debug_verbose=False)
        clf.fit(X_train, y_train)
        pickle_objects(FOLDER_CLFS + CLF_NAME + ".bin", [clf])
        clf.json_dump(FOLDER_CLFS + CLF_NAME + ".json")    
    
    if clf is None and (MEASURE_ACCS_OF_MODEL or ADJUST_DECISION_THRESHOLD_OF_MODEL or DEMO_HAAR_FEATURES_SELECTED or DEMO_DETECT_IN_VIDEO):
        [clf] = unpickle_objects(FOLDER_CLFS + CLF_NAME + ".bin")
        print(f"[unpickled clf {clf} with decision threshold: {clf.decision_threshold_}]")     
    
    if DEMO_HAAR_FEATURES_SELECTED:        
        demo_haar_features(hinds, hcoords, n, selected_indexes=clf.features_selected_)
    
    if MEASURE_ACCS_OF_MODEL:
        measure_accs_of_model(clf, X_train, y_train, X_test, y_test)            
    
    if ADJUST_DECISION_THRESHOLD_OF_MODEL:    
        adjust_decision_threshold_via_precision(clf, X_test, y_test)
        pickle_objects(FOLDER_CLFS + CLF_NAME + ".bin", [clf])
        clf.json_dump(FOLDER_CLFS + CLF_NAME + ".json")    
    
    if DEMO_DETECT_IN_VIDEO:            
        demo_detect_in_video(clf, hcoords, decision_threshold=DETECTION_DECISION_THRESHOLD, computations=DEMO_DETECT_IN_VIDEO_COMPUTATIONS, postprocess=DETECTION_POSTPROCESS, n_jobs=DEMO_DETECT_IN_VIDEO_PARALLEL_JOBS, 
                             detector_title=KIND.upper(), verbose_loop=DEMO_DETECT_IN_VIDEO_VERBOSE_LOOP, verbose_detect=DEMO_DETECT_IN_VIDEO_VERBOSE_DETECT)
    
    if DEMO_DETECT_IN_VIDEO_MULTIPLE_CLFS:        
        clfs_names = MC_CLFS_NAMES
        decision_thresholds = MC_DECISION_THRESHOLDS
        clfs = [unpickle_objects(FOLDER_CLFS + clf_name)[0] for clf_name in clfs_names]
        print(f"[about to start multiple clfs demo; unpickled clfs:]")
        for clf_name, clf in zip(clfs_names, clfs):
            print(f"[{clf_name} -> {clf} with decision_threshold_: {clf.decision_threshold_}]")        
        demo_detect_in_video_multiple_clfs(clfs, hcoords, decision_thresholds, postprocess=DETECTION_POSTPROCESS, 
                                           detector_title="face, hand".upper(), verbose_loop=DEMO_DETECT_IN_VIDEO_VERBOSE_LOOP, verbose_detect=DEMO_DETECT_IN_VIDEO_VERBOSE_DETECT)

    print("MAIN-DETECTOR DONE.")