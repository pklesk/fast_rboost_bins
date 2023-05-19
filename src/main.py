import sys
import numpy as np
from frbb import FastRealBoostBins
from numba import cuda, jit
from numba import void, int8, int16, int32, float32, uint8
from numba.core.errors import NumbaPerformanceWarning
import cv2
from haar import *
import time
import pickle
from joblib import Parallel, delayed
from functools import reduce
import warnings
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt
from numba.core.typing.builtins import CmpOpLt

__version__ = "1.0.0"
__author__ = "Przemysław Klęsk"
__email__ = "pklesk@zut.edu.pl"

warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
np.set_printoptions(linewidth=512)

# main settings
KIND = "face"
S = 5 # "scales" parameter to generete Haar-like features
P = 5 # "positions" parameter to generete Haar-like features
NPI = 100 # no. of negatives (negative windows) to sample per image from FDDB material
AUG = 0 # data augmentation (0 -> none or 1 -> present)
T = 1024 # size of ensemble in FastRealBoostBins (equivalently, no. of boosting rounds when fitting)
B = 8 # no. of bins
SEED = 0 # randomization seed
DEMO_HAAR_FEATURES = False
REGENERATE_DATA_FROM_FDDB = False
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
DETECTION_THRESHOLD = 9.5
DETECTION_POSTPROCESS = "avg" # possible values: None, "nms", "avg"

# folders
FOLDER_DATA = "../data/"
FOLDER_CLFS = "../models/"
FOLDER_EXTRAS = "../extras/"
FOLDER_RAW_DATA_FDDB = "../raw_data_fddb/"

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

def fddb_read_single_fold(path_root, path_fold_relative, n_negs_per_img, hcoords, n, verbose=False, fold_title="", seed=0, data_augmentation=False):
    np.random.seed(seed)    
    
    # settings for sampling negatives
    w_relative_min = 0.1
    w_relative_max = 0.35
    w_relative_spread = w_relative_max - w_relative_min
    neg_max_iou = 0.5
    
    X_list = []
    y_list = []
    
    lines = []
    f = open(path_root + path_fold_relative, "r")
    while True:
        line = f.readline()
        if line != "":
            lines.append(line)
        else:
            break
    f.close()
    
    n_img = 0
    n_faces = 0
    counter = 0
    li = 0 # line index
    line = lines[li].strip()
    li += 1
    augmentations = [None]
    if data_augmentation: 
        augmentations += ["sharpen", "blur", "random_brightness", "random_channel_shift"]    
    while line != "":
        file_name = path_root + line + ".jpg"
        log_line =  str(counter) + ": [" + file_name + "]"
        if fold_title != "":
            log_line += " [" + fold_title + "]" 
        print(log_line)
        counter += 1
        
        i0_original = cv2.imread(file_name)                    
        line = lines[li]
        li += 1
        n_img_faces = int(line)   
        for aug_index, aug in enumerate(augmentations):
            if verbose:
                print(f"[augmentation: {aug}]")
            i0 = np.copy(i0_original)
            if aug_index > 0:
                li -= n_img_faces
            if aug is not None:
                if aug == "sharpen":
                    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                    i0 = cv2.filter2D(i0, -1, kernel)
                if aug == "blur":
                    i0 = cv2.blur(i0, ksize=(5, 5))
                if aug == "random_brightness":
                    value = np.random.uniform(0.5, 2.5)
                    hsv = cv2.cvtColor(i0, cv2.COLOR_BGR2HSV)
                    hsv = np.array(hsv, dtype=np.float64)
                    hsv[:, :, [1, 2]] = value * hsv[:, :, [1, 2]]
                    hsv = np.clip(hsv, 0, 255)
                    hsv = np.array(hsv, dtype=np.uint8)
                    i0 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                if aug == "random_channel_shift":
                    i0 = i0.astype(np.int16)
                    for chnl in range(3):
                        value = int(np.random.uniform(-64, 64))
                        i0[:, :, chnl] += value
                    i0 = (np.clip(i0, 0, 255)).astype(np.uint8)
            i = cv2.cvtColor(i0, cv2.COLOR_BGR2GRAY)                    
            ii = integral_image_numba_jit(i)
            n_img += 1        
            img_faces_coords = []
            for _ in range(n_img_faces):
                line = lines[li].strip()
                li += 1
                r_major, _, _, center_x, center_y, dummy_one = list(map(float, line.split()))
                h = int(1.5 * r_major)
                w = h                
                j0 = int(center_y - h / 2) 
                k0 = int(center_x - w / 2)
                if aug is not None:
                    growth = np.random.uniform(0.95, 1.05)
                    h = int(np.round(h * growth))
                    w = h
                    j0 += int(np.round(np.random.uniform(-0.05, 0.05) * h))
                    k0 += int(np.round(np.random.uniform(-0.05, 0.05) * w))
                img_face_coords = np.array([j0, k0, j0 + h - 1, k0 + w - 1])
                if j0 < 0 or k0 < 0 or j0 + h - 1 >= i.shape[0] or k0 + w - 1 >= i.shape[1]:
                    if verbose:
                        print(f"[window {img_face_coords} out of bounds -> ignored]")
                    continue
                if (h / ii.shape[0] < 0.075): # min relative size of positive window (smaller may lead to division by zero when white regions in haar features have no area)
                    if verbose:
                        print(f"[window {img_face_coords} too small -> ignored]")
                    continue                            
                n_faces += 1
                img_faces_coords.append(img_face_coords)
                if verbose:
                    p1 = (k0, j0)
                    p2 = (k0 + w - 1, j0 + h - 1)    
                    cv2.rectangle(i0, p1, p2, (0, 0, 255), 1)                        
                shcoords_one_window = (np.array([h, w, h, w]) * hcoords).astype(np.int16)                        
                feats = haar_features_one_window_numba_jit(ii, j0, k0, shcoords_one_window, n, np.arange(n, dtype=np.int32))
                if verbose:
                    print(f"[positive window {img_face_coords} accepted; features: {feats}]")
                    cv2.imshow("FDDB [press ESC to continue]", i0)
                    cv2.waitKey(0)
                X_list.append(feats)
                y_list.append(1)
            for _ in range(n_negs_per_img):            
                while True:
                    h = int((np.random.random() * w_relative_spread + w_relative_min) * i.shape[0])
                    w = h
                    j0 = int(np.random.random() * (i.shape[0] - w + 1))
                    k0 = int(np.random.random() * (i.shape[1] - w + 1))                 
                    patch = np.array([j0, k0, j0 + h - 1, k0 + w - 1])
                    ious = list(map(lambda ifc : iou(patch, ifc), img_faces_coords))
                    max_iou = max(ious) if len(ious) > 0 else 0.0
                    if max_iou < neg_max_iou:
                        shcoords_one_window = (np.array([h, w, h, w]) * hcoords).astype(np.int16)
                        feats = haar_features_one_window_numba_jit(ii, j0, k0, shcoords_one_window, n, np.arange(n, dtype=np.int32))
                        X_list.append(feats)
                        y_list.append(-1)                    
                        if verbose:
                            print(f"[negative window {patch} accepted; features: {feats}]")
                            p1 = (k0, j0)
                            p2 = (k0 + w - 1, j0 + h - 1)            
                            cv2.rectangle(i0, p1, p2, (0, 255, 0), 1)
                        break
                    else:                    
                        if verbose:
                            print(f"[negative window {patch} ignored due to max iou: {max_iou}]")
                            p1 = (k0, j0)
                            p2 = (k0 + w - 1, j0 + w - 1)
                            cv2.rectangle(i0, p1, p2, (255, 255, 0), 1)
            if verbose: 
                cv2.imshow("FDDB [press ESC to continue]", i0)
                cv2.waitKey(0)
        if li == len(lines):
            break
        line = lines[li].strip()
        li += 1
    print(f"IMAGES IN THIS FOLD: {n_img}.")
    print(f"ACCEPTED FACES IN THIS FOLD: {n_faces}.")
    f.close()
    X = np.stack(X_list)
    y = np.stack(y_list)
    return X, y

def fddb_read_single_fold_old(path_root, path_fold_relative, n_negs_per_img, hcoords, n, verbose=False, fold_title="", seed=0):
    np.random.seed(seed)    
    
    # settings for sampling negatives
    w_relative_min = 0.1
    w_relative_max = 0.35
    w_relative_spread = w_relative_max - w_relative_min
    neg_max_iou = 0.5
    
    X_list = []
    y_list = []
    
    f = open(path_root + path_fold_relative, "r")
    line = f.readline().strip()
    n_img = 0
    n_faces = 0
    counter = 0
    while line != "":
        file_name = path_root + line + ".jpg"
        log_line =  str(counter) + ": [" + file_name + "]"
        if fold_title != "":
            log_line += " [" + fold_title + "]" 
        print(log_line)
        counter += 1
        
        i0 = cv2.imread(file_name)
        i = cv2.cvtColor(i0, cv2.COLOR_BGR2GRAY)
        ii = integral_image_numba_jit(i)
        n_img += 1        
        n_img_faces = int(f.readline())        
        img_faces_coords = []
        for _ in range(n_img_faces):
            r_major, _, _, center_x, center_y, dummy_one = list(map(float, f.readline().strip().split()))
            h = int(1.5 * r_major)
            w = h                         
            j0 = int(center_y - h / 2) 
            k0 = int(center_x - w / 2)
            img_face_coords = np.array([j0, k0, j0 + h - 1, k0 + w - 1])
            if j0 < 0 or k0 < 0 or j0 + h - 1 >= i.shape[0] or k0 + w - 1 >= i.shape[1]:
                if verbose:
                    print(f"[window {img_face_coords} out of bounds -> ignored]")
                continue
            if (h / ii.shape[0] < 0.075): # min relative size of positive window (smaller may lead to division by zero when white regions in haar features have no area)
                if verbose:
                    print(f"[window {img_face_coords} too small -> ignored]")
                continue                            
            n_faces += 1
            img_faces_coords.append(img_face_coords)
            if verbose:
                p1 = (k0, j0)
                p2 = (k0 + w - 1, j0 + h - 1)    
                cv2.rectangle(i0, p1, p2, (0, 0, 255), 1)                        
            shcoords_one_window = (np.array([h, w, h, w]) * hcoords).astype(np.int16)                        
            feats = haar_features_one_window_numba_jit(ii, j0, k0, shcoords_one_window, n, np.arange(n, dtype=np.int32))
            if verbose:
                print(f"[positive window {img_face_coords} accepted; features: {feats}]")
                cv2.imshow("FDDB [press ESC to continue]", i0)
                cv2.waitKey(0)
            X_list.append(feats)
            y_list.append(1)
        for _ in range(n_negs_per_img):            
            while True:
                h = int((np.random.random() * w_relative_spread + w_relative_min) * i.shape[0])
                w = h
                j0 = int(np.random.random() * (i.shape[0] - w + 1))
                k0 = int(np.random.random() * (i.shape[1] - w + 1))                 
                patch = np.array([j0, k0, j0 + h - 1, k0 + w - 1])
                ious = list(map(lambda ifc : iou(patch, ifc), img_faces_coords))
                max_iou = max(ious) if len(ious) > 0 else 0.0
                if max_iou < neg_max_iou:
                    shcoords_one_window = (np.array([h, w, h, w]) * hcoords).astype(np.int16)
                    feats = haar_features_one_window_numba_jit(ii, j0, k0, shcoords_one_window, n, np.arange(n, dtype=np.int32))
                    X_list.append(feats)
                    y_list.append(-1)                    
                    if verbose:
                        print(f"[negative window {patch} accepted; features: {feats}]")
                        p1 = (k0, j0)
                        p2 = (k0 + w - 1, j0 + h - 1)            
                        cv2.rectangle(i0, p1, p2, (0, 255, 0), 1)
                    break
                else:                    
                    if verbose:
                        print(f"[negative window {patch} ignored due to max iou: {max_iou}]")
                        p1 = (k0, j0)
                        p2 = (k0 + w - 1, j0 + w - 1)
                        cv2.rectangle(i0, p1, p2, (255, 255, 0), 1)
        if verbose: 
            cv2.imshow("FDDB [press ESC to continue]", i0)
            cv2.waitKey(0)
        line = f.readline().strip()
    print(f"IMAGES IN THIS FOLD: {n_img}.")
    print(f"ACCEPTED FACES IN THIS FOLD: {n_faces}.")
    f.close()
    X = np.stack(X_list)
    y = np.stack(y_list)
    return X, y

def fddb_data(path_fddb_root, hfs_coords, n_negs_per_img, n, seed=0, data_augmentation=False):
    n_negs_per_img = n_negs_per_img
       
    fold_paths_train = [
        "FDDB-folds/FDDB-fold-01-ellipseList.txt",
        "FDDB-folds/FDDB-fold-02-ellipseList.txt",
        "FDDB-folds/FDDB-fold-03-ellipseList.txt",
        "FDDB-folds/FDDB-fold-04-ellipseList.txt",
        "FDDB-folds/FDDB-fold-05-ellipseList.txt",
        "FDDB-folds/FDDB-fold-06-ellipseList.txt",
        "FDDB-folds/FDDB-fold-07-ellipseList.txt",
        "FDDB-folds/FDDB-fold-08-ellipseList.txt",
        "FDDB-folds/FDDB-fold-09-ellipseList.txt"
        ] 
    X_train = None 
    y_train = None
    for index, fold_path in enumerate(fold_paths_train):
        print(f"PROCESSING TRAIN FOLD {index + 1}/{len(fold_paths_train)}...")
        t1 = time.time()
        X, y = fddb_read_single_fold(path_fddb_root, fold_path, n_negs_per_img, hfs_coords, n, verbose=False, fold_title=fold_path, seed=seed, data_augmentation=data_augmentation)
        t2 = time.time()
        print(f"PROCESSING TRAIN FOLD {index + 1}/{len(fold_paths_train)} DONE. [time: {t2 - t1} s]")
        print("---")
        if X_train is None:
            X_train = X
            y_train = y
        else:
            X_train = np.r_[X_train, X]
            y_train = np.r_[y_train, y]    
    fold_paths_test = [
        "FDDB-folds/FDDB-fold-10-ellipseList.txt",
        ]     
    X_test = None
    y_test = None
    for index, fold_path in enumerate(fold_paths_test):
        print(f"PROCESSING TEST FOLD {index + 1}/{len(fold_paths_test)}...")
        t1 = time.time()
        X, y = fddb_read_single_fold(path_fddb_root, fold_path, n_negs_per_img, hfs_coords, n, verbose=False, fold_title=fold_path, seed=seed, data_augmentation=data_augmentation)
        t2 = time.time()
        print(f"PROCESSING TEST FOLD {index + 1}/{len(fold_paths_test)} DONE. [time: {t2 - t1} s]")
        print("---")
        if X_test is None:
            X_test = X
            y_test = y
        else:
            X_test = np.r_[X_test, X]
            y_test = np.r_[y_test, y]   
    print(f"TRAIN DATA SHAPE: {X_train.shape}.")
    print(f"TEST DATA SHAPE: {X_test.shape}.") 
    return X_train, y_train, X_test, y_test

def pickle_all(fname, some_list):
    print(f"PICKLE... [to file: {fname}]")
    t1 = time.time()
    try:
        f = open(fname, "wb+")
        pickle.dump(some_list, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
    except IOError:
        sys.exit("[error occurred when trying to open or pickle the file]")
    t2 = time.time()
    print("PICKLE DONE. [time: " + str(t2 - t1) + " s]")

def unpickle_all(fname):
    print(f"UNPICKLE... [from file: {fname}]")
    t1 = time.time()
    try:    
        f = open(fname, "rb")
        some_list = pickle.load(f)
        f.close()
    except IOError:
        sys.exit("[error occurred when trying to open or read the file]")
    t2 = time.time()
    print("UNPICKLE DONE. [time: " + str(t2 - t1) + " s]")
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


def draw_feature_at(i, j0, k0, shcoords_one_feature):
    i_copy = i.copy()
    j, k, h, w = shcoords_one_feature[0]
    cv2.rectangle(i_copy, (k0 + k, j0 + j), (k0 + k + w - 1, j0 + j + h - 1), (0, 0, 0), cv2.FILLED)
    j, k, h, w = shcoords_one_feature[1]
    cv2.rectangle(i_copy, (k0 + k, j0 + j), (k0 + k + w - 1, j0 + j + h - 1), (255, 255, 255), cv2.FILLED)
    return i_copy

def demo_haar_features(hinds, hcoords, n):
    print(f"DEMO OF HAAR FEATURES... [hcoords.shape: {hcoords.shape}]")
    i = cv2.imread(FOLDER_EXTRAS + "photo_for_features_demo.jpg")
    i_resized = resize_image(i)
    i_gray = cv2.cvtColor(i_resized, cv2.COLOR_BGR2GRAY)
    ii = integral_image_numba_jit(i_gray)
    j0, k0 = 116, 100
    w = h = 221
    cv2.rectangle(i_resized, (k0, j0), (k0 + w - 1, j0 + h - 1), (0, 0, 255), 1)
    title = "DEMO OF FEATURES [press any key to continue or ESC to quit]"    
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
    features = haar_features_one_window_numba_jit(ii, j0, k0, hcoords_window_subset, n, np.arange(n))
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

def detect_simple(i, clf, hcoords, n, features_indexes, threshold=0.0, windows=None, shcoords_multiple_scales=None, verbose=False):
    if verbose:
        print("[detect_simple...]")
    t1 = time.time()
    t1_preprocess = time.time()
    i_resized = resize_image(i)
    i_gray = cv2.cvtColor(i_resized, cv2.COLOR_BGR2GRAY)
    i_h, i_w = i_gray.shape
    t2_preprocess = time.time()
    if verbose:
        print(f"[detect_simple: preprocessing done; time: {t2_preprocess - t1_preprocess} s; i_gray.shape: {i_gray.shape}]")
    t1_ii = time.time()
    ii = integral_image_numba_jit(i_gray)
    t2_ii = time.time()
    if verbose:
        print(f"[detect_simple: integral_image_numba_jit done; time: {t2_ii - t1_ii} s]")
    if windows is None:
        t1_prep = time.time()
        windows, shcoords_multiple_scales = prepare_detection_windows_and_scaled_haar_coords(i_h, i_w, hcoords, features_indexes)
        t2_prep = time.time()    
        if verbose:
            print(f"[detect_simple: prepare_detection_windows_and_scaled_haar_coords done; time: {t2_prep - t1_prep} s; windows to check: {windows.shape[0]}]")
    t1_haar_multiple = time.time()    
    X = haar_features_multiple_windows_numba_jit(ii, windows, shcoords_multiple_scales, n, features_indexes)
    t2_haar_multiple = time.time()
    if verbose:
        print(f"[detect_simple: haar_features_multiple_windows done; time: {t2_haar_multiple - t1_haar_multiple} s]")
    t1_df = time.time()    
    responses = clf.decision_function(X)
    t2_df = time.time()
    if verbose:
        print(f"[detect_simple: clf.decision_function done; time: {t2_df - t1_df} s]")
    t1_detections = time.time()                
    detected = responses > threshold
    detections = windows[detected, 1:] # skipping scale index
    responses = responses[detected] 
    t2_detections = time.time()
    if verbose:
        print(f"[detect_simple: finding detections (thresholding and indexing) done; time: {t2_detections - t1_detections} s]")        
    t2 = time.time()
    if verbose:
        print(f"[detect_simple done; time: {t2 - t1} s]")                    
    return detections, responses

def detect_parallel(i, clf, hcoords, n, features_indexes, threshold=0.0, windows=None, shcoords_multiple_scales=None, n_jobs=4, verbose=False):
    if verbose:
        print("[detect_parallel...]")
    t1 = time.time()
    t1_preprocess = time.time()
    i_resized = resize_image(i)
    i_gray = cv2.cvtColor(i_resized, cv2.COLOR_BGR2GRAY)
    i_h, i_w = i_gray.shape
    t2_preprocess = time.time()
    if verbose:
        print(f"[detect_parallel: preprocessing done; time: {t2_preprocess - t1_preprocess} s; i_gray.shape: {i_gray.shape}]")
    t1_ii = time.time()
    ii = integral_image_numba_jit(i_gray)
    t2_ii = time.time()
    if verbose:
        print(f"[detect_parallel: integral_image_numba_jit done; time: {t2_ii - t1_ii} s]")
    if windows is None:
        t1_prep = time.time()
        windows, shcoords_multiple_scales = prepare_detection_windows_and_scaled_haar_coords(i_h, i_w, hcoords, features_indexes)
        t2_prep = time.time()    
        if verbose:
            print(f"[detect_parallel: prepare_detection_windows_and_scaled_haar_coords done; time: {t2_prep - t1_prep} s; windows to check: {windows.shape[0]}]")
    
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
            X = haar_features_multiple_windows_numba_jit_tf(ii, job_windows, shcoords_multiple_scales, n, features_indexes)
            job_responses = clf.decision_function(X)         
            return job_responses 
        workers_results = parallel((delayed(worker)(job_index) for job_index in range(n_calls)))
        responses =  reduce(lambda a, b: np.r_[a, b], [jr for jr in workers_results])
        detected = responses > threshold
        detections = windows[detected, 1:] # skipping scale index
        responses = responses[detected] 
    t2_parallel = time.time()
    if verbose:
        print(f"[detect_parallel: all parallel jobs done (haar_features_multiple_windows_numba_jit_tf, clf.decision_function, finding detections (thresholding and indexing); time {t2_parallel - t1_parallel} s]")    
    t2 = time.time()
    if verbose:
        print(f"[detect_parallel done; time: {t2 - t1} s]")                    
    return detections, responses

def detect_cuda(i, clf, hcoords, features_indexes, threshold=0.0, windows=None, shcoords_multiple_scales=None, 
                dev_windows=None, dev_shcoords_multiple_scales=None, dev_X_selected=None, dev_mins_selected=None, dev_maxes_selected=None, dev_logits=None, dev_responses=None, 
                verbose=False):
    if verbose:
        print("[detect_cuda...]")
    t1 = time.time()
    t1_preprocess = time.time()
    i_resized = resize_image(i)
    i_gray = cv2.cvtColor(i_resized, cv2.COLOR_BGR2GRAY)
    i_h, i_w = i_gray.shape
    t2_preprocess = time.time()
    if verbose:
        print(f"[detect_cuda: preprocessing done; time: {t2_preprocess - t1_preprocess} s; i_gray.shape: {i_gray.shape}]")
    t1_ii = time.time()
    ii = integral_image_numba_jit(i_gray)
    t2_ii = time.time()
    if verbose:
        print(f"[detect_cuda: integral_image_numba_jit done; time: {t2_ii - t1_ii} s]")
    if windows is None:
        t1_prep = time.time()
        windows, shcoords_multiple_scales = prepare_detection_windows_and_scaled_haar_coords(i_h, i_w, hcoords, features_indexes)
        dev_windows = cuda.to_device(windows)
        dev_shcoords_multiple_scales = cuda.to_device(shcoords_multiple_scales)        
        t2_prep = time.time()    
        if verbose:
            print(f"[detect_cuda: prepare_detection_windows_and_scaled_haar_coords done; time: {t2_prep - t1_prep} s; windows to check: {windows.shape[0]}]")
    t1_dev_arrays = time.time()
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
    t2_dev_arrays = time.time()
    if verbose:
        print(f"[detect_cuda: checking presence of device arrays (and creating them if need be) done; time: {t2_dev_arrays - t1_dev_arrays} s]")    
    t1_haar_multiple = time.time()
    tpb = cuda.get_current_device().MAX_THREADS_PER_BLOCK // 2
    bpg = windows.shape[0]
    dev_ii = cuda.to_device(ii)
    haar_features_multiple_windows_numba_cuda[bpg, tpb](dev_ii, dev_windows, dev_shcoords_multiple_scales, dev_X_selected)
    cuda.synchronize()
    t2_haar_multiple = time.time()
    if verbose:
        print(f"[detect_cuda: haar_features_multiple_windows_numba_cuda done; time: {t2_haar_multiple - t1_haar_multiple} s]")
    t1_df = time.time()    
    FastRealBoostBins.decision_function_numba_cuda_job[bpg, tpb](dev_X_selected, dev_mins_selected, dev_maxes_selected, dev_logits, dev_responses)
    responses = dev_responses.copy_to_host()
    cuda.synchronize()
    t2_df = time.time()
    if verbose:
        print(f"[detect_cuda: FastRealBoostBins.decision_function_numba_cuda_job done; time: {t2_df - t1_df} s]")
    t1_detections = time.time()                
    detected = responses > threshold
    detections = windows[detected, 1:] # skipping scale index
    responses = responses[detected] 
    t2_detections = time.time()
    if verbose:
        print(f"[detect_cuda: finding detections (thresholding and indexing) done; time: {t2_detections - t1_detections} s]")        
    t2 = time.time()
    if verbose:
        print(f"[detect_cuda done; time: {t2 - t1} s]")                    
    return detections, responses

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
            if iou2(d[i], d[j]) >= threshold:
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
            if iou2(d[i], d[j]) >= threshold:
                indexes[j] = False
                d_avg.append(d[j])
                r_avg.append(r[j])
        d_avg = (np.round(np.mean(np.array(d_avg), axis=0))).astype(np.int16)
        r_avg = np.mean(r_avg)
        d_final.append(d_avg)
        r_final.append(r_avg)
    return d_final, r_final

def demo_detect_in_video(clf, hcoords, threshold, computations="simple", postprocess="avg", n_jobs=4, verbose_loop=True, verbose_detect=False):
    print("DEMO OF DETECT IN VIDEO...")
    features_indexes = clf.features_indexes_
    video = cv2.VideoCapture(CV2_VIDEO_CAPTURE_CAMERA_INDEX + (cv2.CAP_DSHOW if CV2_VIDEO_CAPTURE_IS_IT_MSWINDOWS else 0))
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    video.set(cv2.CAP_PROP_FPS, 30)    
    _, frame = video.read()
    frame_h, frame_w, _ = frame.shape    
    resized_width = int(np.round(frame.shape[1] / frame.shape[0] * HEIGHT))
    windows, shcoords_multiple_scales = prepare_detection_windows_and_scaled_haar_coords(HEIGHT, resized_width, hcoords, features_indexes)
    print(f"[frame shape: {frame.shape}]")
    print(f"[windows to check per frame: {windows.shape[0]}]")
    print(f"[about to start a camera...]")
    h_scale = frame_h / HEIGHT
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
            detections, responses = detect_simple(frame, clf, hcoords, n, features_indexes, threshold, windows, shcoords_multiple_scales, verbose=verbose_detect)
        elif computations == "parallel":
            detections, responses = detect_parallel(frame, clf, hcoords, n, features_indexes, threshold, windows, shcoords_multiple_scales, n_jobs=n_jobs, verbose=verbose_detect)        
        elif computations == "cuda":
            detections, responses = detect_cuda(frame, clf, hcoords, features_indexes, threshold, windows, shcoords_multiple_scales, 
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
        if n_frames > 0:
            fps_disp_ma = ma_decay * fps_disp_ma + (1.0 - ma_decay) * 1.0 / tpf_prev
            fps_disp = fps_disp_ma / (1.0 - ma_decay**(n_frames + 1))
        fps_comps_ma = ma_decay * fps_comps_ma + (1.0 - ma_decay) * 1.0 / (t2_comps - t1_comps)
        fps_comps = fps_comps_ma / (1.0 - ma_decay**(n_frames + 1))
        time_comps += t2_comps - t1_comps
        cv2.putText(frame, f"FRAME: {n_frames}", (0, 0 + 12), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)
        cv2.putText(frame, f"WINDOWS TO CHECK PER FRAME: {windows.shape[0]}", (0, frame_h - 1 - 32), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)        
        cv2.putText(frame, f"FPS (COMPUTATIONS): {fps_comps:.2f}", (0, frame_h - 1 - 16), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)
        cv2.putText(frame, f"FPS (DISPLAY): {fps_disp:.2f}", (0, frame_h - 1), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)        
        cv2.imshow(f"DEMO: REAL-BOOST + BINS ON HAAR-LIKE FEATURES [press ESC to quit]", frame)        
        if cv2.waitKey(1) & 0xFF == 27: # esc key
            break       
        t2_other = time.time()        
        n_frames += 1
        if verbose_loop:
            print(f"[read time: {t2_read - t1_read} s]")
            print(f"[flip time: {t2_flip - t1_flip} s]")
            print(f"[computations time: {t2_comps - t1_comps} s]")
            print(f"[postprocess time: {t2_post - t1_post} s]")
            print(f"[other time: {t2_other - t1_other} s]")
            print(f"[windows to check per frame: {windows.shape[0]}]")
            print(f"[fps (computations): {fps_comps}]")
            print(f"[fps (display): {fps_disp:.2f}]")
            print(f"[detections in this frame: {len(detections)}]")           
        t2 = time.time()
        tpf_prev = t2 - t1        
    t2_loop = time.time()
    cv2.destroyAllWindows()
    video.release()    
    avg_fps_comps = n_frames / time_comps
    avg_fps_disp = n_frames / (t2_loop - t1_loop)
    print(f"DEMO OF DETECT IN VIDEO DONE. [avg fps (computations): {avg_fps_comps:.2f}, avg fps (display): {avg_fps_disp:.2f}]")
        
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":        
    print("DEMONSTRATION OF \"FAST REAL-BOOST WITH BINS\" ALGORITHM IMPLEMENTED VIA NUMBA.JIT AND NUMBA.CUDA.")

    n = HAAR_TEMPLATES.shape[0] * S**2 * (2 * P - 1)**2    
    hinds = haar_indexes(S, P)
    hcoords = haar_coords(S, P, hinds)
    
    data_suffix = f"{KIND}_n_{n}_S_{S}_P_{P}_NPI_{NPI}_AUG_{AUG}_SEED_{SEED}" 
    DATA_NAME = f"data_{data_suffix}.bin"
    CLF_NAME = f"clf_frbb_{data_suffix}_T_{T}_B_{B}.bin"    
    print(f"DATA_NAME: {DATA_NAME}")
    print(f"CLF_NAME: {CLF_NAME}")
    print(f"GPU_PROPS: {gpu_props()}")
    
    if DEMO_HAAR_FEATURES:
        demo_haar_features(hinds, hcoords, n)      
    
    if REGENERATE_DATA_FROM_FDDB:
        X_train, y_train, X_test, y_test = fddb_data(FOLDER_RAW_DATA_FDDB, hcoords, NPI, n, SEED, data_augmentation=(AUG == 1))
        pickle_all(FOLDER_DATA + DATA_NAME, [X_train, y_train, X_test, y_test])    

    if FIT_OR_REFIT_MODEL or MEASURE_ACCS_OF_MODEL:
        [X_train, y_train, X_test, y_test] = unpickle_all(FOLDER_DATA + DATA_NAME)
        print(f"[X_train.shape: {X_train.shape} (positives: {np.sum(y_train == 1)}), X_test.shape: {X_test.shape} (positives: {np.sum(y_test == 1)})]")
    
    if FIT_OR_REFIT_MODEL: 
        clf = FastRealBoostBins(T=T, B=B, fit_mode="numba_cuda", decision_function_mode="numba_cuda", verbose=True, debug_verbose=False)
        clf.fit(X_train, y_train)
        pickle_all(FOLDER_CLFS + CLF_NAME, [clf])
        
    clf = None
    if MEASURE_ACCS_OF_MODEL or DEMO_DETECT_IN_VIDEO:
        [clf] = unpickle_all(FOLDER_CLFS + CLF_NAME)
        
    if MEASURE_ACCS_OF_MODEL:
        measure_accs_of_model(clf, X_train, y_train, X_test, y_test)
    
    if DEMO_DETECT_IN_VIDEO:
        demo_detect_in_video(clf, hcoords, threshold=DETECTION_THRESHOLD, computations="cuda", postprocess=DETECTION_POSTPROCESS, n_jobs=8, verbose_loop=True, verbose_detect=True)

    print("ALL DONE.")
    
if __name__ == "__rocs__":        
    print("ROCS...")
    
    clfs_settings = [{"S": 5, "P": 5, "NPI": 50, "AUG": 0, "SEED": 0, "T": 512, "B": 8},
                     {"S": 5, "P": 5, "NPI": 10, "AUG": 1, "SEED": 0, "T": 512, "B": 8},
                     ]
    
    for s in clfs_settings:
        S = s["S"]
        P = s["P"]
        NPI = s["NPI"]
        AUG = s["AUG"]
        SEED = s["SEED"]
        T = s["T"]
        B = s["B"] 
        n = HAAR_TEMPLATES.shape[0] * S**2 * (2 * P - 1)**2    
        hinds = haar_indexes(S, P)
        hcoords = haar_coords(S, P, hinds)
        
        data_suffix = f"{KIND}_n_{n}_S_{S}_P_{P}_NPI_{NPI}_AUG_{AUG}_SEED_{SEED}" 
        DATA_NAME = f"data_{data_suffix}.bin"
        DATA_NAME = "data_face_n_18225_S_5_P_5_NPI_50_AUG_0_SEED_0.bin"
        #DATA_NAME = "data_face_n_18225_S_5_P_5_NPI_10_AUG_1_SEED_0.bin"        
        CLF_NAME = f"clf_frbb_{data_suffix}_T_{T}_B_{B}.bin"    
        print("---")
        print(f"DATA_NAME: {DATA_NAME}")
        print(f"CLF_NAME: {CLF_NAME}")            
        [X_train, y_train, X_test, y_test] = unpickle_all(FOLDER_DATA + DATA_NAME)        
        [clf] = unpickle_all(FOLDER_CLFS + CLF_NAME)                
        responses_test = clf.decision_function(X_test)
        roc = roc_curve(y_test, responses_test)
        fars, sens, thrs = roc
        roc_arr = np.array([fars, sens, thrs]).T        
        plt.plot(fars, sens, label=CLF_NAME)
        print(f"X_train.shape: {X_train.shape}, X_test.shape: {X_test.shape}")        
        with np.printoptions(threshold=np.inf):
            print(roc_arr)        
    plt.xscale("log")
    plt.xlabel("FAR")
    plt.ylabel("SENSITIVITY")
    plt.legend(loc="lower right", fontsize=8)
    plt.show()