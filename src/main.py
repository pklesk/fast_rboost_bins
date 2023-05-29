import sys
import os
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

__version__ = "1.0.0"
__author__ = "Przemysław Klęsk"
__email__ = "pklesk@zut.edu.pl"

warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
np.set_printoptions(linewidth=512)


# main settings
KIND = "hand"
S = 5 # parameter "scales" to generete Haar-like features
P = 5 # parameter "positions" to generete Haar-like features
AUG = False # data augmentation (0 -> none or 1 -> present)
KOP = 10 # "kilos of positives " - no. of thousands of positives (positive windows) to generate (in case of synthetic data only; 0 value for real data, meaning 'not applicable')
NPI = 80 # "negatives per image" - no. of negatives (negative windows) to sample per image (image real or generated synthetically) 
T = 2048 # size of ensemble in FastRealBoostBins (equivalently, no. of boosting rounds when fitting)
B = 8 # no. of bins
SEED = 0 # randomization seed
DEMO_HAAR_FEATURES_ALL = False
DEMO_HAAR_FEATURES_SELECTED = False
REGENERATE_DATA = True
FIT_OR_REFIT_MODEL = True
MEASURE_ACCS_OF_MODEL = True
DEMO_DETECT_IN_VIDEO = True

# cv2 camera settings
CV2_VIDEO_CAPTURE_CAMERA_INDEX = 0
CV2_VIDEO_CAPTURE_IS_IT_MSWINDOWS = True

# detection procedure settings
DETECTION_SCALES = 10
DETECTION_WINDOW_HEIGHT_MIN = 96
DETECTION_WINDOW_WIDTH_MIN = 96
DETECTION_WINDOW_GROWTH = 1.2
DETECTION_WINDOW_JUMP = 0.05
DETECTION_THRESHOLD = 7.0
DETECTION_POSTPROCESS = "avg" # possible values: None, "nms", "avg"

# folders
FOLDER_DATA = "../data/"
FOLDER_CLFS = "../models/"
FOLDER_EXTRAS = "../extras/"
FOLDER_RAW_DATA_FDDB = "../raw_data_fddb/"
FOLDER_RAW_DATA_HAND = "../raw_data_hand/"

# synthetic data generation constants
SYNTHETIC_ROTATION_RANGE = np.pi / 16
SYNTHETIC_TRAIN_RATIO = 0.75


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

def fddb_read_single_fold(path_root, path_fold_relative, hcoords, n, data_augmentation=False, n_negs_per_img=10, seed=0, fold_title="", verbose=False):
    np.random.seed(seed)        
    relative_min = 0.1 # when sampling negatives
    relative_max = 0.35 # when sampling negatives
    relative_spread = relative_max - relative_min
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
    augmentations_extras = []
    if data_augmentation: 
        augmentations += ["sharpen", "blur", "random_brightness", "random_channel_shift"]
        augmentations_extras += ["random_window_distort", "random_horizontal_flip"]
    while line != "":
        file_name = path_root + line + ".jpg"
        fold_title_str = f"({fold_title})" if fold_title != "" else ""
        log_line = f"[{counter}: {file_name} {fold_title_str}]"
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
            flipped = False
            if aug_index > 0:
                li -= n_img_faces
            if aug is not None:
                if aug == "sharpen":
                    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                    i0 = cv2.filter2D(i0, -1, kernel)
                elif aug == "blur":
                    i0 = cv2.blur(i0, ksize=(5, 5))
                elif aug == "random_brightness":
                    value = np.random.uniform(0.5, 2.0)
                    hsv = cv2.cvtColor(i0, cv2.COLOR_BGR2HSV)
                    hsv = np.array(hsv, dtype=np.float64)
                    hsv[:, :, [1, 2]] = value * hsv[:, :, [1, 2]]
                    hsv = np.clip(hsv, 0, 255)
                    hsv = np.array(hsv, dtype=np.uint8)
                    i0 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                elif aug == "random_channel_shift":
                    i0 = i0.astype(np.int16)
                    for chnl in range(3):
                        value = int(np.random.uniform(-64, 64))
                        i0[:, :, chnl] += value
                    i0 = (np.clip(i0, 0, 255)).astype(np.uint8)
                if "random_horizontal_flip" in augmentations_extras:
                    if np.random.rand() < 0.5:
                        i0 = np.copy(np.fliplr(i0))
                        flipped = True
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
                if flipped:
                    k0 = i.shape[1] - 1 - k0 - w + 1
                if aug is not None and "random_window_distort" in augmentations_extras:
                    growth = np.random.uniform(0.95, 1.05)                    
                    hn = int(np.round(h * growth))
                    wn = int(np.round(w * growth))
                    j0n = j0 + (hn - h) // 2
                    k0n = k0 + (wn - w) // 2
                    j0n += int(np.round(np.random.uniform(-0.05, 0.05) * hn))
                    k0n += int(np.round(np.random.uniform(-0.05, 0.05) * wn))
                    if j0n >= 0 and j0n + hn - 1 < b.shape[0] and k0n >= 0 and k0n + wn - 1 < b.shape[1]:
                        j0 = j0n
                        k0 = k0n
                        h = hn
                        w = wn
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
                    h = int((np.random.random() * relative_spread + relative_min) * i.shape[0])
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
                cv2.destroyAllWindows()
        if li == len(lines):
            break
        line = lines[li].strip()
        li += 1
    aug_str = "(after data augmentation)" if data_augmentation else "(without data augmentation)" 
    print(f"[total of images {aug_str} in this fold: {n_img}, accepted faces: {n_faces}]")
    f.close()
    X = np.stack(X_list)
    y = np.stack(y_list)
    return X, y

def fddb_data(path_fddb_root, hcoords, n, data_augmentation=False, n_negs_per_img=10, seed=0, verbose=False):
    print("FDDB DATA...")
    t1 = time.time()
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
        print(f"[processing train fold {index + 1}/{len(fold_paths_train)}...]")
        t1 = time.time()        
        X, y = fddb_read_single_fold(path_fddb_root, fold_path, hcoords, n, data_augmentation=data_augmentation, n_negs_per_img=n_negs_per_img, seed=seed, fold_title=fold_path, verbose=verbose)
        t2 = time.time()
        print(f"[processing train fold {index + 1}/{len(fold_paths_train)} done; time: {t2 - t1} s]")
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
        print(f"[processing test fold {index + 1}/{len(fold_paths_test)}...]")
        t1 = time.time()               
        X, y = fddb_read_single_fold(path_fddb_root, fold_path, hcoords, n, data_augmentation=data_augmentation, n_negs_per_img=n_negs_per_img, seed=seed, fold_title=fold_path, verbose=verbose)
        t2 = time.time()
        print(f"[processing test fold {index + 1}/{len(fold_paths_test)} done; time: {t2 - t1} s]")
        if X_test is None:
            X_test = X
            y_test = y
        else:
            X_test = np.r_[X_test, X]
            y_test = np.r_[y_test, y]   
    t2 = time.time()
    print(f"FDDB DATA DONE. [time: {t2 - t1} s, X_train.shape: {X_train.shape}, X_test.shape: {X_test.shape}]")
    return X_train, y_train, X_test, y_test

def rotate_bound(image, angle):
    h, w = image.shape[:2]
    cj, ck = w // 2, h // 2
    M = cv2.getRotationMatrix2D((cj, ck), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int(h * sin + w * cos)
    nH = int(h * cos + w * sin)
    M[0, 2] += 0.5 * nW - cj
    M[1, 2] += 0.5 * nH - ck
    return cv2.warpAffine(image, M, (nW, nH))

def synthetic_data(folder_backgrounds, folder_targets, hcoords, n, data_augmentation=False, n_poss=1, n_negs_per_img=10, seed=0, verbose=False,
                   rotation_range=SYNTHETIC_ROTATION_RANGE, train_ratio=SYNTHETIC_TRAIN_RATIO):
    print("SYNTHETIC DATA...")
    t1 = time.time()
    relative_min = 0.25 # for both positives and negatives
    relative_max = 0.75 # for both positives and negatives
    neg_max_iou = 0.5
    margin_pixels = 8
    np.random.seed(seed)    
    b_names = os.listdir(folder_backgrounds)
    n_b = len(b_names)
    t_names = os.listdir(folder_targets)
    n_t = len(t_names)
    X_list = []
    y_list = []
    augmentations = [None]
    augmentations_extras = []
    if data_augmentation: 
        augmentations += ["sharpen", "blur", "random_brightness", "random_channel_shift"]
        #augmentations_extras += ["random_window_distort", "random_horizontal_flip"]
        augmentations_extras += ["random_horizontal_flip"]
    imshow_title = "SYNTHETIC IMAGE [press ESC to continue]"
    m = n_poss    
    aug_str = " (with data augmentation)" if data_augmentation else ""
    for index in range(m):
        b_fname = folder_backgrounds + b_names[np.random.randint(n_b)] 
        b_original = cv2.imread(b_fname)
        bh, bw = b_original.shape[:2]
        side = min(bw, bh)
        t_fname = folder_targets + t_names[np.random.randint(n_t)]
        t = cv2.imread(t_fname)                
        print(f"[{index + 1}/{m}: background: {b_fname}, target: {t_fname}{aug_str}]")
        for aug in augmentations:
            b = np.copy(b_original)
            if verbose:
                print(f"[augmentation: {aug}]")
            if aug is not None and "random_horizontal_flip" in augmentations_extras:
                if np.random.rand() < 0.5:
                    t = np.fliplr(t)
                if np.random.rand() < 0.5:
                    b = np.fliplr(b)
            th, tw = t.shape[:2]
            ratio = np.random.uniform(relative_min, relative_max)
            rside = ratio * side
            if th < tw:
                ts = cv2.resize(t, (round(rside), round(rside * th / tw)))
            else:
                ts = cv2.resize(t, (round(rside * tw / th), round(rside)))
            h, w = ts.shape[:2]
            c = np.array([h, w]) / 2.0
            angle_deg = 0.5 * rotation_range * 180.0 / np.pi
            if aug is None:
                angle_deg = 0.0
            ts = rotate_bound(ts, np.random.uniform(-angle_deg, angle_deg))
            hr, wr = ts.shape[:2]
            cr = np.array([hr, wr]) / 2.0
            h = (h + hr) // 2
            w = (w + hr) // 2            
            c = np.array([h, w]) / 2.0
            ts[ts == 0] = 255        
            ts = cv2.medianBlur(ts, 3)            
            correction_shift = (np.round(cr - c)).astype(np.int16)
            j0 = margin_pixels + correction_shift[0] + np.random.randint(bh - hr - 2 * margin_pixels - correction_shift[0])
            k0 = margin_pixels + correction_shift[1] + np.random.randint(bw - wr - 2 * margin_pixels - correction_shift[1])
            roi = b[j0 : j0 + hr, k0 : k0 + wr]
            ts_gray = cv2.cvtColor(ts, cv2.COLOR_BGR2GRAY)    
            _, mask = cv2.threshold(ts_gray, 245, 255, cv2.THRESH_BINARY_INV)    
            mask_inv = cv2.bitwise_not(mask)
            b_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
            t_fg = cv2.bitwise_and(ts, ts, mask=mask)
            overlay = cv2.add(b_bg, t_fg)
            overlay = cv2.medianBlur(overlay, 3)
            b[j0 : j0 + hr, k0 : k0 + wr] = overlay
            j0 += correction_shift[0]
            k0 += correction_shift[1]
            if aug is not None:
                if aug == "sharpen":
                    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                    b = cv2.filter2D(b, -1, kernel)
                elif aug == "blur":
                    b = cv2.blur(b, ksize=(5, 5))
                elif aug == "random_brightness":
                    value = np.random.uniform(0.5, 2.0)
                    hsv = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)
                    hsv = np.array(hsv, dtype=np.float64)
                    hsv[:, :, [1, 2]] = value * hsv[:, :, [1, 2]]
                    hsv = np.clip(hsv, 0, 255)
                    hsv = np.array(hsv, dtype=np.uint8)
                    b = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                elif aug == "random_channel_shift":
                    b = b.astype(np.int16)
                    for chnl in range(3):
                        value = int(np.random.uniform(-64, 64))
                        b[:, :, chnl] += value
                    b = (np.clip(b, 0, 255)).astype(np.uint8)
                if "random_window_distort" in augmentations_extras:
                    growth = np.random.uniform(0.95, 1.05)
                    hn = int(np.round(h * growth))
                    wn = int(np.round(w * growth))
                    j0n = j0 + (hn - h) // 2
                    k0n = k0 + (wn - w) // 2
                    j0n += int(np.round(np.random.uniform(-0.05, 0.05) * hn))
                    k0n += int(np.round(np.random.uniform(-0.05, 0.05) * wn))
                    if j0n >= 0 and j0n + hn - 1 < b.shape[0] and k0n >= 0 and k0n + wn - 1 < b.shape[1]:
                        j0 = j0n
                        k0 = k0n
                        h = hn
                        w = wn                         
            target_coords = np.array([j0, k0, j0 + h - 1, k0 + w - 1])                                
            i = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)       
            ii = integral_image_numba_jit(i)                
            if verbose:
                b_copy = np.copy(b)
                p1 = (k0, j0)
                p2 = (k0 + w - 1, j0 + h - 1)
                cv2.rectangle(b_copy, p1, p2, (0, 0, 255), 1)
            shcoords_one_window = (np.array([h, w, h, w]) * hcoords).astype(np.int16)                        
            feats = haar_features_one_window_numba_jit(ii, j0, k0, shcoords_one_window, n, np.arange(n, dtype=np.int32))
            if verbose:
                print(f"[positive window {target_coords} accepted; features: {feats}]")
                cv2.imshow(imshow_title, b_copy)
                cv2.waitKey(0)
            X_list.append(feats)
            y_list.append(1)
            for _ in range(n_negs_per_img):            
                while True:
                    ratio = np.random.uniform(relative_min, relative_max)
                    rside = ratio * side
                    if tw < th:
                        h = int(rside)
                        w = int(rside * tw / th)
                    else:
                        w = int(rside)
                        h = int(rside * th / tw)                
                    j0 = int(np.random.random() * (i.shape[0] - h + 1))
                    k0 = int(np.random.random() * (i.shape[1] - w + 1))                 
                    patch = np.array([j0, k0, j0 + h - 1, k0 + w - 1])
                    max_iou = iou(patch, target_coords)       
                    if max_iou < neg_max_iou:
                        shcoords_one_window = (np.array([h, w, h, w]) * hcoords).astype(np.int16)
                        feats = haar_features_one_window_numba_jit(ii, j0, k0, shcoords_one_window, n, np.arange(n, dtype=np.int32))
                        X_list.append(feats)
                        y_list.append(-1)                    
                        if verbose:
                            print(f"[negative window {patch} accepted; features: {feats}]")
                            p1 = (k0, j0)
                            p2 = (k0 + w - 1, j0 + h - 1)            
                            cv2.rectangle(b_copy, p1, p2, (0, 255, 0), 1)
                        break                        
                    else:                    
                        if verbose:
                            print(f"[negative window {patch} ignored due to max iou: {max_iou}]")
                            p1 = (k0, j0)
                            p2 = (k0 + w - 1, j0 + w - 1)
                            cv2.rectangle(b_copy, p1, p2, (255, 255, 0), 1)                    
            if verbose: 
                cv2.imshow(imshow_title, b_copy)
                cv2.waitKey(0)
                cv2.destroyAllWindows()            
    m_train = int(np.round(train_ratio * len(X_list)))
    X_train = np.stack(X_list[:m_train])
    y_train = np.stack(y_list[:m_train])
    X_test = np.stack(X_list[m_train:])
    y_test = np.stack(y_list[m_train:])    
    t2 = time.time()
    print(f"SYNTHETIC DATA DONE. [time: {t2 - t1} s, X_train.shape: {X_train.shape}, X_test.shape: {X_test.shape}]")    
    return X_train, y_train, X_test, y_test            
                            
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
    i_resized = resize_image(i)
    i_gray = cv2.cvtColor(i_resized, cv2.COLOR_BGR2GRAY)
    ii = integral_image_numba_jit(i_gray)
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
    times = {}
    t1_preprocess = time.time()
    i_resized = resize_image(i)
    i_gray = cv2.cvtColor(i_resized, cv2.COLOR_BGR2GRAY)
    i_h, i_w = i_gray.shape
    t2_preprocess = time.time()
    dt_preprocess = t2_preprocess - t1_preprocess
    times["preprocess"] = dt_preprocess
    if verbose:
        print(f"[detect_simple: preprocessing done; time: {dt_preprocess} s, i_gray.shape: {i_gray.shape}]")
    t1_ii = time.time()
    ii = integral_image_numba_jit(i_gray)
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
    X = haar_features_multiple_windows_numba_jit(ii, windows, shcoords_multiple_scales, n, features_indexes)
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

def detect_parallel(i, clf, hcoords, n, features_indexes, threshold=0.0, windows=None, shcoords_multiple_scales=None, n_jobs=4, verbose=False):
    if verbose:
        print("[detect_parallel...]")
    t1 = time.time()
    times = {}
    t1_preprocess = time.time()
    i_resized = resize_image(i)
    i_gray = cv2.cvtColor(i_resized, cv2.COLOR_BGR2GRAY)
    i_h, i_w = i_gray.shape
    t2_preprocess = time.time()
    dt_preprocess = t2_preprocess - t1_preprocess
    times["preprocess"] = dt_preprocess
    if verbose:
        print(f"[detect_parallel: preprocessing done; time: {dt_preprocess} s; i_gray.shape: {i_gray.shape}]")
    t1_ii = time.time()
    ii = integral_image_numba_jit(i_gray)
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
            X = haar_features_multiple_windows_numba_jit_tf(ii, job_windows, shcoords_multiple_scales, n, features_indexes)
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
    i_resized = resize_image(i)
    i_gray = cv2.cvtColor(i_resized, cv2.COLOR_BGR2GRAY)
    i_h, i_w = i_gray.shape
    t2_preprocess = time.time()
    dt_preprocess = t2_preprocess - t1_preprocess
    times["preprocess"] = dt_preprocess
    if verbose:
        print(f"[detect_cuda: preprocessing done; time: {dt_preprocess} s; i_gray.shape: {i_gray.shape}]")
    t1_ii = time.time()
    ii = integral_image_numba_jit(i_gray)
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
    haar_features_multiple_windows_numba_cuda[bpg, tpb](dev_ii, dev_windows, dev_shcoords_multiple_scales, dev_X_selected)
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
    gpu_name = gpu_props()["name"]
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
    print(f"[windows per frame: {windows.shape[0]}]")
    print(f"[terms per window: {clf.T_}]")
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
            comps_details += f"[HAAR: {times['haar'] * 1000:05.1f} ms"            
            comps_details += f", FRBB: {times['frbb'] * 1000:05.1f} ms]"
            time_comps_haar += times["haar"]
            time_comps_frbb += times["frbb"]
        cv2.putText(frame, f"FPS (COMPUTATIONS): {fps_comps:.1f} {comps_details}", (0, frame_h - 1 - 1 * text_shift), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)
        cv2.putText(frame, f"FPS (DISPLAY): {fps_disp:.1f}", (0, frame_h - 1), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)                    
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
            print(f"[fps (computations): {fps_comps:.1f}]")
            print(f"[fps (display): {fps_disp:.1f}]")
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
    print(f"DEMO OF DETECT IN VIDEO DONE. [avg fps (computations): {avg_fps_comps:.1f}, avg time haar: {avg_time_comps_haar * 1000:.1f} ms, avg time frbb: {avg_time_comps_frbb * 1000:.1f} ms; avg fps (display): {avg_fps_disp:.1f}]")

def best_prec_threshold(roc, y_test):
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
    print(f"[best_prec_threshold -> best_thr: {best_thr}, best_prec: {best_prec}; py: {py}, fprs_sub[best_index]: {fprs_sub[best_index]}, tprs_sub[best_index]: {tprs_sub[best_index]}]")
    return best_thr, best_prec


        
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":        
    print("DEMONSTRATION OF \"FAST REAL-BOOST WITH BINS\" ALGORITHM IMPLEMENTED VIA NUMBA.JIT AND NUMBA.CUDA.")

    n = HAAR_TEMPLATES.shape[0] * S**2 * (2 * P - 1)**2    
    hinds = haar_indexes(S, P)
    hcoords = haar_coords(S, P, hinds)
    
    data_suffix = f"{KIND}_n_{n}_S_{S}_P_{P}_AUG_{np.int8(AUG)}_KOP_{KOP}_NPI_{NPI}_SEED_{SEED}" 
    DATA_NAME = f"data_{data_suffix}.bin"
    CLF_NAME = f"clf_frbb_{data_suffix}_T_{T}_B_{B}.bin"    
    print(f"DATA_NAME: {DATA_NAME}")
    print(f"CLF_NAME: {CLF_NAME}")
    print(f"GPU_PROPS: {gpu_props()}")
    
    if DEMO_HAAR_FEATURES_ALL:
        demo_haar_features(hinds, hcoords, n)      
    
    if REGENERATE_DATA:
        if KIND == "face":
            X_train, y_train, X_test, y_test = fddb_data(FOLDER_RAW_DATA_FDDB, hcoords, n, AUG, NPI, seed=SEED, verbose=False)
        elif KIND == "hand":
            X_train, y_train, X_test, y_test = synthetic_data(FOLDER_RAW_DATA_HAND + "backgrounds/", FOLDER_RAW_DATA_HAND + "targets/", hcoords, n, AUG, KOP * 1000, NPI, seed=SEED, verbose=False)
        pickle_objects(FOLDER_DATA + DATA_NAME, [X_train, y_train, X_test, y_test])
    
    if FIT_OR_REFIT_MODEL or MEASURE_ACCS_OF_MODEL:
        [X_train, y_train, X_test, y_test] = unpickle_objects(FOLDER_DATA + DATA_NAME)
        print(f"[X_train.shape: {X_train.shape} (positives: {np.sum(y_train == 1)}), X_test.shape: {X_test.shape} (positives: {np.sum(y_test == 1)})]")
    
    if FIT_OR_REFIT_MODEL: 
        clf = FastRealBoostBins(T=T, B=B, fit_mode="numba_cuda", decision_function_mode="numba_cuda", verbose=True, debug_verbose=False)
        clf.fit(X_train, y_train)
        pickle_objects(FOLDER_CLFS + CLF_NAME, [clf])
        
    clf = None
    if MEASURE_ACCS_OF_MODEL or DEMO_DETECT_IN_VIDEO:
        [clf] = unpickle_objects(FOLDER_CLFS + CLF_NAME)
    
    if DEMO_HAAR_FEATURES_SELECTED and clf is not None:
        selected = features_indexes_
        demo_haar_features(hinds[selected], hcoords[selected], selected.size)
        
    if MEASURE_ACCS_OF_MODEL:
        measure_accs_of_model(clf, X_train, y_train, X_test, y_test)        
    
    if DEMO_DETECT_IN_VIDEO:
        demo_detect_in_video(clf, hcoords, threshold=DETECTION_THRESHOLD, computations="cuda", postprocess=DETECTION_POSTPROCESS, n_jobs=8, verbose_loop=True, verbose_detect=True)

    print("ALL DONE.")
    
    
    
    
if __name__ == "__rocs__":        
    print("ROCS...")
    
    clfs_settings = [#{"KIND": "hand", "S": 5, "P": 5, "AUG": 1, "KOP": 5, "NPI": 20, "SEED": 0, "T": 1024, "B": 8},
                     #{"KIND": "hand", "S": 5, "P": 5, "AUG": 1, "KOP": 5, "NPI": 20, "SEED": 0, "T": 2048, "B": 16},
                     {"KIND": "hand", "S": 5, "P": 5, "AUG": 1, "KOP": 2, "NPI": 20, "SEED": 0, "T": 1024, "B": 8},
                     {"KIND": "hand", "S": 5, "P": 5, "AUG": 1, "KOP": 2, "NPI": 50, "SEED": 0, "T": 1024, "B": 8},
                     {"KIND": "hand", "S": 5, "P": 5, "AUG": 0, "KOP": 10, "NPI": 50, "SEED": 0, "T": 1024, "B": 8}
                     ]
    
    for s in clfs_settings:
        KIND = s["KIND"]
        S = s["S"]
        P = s["P"]
        AUG = s["AUG"]
        KOP = s["KOP"]
        NPI = s["NPI"]        
        SEED = s["SEED"]
        T = s["T"]
        B = s["B"] 
        n = HAAR_TEMPLATES.shape[0] * S**2 * (2 * P - 1)**2    
        hinds = haar_indexes(S, P)
        hcoords = haar_coords(S, P, hinds)            
        data_suffix = f"{KIND}_n_{n}_S_{S}_P_{P}_AUG_{AUG}_KOP_{KOP}_NPI_{NPI}_SEED_{SEED}"                                      
        #DATA_NAME = "data_face_n_18225_S_5_P_5_AUG_0_KOP_0_NPI_100_SEED_0.bin"
        DATA_NAME = "data_hand_n_18225_S_5_P_5_AUG_1_KOP_5_NPI_10_SEED_0.bin"
        #DATA_NAME = "data_hand_n_18225_S_5_P_5_AUG_1_KOP_2_NPI_20_SEED_0.bin"
        [X_train, y_train, X_test, y_test] = unpickle_objects(FOLDER_DATA + DATA_NAME)        
        CLF_NAME = f"clf_frbb_{data_suffix}_T_{T}_B_{B}.bin"            
        print("---")
        print(f"DATA_NAME: {DATA_NAME}")
        print(f"CLF_NAME: {CLF_NAME}")                            
        [clf] = unpickle_objects(FOLDER_CLFS + CLF_NAME)                
        responses_test = clf.decision_function(X_test)
        roc = roc_curve(y_test, responses_test)
        best_thr, best_prec = best_prec_threshold(roc, y_test) 
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