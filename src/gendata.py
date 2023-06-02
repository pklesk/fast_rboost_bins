import sys
import os
import numpy as np
import cv2
from haar import *
import time
import json

__version__ = "1.0.0"
__author__ = "Przemysław Klęsk"
__email__ = "pklesk@zut.edu.pl"

# folders
FOLDER_DATA_RAW = "../data_raw/"
FOLDER_DATA_RAW_FDDB = FOLDER_DATA_RAW + "fddb/"
FOLDER_DATA_RAW_HAGRID = FOLDER_DATA_RAW + "hagrid/"
FOLDER_DATA_RAW_SYNTHETIC_HAND = FOLDER_DATA_RAW + "synthetic_hand/"

# synthetic data generation constants
SYNTHETIC_ROTATION_RANGE = np.pi / 16
SYNTHETIC_TRAIN_RATIO = 0.75
SYNTHETIC_FORCE_RANDOM_ROTATIONS = True
SYNTHETIC_FORCE_RANDOM_HORIZONTAL_FLIPS = True


def fddb_data(hcoords, n, data_augmentation=False, n_negs_per_img=10, seed=0, verbose=False):
    path_fddb_root = FOLDER_DATA_RAW_FDDB
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
        augmentations_extras += ["random_horizontal_flip"]
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
                    cv2.imshow("FDDB IMAGE [press ESC to continue]", i0)
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
                cv2.imshow("FDDB IMAGE [press ESC to continue]", i0)
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

def hagrid_data(hcoords, n, n_negs_per_img, train_ratio=0.75, seed=0, verbose=False):
    print(f"HAGRID DATA...")
    t1 = time.time()    
    relative_min = 0.1 # when sampling negatives
    relative_max = 0.35 # when sampling negatives
    relative_spread = relative_max - relative_min
    neg_max_iou = 0.5
    np.random.seed(seed)     
    f = open(FOLDER_DATA_RAW_HAGRID + "palm.json", "r") # only "palm gesture" hands take part in this material
    data_dict = json.load(f)
    f.close()  
    keys = list(data_dict.keys())
    images_path = FOLDER_DATA_RAW_HAGRID + "train_val_palm/"  
    X_list = []
    y_list = []
    for index, key in enumerate(keys):
        file_name = images_path + key + ".jpg"
        log_line = f"[{index + 1}/{len(keys)}: {file_name}]"
        print(log_line)
        i0_original = cv2.imread(file_name)        
        i0 = resize_image(i0_original)
        H, W, _ = i0.shape
        i = cv2.cvtColor(i0, cv2.COLOR_BGR2GRAY)               
        ii = integral_image_numba_jit(i)
        bboxes = data_dict[key]["bboxes"]
        labels = data_dict[key]["labels"]
        targets_coords = []
        for b, l in zip(bboxes, labels):
            if l != "palm":
                continue
            j0 = b[1] * H
            k0 = b[0] * W                                                 
            h = b[3] * H
            w = b[2] * W
            side = max(h, w)
            j0 = j0 + 0.5 * (h - side)
            k0 = k0 + 0.5 * (w - side)
            side = int(np.round(side))
            h = side
            w = side
            j0 = int(np.round(j0))
            k0 = int(np.round(k0))
            target_coords = np.array([j0, k0, j0 + h - 1, k0 + w - 1])
            if j0 < 0 or k0 < 0 or j0 + h - 1 >= i.shape[0] or k0 + w - 1 >= i.shape[1]:
                if verbose:
                    print(f"[window {target_coords} out of bounds -> ignored]")
                continue
            if (h / H < 0.075): # min relative size of positive window (smaller may lead to division by zero when white regions in haar features have no area)
                if verbose:
                    print(f"[window {target_coords} too small -> ignored]")
                continue                            
            targets_coords.append(target_coords)                        
            shcoords_one_window = (np.array([h, w, h, w]) * hcoords).astype(np.int16)                        
            feats = haar_features_one_window_numba_jit(ii, j0, k0, shcoords_one_window, n, np.arange(n, dtype=np.int32))                
            if verbose:
                print(f"[positive window {target_coords} accepted; features: {feats}]")
                p1 = (k0, j0)
                p2 = (k0 + w - 1, j0 + h - 1)    
                cv2.rectangle(i0, p1, p2, (0, 0, 255), 1)                    
                cv2.imshow("HAGRID IMAGE [press ESC to continue]", i0)
                cv2.waitKey(0)
            X_list.append(feats)
            y_list.append(1)
        for _ in range(n_negs_per_img):            
            while True:
                h = int((np.random.random() * relative_spread + relative_min) * i.shape[0])
                w = h
                j0 = int(np.random.random() * (H - h + 1))
                k0 = int(np.random.random() * (W - w + 1))                 
                patch = np.array([j0, k0, j0 + h - 1, k0 + w - 1])
                ious = list(map(lambda ifc : iou(patch, ifc), targets_coords))
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
            cv2.imshow("HAGRID IMAGE [press ESC to continue]", i0)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    m_train = int(np.round(train_ratio * len(X_list)))
    X_train = np.stack(X_list[:m_train])
    y_train = np.stack(y_list[:m_train])
    X_test = np.stack(X_list[m_train:])
    y_test = np.stack(y_list[m_train:])    
    t2 = time.time()
    print(f"HAGRID DATA DONE. [time: {t2 - t1} s; X_train.shape: {X_train.shape}, positives: {np.sum(y_train == 1)}; X_test.shape: {X_test.shape}, positives: {np.sum(y_test == 1)}]")    
    return X_train, y_train, X_test, y_test             
    
def synthetic_data(folder_backgrounds, folder_targets, hcoords, n, data_augmentation=False, n_poss=1, n_negs_per_img=10, seed=0, verbose=False,
                   rotation_range=SYNTHETIC_ROTATION_RANGE, train_ratio=SYNTHETIC_TRAIN_RATIO, 
                   force_random_rotations=SYNTHETIC_FORCE_RANDOM_ROTATIONS, force_random_horizontal_flips=SYNTHETIC_FORCE_RANDOM_HORIZONTAL_FLIPS):
    print("SYNTHETIC DATA...")
    t1 = time.time()
    relative_min = 0.2 # for both positives and negatives
    relative_max = 0.6 # for both positives and negatives
    neg_max_iou = 0.5
    margin_pixels = 4
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
            if (aug is not None and "random_horizontal_flip" in augmentations_extras) or force_random_horizontal_flips:
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
            if aug is None and not force_random_rotations:
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
