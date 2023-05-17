import numpy as np
import cv2
from numba import cuda, jit
from numba import void, int16, int32, float32, uint8
from numba.core.errors import NumbaPerformanceWarning
import warnings

__version__ = "1.0.0"
__author__ = "Przemysław Klęsk"
__email__ = "pklesk@zut.edu.pl"

warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
np.set_printoptions(linewidth=512)

# haar-related constants
HSQRT2 = 0.5 * np.sqrt(2.0)
HAAR_TEMPLATES = np.array([ # each row describes a template - a white rectangle (j, k, h, w) placed within a black unit square
    [0.0, 0.0, 0.5, 1.0],   # "top-down edge"
    [0.0, 0.0, 1.0, 0.5],   # "left-right edge"
    [0.25, 0.0, 0.5, 1.0], # "horizontal middle edge"
    [0.0, 0.25, 1.0, 0.5], # "vertical middle edge"
    [0.5 * (1.0 - HSQRT2), 0.5 * (1.0 - HSQRT2), HSQRT2, HSQRT2], # "center"
    [0.0, 0.0, HSQRT2, HSQRT2], # "top-left corner"
    [0.0, 1.0 - HSQRT2, HSQRT2, HSQRT2], # "top-right corner"
    [1.0 - HSQRT2, 0.0, HSQRT2, HSQRT2], # "bottom-left corner"
    [1.0 - HSQRT2, 1.0 - HSQRT2, HSQRT2, HSQRT2] # "bottom-left corner"    
    ], dtype=np.float32)
FEATURE_MIN = 0.25
FEATURE_MAX = 0.75
HEIGHT = 480 # defualt height to which image or video frame scaled before further computations 

def resize_image(i):
    h, w, _ = i.shape
    return cv2.resize(i, (round(w * HEIGHT / h), HEIGHT))

def haar_indexes(s, p):
    hinds = []
    for t in range(HAAR_TEMPLATES.shape[0]):
        for s_j in range(s):
            for s_k in range(s):
                for p_j in range(-p + 1, p, 1):
                    for p_k in range(-p + 1, p, 1):
                        hinds.append(np.array([t, s_j, s_k, p_j, p_k]))
    return np.array(hinds)

def haar_coords(s, p, hinds):
    hcoords = []
    for t, s_j, s_k, p_j, p_k in hinds:
        f_h = FEATURE_MIN + s_j * (FEATURE_MAX - FEATURE_MIN) / (s - 1) if s > 1 else FEATURE_MIN
        f_w = FEATURE_MIN + s_k * (FEATURE_MAX - FEATURE_MIN) / (s - 1) if s > 1 else FEATURE_MIN
        shift_h = (1.0 - f_h) / (2 * p - 2) if p > 1 else 0.0
        shift_w = (1.0 - f_w) / (2 * p - 2) if p > 1 else 0.0
        pos_j = 0.5 + p_j * shift_h - 0.5 * f_h
        pos_k = 0.5 + p_k * shift_w - 0.5 * f_w
        single_hcoords = [np.array([pos_j, pos_k, f_h, f_w], dtype=np.float32)] # background of whole feature (useful later for feature computation)
        white = HAAR_TEMPLATES[t]         
        single_hcoords.append(white * np.array([f_h, f_w, f_h, f_w], dtype=np.float32) + np.array([pos_j, pos_k, 0.0, 0.0], dtype=np.float32))
        hcoords.append(np.array(single_hcoords))
    return np.array(hcoords)

def integral_image_python(i):
    h, w = i.shape
    ii = np.zeros(i.shape, dtype=np.int32)
    ii_row = np.zeros(w, dtype=np.int32)
    for j in range(h):
        for k in range(w):
            ii_row[k] = i[j, k]
            if k > 0:
                ii_row[k] += ii_row[k - 1]
            ii[j, k] = ii_row[k]
            if j > 0:
                ii[j, k] += ii[j - 1, k]
    return ii

def integral_image_numpy(i):
    return np.cumsum(np.cumsum(i, axis=0), axis=1)

@jit(int32[:,:](uint8[:,:]), nopython=True, cache=True)
def integral_image_numba_jit(i):
    h, w = i.shape
    ii = np.zeros(i.shape, dtype=int32)
    ii_row = np.zeros(w, dtype=int32)
    for j in range(h):
        for k in range(w):
            ii_row[k] = i[j, k]
            if k > 0:
                ii_row[k] += ii_row[k - 1]
            ii[j, k] = ii_row[k]
            if j > 0:
                ii[j, k] += ii[j - 1, k]
    return ii

@jit(int32(int32[:,:], int16, int16, int16, int16), nopython=True, cache=True)
def ii_delta_numba_jit(ii, j1, k1, j2, k2): 
    delta = ii[j2, k2]
    if j1 > 0: 
        delta -= ii[j1 - 1, k2]
    if k1 > 0:        
        delta -= ii[j2, k1 - 1] 
    if j1 > 0 and k1 > 0: 
        delta += ii[j1 - 1, k1 - 1]
    return delta

# TODO check if this device function can be supported with specific types in decorator
@cuda.jit(device=True)
def ii_delta_numba_cuda(ii, j1, k1, j2, k2): 
    delta = ii[j2, k2]
    if j1 > 0: 
        delta -= ii[j1 - 1, k2]
    if k1 > 0:        
        delta -= ii[j2, k1 - 1] 
    if j1 > 0 and k1 > 0: 
        delta += ii[j1 - 1, k1 - 1]
    return delta

@jit(int16(int32[:,:], int16, int16, int16[:, :]), nopython=True, cache=True)
def haar_feature_numba_jit(ii, j0, k0, shcoords_one_feature): # (j0, k0) - window top-left corner, shcoords - scaled coordinates (in pixels) of a single haar feature 
    j, k, h, w = shcoords_one_feature[0] # whole feature background
    j1 = j0 + j
    k1 = k0 + k
    total_area = h * w
    total_intensity = ii_delta_numba_jit(ii, j1, k1, j1 + h - 1, k1 + w - 1)
    j, k, h, w = shcoords_one_feature[1] # white rectangle 
    j1 = j0 + j
    k1 = k0 + k
    white_area = h * w
    white_intensity = ii_delta_numba_jit(ii, j1, k1, j1 + h - 1, k1 + w - 1)
    black_area = total_area - white_area
    black_intensity = total_intensity - white_intensity
    return np.int16(white_intensity / white_area - black_intensity / black_area)

@jit(int16[:](int32[:, :], int16, int16, int16[:, :, :], int32, int32[:]), nopython=True, cache=True)        
def haar_features_one_window_numba_jit(ii, j0, k0, shcoords_one_window, n, features_indexes):
    features = np.zeros(n, dtype=np.int16)
    for j in range(features_indexes.size):
        features[features_indexes[j]] = haar_feature_numba_jit(ii, j0, k0, shcoords_one_window[j])
    return features 

@jit(int16[:, :](int32[:, :], int16[:, :], int16[:, :, :, :], int32, int32[:]), nopython=True, cache=True)        
def haar_features_multiple_windows_numba_jit(ii, windows, shcoords_multiple_scales, n, features_indexes):
    m = windows.shape[0]
    X = np.zeros((m, n), dtype=np.int16)
    for i in range(m):        
        s, j0, k0, _, _ = windows[i]
        X[i] = haar_features_one_window_numba_jit(ii, j0, k0, shcoords_multiple_scales[s], n, features_indexes)
    return X
        
@cuda.jit(void(int32[:, :], int16[:, :], int16[:, :, :, :], int16[:, :]))
def haar_features_multiple_windows_numba_cuda(ii, windows, shcoords_multiple_scales, X_selected):           
    i = cuda.blockIdx.x
    tpb = cuda.blockDim.x
    tx = cuda.threadIdx.x
    T = X_selected.shape[1]
    fpt = int((T + tpb - 1) / tpb) # features to calculate per thread
    t = tx # feature index
    s, j0, k0, _, _ = windows[i]
    for _ in range(fpt):
        if t < T:
            j, k, h, w = shcoords_multiple_scales[s, t, 0]
            j1 = j0 + j
            k1 = k0 + k
            total_area = float32(h * w)
            total_intensity = float32(ii_delta_numba_cuda(ii, j1, k1, j1 + h - 1, k1 + w - 1))
            j, k, h, w = shcoords_multiple_scales[s, t, 1] # white rectangle 
            j1 = j0 + j
            k1 = k0 + k
            white_area = float32(h * w)
            white_intensity = float32(ii_delta_numba_cuda(ii, j1, k1, j1 + h - 1, k1 + w - 1))
            black_area = total_area - white_area
            black_intensity = total_intensity - white_intensity
            X_selected[i, t] = int16(white_intensity / white_area - black_intensity / black_area)            
        t += tpb

@jit(nopython=True)
def ii_delta_numba_jit_tf(ii, j1, k1, j2, k2): 
    delta = ii[j2, k2]
    if j1 > 0: 
        delta -= ii[j1 - 1, k2]
    if k1 > 0:        
        delta -= ii[j2, k1 - 1] 
    if j1 > 0 and k1 > 0: 
        delta += ii[j1 - 1, k1 - 1]
    return delta

@jit(nopython=True)
def haar_feature_numba_jit_tf(ii, j0, k0, shcoords_one_feature): # same functionality as in haar_feature_numba_jit but types-free signature (tf)
    j, k, h, w = shcoords_one_feature[0] 
    j1 = j0 + j
    k1 = k0 + k
    total_area = h * w
    total_intensity = ii_delta_numba_jit_tf(ii, j1, k1, j1 + h - 1, k1 + w - 1)
    j, k, h, w = shcoords_one_feature[1]  
    j1 = j0 + j
    k1 = k0 + k
    white_area = h * w
    white_intensity = ii_delta_numba_jit_tf(ii, j1, k1, j1 + h - 1, k1 + w - 1)
    black_area = total_area - white_area
    black_intensity = total_intensity - white_intensity
    return np.int16(white_intensity / white_area - black_intensity / black_area)
        
@jit(nopython=True)
def haar_features_one_window_numba_jit_tf(ii, j0, k0, shcoords_one_window, n, features_indexes): # same functionality as in haar_features_one_window_numba_jit but types-free signature (tf)  
    features = np.zeros(n, dtype=np.int16)
    for j in range(features_indexes.size):
        features[features_indexes[j]] = haar_feature_numba_jit_tf(ii, j0, k0, shcoords_one_window[j])
    return features 

@jit(nopython=True)        
def haar_features_multiple_windows_numba_jit_tf(ii, windows, shcoords_multiple_scales, n, features_indexes): # same functionality as in haar_features_multiple_windows_numba_jit but types-free signature (tf)
    m = windows.shape[0]
    X = np.zeros((m, n), dtype=np.int16)
    for i in range(m):        
        s, j0, k0, _, _ = windows[i]
        X[i] = haar_features_one_window_numba_jit_tf(ii, j0, k0, shcoords_multiple_scales[s], n, features_indexes)
    return X

def iou(coords1, coords2): # coords of two rectangles given in the form: (j1, k1, j2, k2), where: (j1, k1) - top-left corner, (j2, k2) - bottom-right corner  
    j11, k11, j12, k12 = coords1
    j21, k21, j22, k22 = coords2    
    dj = np.min([j12, j22]) - np.max([j21, j11]) + 1 
    if dj <= 0: 
        return 0.0
    dk = np.min([k12, k22]) - np.max([k21, k11]) + 1
    if dk <= 0: 
        return 0.0
    i = dj * dk
    u = (j12 - j11 + 1) * (k12 - k11 + 1) + (j22 - j21 + 1) * (k22 - k21 + 1) - i
    return i / u

def iou2(jkhw1, jkhw2): # coords of two rectangles given in the form: (j, k, h, w), where (j, k) - top-left corner, (h, w) - height and width
    j11 = jkhw1[0] 
    k11 = jkhw1[1] 
    j12 = j11 + jkhw1[2] - 1
    k12 = k11 + jkhw1[3] - 1
    j21 = jkhw2[0] 
    k21 = jkhw2[1]
    j22 = j21 + jkhw2[2] - 1
    k22 = k21 + jkhw2[3] - 1    
    dj = np.min([j12, j22]) - np.max([j21, j11]) + 1 
    if dj <= 0: 
        return 0.0
    dk = np.min([k12, k22]) - np.max([k21, k11]) + 1
    if dk <= 0: 
        return 0.0
    i = dj * dk
    u = (j12 - j11 + 1) * (k12 - k11 + 1) + (j22 - j21 + 1) * (k22 - k21 + 1) - i
    return i / u