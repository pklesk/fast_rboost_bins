"""
This module contains the core machine learning functionalities of the project, embodied by the class `FastRealBoostBins` (compliant with `scikit-learn`).
The module includes:

- `FastRealBoostBins`: class representing an ensemble classifier for fast predictions implemented using `numba.jit` and `numba.cuda`,

- `_lock`, `_unlock`: utility functions (placed outside the class, related to mutex mechanisms in case of numba.cuda-based fit).

In ``FastRealBoostBins`` class, attributes estimated by the ``fit`` function are named with trailing underscores (e.g. ``features_selected_``, ``logits_``, etc.) as indicated in the scikit-learn guidelines. 
Private functions are named with single leading underscores and some of them are additionally described by ``@jit`` or ``@cuda.jit`` decorators coming from ``numba`` module (intended to be compiled by `Numba`).

Documentation note: this documentation was built with `Sphinx` tool, which does not correctly process docstrings for CUDA kernel functions, 
i.e. functions decorated with ``@cuda.jit`` that produce ``numba.cuda.compiler.Dispatcher`` objects as outcomes. 
For actual docstrings associated with those functions see the source code. 

Installation
------------

.. code-block:: console
    
    pip install frbb
    
Note: for further usage, NVIDIA CUDA drivers must be present in the operating system.

Example Usage
-------------
With ``frbb`` module installed, one can write e.g.:

.. code-block:: python

    from frbb import FastRealBoostBins
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split    
    if __name__ == "__main__":    
        X, y = load_breast_cancer(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
        clf = FastRealBoostBins()
        clf.fit(X_train, y_train)
        print(f"CLF: {clf}")
        print(f"TRAIN ACC: {clf.score(X_train, y_train)}")
        print(f"TEST ACC: {clf.score(X_test, y_test)}")

Running the script above produces the following output:
        
.. code-block:: console

    CLF: FastRealBoostBins(T=256, B=8, outliers_ratio=0.05, logit_max: 2.0, fit_mode='numba_cuda', decision_function_mode='numba_cuda')
    TRAIN ACC: 1.0
    TEST ACC: 0.958041958041958

Dependencies
------------
- ``numpy``, ``math``: required for mathematical computations.

- ``numba``: required for just-in-time compilation of crucial computational functions and CUDA kernels (decorated by ``@jit`` and ``@cuda.jit`` imported from ``numba``). 

- ``sklearn``: required for inheritence and other sklearn API purposes.

Link to project repository
--------------------------
`https://github.com/pklesk/fast_rboost_bins <https://github.com/pklesk/fast_rboost_bins>`_
"""

import numpy as np
from numpy import inf
from numba import cuda, jit
from numba import void, int8, int16, int32, int64, float32, float64, uint8, uint16, uint32, uint64
from numba import types as nbtypes
from numba.core.errors import NumbaPerformanceWarning
import math
import time
import warnings
import json
import sys
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import column_or_1d, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
        
__version__ = "1.0.0"
__author__ = "Przemysław Klęsk"
__email__ = "pklesk@zut.edu.pl" 
        
warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
np.set_printoptions(linewidth=512)

# mutex-related cuda utility functions 
@cuda.jit(device=True)
def _lock(mutex):
    """Device-side function that locks the wanted critical section (mutex mechanism). Locking means setting the value of variable or array cell (passed by reference) from 0 to 1."""     
    while cuda.atomic.compare_and_swap(mutex, 0, 1) != 0:
        pass
    cuda.threadfence()    
    
@cuda.jit(device=True)
def _unlock(mutex):
    """Device-side function that unlocks the wanted critical section (mutex mechanism). Unlocking means setting the value of variable or array cell (passed by reference) from 1 to 0."""
    cuda.threadfence()
    cuda.atomic.exch(mutex, 0, 0)

# the class
class FastRealBoostBins(BaseEstimator, ClassifierMixin):
    """
    An ensemble classifier for fast predictions implemented using numba.jit and numba.cuda. 
    Bins with logit transform values play the role of "weak learners".
    
    Parameters:
        T (int): 
            number of boosting rounds (equivalently, number of weak estimators), defaults to ``256``.            
        B (int): 
            number of bins, defaults to ``8``.            
        outliers_ratio (float): 
            fraction of outliers to skip (on each end) when establishing features’ variability ranges, defaults to ``0.05``.
        logit_max (np.float32):
            maximum absolute value of logit transform, outcomes clipped to interval [``-logit_max``, ``logit_max``], defaults to ``np.float32(2.0)``.
        fit_mode (str):
            choice of fit method from {``"numpy"``, ``"numba_jit"``, ``"numba_cuda"``}, defaults to ``"numba_cuda"``.
        decision_function_mode (str):
            choice of decision function method from {``"numpy"``, ``"numba_jit"``, ``"numba_cuda"``} (called e.g. within ``predict``), defaults to ``"numba_cuda"``.
        verbose (bool):
            verbosity flag, if ``True`` then fit progress and auxiliary information are printed to console, defaults to ``False``.
        debug_verbose (bool):
            detailed verbosity (only for ``'numba_cuda'`` fit), defaults to ``False``. 
        
    Attributes:
        features_selected_ (ndarray[np.int32]):
            indexes of selected features, array of shape ``(T,)``.
        dtype_ (np.dtype): 
            type of input data array, one of {``np.int8``, ``np.uint8``, ..., ``np.int64``, ``np.uint64``} or {``np.float32``, ``np.float64``} - numeric types only allowed.       
        mins_selected_ (ndarray[dtype_]): 
            left ends of ranges for selected features, array of shape ``(T,)``.
        maxes_selected_ (ndarray[dtype_]):
            right ends of ranges for selected features, array of shape ``(T,)``.
        logits_ (ndarray[np.float32]): 
            binned logit values for selected features, array of shape ``(T, B)``.            
        decision_function_numba_cuda_job_name_ (str): 
            name, implied by ``dtype_``, of decision function to be called in case of ``"numba_cuda"`` mode (e.g. ``_decision_function_numba_cuda_job_int16``).
        decision_threshold_ (float): 
            threshold value used inside ``predict`` function, defaults to ``0.0``.
        classes_ (ndarray): 
            original class labels (scikit-learn requirement).
        n_features_in_ (int): 
            number of features registered in ``fit`` call and expected for subsequent ``predict`` calls (scikit-learn requirement).            
    """

    # constants
    T_DEFAULT = 256
    B_DEFAULT = 8
    OUTLIERS_RATIO_DEFAULT = 0.05
    LOGIT_MAX_DEFAULT = np.float32(2.0)
    FIT_MODE_DEFAULT = "numba_cuda"
    DECISION_FUNCTION_MODE_DEFAULT = "numba_cuda"
    VERBOSE_DEFAULT = False
    DEBUG_VERBOSE_DEFAULT = False        
    B_MAX = 32                    
    OUTLIERS_RATIO_MAX = 0.25
    LOGIT_MAX_MAX = np.float32(8.0)
    FIT_MODES = ["numpy", "numba_jit", "numba_cuda"]
    DECISION_FUNCTION_MODES = ["numpy", "numba_jit", "numba_cuda"]    
    CUDA_MAX_MEMORY_PER_CALL = 8 * 1024**2 # applicable only for cuda-based fit, can be adjusted for given gpu device     

    # error messages
    SKLEARN_ERR_MESSAGE_UNKNOWN_LABEL_TYPE = "Unknown label type"
    SKLEARN_ERR_MESSAGE_DISCREPANCY_IN_NO_OF_FEATURES = "Number of features in predict or decision_function is different from the number of features in fit"
    
    def __init__(self, T=T_DEFAULT, B=B_DEFAULT, outliers_ratio=OUTLIERS_RATIO_DEFAULT, logit_max=LOGIT_MAX_DEFAULT, 
                 fit_mode=FIT_MODE_DEFAULT, decision_function_mode=DECISION_FUNCTION_MODE_DEFAULT, 
                 verbose=VERBOSE_DEFAULT, debug_verbose=DEBUG_VERBOSE_DEFAULT):
        """
        Constructor of ``FastRealBoostBins`` instances.
         
        Args:
            T (int): 
                number of boosting rounds (equivalently, number of weak estimators), defaults to ``256``.            
            B (int): 
                number of bins, defaults to ``8``.            
            outliers_ratio (float): 
                fraction of outliers to skip (on each end) when establishing features’ variability ranges, defaults to ``0.05``.
            logit_max (np.float32):
                maximum absolute value of logit transform, outcomes clipped to interval [``-logit_max``, ``logit_max``], defaults to ``np.float32(2.0)``.
            fit_mode (str):
                choice of fit method from {``"numpy"``, ``"numba_jit"``, ``"numba_cuda"``}, defaults to ``"numba_cuda"``.
            decision_function_mode (str):
                choice of decision method from {``"numpy"``, ``"numba_jit"``, ``"numba_cuda"``} (called e.g. within ``predict``), defaults to ``"numba_cuda"``.
            verbose (bool):
                verbosity flag, if ``True`` then fit progress and auxiliary information are printed to console, defaults to ``False``.
            debug_verbose (bool):
                detailed verbosity (only for ``'numba_cuda'`` fit), defaults to ``False``.            
        """
        super().__init__()
        self.T = T
        self.B = B    
        self.outliers_ratio = outliers_ratio
        self.logit_max = logit_max
        self.fit_mode = fit_mode
        self.decision_function_mode = decision_function_mode
        self.verbose = verbose
        self.debug_verbose = debug_verbose        

    def _get_tags(self=None):
        """Returns a dictionary with particular properties of this estimator (compliant with scikit-learn guidelines)."""
        tags = super()._get_tags()
        tags["binary_only"] = True
        tags["non_deterministic"] = True # in case of cuda computations  
        return tags
    
    def __str__(self):
        """
        Returns a string representation of this classifier.
        
        Returns:
            str: string representation of this classifier.
        """
        return f"{self.__class__.__name__}(T={self.T}, B={self.B}, outliers_ratio={self.outliers_ratio}, logit_max: {self.logit_max}, fit_mode='{self.fit_mode}', decision_function_mode='{self.decision_function_mode}')"
            
    def __repr__(self):
        """
        Returns a detailed string representation of this classifier.
        
        Returns:
            str: detailed string representation of this classifier.
        """
        repr_str = f"{self.__class__.__name__}(T={self.T}, B={self.B}, outliers_ratio={self.outliers_ratio}, logit_max: {self.logit_max}, fit_mode='{self.fit_mode}', decision_function_mode='{self.decision_function_mode}',\n"
        repr_str += f"  verbose={self.verbose}, debug_verbose={self.debug_verbose}"
        if hasattr(self, "classes_"):
            repr_str += ",\n"                    
            repr_str += f"  classes_={self.classes_}, dtype_={self.dtype_}, decision_function_numba_cuda_job_name_={self.decision_function_numba_cuda_job_name_}, decision_threshold_={self.decision_threshold_},\n"
            repr_str += f"  features_selected_={self.features_selected_},\n"
            repr_str += f"  mins_selected_={self.mins_selected_},\n"
            repr_str += f"  maxes_selected_={self.maxes_selected_},\n"
            repr_str += f"  logits_={self.logits_}"
        repr_str += ")"
        return repr_str

    def _validate_param(self, name, value, ptype, leq, low, geq, high, default):
        """Validates a parameter - is it of correct type and within given range (either end of the range can be open or closed)."""     
        invalid = value <= low if leq else value < low
        if not invalid:
            invalid = value >= high if geq else value > high
        if not invalid:
            invalid = not isinstance(value, ptype)
        if invalid:
            correct_range_str = ("(" if leq else "[") + f"{low}, {high}" + (")" if geq else "]")
            print(f"[error -> invalid param {name}: {value} changed to default: {default}; correct range: {correct_range_str}, correct type: {ptype}]")
            raise ValueError(self.SKLEARN_ERR_MESSAGE_UNKNOWN_LABEL_TYPE)
    
    def _set_cuda_constants(self):
        """Investigates (via numba module) if CUDA-based computations are available and, if so, sets suitable constants."""
        self._cuda_available = cuda.is_available() 
        self._cuda_tpb_default = cuda.get_current_device().MAX_THREADS_PER_BLOCK // 2 if self._cuda_available else None
        self._cuda_tpb_bin_add_weights = 128 if self._cuda_available else None
        self._cuda_n_streams = cuda.get_current_device().ASYNC_ENGINE_COUNT if self._cuda_available else None
    
    def _set_modes(self, fit_mode="numba_cuda", decision_function_mode="numba_cuda"):
        """Sets modes for fit and decision_function functions."""
        if not self.fit_mode in self.FIT_MODES:
            invalid_mode = fit_mode
            fit_mode = self.FIT_MODE_DEFAULT
            print(f"[invalid fit mode: '{invalid_mode}' changed to '{fit_mode}'; possible modes: {self.FIT_MODES}]")        
        if not self.decision_function_mode in self.DECISION_FUNCTION_MODES:
            invalid_mode = decision_function_mode
            decision_function_mode = self.DECISION_FUNCTION_MODE_DEFAULT 
            print(f"[invalid decision function mode: '{invalid_mode}' changed to '{decision_function_mode}'; possible modes: {self.DECISION_FUNCTION_MODES}]")
        self.fit_mode = fit_mode
        self.decision_function_mode = decision_function_mode
        if self.fit_mode == "numba_cuda" and not self._cuda_available:
            self.fit_mode = "numba_jit"
            print(f"[changing fit mode to '{self.fit_mode}' due to cuda functionality not available on this machine]")
        if self.decision_function_mode == "numba_cuda" and not self._cuda_available:
            self.decision_function_mode = 'numba_jit'
            print(f"[changing decision function mode to '{self.decision_function_mode}' due to cuda functionality not available on this machine]")       
        self._fit_method = getattr(self, "_fit_" + self.fit_mode)
        self._decision_function_method = getattr(self, "_decision_function_" + self.decision_function_mode)                 
                                                           
    def _logit(self, W_p, W_n):
        """Computes a logit transform value based on sums of current boosting weights (for positive and negative examples) with proper clipping and handling of zeros in either numerator or denominator."""        
        if W_p == W_n:
            return np.float32(0.0)
        elif W_p == 0.0:
            return -self.logit_max
        elif W_n == 0.0:
            return self.logit_max
        return np.clip(0.5 * np.log(W_p / W_n), -self.logit_max, self.logit_max)
                    
    def fit(self, X, y):
        """
        Performs the fit operation according to a general scheme of RealBoost algorithm (data reweighting, real-valued responses)
        and using an approach where bins with logit transform values play the role of "weak learners". 
        Each weak learner is based on one selected feature - the minimizer of exponential criterion (taking into account weights of data examples from a current boosting round).
        Computations are carried out according to the formerly chosen ``fit_mode`` i.e. one of {``"numpy"``, ``"numba_jit"``, ``"numba_cuda"``}.
        Depending on the mode, the function calls one of the following functions: ``_fit_numpy``, ``_fit_numba_jit``, or ``_fit_numba_cuda``.
        
        Args:
            X (ndarray): 
                two-dimensional data array of numeric type with examples written as rows and features as columns.
            y (ndarray): 
                one-dimensional array containing class labels associated with data examples.
            
        Returns:
            self (FastRealBoostBins): 
                reference to self (in compliance with scikit-learn guidelines).
        """
        # sklearn checks
        y = column_or_1d(y)
        X, y = check_X_y(X, y)                    
        check_classification_targets(y)
        # actual functions
        self._fit_init(X, y)
        self._fit_method(X, y)
        return self
        
    def _fit_init(self, X, y):
        """Validates parameters and initializes some of the attributes and constants needed for actual fitting (taking into account information about the input data)."""
        # validation
        self._validate_param("T", self.T, int, False, 1, True, inf, self.T_DEFAULT)
        self._validate_param("B", self.B, int, False, 1, False, self.B_MAX, self.B_DEFAULT)    
        self._validate_param("outliers_ratio", self.outliers_ratio, float, False, 0.0, False, self.OUTLIERS_RATIO_MAX, self.OUTLIERS_RATIO_DEFAULT)
        self._validate_param("logit_max", self.logit_max, np.float32, True, 0.0, False, self.LOGIT_MAX_MAX, self.LOGIT_MAX_DEFAULT)
        self._validate_param("verbose", self.verbose, bool, False, False, False, True, self.VERBOSE_DEFAULT) 
        self._validate_param("debug_verbose", self.debug_verbose, bool, False, False, False, True, self.DEBUG_VERBOSE_DEFAULT)
        self._set_cuda_constants()
        self._set_modes(self.fit_mode, self.decision_function_mode)
        # initialization of attributes to be estimated (names with trailing underscores)
        self.features_selected_ = np.zeros(self.T, dtype=np.int32) # indexes of selected features
        self.logits_ = np.zeros((self.T, self.B), dtype=np.float32) # binned logits for selected features
        self.decision_threshold_ = 0.0 # default decision threshold
        self.classes_ = np.unique(y) # unique class labels, we assume exactly 2 classes with first class negative
        self.dtype_ = X.dtype # dtype of input array 
        self.decision_function_numba_cuda_job_name_ = f"_decision_function_numba_cuda_job_{str(self.dtype_)}"
        # memorizing incoming number of features (same value expected later at predict stage according to sklearn)
        self.n_features_in_ = X.shape[1]             
    
    def _bin_data(self, X, mins, maxes):
        """Returns a binned version of data array using given variability ranges and knowing the number of bins. In binning arithmetics, suitably handles outliers and broadens data type (temporarily)."""
        X_binned = None
        spreads = maxes - mins
        info = np.iinfo(self.dtype_) if np.issubdtype(self.dtype_, np.integer) else np.finfo(self.dtype_) 
        if np.issubdtype(self.dtype_, np.integer):
            if np.any(np.float64(self.B) * spreads > info.max):
                broader_dtype = np.int16
                if self.dtype_ == np.int16 or self.dtype_ == np.uint16:
                    broader_dtype = np.int32
                elif self.dtype_ == np.int32 or self.dtype_ == np.uint32:
                    broader_dtype = np.int64
                spreads[spreads == 0] = info.max                
                if self.dtype_ == np.int64 or self.dtype_ == np.uint64:
                    if self.verbose:
                        print(f"[warning: temporarily changing dtype = {self.dtype_} to {np.float64} while binning to prevent overflow]")
                    X_binned = np.clip(np.int8(np.float64(self.B) * (X - mins) / spreads), 0, self.B - 1)        
                else:                    
                    if self.verbose:
                        print(f"[warning: temporarily extending dtype = {self.dtype_} to {broader_dtype} while binning to prevent overflow]")                 
                    X = X.astype(broader_dtype)
                    X_binned = np.clip(np.int8(self.B * (X - mins) // spreads), 0, self.B - 1)
                    X = X.astype(self.dtype_)
            else:                 
                spreads[spreads == 0] = info.max
                X_binned = np.clip(np.int8(self.B * (X - mins) // spreads), 0, self.B - 1)
        else:
            spreads[spreads == 0] = info.max
            X_binned = np.clip(np.int8(self.B * (X - mins) / spreads), 0, self.B - 1)
        return X_binned

    def _find_ranges(self, X):
        """Finds and returns variability ranges after skipping a certain ratio of outliers on both ends."""
        m, n = X.shape
        mins = np.zeros(n, dtype=X.dtype)
        maxes = np.zeros(n, dtype=X.dtype)
        if self.outliers_ratio > 0.0:
            for j in range(n):
                X_j_sorted = np.sort(X[:, j])
                mins[j] = X_j_sorted[int(np.ceil(self.outliers_ratio * m))]
                maxes[j] = X_j_sorted[int(np.floor((1.0 - self.outliers_ratio) * m))]
                if mins[j] >= maxes[j]:
                    mins[j] = X_j_sorted[0]
                    maxes[j] = X_j_sorted[-1]
        else:
            mins = np.min(X, axis=0)
            maxes = np.max(X, axis=0)
        return mins, maxes        
           
    def _fit_numpy(self, X, y):
        """Performs the actual fit with computations carried out in ``"numpy"`` mode (the slowest one).""" 
        if self.verbose:
            print(f"FIT... [fit_numpy, X.shape: {X.shape}, X.dtype={X.dtype}, T: {self.T}, B: {self.B}]")
        t1 = time.time()        
        m, n = X.shape
        yy = np.ones(m, dtype=np.int8)
        yy[y == self.classes_[0]] = -1        

        if self.verbose:
            print("[finding ranges of features...]")        
        t1_ranges = time.time()
        mins, maxes = self._find_ranges(X)
        t2_ranges = time.time()
        if self.verbose:
            print(f"[finding ranges of features done; time: {t2_ranges - t1_ranges} s]")
        
        if self.verbose:
            print("[binning...]")
        t1_binning = time.time()
        X_binned = self._bin_data(X, mins, maxes)
        t2_binning = time.time()
        if self.verbose:
            print(f"[binning done; time: {t2_binning - t1_binning} s]")                
        if self.verbose:
            print(f"[preparing indexing helpers...]")
        t1_indexer = time.time()
        ind_p = yy == 1
        ind_n = yy == -1
        indexer_p = np.zeros((n, self.B, m), dtype=bool)
        indexer_n = np.zeros((n, self.B, m), dtype=bool) 
        for j in range(n):
            for b in range(self.B):
                j_in_b = X_binned[:, j] == b                    
                indexer_p[j, b] = np.logical_and(j_in_b, ind_p)
                indexer_n[j, b] = np.logical_and(j_in_b, ind_n)
        t2_indexer = time.time()
        if self.verbose:
            print(f"[preparing indexing helpers; time: {t2_indexer - t1_indexer} s, shape: 2 x {indexer_p.shape}, size: 2 x {indexer_p.nbytes / 1024**3:.2f} GB]")
                
        w = np.ones(m, dtype=np.float32) / np.float32(m) # boosting weights of data examples
        
        if self.verbose:
            print("[main boosting loop...]")
        t1_loop = time.time()
        for t in range(self.T):
            t1_round = time.time()
            if self.verbose:
                print(f"[{t + 1}/{self.T}...]")
            best_err_exp = np.inf
            best_j = -1
            for j in range(n):
                W_p = np.zeros(self.B, dtype=np.float32)
                W_n = np.zeros(self.B, dtype=np.float32)
                logits_j = np.zeros(self.B, dtype=np.float32)
                for b in range(self.B):                                
                    W_p[b] = np.sum(w[indexer_p[j, b]])
                    W_n[b] = np.sum(w[indexer_n[j, b]])
                    logits_j[b] = self._logit(W_p[b], W_n[b])
                err_exp_j = np.sum(w * np.exp(-yy * logits_j[X_binned[:, j]]))
                if err_exp_j < best_err_exp:
                    best_err_exp = err_exp_j
                    best_j = j
                    self.logits_[t] = logits_j
            self.features_selected_[t] = best_j
            w = w * np.exp(-yy * self.logits_[t, X_binned[:, best_j]]) / best_err_exp
            t2_round = time.time()
            if self.verbose:
                print(f"[{t + 1}/{self.T} done; best_j: {best_j}, best_err_exp: {best_err_exp:.8f}, best_logits: {self.logits_[t]}, time: {t2_round - t1_round} s]")
        t2_loop = time.time()
        if self.verbose:        
            print(f"[main boosting loop done; time: {t2_loop - t1_loop} s]")
        
        self.mins_selected_ = mins[self.features_selected_]
        self.maxes_selected_ = maxes[self.features_selected_]
                
        t2 = time.time()
        if self.verbose:
            print(f"FIT DONE. [fit_numpy; time: {t2 - t1} s]")    
    
    def _fit_numba_jit(self, X, y):
        """
        Performs the actual fit with computations carried out in ``"numba_jit"`` mode.
        Inside the main boosting loop, all operations (weights binning, computing logits, computing exponential errors, finding the error minimizer, and examples reweighting)
        are perfomed  within a call to ``_fit_numba_jit_job`` function.
        """
        if self.verbose:
            print(f"FIT... [fit_numba_jit, X.shape: {X.shape}, X.dtype={X.dtype}, T: {self.T}, B: {self.B}]")
        t1 = time.time()
        m = X.shape[0]
        yy = np.ones(m, dtype=np.int8)
        yy[y == self.classes_[0]] = -1        

        if self.verbose:
            print("[finding ranges of features...]")        
        t1_ranges = time.time()
        mins, maxes = self._find_ranges(X)              
        t2_ranges = time.time()
        if self.verbose:
            print(f"[finding ranges of features done; time: {t2_ranges - t1_ranges} s]")
        
        if self.verbose:
            print("[binning...]")
        t1_binning = time.time()
        X_binned = self._bin_data(X, mins, maxes)
        t2_binning = time.time()
        if self.verbose:
            print(f"[binning done; time: {t2_binning - t1_binning} s]")                
                
        w = np.ones(m, dtype=np.float32) / np.float32(m) # boosting weights of data examples
                
        if self.verbose:
            print("[main boosting loop...]")
        t1_loop = time.time()
        for t in range(self.T):
            t1_round = time.time()
            if self.verbose:
                print(f"[{t + 1}/{self.T}...]")
            best_j, best_err_exp, best_logits, w = FastRealBoostBins._fit_numba_jit_job(X_binned, yy, w, self.B, self.logit_max)
            self.features_selected_[t] = best_j
            self.logits_[t] = np.copy(best_logits)            
            t2_round = time.time()
            if self.verbose:
                print(f"[{t + 1}/{self.T} done; best_j: {best_j}, best_err_exp: {best_err_exp:.8f}, best_logits: {self.logits_[t]}, time: {t2_round - t1_round} s]")
        t2_loop = time.time()
        if self.verbose:        
            print(f"[main boosting loop done; time: {t2_loop - t1_loop} s]")
        
        self.mins_selected_ = mins[self.features_selected_]
        self.maxes_selected_ = maxes[self.features_selected_]
                
        t2 = time.time()
        if self.verbose:
            print(f"FIT DONE. [fit_numba_jit; time: {t2 - t1} s]")

    @staticmethod
    @jit(nbtypes.Tuple((int32, float32, float32[:], float32[:]))(int8[:, :], int8[:], float32[:], int8, float32), nopython=True, cache=True)    
    def _fit_numba_jit_job(X_binned, yy, w, B, logit_max):
        """Body of the main boosting loop during fit carried out in ``"numba_jit"`` mode; called from within ``_fit_numba_jit`` function."""
        m, n = X_binned.shape           
        best_err_exp = np.finfo(np.float32).max
        best_j = -1
        best_logits = np.zeros(B, dtype=np.float32) 
        for j in range(n):
            W_p = np.zeros(B, dtype=np.float32)
            W_n = np.zeros(B, dtype=np.float32)
            logits_j = np.zeros(B, dtype=np.float32)
            for i in range(m):
                if yy[i] == 1:
                    W_p[X_binned[i, j]] += w[i]
                else:
                    W_n[X_binned[i, j]] += w[i]
            for b in range(B):
                W_p_b = W_p[b]
                W_n_b = W_n[b]            
                logit_value = None
                if W_p_b == 0.0  and W_n_b == 0.0:
                    logit_value = np.float32(0.0)
                elif W_p_b == 0.0 and W_n_b > np.float32(0.0):
                    logit_value = -logit_max
                elif W_n_b == 0.0 and W_p_b > np.float32(0.0):
                    logit_value = logit_max
                else:
                    logit_value = 0.5 * np.log(W_p_b / W_n_b)
                    if logit_value > logit_max:
                        logit_value = logit_max
                    elif logit_value < -logit_max:
                        logit_value = -logit_max             
                logits_j[b] = logit_value
            err_exp_j = np.sum(w * np.exp(-yy * logits_j[X_binned[:, j]]))
            if err_exp_j < best_err_exp:
                best_err_exp = err_exp_j
                best_j = j
                best_logits = logits_j
        w = w * np.exp(-yy * best_logits[X_binned[:, best_j]]) / best_err_exp
        return best_j, best_err_exp, best_logits, w 
    
    @staticmethod
    def _prepare_cuda_call_ranges(m, n_calls_min, power_two_sizes=False):
        """Prepares ranges of indexes (of data examples) for wanted number of subsequent calls for some CUDA kernel."""    
        if n_calls_min > m:                
            print(f"[warning: wanted n_calls_min = {n_calls_min} greater than m = {m} in prepare_cuda_call_ranges(...); hence, setting n_calls_min to m]")
            n_calls_min = m
        if not power_two_sizes:
            n_calls = n_calls_min
            call_size = m // n_calls
            call_ranges = call_size * np.ones(n_calls, dtype=np.int32)        
            call_ranges[:m % n_calls] += 1      
        else:
            call_size = 2**int(np.log2(m // n_calls_min))
            n_calls = int(np.ceil(m / call_size))
            call_ranges = call_size * np.ones(n_calls, dtype=np.int32)
            call_ranges[-1] = m % call_size
        call_ranges = np.r_[0, np.cumsum(call_ranges)]        
        return n_calls, call_ranges                                            
               
    def _fit_numba_cuda(self, X, y):
        """
        Performs the actual fit with computations carried out in ``"numba_cuda"`` mode.
        Inside the main boosting loop, all operations (weights binning, computing logits, computing exponential errors, finding the error minimizer, and examples reweighting)
        are perfomed, respectively, by the following CUDA kernel functions: ``_bin_add_weights_numba_cuda``, ``_logits_numba_cuda``, ``_errs_exp_numba_cuda``, ``_argmin_errs_exp_numba_cuda``, and ``_reweight_numba_cuda``.
        Three from those five kernels (1st, 3rd, 5th) are perfomed "in chunks", i.e. for successive data slices, relying on CUDA streams mechanism (partial parallelization of computations and data transfers).        
        """
        if self.verbose:
            print(f"FIT... [fit_numba_cuda, X.shape: {X.shape}, X.dtype={X.dtype}, T: {self.T}, B: {self.B}]")
        t1 = time.time()
        m, n = X.shape
        yy = np.ones(m, dtype=np.int8)
        yy[y == self.classes_[0]] = -1    
        
        if self.verbose:
            print("[finding ranges of features...]")
        t1_ranges = time.time()
        mins, maxes = self._find_ranges(X)
        t2_ranges = time.time()
        if self.verbose:
            print(f"[finding ranges of features done; time: {t2_ranges - t1_ranges} s]")          
        
        if self.verbose:
            print("[binning...]")
        t1_binning = time.time()            
        X_binned = self._bin_data(X, mins, maxes)        
        t2_binning = time.time()
        if self.verbose:
            print(f"[binning done; time: {t2_binning - t1_binning} s]")    
        
        w = np.ones(m, dtype=np.float32) / np.float32(m) # boosting weights of data examples
        
        t1_loop = time.time()
        if self.verbose:
            print("[main boosting loop...]")                    
        dev_mutexes = cuda.to_device(np.zeros((n, 1), dtype=np.int32)) # in most cases per-feature mutexes  are applied (only in argmin case a single mutex)
        dev_logits = cuda.device_array((n, self.B), dtype=np.float32)                                
        for t in range(self.T):
            t1_round = time.time()
            if self.verbose:
                print(f"[{t + 1}/{self.T}...]")
        
            t1_bin_add_weights = time.time()                        
            memory = X_binned.nbytes + yy.nbytes + w.nbytes 
            ratio = memory / self.CUDA_MAX_MEMORY_PER_CALL
            if ratio < 1.0:
                ratio = 1.0
            n_calls, call_ranges = FastRealBoostBins._prepare_cuda_call_ranges(m, int(np.ceil(ratio)))
            streams = []
            for _ in range(min(self._cuda_n_streams, n_calls)):
                streams.append(cuda.stream())
            dev_W_p = cuda.to_device(np.zeros((n, self.B), dtype=np.float32))
            dev_W_n = cuda.to_device(np.zeros((n, self.B), dtype=np.float32))
            tpb =  self._cuda_tpb_bin_add_weights                          
            with cuda.pinned(X_binned, yy, w):
                for i in range(n_calls):     
                    stream = streams[i % self._cuda_n_streams]
                    call_slice = slice(call_ranges[i], call_ranges[i + 1])
                    X_binned_sub = X_binned[call_slice]
                    dev_X_binned_sub = cuda.to_device(X_binned_sub, stream=stream)
                    dev_yy_sub = cuda.to_device(yy[call_slice], stream=stream)
                    dev_w_sub = cuda.to_device(w[call_slice], stream=stream)
                    bpg = (n, (X_binned_sub.shape[0] + tpb - 1) // tpb)
                    FastRealBoostBins._bin_add_weights_numba_cuda[bpg, tpb, stream](dev_X_binned_sub, dev_yy_sub, dev_w_sub, dev_W_p, dev_W_n, dev_mutexes)
                cuda.synchronize()
            t2_bin_add_weights = time.time()
            if self.debug_verbose:
                print(f"[bin_add_weights_numba_cuda done; n_calls: {n_calls}; time: {t2_bin_add_weights - t1_bin_add_weights} s]")

            t1_logits = time.time()   
            tpb = self.B
            bpg = n
            FastRealBoostBins._logits_numba_cuda[bpg, tpb](dev_W_p, dev_W_n, self.logit_max, dev_logits)
            cuda.synchronize()
            t2_logits = time.time()
            if self.debug_verbose:
                print(f"[logits_numba_cuda done; time: {t2_logits - t1_logits} s]")

            t1_errs_exp = time.time()            
            memory = X_binned.nbytes + yy.nbytes + w.nbytes
            ratio = memory / self.CUDA_MAX_MEMORY_PER_CALL
            if ratio < 1.0:
                ratio = 1.0
            n_calls, call_ranges = FastRealBoostBins._prepare_cuda_call_ranges(m, int(np.ceil(ratio)))
            streams = []            
            for _ in range(min(self._cuda_n_streams, n_calls)):
                streams.append(cuda.stream()) 
            dev_errs_exp = cuda.to_device(np.zeros(n, dtype=np.float32))
            tpb = self._cuda_tpb_default              
            with cuda.pinned(X_binned, yy, w):                                
                for i in range(n_calls):     
                    stream = streams[i % self._cuda_n_streams]
                    call_slice = slice(call_ranges[i], call_ranges[i + 1])
                    X_binned_sub = X_binned[call_slice]           
                    dev_X_binned_sub = cuda.to_device(X_binned_sub, stream=stream)
                    dev_yy_sub = cuda.to_device(yy[call_slice], stream=stream)
                    dev_w_sub = cuda.to_device(w[call_slice], stream=stream)
                    bpg = (n, (X_binned_sub.shape[0] + tpb - 1) // tpb)
                    FastRealBoostBins._errs_exp_numba_cuda[bpg, tpb, stream](dev_X_binned_sub, dev_yy_sub, dev_w_sub, dev_logits, dev_errs_exp, dev_mutexes)
                cuda.synchronize()
            t2_errs_exp = time.time()
            if self.debug_verbose:
                print(f"[errs_exp_numba_cuda done; n_calls: {n_calls}; time: {t2_errs_exp - t1_errs_exp} s]")

            t1_argmin_errs_exp = time.time()                                    
            best_err_exp = np.inf * np.ones(1, dtype=np.float32)
            best_j = -1 * np.ones(1, dtype=np.int32)
            best_logits = np.zeros(self.B, dtype=np.float32)
            dev_best_err_exp = cuda.to_device(best_err_exp)
            dev_best_j = cuda.to_device(best_j)
            dev_best_logits = cuda.to_device(best_logits)
            tpb = self._cuda_tpb_default
            bpg = (n + tpb - 1) // tpb
            FastRealBoostBins._argmin_errs_exp_numba_cuda[bpg, tpb](dev_errs_exp, dev_logits, dev_best_err_exp, dev_best_j, dev_best_logits, dev_mutexes)
            dev_best_err_exp.copy_to_host(ary=best_err_exp)
            dev_best_j.copy_to_host(ary=best_j)
            dev_best_logits.copy_to_host(ary=best_logits)
            cuda.synchronize()
            self.features_selected_[t] = best_j[0]
            self.logits_[t] = best_logits
            t2_argmin_errs_exp = time.time()
            if self.debug_verbose:
                print(f"[argmin_errs_exp_numba_cuda done; time: {t2_argmin_errs_exp - t1_argmin_errs_exp} s]")

            t1_reweight = time.time()                
            memory = X_binned.nbytes + yy.nbytes + w.nbytes
            ratio = memory / self.CUDA_MAX_MEMORY_PER_CALL
            if ratio < 1.0:
                ratio = 1.0
            n_calls, call_ranges = FastRealBoostBins._prepare_cuda_call_ranges(m, int(np.ceil(ratio)))
            streams = []            
            for _ in range(min(self._cuda_n_streams, n_calls)):
                streams.append(cuda.stream())
            tpb = self._cuda_tpb_default      
            with cuda.pinned(X_binned, yy, w):
                for i in range(n_calls):
                    stream = streams[i % self._cuda_n_streams]
                    call_slice = slice(call_ranges[i], call_ranges[i + 1])
                    X_binned_sub = X_binned[call_slice]           
                    dev_X_binned_sub = cuda.to_device(X_binned_sub, stream=stream)
                    dev_yy_sub = cuda.to_device(yy[call_slice], stream=stream)
                    dev_w_sub = cuda.to_device(w[call_slice], stream=stream)
                    bpg = (X_binned_sub.shape[0] + tpb - 1) // tpb
                    FastRealBoostBins._reweight_numba_cuda[bpg, tpb, stream](dev_X_binned_sub, dev_yy_sub, dev_w_sub, dev_best_j, dev_best_err_exp, dev_best_logits)
                    dev_w_sub.copy_to_host(ary=w[call_slice], stream=stream)
                cuda.synchronize()
            t2_reweight = time.time()
            if self.debug_verbose:
                print(f"[reweight_numba_cuda done; n_calls: {n_calls}; time: {t2_reweight - t1_reweight} s]")
            t2_round = time.time()
            if self.verbose:
                print(f"[{t + 1}/{self.T} done; best_j: {best_j[0]}, best_err_exp: {best_err_exp[0]:.8f}, best_logits: {best_logits}, time: {t2_round - t1_round} s]")
                        
        t2_loop = time.time()
        if self.verbose:
            print(f"[main boosting loop done; time: {t2_loop - t1_loop} s]")        
        
        self.mins_selected_ = mins[self.features_selected_]
        self.maxes_selected_ = maxes[self.features_selected_]
        
        t2 = time.time()
        if self.verbose:
            print(f"FIT DONE. [fit_numba_cuda; time: {t2 - t1} s]")      
                
    @staticmethod
    @cuda.jit(void(int8[:, :], int8[:], float32[:], float32[:, :], float32[:, :], int32[:, :]))    
    def _bin_add_weights_numba_cuda(X_binned_sub, yy_sub, w_sub, W_p, W_n, mutexes):
        """CUDA kernel responsible for binning and adding weights (within the main boosting loop)."""         
        shared_w_p = cuda.shared.array((128, 32), dtype=float32) # assumed max constants for shared memory: 128 - subsample size (equal to self._cuda_tpb_bin_add_weights), 32 - no. of bins
        shared_w_n = cuda.shared.array((128, 32), dtype=float32) 
        m, _ = X_binned_sub.shape        
        j = cuda.blockIdx.x
        tpb = cuda.blockDim.x
        tx = cuda.threadIdx.x
        i = cuda.blockIdx.y * tpb + tx # local data point index within current data sub (not global)
        _, B = W_p.shape
        if i < m:
            b_i_j = X_binned_sub[i, j]
        else:
            b_i_j = -1
        for b in range(B):
            if b == b_i_j:
                if yy_sub[i] == 1:
                    shared_w_p[tx, b] = w_sub[i]
                    shared_w_n[tx, b] = float32(0.0)
                else:
                    shared_w_p[tx, b] = float32(0.0)
                    shared_w_n[tx, b] = w_sub[i]                    
            else:
                shared_w_p[tx, b] = float32(0.0)
                shared_w_n[tx, b] = float32(0.0)
        cuda.syncthreads()
        stride = tpb >> 1 # half of tpb
        while stride > 0: # sum-reduction pattern
            if tx < stride:
                for b in range(B):
                    shared_w_p[tx, b] += shared_w_p[tx + stride, b]
                    shared_w_n[tx, b] += shared_w_n[tx + stride, b]
            cuda.syncthreads()
            stride >>= 1
        if tx == 0:           
            _lock(mutexes[j]) 
            for b in range(B):                                 
                W_p[j, b] += shared_w_p[0, b]
                W_n[j, b] += shared_w_n[0, b]                                     
            _unlock(mutexes[j])
    
    @staticmethod
    @cuda.jit(void(float32[:, :], float32[:, :], float32, float32[:, :]))
    def _logits_numba_cuda(W_p, W_n, logit_max, logits):
        """CUDA kernel responsible for computing binned logit values (within the main boosting loop)."""
        j = cuda.blockIdx.x
        b = cuda.threadIdx.x
        W_p_j_b = W_p[j, b]
        W_n_j_b = W_n[j, b] 
        if W_p_j_b == W_n_j_b:
            logits[j, b] = float32(0.0)
        elif W_p_j_b == float32(0.0):
            logits[j, b] = -logit_max
        elif W_n_j_b == float32(0.0):
            logits[j, b] = logit_max # equal to +LOGIT_MAX
        else:
            temp = 0.5 * math.log(W_p_j_b / W_n_j_b)
            if temp > logit_max:
                temp = logit_max
            elif temp < -logit_max:
                temp = -logit_max
            logits[j, b] = temp                          

    @staticmethod
    @cuda.jit(void(int8[:, :], int8[:], float32[:], float32[:, :], float32[:], int32[:, :]))
    def _errs_exp_numba_cuda(X_binned_sub, yy_sub, w_sub, logits, errs_exp, mutexes):
        """CUDA kernel responsible for computing exponential errors (within the main boosting loop)."""
        shared_errs_exp = cuda.shared.array((512), dtype=float32) # assumed max constant: 512 - subsample size                
        m, _ = X_binned_sub.shape                                                
        tpb = cuda.blockDim.x
        tx = cuda.threadIdx.x
        i = cuda.blockIdx.y * tpb + tx
        j = cuda.blockIdx.x
        if i < m:
            err = w_sub[i] * math.exp(-yy_sub[i] * logits[j, X_binned_sub[i, j]])
        else:
            err = float32(0.0)
        shared_errs_exp[tx] = err
        cuda.syncthreads()
        stride = tpb >> 1 # half of tpb
        while stride > 0: # sum-reduction pattern
            if tx < stride:
                shared_errs_exp[tx] += shared_errs_exp[tx + stride]
            cuda.syncthreads()
            stride >>= 1   
        if tx == 0:
            _lock(mutexes[j])
            errs_exp[j] += shared_errs_exp[0]
            _unlock(mutexes[j])

    @staticmethod
    @cuda.jit(void(float32[:], float32[:, :], float32[:], int32[:], float32[:], int32[:, :]))
    def _argmin_errs_exp_numba_cuda(errs_exp, logits, best_err_exp, best_j, best_logits, mutexes):
        """CUDA kernel responsible for finding the error minimizer (within the main boosting loop)."""
        shared_errs_exp = cuda.shared.array((512), dtype=float32) # assumed max tpb
        shared_best_j = cuda.shared.array((512), dtype=int32) # assumed max tpb
        n = errs_exp.size                
        j = cuda.grid(1)
        tx = cuda.threadIdx.x
        tpb = cuda.blockDim.x
        if j < n:
            shared_errs_exp[tx] = errs_exp[j]
            shared_best_j[tx] = j
        else:
            shared_errs_exp[tx] = float32(inf)
            shared_best_j[tx] = int32(-1)
        cuda.syncthreads()
        stride = tpb >> 1 # half of tpb
        while stride > 0: # sum-reduction pattern
            if tx < stride:
                tx_stride = tx + stride
                if shared_errs_exp[tx] > shared_errs_exp[tx_stride]:
                    shared_errs_exp[tx] = shared_errs_exp[tx_stride]
                    shared_best_j[tx] = shared_best_j[tx_stride]                
            cuda.syncthreads()
            stride >>= 1            
        if tx == 0:
            _lock(mutexes[0])
            if shared_errs_exp[0] < best_err_exp[0]:
                best_err_exp[0] = shared_errs_exp[0]
                the_best_j = shared_best_j[0]
                best_j[0] = the_best_j
                for b in range(best_logits.size):
                    best_logits[b] = logits[the_best_j, b]
            _unlock(mutexes[0])

    @staticmethod
    @cuda.jit(void(int8[:, :], int8[:], float32[:], int32[:], float32[:], float32[:]))
    def _reweight_numba_cuda(X_binned_sub, yy_sub, w_sub, best_j, best_exp_err, best_logits):
        """CUDA kernel responsible for reweighting data examples (within the main boosting loop)."""           
        m, _ = X_binned_sub.shape
        i = cuda.grid(1)        
        if i < m:
            w_sub[i] = w_sub[i] * math.exp(-yy_sub[i] * best_logits[X_binned_sub[i, best_j[0]]]) / best_exp_err[0]

    def decrease_T(self, T):
        """
        Decreases the number of weak estimators in this ensemble classifier. Attention: can be used only when the classifier has been fit (after ``fit`` function was called).
        Plays the role of a utility allowing one to easily reduce the classifier (e.g. to check accuracy of the smaller one) without having to refit it for a smaller ``T`` value.
        
        Args:
            T (int): 
                new wanted number weak estimators.
        """
        self.T = T
        self.features_selected_ = self.features_selected_[:T]
        self.logits_ = self.logits_[:T]
        self.mins_selected_ = self.mins_selected_[:T]
        self.maxes_selected_ = self.maxes_selected_[:T]
        params = self.get_params(deep=True)
        params["T"] = self.T
        self.set_params(**params)

    def decision_function(self, X):
        """
        Computes real-valued responses of ensemble for given data array.
        Depending on the mode, delegates actual computations to one of the following functions: ``_decision_function_numpy``, ``_decision_function_numba_jit``, or ``_decision_function_numba_cuda``.
        
        Args:
            X (ndarray[dtype_]):
                two-dimensional data array of numeric type with examples written as rows and features as columns (must have the same number of features as registered at fit stage).
                
        Returns:
            responses (ndarray[np.float32]): 
                one-dimensional array of ensemble responses. 
        """
        # sklearn checks
        check_is_fitted(self)
        X = check_array(X)        
        if self.n_features_in_ != X.shape[1]:
            raise ValueError(self.SKLEARN_ERR_MESSAGE_DISCREPANCY_IN_NO_OF_FEATURES)        
        # actual function        
        return self._decision_function_method(X)
        
    def _decision_function_numpy(self, X):
        """Performs the actual decision function with computations carried out in ``"numpy"`` mode (the slowest one)."""
        X_selected = X[:, self.features_selected_]
        X_binned = self.bin_data(X_selected, self.mins_selected_, self.maxes_selected_)
        m = X_binned.shape[0]
        responses = np.zeros(m)
        for i in range(m):
            responses[i] = np.sum(self.logits_[np.arange(self.T), X_binned[i]])     
        return responses

    def _decision_function_numba_jit(self, X):
        """Performs the actual decision function with computations carried out in ``"numba_jit"`` mode."""
        X_selected = X[:, self.features_selected_]
        X_binned = self._bin_data(X_selected, self.mins_selected_, self.maxes_selected_)        
        return FastRealBoostBins._decision_function_numba_jit_job(X_binned, self.logits_)
    
    @staticmethod
    @jit(float32[:](int8[:, :], float32[:, :]), nopython=True, cache=True)
    def _decision_function_numba_jit_job(X_binned, logits):
        """Body of the decision function carried out in ``"numba_jit"`` mode; called from within ``_decision_function_numba_jit`` function."""
        m, T = X_binned.shape
        responses = np.zeros(m, dtype=np.float32)
        for i in range(m):
            for t in range(T):  
                responses[i] += logits[t, X_binned[i, t]]
        return responses
    
    def _decision_function_numba_cuda(self, X):
        """
        Performs the actual decision function with computations carried out in ``"numba_cuda"`` mode.
        Depending on the type of input data array, delegates actual computations to one of the following kernel functions: 
        ``_decision_function_numba_cuda_job_int8``, ``_decision_function_numba_cuda_job_uint8``, ..., ``_decision_function_numba_cuda_job_int64``, ``_decision_function_numba_cuda_job_uint64``,
        or ``_decision_function_numba_cuda_job_float32``, ``_decision_function_numba_cuda_job_float64``.
        """
        X_selected = X[:, self.features_selected_]
        m = X_selected.shape[0]
        dev_X_selected = cuda.to_device(X_selected)
        dev_mins_selected = cuda.to_device(self.mins_selected_)
        dev_maxes_selected = cuda.to_device(self.maxes_selected_)
        dev_logits = cuda.to_device(self.logits_)                
        dev_responses = cuda.device_array(m, dtype=np.float32)
        tpb = self._cuda_tpb_default
        bpg = m    
        decision_function_numba_cuda_job_method = getattr(FastRealBoostBins, self.decision_function_numba_cuda_job_name_)
        decision_function_numba_cuda_job_method[bpg, tpb](dev_X_selected, dev_mins_selected, dev_maxes_selected, dev_logits, dev_responses)
        cuda.synchronize()
        responses = dev_responses.copy_to_host()        
        return responses

    @staticmethod
    @cuda.jit(void(int8[:, :], int8[:], int8[:], float32[:, :], float32[:]))
    def _decision_function_numba_cuda_job_int8(X_selected, mins_selected, maxes_selected, logits, responses):
        """Body of the decision function carried out in ``"numba_cuda"`` mode suitable for input arrays of type ``int8``; called from within ``_decision_function_numba_cuda`` function."""
        shared_logits = cuda.shared.array(1024, dtype=float32) # 1024 - corresponds to assumed max tpb           
        i = cuda.blockIdx.x
        tpb = cuda.blockDim.x
        tx = cuda.threadIdx.x
        T, B = logits.shape
        fpt = int((T + tpb - 1) / tpb) # features per thread to be translated onto appropriate logits and stored in shared memory (summed later if fpt > 1)
        t = tx # feature index
        shared_logits[tx] = float32(0.0)
        cuda.syncthreads()
        for _ in range(fpt):
            if t < T:
                b = int8(B * (int16(X_selected[i, t]) - int16(mins_selected[t])) // (int16(maxes_selected[t]) - int16(mins_selected[t])))                
                if b < 0:
                    b = 0
                elif b >= B:
                    b = B - 1
                shared_logits[tx] += logits[t, b]
            t += tpb
        cuda.syncthreads()
        stride = tpb >> 1 # half of tpb
        while stride > 0: # sum-reduction pattern
            if tx < stride:
                shared_logits[tx] += shared_logits[tx + stride]
            cuda.syncthreads()
            stride >>= 1
        if tx == 0:
            responses[i] = shared_logits[0]
    
    @staticmethod
    @cuda.jit(void(uint8[:, :], uint8[:], uint8[:], float32[:, :], float32[:]))
    def _decision_function_numba_cuda_job_uint8(X_selected, mins_selected, maxes_selected, logits, responses):
        """Body of the decision function carried out in ``"numba_cuda"`` mode suitable for input arrays of type ``uint8``; called from within ``_decision_function_numba_cuda`` function."""
        shared_logits = cuda.shared.array(1024, dtype=float32) # 1024 - corresponds to assumed max tpb           
        i = cuda.blockIdx.x
        tpb = cuda.blockDim.x
        tx = cuda.threadIdx.x
        T, B = logits.shape
        fpt = int((T + tpb - 1) / tpb) # features per thread to be translated onto appropriate logits and stored in shared memory (summed if fpt > 1)
        t = tx # feature index
        shared_logits[tx] = float32(0.0)
        cuda.syncthreads()
        for _ in range(fpt):
            if t < T:
                b = int8(B * (int16(X_selected[i, t]) - int16(mins_selected[t])) // (int16(maxes_selected[t]) - int16(mins_selected[t])))
                if b < 0:
                    b = 0
                elif b >= B:
                    b = B - 1
                shared_logits[tx] += logits[t, b]
            t += tpb
        cuda.syncthreads()
        stride = tpb >> 1 # half of tpb
        while stride > 0: # sum-reduction pattern
            if tx < stride:
                shared_logits[tx] += shared_logits[tx + stride]
            cuda.syncthreads()
            stride >>= 1
        if tx == 0:
            responses[i] = shared_logits[0]    
    
    @staticmethod
    @cuda.jit(void(int16[:, :], int16[:], int16[:], float32[:, :], float32[:]))
    def _decision_function_numba_cuda_job_int16(X_selected, mins_selected, maxes_selected, logits, responses):
        """Body of the decision function carried out in ``"numba_cuda"`` mode suitable for input arrays of type ``int16``; called from within ``_decision_function_numba_cuda`` function."""
        shared_logits = cuda.shared.array(1024, dtype=float32) # 1024 - corresponds to assumed max tpb
        i = cuda.blockIdx.x
        tpb = cuda.blockDim.x
        tx = cuda.threadIdx.x
        T, B = logits.shape
        fpt = int((T + tpb - 1) / tpb) # features per thread to be translated onto appropriate logits and stored in shared memory (summed if fpt > 1)
        t = tx # feature index
        shared_logits[tx] = float32(0.0)
        cuda.syncthreads()
        for _ in range(fpt):
            if t < T:
                b = int8(B * (int32(X_selected[i, t]) - int32(mins_selected[t])) // (int32(maxes_selected[t]) - int32(mins_selected[t])))                
                if b < 0:
                    b = 0
                elif b >= B:
                    b = B - 1
                shared_logits[tx] += logits[t, b]
            t += tpb
        cuda.syncthreads()
        stride = tpb >> 1 # half of tpb
        while stride > 0: # sum-reduction pattern
            if tx < stride:
                shared_logits[tx] += shared_logits[tx + stride]
            cuda.syncthreads()
            stride >>= 1
        if tx == 0:
            responses[i] = shared_logits[0]
            
    @staticmethod
    @cuda.jit(void(uint16[:, :], uint16[:], uint16[:], float32[:, :], float32[:]))
    def _decision_function_numba_cuda_job_uint16(X_selected, mins_selected, maxes_selected, logits, responses):
        """Body of the decision function carried out in ``"numba_cuda"`` mode suitable for input arrays of type ``uint16``; called from within ``_decision_function_numba_cuda`` function."""
        shared_logits = cuda.shared.array(1024, dtype=float32) # 1024 - corresponds to assumed max tpb           
        i = cuda.blockIdx.x
        tpb = cuda.blockDim.x
        tx = cuda.threadIdx.x
        T, B = logits.shape
        fpt = int((T + tpb - 1) / tpb) # features per thread to be translated onto appropriate logits and stored in shared memory (summed if fpt > 1)
        t = tx # feature index
        shared_logits[tx] = float32(0.0)
        cuda.syncthreads()
        for _ in range(fpt):
            if t < T:
                b = int8(B * (int32(X_selected[i, t]) - int32(mins_selected[t])) // (int32(maxes_selected[t]) - int32(mins_selected[t])))
                if b < 0:
                    b = 0
                elif b >= B:
                    b = B - 1
                shared_logits[tx] += logits[t, b]
            t += tpb
        cuda.syncthreads()
        stride = tpb >> 1 # half of tpb
        while stride > 0: # sum-reduction pattern
            if tx < stride:
                shared_logits[tx] += shared_logits[tx + stride]
            cuda.syncthreads()
            stride >>= 1
        if tx == 0:
            responses[i] = shared_logits[0]            
            
    @staticmethod
    @cuda.jit(void(int32[:, :], int32[:], int32[:], float32[:, :], float32[:]))
    def _decision_function_numba_cuda_job_int32(X_selected, mins_selected, maxes_selected, logits, responses):
        """Body of the decision function carried out in ``"numba_cuda"`` mode suitable for input arrays of type ``int32``; called from within ``_decision_function_numba_cuda`` function."""
        shared_logits = cuda.shared.array(1024, dtype=float32) # 1024 - corresponds to assumed max tpb           
        i = cuda.blockIdx.x
        tpb = cuda.blockDim.x
        tx = cuda.threadIdx.x
        T, B = logits.shape
        fpt = int((T + tpb - 1) / tpb) # features per thread to be translated onto appropriate logits and stored in shared memory (summed if fpt > 1)
        t = tx # feature index
        shared_logits[tx] = float32(0.0)
        cuda.syncthreads()
        for _ in range(fpt):
            if t < T:
                b = int8(B * (int64(X_selected[i, t]) - int64(mins_selected[t])) // (int64(maxes_selected[t]) - int64(mins_selected[t])))
                if b < 0:
                    b = 0
                elif b >= B:
                    b = B - 1
                shared_logits[tx] += logits[t, b]
            t += tpb
        cuda.syncthreads()
        stride = tpb >> 1 # half of tpb
        while stride > 0: # sum-reduction pattern
            if tx < stride:
                shared_logits[tx] += shared_logits[tx + stride]
            cuda.syncthreads()
            stride >>= 1
        if tx == 0:
            responses[i] = shared_logits[0]            

    @staticmethod
    @cuda.jit(void(uint32[:, :], uint32[:], uint32[:], float32[:, :], float32[:]))
    def _decision_function_numba_cuda_job_uint32(X_selected, mins_selected, maxes_selected, logits, responses):
        """Body of the decision function carried out in ``"numba_cuda"`` mode suitable for input arrays of type ``uint32``; called from within ``_decision_function_numba_cuda`` function."""
        shared_logits = cuda.shared.array(1024, dtype=float32) # 1024 - corresponds to assumed max tpb           
        i = cuda.blockIdx.x
        tpb = cuda.blockDim.x
        tx = cuda.threadIdx.x
        T, B = logits.shape
        fpt = int((T + tpb - 1) / tpb) # features per thread to be translated onto appropriate logits and stored in shared memory (summed if fpt > 1)
        t = tx # feature index
        shared_logits[tx] = float32(0.0)
        cuda.syncthreads()
        for _ in range(fpt):
            if t < T:
                b = int8(B * (int64(X_selected[i, t]) - int64(mins_selected[t])) // (int64(maxes_selected[t]) - int64(mins_selected[t])))
                if b < 0:
                    b = 0
                elif b >= B:
                    b = B - 1
                shared_logits[tx] += logits[t, b]
            t += tpb
        cuda.syncthreads()
        stride = tpb >> 1 # half of tpb
        while stride > 0: # sum-reduction pattern
            if tx < stride:
                shared_logits[tx] += shared_logits[tx + stride]
            cuda.syncthreads()
            stride >>= 1
        if tx == 0:
            responses[i] = shared_logits[0]

    @staticmethod
    @cuda.jit(void(int64[:, :], int64[:], int64[:], float32[:, :], float32[:]))
    def _decision_function_numba_cuda_job_int64(X_selected, mins_selected, maxes_selected, logits, responses):
        """Body of the decision function carried out in ``"numba_cuda"`` mode suitable for input arrays of type ``int64``; called from within ``_decision_function_numba_cuda`` function."""
        shared_logits = cuda.shared.array(1024, dtype=float32) # 1024 - corresponds to assumed max tpb           
        i = cuda.blockIdx.x
        tpb = cuda.blockDim.x
        tx = cuda.threadIdx.x
        T, B = logits.shape
        fpt = int((T + tpb - 1) / tpb) # features per thread to be translated onto appropriate logits and stored in shared memory (summed if fpt > 1)
        t = tx # feature index
        shared_logits[tx] = float32(0.0)
        cuda.syncthreads()
        for _ in range(fpt):
            if t < T:
                b = int8(B * (X_selected[i, t] - mins_selected[t]) // (maxes_selected[t] - mins_selected[t]))
                if b < 0:
                    b = 0
                elif b >= B:
                    b = B - 1
                shared_logits[tx] += logits[t, b]
            t += tpb
        cuda.syncthreads()
        stride = tpb >> 1 # half of tpb
        while stride > 0: # sum-reduction pattern
            if tx < stride:
                shared_logits[tx] += shared_logits[tx + stride]
            cuda.syncthreads()
            stride >>= 1
        if tx == 0:
            responses[i] = shared_logits[0]
            
    @staticmethod
    @cuda.jit(void(uint64[:, :], uint64[:], uint64[:], float32[:, :], float32[:]))
    def _decision_function_numba_cuda_job_uint64(X_selected, mins_selected, maxes_selected, logits, responses):
        """Body of the decision function carried out in ``"numba_cuda"`` mode suitable for input arrays of type ``uint32``; called from within ``_decision_function_numba_cuda`` function."""
        shared_logits = cuda.shared.array(1024, dtype=float32) # 1024 - corresponds to assumed max tpb           
        i = cuda.blockIdx.x
        tpb = cuda.blockDim.x
        tx = cuda.threadIdx.x
        T, B = logits.shape
        fpt = int((T + tpb - 1) / tpb) # features per thread to be translated onto appropriate logits and stored in shared memory (summed if fpt > 1)
        t = tx # feature index
        shared_logits[tx] = float32(0.0)
        cuda.syncthreads()
        for _ in range(fpt):
            if t < T:
                b = int8(B * (X_selected[i, t] - mins_selected[t]) // (maxes_selected[t] - mins_selected[t]))
                if b < 0:
                    b = 0
                elif b >= B:
                    b = B - 1
                shared_logits[tx] += logits[t, b]
            t += tpb
        cuda.syncthreads()
        stride = tpb >> 1 # half of tpb
        while stride > 0: # sum-reduction pattern
            if tx < stride:
                shared_logits[tx] += shared_logits[tx + stride]
            cuda.syncthreads()
            stride >>= 1
        if tx == 0:
            responses[i] = shared_logits[0]            
            
    @staticmethod
    @cuda.jit(void(float32[:, :], float32[:], float32[:], float32[:, :], float32[:]))
    def _decision_function_numba_cuda_job_float32(X_selected, mins_selected, maxes_selected, logits, responses):
        shared_logits = cuda.shared.array(1024, dtype=float32) # 1024 - corresponds to assumed max tpb           
        i = cuda.blockIdx.x
        tpb = cuda.blockDim.x
        tx = cuda.threadIdx.x
        T, B = logits.shape
        fpt = int((T + tpb - 1) / tpb) # features per thread to be translated onto appropriate logits and stored in shared memory (summed if fpt > 1)
        t = tx # feature index
        shared_logits[tx] = float32(0.0)
        cuda.syncthreads()
        for _ in range(fpt):
            if t < T:
                b = int8(B * (X_selected[i, t] - mins_selected[t]) / (maxes_selected[t] - mins_selected[t]))
                if b < 0:
                    b = 0
                elif b >= B:
                    b = B - 1
                shared_logits[tx] += logits[t, b]
            t += tpb
        cuda.syncthreads()
        stride = tpb >> 1 # half of tpb
        while stride > 0: # sum-reduction pattern
            if tx < stride:
                shared_logits[tx] += shared_logits[tx + stride]
            cuda.syncthreads()
            stride >>= 1
        if tx == 0:
            responses[i] = shared_logits[0]            
            
    @staticmethod
    @cuda.jit(void(float64[:, :], float64[:], float64[:], float32[:, :], float32[:]))
    def _decision_function_numba_cuda_job_float64(X_selected, mins_selected, maxes_selected, logits, responses):
        shared_logits = cuda.shared.array(1024, dtype=float32) # 1024 - corresponds to assumed max tpb           
        i = cuda.blockIdx.x
        tpb = cuda.blockDim.x
        tx = cuda.threadIdx.x
        T, B = logits.shape
        fpt = int((T + tpb - 1) / tpb) # features per thread to be translated onto appropriate logits and stored in shared memory (summed if fpt > 1)
        t = tx # feature index
        shared_logits[tx] = float32(0.0)
        cuda.syncthreads()
        for _ in range(fpt):
            if t < T:
                b = int8(B * (X_selected[i, t] - mins_selected[t]) / (maxes_selected[t] - mins_selected[t]))
                if b < 0:
                    b = 0
                elif b >= B:
                    b = B - 1
                shared_logits[tx] += logits[t, b]
            t += tpb
        cuda.syncthreads()
        stride = tpb >> 1 # half of tpb
        while stride > 0: # sum-reduction pattern
            if tx < stride:
                shared_logits[tx] += shared_logits[tx + stride]
            cuda.syncthreads()
            stride >>= 1
        if tx == 0:
            responses[i] = shared_logits[0]            
         
    def predict(self, X):
        """
        Returns class labels predicted for given data array.
        Delegates actual computations to ``decision_function`` and maps real-valued responses obtained from it to one of two class labels (negative or positive) kept in ``self.classes_``, 
        taking into account the ``self.decision_threshold_`` attribute.   
        
        Args:
            X (ndarray[dtype_]):
                two-dimensional data array of numeric type with examples written as rows and features as columns (must have the same number of features as registered at fit stage).
                
        Returns:
            class labels (ndarray): 
                one-dimensional array of predicted class labels. 
        """        
        return self.classes_[1 * (self.decision_function(X) > self.decision_threshold_)]
    
    def json_dump(self, fname):
        """
        Dumps (saves) this ensemble classifier to a text file in json format.   
        
        Args:
            fname (string):
                file name.             
        """        
        if self.verbose:
            print(f"JSON DUMP... [to file: {fname}]")
        t1 = time.time()        
        d = {}
        d["T"] = self.T
        d["B"] = self.B
        d["outliers_ratio"] = self.outliers_ratio
        d["logit_max"] = float(self.logit_max)
        d["fit_mode"] = self.fit_mode
        d["decision_function_mode"] = self.decision_function_mode
        d["verbose"] = self.verbose
        d["debug_verbose"] = self.verbose
        if self.classes_ is not None:
            d["classes_"] = self.classes_.tolist()
            d["n_features_in_"] = self.n_features_in_
            d["dtype_"] = str(self.dtype_)            
            d["decision_function_numba_cuda_job_name_"] = self.decision_function_numba_cuda_job_name_
            d["decision_threshold_"] = self.decision_threshold_
            d["features_selected_"] = self.features_selected_.tolist()
            d["mins_selected_"] = self.mins_selected_.tolist()
            d["maxes_selected_"] = self.maxes_selected_.tolist()
            d["logits_"] = self.logits_.tolist()
        try:
            f = open(fname, "w+")
            json.dump(d, f, indent=2)
            f.close()
        except IOError:
            sys.exit(f"[error occurred when trying to dump clf as json to file: {fname}]")
        t2 = time.time()
        if self.verbose:
            print(f"JSON DUMP DONE. [time: {t2 - t1} s]")
            
    @staticmethod
    def json_load(fname, verbose=True):        
        """
        Creates and returns an instance of ``FastRealBoostBins`` from a text file in json format given its file path.   
        
        Args:
            fname (string):
                file name.
            verbose (bool):
                verbosity flag.        
        """
        if verbose:
            print(f"JSON LOAD... [from file: {fname}]")
        t1 = time.time()
        try:
            f = open(fname, "r")
            d = json.load(f)
            params = {}
            params["T"] = d["T"]
            params["B"] = d["B"]
            params["outliers_ratio"] = d["outliers_ratio"]
            params["logit_max"] = np.float32(d["logit_max"])
            params["fit_mode"] = d["fit_mode"]
            params["decision_function_mode"] = d["decision_function_mode"]
            params["verbose"] = d["verbose"]
            params["debug_verbose"] = d["debug_verbose"]                            
            clf = FastRealBoostBins(**params)
            clf._set_cuda_constants()
            clf._set_modes(clf.fit_mode, clf.decision_function_mode)
            clf.classes_ = np.array(d["classes_"])
            clf.n_features_in_ = d["n_features_in_"]
            clf.dtype_ = d["dtype_"]
            clf.decision_function_numba_cuda_job_name_ = d["decision_function_numba_cuda_job_name_"]
            clf.decision_threshold_ = d["decision_threshold_"]            
            clf.features_selected_ = np.array(d["features_selected_"], dtype=np.int32)
            clf.mins_selected_ = np.array(d["mins_selected_"], dtype=clf.dtype_)
            clf.maxes_selected_ = np.array(d["maxes_selected_"], dtype=clf.dtype_)
            clf.logits_ = np.array(d["logits_"], dtype=np.float32)
            f.close()
        except IOError:
            sys.exit(f"[error occurred when trying to load clf from json file: {fname}]")
        t2 = time.time()
        if verbose:
            print(f"JSON LOAD DONE. [time: {t2 - t1} s]")    
        return clf