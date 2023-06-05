from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from numpy import inf
import time
from numba import cuda, jit
from numba import void, int8, int16, int32, float32
import math
from numba.core.errors import NumbaPerformanceWarning
import warnings
        
__version__ = "0.8.0"
__author__ = "Przemysław Klęsk"
__email__ = "pklesk@zut.edu.pl"
        
warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
np.set_printoptions(linewidth=512)

# mutex-related cuda utility functions 
@cuda.jit(device=True)
def lock(mutex):        
    while cuda.atomic.compare_and_swap(mutex, 0, 1) != 0:
        pass
    cuda.threadfence()    
    
@cuda.jit(device=True)
def unlock(mutex):
    cuda.threadfence()
    cuda.atomic.exch(mutex, 0, 0)


class FastRealBoostBins(BaseEstimator, ClassifierMixin):

    # constants
    OUTLIERS_RATIO = 0.05
    LOGIT_MAX = np.float32(2.0)
    B_MAX = 32
    CUDA_MAX_MEMORY_PER_CALL = 8 * 1024**2 # can be adjusted for given gpu device     
    
    def __init__(self, T=8, B=8, fit_mode="numba_cuda", decision_function_mode="numba_cuda", verbose=True, debug_verbose=False):
        super().__init__()
        self.T = T # no. of boosting rounds
        self.B = B # no. of bins
        if self.B > self.B_MAX:
            self.B = self.B_MAX
            print(f"[changing wanted number of bins to the allowed limit: {B_MAX}]")            
        self.verbose = verbose
        self.debug_verbose = debug_verbose                        
        self.cuda_available = cuda.is_available() 
        self.cuda_tpb_default = cuda.get_current_device().MAX_THREADS_PER_BLOCK // 2 if self.cuda_available else None
        self.cuda_tpb_bin_add_weights = 128 if self.cuda_available else None
        self.cuda_n_streams = cuda.get_current_device().ASYNC_ENGINE_COUNT if self.cuda_available else None
        self.set_modes(fit_mode, decision_function_mode)        
        
    def set_modes(self, fit_mode="numba_cuda", decision_function_mode="numba_cuda"):
        self.fit_mode = fit_mode
        self.decision_function_mode = decision_function_mode
        if self.fit_mode == "numba_cuda" and not self.cuda_available:
            self.fit_mode = "numpy"
            print(f"[changing fit mode to \"{self.fit_mode}\" due to cuda functionality not available on this machine]")
        if self.decision_function_mode == "numba_cuda" and not self.cuda_available:
            self.decision_function_mode = "numba_jit"
            print(f"[changing decision function mode \"{self.decision_function_mode}\" to numba_jit due to cuda functionality not available on this machine]")
        self.fit_attr = getattr(self, "fit_" + self.fit_mode)
        self.decision_function_attr = getattr(self, "decision_function_" + self.decision_function_mode)            
                                                           
    def logit(self, W_p, W_n):
        if W_p == W_n:
            return np.float32(0.0)
        elif W_p == 0.0:
            return -self.LOGIT_MAX
        elif W_n == 0.0:
            return self.LOGIT_MAX
        return np.clip(0.5 * np.log(W_p / W_n), -self.LOGIT_MAX, self.LOGIT_MAX)         
                
    def fit(self, X, y):
        self.fit_init()
        self.fit_attr(X, y)
        return self
    
    def fit_init(self):
        self.T_ = self.T
        self.B_ = self.B
        self.features_indexes_ = np.zeros(self.T_, dtype=np.int32) # indexes of selected features
        self.logits_ = np.zeros((self.T_, self.B_), dtype=np.float32)
    
    def fit_numpy(self, X, y):
        if self.verbose:
            print(f"FIT... [fit_numpy, X.shape: {X.shape}, X.dtype={X.dtype}, T: {self.T_}, B: {self.B_}]")
        t1 = time.time()
        self.class_labels_ = np.unique(y) # we assume exactly 2 classes with first class negative
        m, n = X.shape
        yy = np.ones(m, dtype=np.int8)
        yy[y == self.class_labels_[0]] = -1        

        if self.verbose:
            print("[finding ranges of features...]")        
        t1_ranges = time.time()
        self.mins_ = np.zeros(n, dtype=X.dtype)
        self.maxes_ = np.zeros(n, dtype=X.dtype)
        for j in range(n):
            X_j_sorted = np.sort(X[:, j])
            self.mins_[j] = X_j_sorted[int(np.ceil(self.OUTLIERS_RATIO * m))]
            self.maxes_[j] = X_j_sorted[int(np.floor((1.0 - self.OUTLIERS_RATIO) * m))]
        t2_ranges = time.time()
        if self.verbose:
            print(f"[finding ranges of features done; time: {t2_ranges - t1_ranges} s]")
        
        if self.verbose:
            print("[binning...]")
        t1_binning = time.time()
        X_binned = np.clip(np.int8(self.B_ * (X - self.mins_) // (self.maxes_ - self.mins_)), 0, self.B_ - 1)
        t2_binning = time.time()
        if self.verbose:
            print(f"[binning done; time: {t2_binning - t1_binning} s]")                
        if self.verbose:
            print(f"[preparing indexing helpers...]")
        t1_indexer = time.time()
        ind_p = yy == 1
        ind_n = yy == -1
        indexer_p = np.zeros((n, self.B_, m), dtype=bool)
        indexer_n = np.zeros((n, self.B_, m), dtype=bool) 
        for j in range(n):
            for b in range(self.B_):
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
        for t in range(self.T_):
            if self.verbose:
                print(f"{t + 1}/{self.T_}")
            best_err_exp = np.inf
            best_j = -1
            for j in range(n):
                W_p = np.zeros(self.B_, dtype=np.float32)
                W_n = np.zeros(self.B_, dtype=np.float32)
                logits_j = np.zeros(self.B_, dtype=np.float32)
                for b in range(self.B_):                                
                    W_p[b] = np.sum(w[indexer_p[j, b]])
                    W_n[b] = np.sum(w[indexer_n[j, b]])
                    logits_j[b] = self.logit(W_p[b], W_n[b])
                err_exp_j = np.sum(w * np.exp(-yy * logits_j[X_binned[:, j]]))
                if err_exp_j < best_err_exp:
                    best_err_exp = err_exp_j
                    best_j = j
                    self.logits_[t] = logits_j
            if self.verbose:
                print(f"[best_j: {best_j}, best_err_exp: {best_err_exp:.8f}, best_logits: {self.logits_[t]}]")
            self.features_indexes_[t] = best_j
            w = w * np.exp(-yy * self.logits_[t, X_binned[:, best_j]]) / best_err_exp
        t2_loop = time.time()
        if self.verbose:        
            print(f"[main boosting loop done; time: {t2_loop - t1_loop} s]")
        
        self.mins_selected_ = self.mins_[self.features_indexes_]
        self.maxes_selected_ = self.maxes_[self.features_indexes_]
                
        t2 = time.time()
        if self.verbose:
            print(f"FIT DONE. [fit_numpy; time: {t2 - t1} s]")
    
    @staticmethod
    def prepare_cuda_call_ranges(m, n_calls_min, power_two_sizes=False):
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
               
    def fit_numba_cuda(self, X, y):
        if self.verbose:
            print(f"FIT... [fit_numba_cuda, X.shape: {X.shape}, X.dtype={X.dtype}, T: {self.T_}, B: {self.B_}]")
        t1 = time.time()
        self.class_labels_ = np.unique(y) # we assume exactly 2 classes with first class negative
        m, n = X.shape
        yy = np.ones(m, dtype=np.int8)
        yy[y == self.class_labels_[0]] = -1    
        
        if self.verbose:
            print("[finding ranges of features...]")
        t1_ranges = time.time()
        self.mins_ = np.zeros(n, dtype=np.int16)
        self.maxes_ = np.zeros(n, dtype=np.int16)
        left = int(np.ceil(self.OUTLIERS_RATIO * m))
        right = int(np.floor((1.0 - self.OUTLIERS_RATIO) * m))
        for j in range(n):
            X_j_sorted = np.sort(X[:, j])
            self.mins_[j] = X_j_sorted[left]
            self.maxes_[j] = X_j_sorted[right]
        t2_ranges = time.time()
        if self.verbose:
            print(f"[finding ranges of features done; time: {t2_ranges - t1_ranges} s]")          

        t1_binning = time.time()
        if self.verbose:
            print("[binning...]")
        X_binned = np.clip(np.int8(self.B_ * (X - self.mins_) // (self.maxes_ - self.mins_)), 0, self.B_ - 1)
        t2_binning = time.time()
        if self.verbose:
            print(f"[binning done; time: {t2_binning - t1_binning} s]")    
        
        w = np.ones(m, dtype=np.float32) / np.float32(m) # boosting weights of data examples
        
        t1_loop = time.time()
        if self.verbose:
            print("[main boosting loop...]")                    
        dev_mutexes = cuda.to_device(np.zeros((n, 1), dtype=np.int32)) # in most cases per-feature mutexes  are applied (only in argmin case a single mutex)
        dev_logits = cuda.device_array((n, self.B_), dtype=np.float32)                                
        for t in range(self.T_):
            if self.verbose:
                print(f"{t + 1}/{self.T_}")
        
            t1_bin_add_weights = time.time()                        
            memory = X_binned.nbytes + yy.nbytes + w.nbytes 
            ratio = memory / self.CUDA_MAX_MEMORY_PER_CALL
            if ratio < 1.0:
                ratio = 1.0
            n_calls, call_ranges = FastRealBoostBins.prepare_cuda_call_ranges(m, int(np.ceil(ratio)))
            streams = []
            for _ in range(min(self.cuda_n_streams, n_calls)):
                streams.append(cuda.stream())
            dev_W_p = cuda.to_device(np.zeros((n, self.B_), dtype=np.float32))
            dev_W_n = cuda.to_device(np.zeros((n, self.B_), dtype=np.float32))
            tpb =  self.cuda_tpb_bin_add_weights                                
            with cuda.pinned(X_binned, yy, w):                
                for i in range(n_calls):     
                    stream = streams[i % self.cuda_n_streams]
                    call_slice = slice(call_ranges[i], call_ranges[i + 1])
                    X_binned_sub = X_binned[call_slice]
                    dev_X_binned_sub = cuda.to_device(X_binned_sub, stream=stream)
                    dev_yy_sub = cuda.to_device(yy[call_slice], stream=stream)
                    dev_w_sub = cuda.to_device(w[call_slice], stream=stream)
                    bpg = ((X_binned_sub.shape[0] + tpb - 1) // tpb, n)
                    FastRealBoostBins.bin_add_weights_numba_cuda[bpg, tpb, stream](dev_X_binned_sub, dev_yy_sub, dev_w_sub, dev_W_p, dev_W_n, dev_mutexes)
                cuda.synchronize()
            t2_bin_add_weights = time.time()
            if self.debug_verbose:
                print(f"[bin_add_weights_numba_cuda done; n_calls: {n_calls}; time: {t2_bin_add_weights - t1_bin_add_weights} s]")

            t1_logits = time.time()            
            tpb = self.B_
            bpg = n
            FastRealBoostBins.logits_numba_cuda[bpg, tpb](dev_W_p, dev_W_n, dev_logits)
            cuda.synchronize()
            t2_logits = time.time()
            if self.debug_verbose:
                print(f"[logits_numba_cuda done; time: {t2_logits - t1_logits} s]")

            t1_errs_exp = time.time()            
            memory = X_binned.nbytes + yy.nbytes + w.nbytes
            ratio = memory / self.CUDA_MAX_MEMORY_PER_CALL
            if ratio < 1.0:
                ratio = 1.0
            n_calls, call_ranges = FastRealBoostBins.prepare_cuda_call_ranges(m, int(np.ceil(ratio)))
            streams = []            
            for _ in range(min(self.cuda_n_streams, n_calls)):
                streams.append(cuda.stream()) 
            dev_errs_exp = cuda.to_device(np.zeros(n, dtype=np.float32))
            tpb = self.cuda_tpb_default              
            with cuda.pinned(X_binned, yy, w):                                
                for i in range(n_calls):     
                    stream = streams[i % self.cuda_n_streams]
                    call_slice = slice(call_ranges[i], call_ranges[i + 1])
                    X_binned_sub = X_binned[call_slice]           
                    dev_X_binned_sub = cuda.to_device(X_binned_sub, stream=stream)
                    dev_yy_sub = cuda.to_device(yy[call_slice], stream=stream)
                    dev_w_sub = cuda.to_device(w[call_slice], stream=stream)
                    bpg = ((X_binned_sub.shape[0] + tpb - 1) // tpb, n)
                    FastRealBoostBins.errs_exp_numba_cuda[bpg, tpb, stream](dev_X_binned_sub, dev_yy_sub, dev_w_sub, dev_logits, dev_errs_exp, dev_mutexes)
                cuda.synchronize()
            t2_errs_exp = time.time()
            if self.debug_verbose:
                print(f"[errs_exp_numba_cuda done; n_calls: {n_calls}; time: {t2_errs_exp - t1_errs_exp} s]")

            t1_argmin_errs_exp = time.time()                                    
            best_err_exp = np.inf * np.ones(1, dtype=np.float32)
            best_j = -1 * np.ones(1, dtype=np.int32)
            best_logits = np.zeros(self.B_, dtype=np.float32)
            dev_best_err_exp = cuda.to_device(best_err_exp)
            dev_best_j = cuda.to_device(best_j)
            dev_best_logits = cuda.to_device(best_logits)
            tpb = self.cuda_tpb_default
            bpg = (n + tpb - 1) // tpb
            FastRealBoostBins.argmin_errs_exp_numba_cuda[bpg, tpb](dev_errs_exp, dev_logits, dev_best_err_exp, dev_best_j, dev_best_logits, dev_mutexes)
            dev_best_err_exp.copy_to_host(ary=best_err_exp)
            dev_best_j.copy_to_host(ary=best_j)
            dev_best_logits.copy_to_host(ary=best_logits)
            cuda.synchronize()
            self.features_indexes_[t] = best_j[0]
            self.logits_[t] = best_logits
            t2_argmin_errs_exp = time.time()
            if self.debug_verbose:
                print(f"[argmin_errs_exp_numba_cuda done; time: {t2_argmin_errs_exp - t1_argmin_errs_exp} s]")

            t1_reweight = time.time()                
            memory = X_binned.nbytes + yy.nbytes + w.nbytes
            ratio = memory / self.CUDA_MAX_MEMORY_PER_CALL
            if ratio < 1.0:
                ratio = 1.0
            n_calls, call_ranges = FastRealBoostBins.prepare_cuda_call_ranges(m, int(np.ceil(ratio)))
            streams = []            
            for _ in range(min(self.cuda_n_streams, n_calls)):
                streams.append(cuda.stream())
            tpb = self.cuda_tpb_default          
            with cuda.pinned(X_binned, yy, w):
                for i in range(n_calls):
                    stream = streams[i % self.cuda_n_streams]
                    call_slice = slice(call_ranges[i], call_ranges[i + 1])
                    X_binned_sub = X_binned[call_slice]           
                    dev_X_binned_sub = cuda.to_device(X_binned_sub, stream=stream)
                    dev_yy_sub = cuda.to_device(yy[call_slice], stream=stream)
                    dev_w_sub = cuda.to_device(w[call_slice], stream=stream)
                    bpg = (X_binned_sub.shape[0] + tpb - 1) // tpb
                    FastRealBoostBins.reweight_numba_cuda[bpg, tpb, stream](dev_X_binned_sub, dev_yy_sub, dev_w_sub, dev_best_j, dev_best_err_exp, dev_best_logits)
                    dev_w_sub.copy_to_host(ary=w[call_slice], stream=stream)
                cuda.synchronize()
            t2_reweight = time.time()
            if self.debug_verbose:
                print(f"[reweight_numba_cuda done; n_calls: {n_calls}; time: {t2_reweight - t1_reweight} s]")            
            if self.verbose:
                print(f"[best_j: {best_j[0]}, best_err_exp: {best_err_exp[0]:.8f}, best_logits: {best_logits}]")
                        
        t2_loop = time.time()
        if self.verbose:
            print(f"[main boosting loop done; time: {t2_loop - t1_loop} s]")        
        
        self.mins_selected_ = self.mins_[self.features_indexes_]
        self.maxes_selected_ = self.maxes_[self.features_indexes_]
        
        t2 = time.time()
        if self.verbose:
            print(f"FIT DONE. [fit_numba_cuda; time: {t2 - t1} s]")      

    @staticmethod
    @cuda.jit(void(int8[:, :], int8[:], float32[:], float32[:, :], float32[:, :], int32[:, :]))
    def bin_add_weights_numba_cuda(X_binned_sub, yy_sub, w_sub, W_p, W_n, mutexes):        
        shared_w_p = cuda.shared.array((128, 32), dtype=float32) # assumed max constants for shared memory: 128 - subsample size (equal to self.cuda_tpb_bin_add_weights), 32 - no. of bins
        shared_w_n = cuda.shared.array((128, 32), dtype=float32) 
        m, _ = X_binned_sub.shape        
        j = cuda.blockIdx.y
        tpb = cuda.blockDim.x
        tx = cuda.threadIdx.x
        i = cuda.blockIdx.x * tpb + tx # local data point index within current data sub (not global)
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
            lock(mutexes[j]) 
            for b in range(B):                                 
                W_p[j, b] += shared_w_p[0, b]
                W_n[j, b] += shared_w_n[0, b]                                     
            unlock(mutexes[j])
    
    @staticmethod
    @cuda.jit(void(float32[:, :], float32[:, :], float32[:, :]))
    def logits_numba_cuda(W_p, W_n, logits):
        j = cuda.blockIdx.x
        b = cuda.threadIdx.x
        W_p_j_b = W_p[j, b]
        W_n_j_b = W_n[j, b] 
        if W_p_j_b == W_n_j_b:
            logits[j, b] = float32(0.0)
        elif W_p_j_b == float32(0.0):
            logits[j, b] = float32(-2.0) # equal to -LOGIT_MAX
        elif W_n_j_b == float32(0.0):
            logits[j, b] = float32(2.0) # equal to +LOGIT_MAX
        else:
            temp = 0.5 * math.log(W_p_j_b / W_n_j_b)
            if temp > float32(2.0):
                temp = float32(2.0)
            elif temp < float32(-2.0):
                temp = float32(-2.0)
            logits[j, b] = temp                          

    @staticmethod
    @cuda.jit(void(int8[:, :], int8[:], float32[:], float32[:, :], float32[:], int32[:, :]))
    def errs_exp_numba_cuda(X_binned_sub, yy_sub, w_sub, logits, errs_exp, mutexes):
        shared_errs_exp = cuda.shared.array((512), dtype=float32) # assumed max constant: 512 - subsample size                
        m, _ = X_binned_sub.shape                                                
        tpb = cuda.blockDim.x
        tx = cuda.threadIdx.x
        i = cuda.blockIdx.x * tpb + tx
        j = cuda.blockIdx.y
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
            lock(mutexes[j])
            errs_exp[j] += shared_errs_exp[0]
            unlock(mutexes[j])

    @staticmethod
    @cuda.jit(void(float32[:], float32[:, :], float32[:], int32[:], float32[:], int32[:, :]))
    def argmin_errs_exp_numba_cuda(errs_exp, logits, best_err_exp, best_j, best_logits, mutexes):
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
            lock(mutexes[0])
            if shared_errs_exp[0] < best_err_exp[0]:
                best_err_exp[0] = shared_errs_exp[0]
                the_best_j = shared_best_j[0]
                best_j[0] = the_best_j
                for b in range(best_logits.size):
                    best_logits[b] = logits[the_best_j, b]
            unlock(mutexes[0])

    @staticmethod
    @cuda.jit(void(int8[:, :], int8[:], float32[:], int32[:], float32[:], float32[:]))
    def reweight_numba_cuda(X_binned_sub, yy_sub, w_sub, best_j, best_exp_err, best_logits):           
        m, _ = X_binned_sub.shape
        i = cuda.grid(1)        
        if i < m:
            w_sub[i] = w_sub[i] * math.exp(-yy_sub[i] * best_logits[X_binned_sub[i, best_j[0]]]) / best_exp_err[0]

    def decrease_T(self, T):
        self.T_ = T
        self.features_indexes_ = self.features_indexes_[:T]
        self.logits_ = self.logits_[:T]
        self.mins_selected_ = self.mins_selected_[:T]
        self.maxes_selected_ = self.maxes_selected_[:T]

    def decision_function(self, X):
        return self.decision_function_attr(X)
        
    def decision_function_numpy(self, X):
        X_selected = X[:, self.features_indexes_]
        X_binned = np.clip(np.int8(self.B_ * (X_selected - self.mins_selected_) // (self.maxes_selected_ - self.mins_selected_)), 0, self.B_ - 1)
        m = X_binned.shape[0]
        responses = np.zeros(m)
        for i in range(m):
            responses[i] = np.sum(self.logits_[np.arange(self.T_), X_binned[i]])     
        return responses

    def decision_function_numba_jit(self, X):
        X_selected = X[:, self.features_indexes_]
        X_binned = np.clip(np.int8(self.B_ * (X_selected - self.mins_selected_) // (self.maxes_selected_ - self.mins_selected_)), 0, self.B_ - 1)
        return FastRealBoostBins.decision_function_numba_jit_job(X_binned, self.logits_)
    
    @staticmethod
    @jit(float32[:](int8[:, :], float32[:, :]), nopython=True, cache=True)
    def decision_function_numba_jit_job(X_binned, logits):           
        m, T = X_binned.shape
        responses = np.zeros(m, dtype=np.float32)
        for i in range(m):
            for t in range(T):  
                responses[i] += logits[t, X_binned[i, t]]
        return responses
    
    def decision_function_numba_cuda(self, X):
        X_selected = X[:, self.features_indexes_]
        if not X_selected.data.c_contiguous: 
            X_selected = np.ascontiguousarray(X_selected)
        m = X_selected.shape[0]
        dev_X_selected = cuda.to_device(X_selected)
        dev_mins_selected = cuda.to_device(self.mins_selected_)
        dev_maxes_selected = cuda.to_device(self.maxes_selected_)
        dev_logits = cuda.to_device(self.logits_)                
        dev_responses = cuda.device_array(m, dtype=np.float32)
        tpb = self.cuda_tpb_default
        bpg = m
        FastRealBoostBins.decision_function_numba_cuda_job[bpg, tpb](dev_X_selected, dev_mins_selected, dev_maxes_selected, dev_logits, dev_responses)
        cuda.synchronize()
        responses = dev_responses.copy_to_host()        
        return responses
    
    @staticmethod
    @cuda.jit(void(int16[:, :], int16[:], int16[:], float32[:, :], float32[:]))
    def decision_function_numba_cuda_job(X_selected, mins_selected, maxes_selected, logits, responses):
        shared_logits = cuda.shared.array(4096, dtype=float32) # 4096 - assumed limit of features used at detection stage           
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
                b = int8(B * (X_selected[i, t] - mins_selected[t]) / float32(maxes_selected[t] - mins_selected[t]))
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
        return self.class_labels_[1 * (self.decision_function(X) > 0.0)]