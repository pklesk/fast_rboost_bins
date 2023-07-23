from frbb import FastRealBoostBins
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import time

if __name__ == "__main__":  
    T = 128
    clfs = [
        AdaBoostClassifier(n_estimators=T),
        FastRealBoostBins(T=T, fit_mode="numba_jit", decision_function_mode="numba_jit"),
        FastRealBoostBins(T=T, fit_mode="numba_cuda", decision_function_mode="numba_cuda")
        ]
    n = 1000
    np.random.seed(0) # setting some randomization seed
    for m in [1000, 10000]:
        print(f"DATA SHAPE (TRAIN AND TEST): {m} x {n}")
        # generating fake random train data
        X_train = np.random.rand(m, n)
        y_train = np.random.randint(2, size=m)
        X_test = np.random.rand(m, n)
        y_test = np.random.randint(2, size=m)
        # checking classifiers    
        for clf in clfs:
            t1_fit = time.time()      
            clf.fit(X_train, y_train)
            t2_fit = time.time()
            t1_predict_train = time.time()
            acc_train = clf.score(X_train, y_train)
            t2_predict_train = time.time()
            t1_predict_train = time.time()
            acc_train = clf.score(X_train, y_train)
            t2_predict_train = time.time()            
            t1_predict_test = time.time()
            acc_test = clf.score(X_train, y_train)
            t2_predict_test = time.time()
            print(f"  CLF: {clf}")            
            print(f"    ACCs -> TRAIN {clf.score(X_train, y_train)}, TEST: {clf.score(X_test, y_test)}")            
            print(f"    TIMES [s] -> FIT: {t2_fit - t1_fit}, PREDICT (TRAIN): {t2_predict_train - t1_predict_train}, PREDICT (TEST): {t2_predict_test - t1_predict_test}")            