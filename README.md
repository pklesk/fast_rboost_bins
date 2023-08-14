# FastRealBoostBins: An ensemble classifier for fast predictions implemented in Python using numba.jit and numba.cuda
<table>
   <tr>
        <td><img src="/extras/fig_experiment_real_1903270360_20230625_fddb-patches_time_predict_test.png"/></td>
        <td><br/><a href="https://github.com/pklesk/fast_rboost_bins/assets/23095311/bbcd05d0-24f6-49cf-be5e-210a71d2595c"><img src="/extras/screenshot_video_1_geforce_rtx_3090__1280_960.jpg"/></td>
        <td><br/><a href="https://github.com/pklesk/fast_rboost_bins/assets/23095311/df08ca75-2cc2-4608-bcbd-e7019134030c"><img src="/extras/screenshot_video_2_quadro_m4000m__1280_960.jpg"/></td>
    </tr>
</table>

Taking advantage of [Numba](https://numba.pydata.org/) (a high-performance just-in-time Python compiler) 
we provide a fast operating implementation of a boosting algorithm in which *bins* with *logit* transform values play the role of "weak learners".

The software comes as a Python class compliant with [scikit-learn](https://scikit-learn.org) library.
It allows to choose between CPU and GPU computations for each of the two stages: fit and predict (decision function). 
The efficiency of implementation has been confirmed on large data sets where the total of array entries (sample
size $\times$ features count) was of order 10<sup>10</sup> at fit stage and 10<sup>8</sup> at predict stage.
In case of GPU-based fit, the main boosting loop is designed as five CUDA kernels responsible for: 
weights binning, computing logits, computing exponential errors, finding the error minimizer, and examples reweighting. 
The GPU-based predict is computed by a single CUDA kernel. We apply suitable *reduction* patterns and *mutexes*
to carry out summations and 'argmin' operations. 

To test the predict stage performance we compare `FastRealBoostBins` against state-of-the-art classifiers from `sklearn.ensemble` 
using large data sets and focusing on response times. In an additional experiment, we make our
classifiers operate as object detectors under heavy computational load (over 60k queries per a video frame using ensembles of size 2048).

## Installation
```bash
pip install frbb
```
Note: for further usage, NVIDIA CUDA drivers must be present in the operating system.

## Example usage
With `frbb` module installed, one can write e.g.:
```python
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
```
Running the script above produces the following output:
```bash
CLF: FastRealBoostBins(T=256, B=8, outliers_ratio=0.05, logit_max: 2.0, fit_mode='numba_cuda', decision_function_mode='numba_cuda')
TRAIN ACC: 1.0
TEST ACC: 0.958041958041958
```

## Simple example of time performance
The code below compares `sklearn.ensemble.AdaBoostClassifier` against two instances of `FastRealBoostBins` (with different fit / predict modes) on two random data sets, using ensembles of size 1024.
```python
from frbb import FastRealBoostBins
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import time

if __name__ == "__main__":  
    T = 1024
    clfs = [
        AdaBoostClassifier(n_estimators=T),
        FastRealBoostBins(T=T, fit_mode="numba_jit", decision_function_mode="numba_jit"),
        FastRealBoostBins(T=T, fit_mode="numba_cuda", decision_function_mode="numba_cuda")
        ]
    n = 1000
    np.random.seed(0) # setting some randomization seed
    for m in [1000, 10000]:
        print(f"DATA SHAPE (TRAIN AND TEST): {m} x {n}")
        # generating fake random data
        X_train = np.random.rand(m, n)
        y_train = np.random.randint(2, size=m)
        X_test = np.random.rand(m, n)
        y_test = np.random.randint(2, size=m)
        # iterating over classifiers    
        for clf in clfs:
            print(f"  CLF: {clf}...")
            t1_fit = time.time()      
            clf.fit(X_train, y_train)
            t2_fit = time.time()
            t1_predict_train = time.time()
            acc_train = clf.score(X_train, y_train)
            t2_predict_train = time.time()
            t1_predict_test = time.time()
            acc_test = clf.score(X_train, y_train)
            t2_predict_test = time.time()                        
            print(f"    ACCs -> TRAIN {clf.score(X_train, y_train)}, TEST: {clf.score(X_test, y_test)}")            
            print(f"    TIMES [s] -> FIT: {t2_fit - t1_fit}, PREDICT (TRAIN): {t2_predict_train - t1_predict_train}, PREDICT (TEST): {t2_predict_test - t1_predict_test}")            
```
And produces the following output (actual observed times are machine dependent but their proportions should be roughly preserved):
```bash
DATA SHAPE (TRAIN AND TEST): 1000 x 1000
  CLF: AdaBoostClassifier(n_estimators=1024)...
    ACCs -> TRAIN 1.0, TEST: 0.491
    TIMES [s] -> FIT: 68.10534024238586, PREDICT (TRAIN): 0.9158439636230469, PREDICT (TEST): 0.9115560054779053
  CLF: FastRealBoostBins(T=1024, B=8, outliers_ratio=0.05, logit_max: 2.0, fit_mode='numba_jit', decision_function_mode='numba_jit')...
    ACCs -> TRAIN 1.0, TEST: 0.521
    TIMES [s] -> FIT: 9.541089534759521, PREDICT (TRAIN): 0.008351802825927734, PREDICT (TEST): 0.008364677429199219
  CLF: FastRealBoostBins(T=1024, B=8, outliers_ratio=0.05, logit_max: 2.0, fit_mode='numba_cuda', decision_function_mode='numba_cuda')...
    ACCs -> TRAIN 1.0, TEST: 0.521
    TIMES [s] -> FIT: 6.236358404159546, PREDICT (TRAIN): 0.004852771759033203, PREDICT (TEST): 0.0038003921508789062
DATA SHAPE (TRAIN AND TEST): 10000 x 1000
  CLF: AdaBoostClassifier(n_estimators=1024)...
    ACCs -> TRAIN 0.8686, TEST: 0.4957
    TIMES [s] -> FIT: 851.2353613376617, PREDICT (TRAIN): 12.599870443344116, PREDICT (TEST): 12.600045204162598
  CLF: FastRealBoostBins(T=1024, B=8, outliers_ratio=0.05, logit_max: 2.0, fit_mode='numba_jit', decision_function_mode='numba_jit')...
    ACCs -> TRAIN 0.8535, TEST: 0.5054
    TIMES [s] -> FIT: 103.08070993423462, PREDICT (TRAIN): 0.11068463325500488, PREDICT (TEST): 0.11023759841918945
  CLF: FastRealBoostBins(T=1024, B=8, outliers_ratio=0.05, logit_max: 2.0, fit_mode='numba_cuda', decision_function_mode='numba_cuda')...
    ACCs -> TRAIN 0.8553, TEST: 0.5122
    TIMES [s] -> FIT: 16.11052441596985, PREDICT (TRAIN): 0.07710742950439453, PREDICT (TEST): 0.07628917694091797
```

## Documentation
Complete developer documentation of the project is accessible at: [https://pklesk.github.io/fast_rboost_bins](https://pklesk.github.io/fast_rboost_bins). <br/>
Documentation for the `FastRealBoostBins` class alone is at: [https://pklesk.github.io/fast_rboost_bins/frbb.html](https://pklesk.github.io/fast_rboost_bins/frbb.html).

## Constructor parameters
| parameter                      | description                                                                                                                                       |
|:-------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------|
| `T (int)`                      | number of boosting rounds (equivalently, number of weak estimators), defaults to `256`                                                            |
| `B (int)`                      | number of bins, defaults to `8`                                                                                                                   |
| `outliers_ratio (float)`       | fraction of outliers to skip (on each end) when establishing features’ variability ranges, defaults to `0.05`                                     |
| `logit_max (np.float32)`       | maximum absolute value of logit transform, outcomes clipped to interval [−`logit_max`, `logit_max`], defaults to `np.float32(2.0)`                |
| `fit_mode (str)`               | choice of fit method from {`"numpy"`, `"numba_jit"`, `"numba_cuda"`}, defaults to `"numba_cuda"`                                                  |
| `decision_function_mode (str)` | choice of decision function method from {`"numpy"`, `"numba_jit"`, `"numba_cuda"`} (called e.g. within `predict`), defaults to `"numba_cuda"`     |
| `verbose (bool)`               | verbosity flag, if `True` then fit progress and auxiliary information are printed to console, defaults to `False`                                 |
| `debug_verbose (bool)`         | detailed verbosity (only for `"numba_cuda"` fit), defaults to `False`                                                                             |

## Estimated attributes
| attribute                                      | description                                                                                                                                           |
|:-----------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------|
| `features_selected_ (ndarray[np.int32])`       | indexes of selected features, array of shape `(T,)`                                                                                                   |
| `dtype_ (np.dtype)`                            | type of input data array, one of {`np.int8`, `np.uint8`, …, `np.int64`,` np.uint64`} or {`np.float32`, `np.float64`} - numeric types  only allowed    | 
| `mins_selected_ (ndarray[dtype_])`             | left ends of ranges for selected features, array of shape `(T,)`                                                                                      |
| `maxes_selected_ (ndarray[dtype_])`            | right ends of ranges for selected features, array of shape `(T,)`                                                                                     |
| `logits_ (ndarray[np.float32])`                | binned logit values for selected features, array of shape `(T, B)`                                                                                    |
| `decision_function_numba_cuda_job_name_ (str)` | name, implied by `dtype_`, of decision function to be called in case of `numba_cuda` mode (e.g.`_decision_function_numba_cuda_job_int16)`             |
| `decision_threshold_ (float)`                  | threshold value used inside `predict` function, defaults to `0.0`                                                                                     |
| `classes_ (ndarray)`                           | original class labels (scikit-learn requirement)                                                                                                      |
| `n_features_in_ (int)`                         | number of features registered in ``fit`` call and expected for subsequent ``predict`` calls (scikit-learn requirement).                              |
                   

## Selected experimental results

### Comparison against state-of-the-art classifiers from `sklearn.ensemble`

In tables below we write for shortness `FastRealBoostBins("numba_jit")` which, in fact, stands for `FastRealBoostBins(fit_mode="numba_jit", decision_function_mode="numba_jit")`,
and `FastRealBoostBins("numba_cuda")` stands for `FastRealBoostBins(fit_mode="numba_cuda", decision_function_mode="numba_cuda")`.
To have approximately equal conditions for comparison (e.g. same number of weak learners, each learner based on 1 feature), we forced the following settings on classifiers
from `sklearn.ensemble`:
```python
AdaBoostClassifier(algorithm="SAMME.R", max_depth=1, n_estimators=T)
GradientBoostingClassifier(max_depth=1, n_estimators=T)
HistGradientBoostingClassifier(early_stopping=False, max_iter=T, max_bins=B)
```
where `T` (ensemble size) and `B` (number of bins) represent the values simultaneously imposed on `FastRealBoostBins` instances.

Hardware environment: Intel(R) Core(TM) i7-10700 CPU @ 2.90GHz, 125.7 GB RAM, NVIDIA GeForce RTX 3090 GPU.<br/>
Software environment: Linux 5.15.0-71-generic, Python 3.8.10, GCC 9.4.0, numpy 1.22.3, numba 0.57.0, sklearn 1.0.2, cv2 4.6.0, nvcc 11.7.

Results presented below pertain to three selected data sets named: 'FDDB-PATCHES (3NPI)', 'MNIST-B (DIGIT 0)', and 'HaGRID-HFs (PALM, 10NPI)';
generated based on known image databases: [FDDB](http://vis-www.cs.umass.edu/fddb), [MNIST](https://ieeexplore.ieee.org/document/6296535), and [HaGRID](https://github.com/hukenovs/hagrid) respectively.
More results and more details on these and other data sets can be found in the research paper associated with the project.

#### 'FDDB-PATCHES (3NPI)', T=1024: &nbsp;&nbsp; train data: (11 775 $\times$ 3 072) ~ 10<sup>7.6</sup>, &nbsp; test data: (1 303 $\times$ 3 072 $\rightarrow$ 1 024) ~ 10<sup>6.1</sup>
| classifier                        | fit time [s] | fit speedup $\approx$ | predict time [s]         | predict speedup $\approx$ | acc [%]     | predict time [s] | predict speedup $\approx$ | acc [%]    |
|:----------------------------------|-------------:|----------------------:|-------------------------:|--------------------------:|------------:|-----------------:|--------------------------:|-----------:|
|                                   |  **(train)** |           **(train)** |              **(train)** |               **(train)** | **(train)** |       **(test)** |                **(test)** | **(test)** |
| `AdaBoostClassifier`              |        1 421 |            $\times$ 1 |                   33.285 |                $\times$ 1 |       99.94 |            2.702 |                $\times$ 1 |      89.56 |
| `GradientBoostingClassifier`      |        1 341 |            $\times$ 1 |                    0.191 |              $\times$ 174 |       94.04 |            0.013 |              $\times$ 208 |      90.18 |
| `HistGradientBoostingClassifier`  |           14 |          $\times$ 102 |                    0.288 |              $\times$ 116 |       93.17 |            0.030 |               $\times$ 90 |      90.02 |
| `FastRealBoostBins("numba_jit")`  |          395 |            $\times$ 4 |                    0.096 |              $\times$ 347 |       99.98 |            0.009 |              $\times$ 300 |      88.41 |
| `FastRealBoostBins("numba_cuda")` |           43 |           $\times$ 33 |                    0.068 |              $\times$ 489 |       99.97 |            0.003 |              $\times$ 901 |      88.33 |

|fit times along growing T|predict times (test) along growing T|
|-|-|
|<img src="/extras/fig_experiment_real_1903270360_20230625_fddb-patches_time_fit.png"/>|<img src="/extras/fig_experiment_real_1903270360_20230625_fddb-patches_time_predict_test.png"/>|

#### 'MNIST-B (DIGIT 0)', T=1024: &nbsp;&nbsp; train data: (60 000 $\times$ 784) ~ 10<sup>7.7</sup>, &nbsp; test data: (10 000 $\times$ 784 $\rightarrow$ 1 024) ~ 10<sup>7.0</sup>
| classifier                        | fit time [s] | fit speedup $\approx$ | predict time [s]         | predict speedup $\approx$ | acc [%]     | predict time [s] | predict speedup $\approx$ | acc [%]    |
|:----------------------------------|-------------:|----------------------:|-------------------------:|--------------------------:|------------:|-----------------:|--------------------------:|-----------:|
|                                   |  **(train)** |           **(train)** |              **(train)** |               **(train)** | **(train)** |       **(test)** |                **(test)** | **(test)** |
| `AdaBoostClassifier`              |        1 754 |            $\times$ 1 |                   44.398 |                $\times$ 1 |      100.00 |            6.013 |                $\times$ 1 |      98.69 |
| `GradientBoostingClassifier`      |        1 641 |            $\times$ 1 |                    0.665 |              $\times$  67 |       99.13 |            0.089 |               $\times$ 68 |      98.85 |
| `HistGradientBoostingClassifier`  |            9 |          $\times$ 195 |                    0.457 |              $\times$  97 |       98.99 |            0.072 |               $\times$ 84 |      98.89 |
| `FastRealBoostBins("numba_jit")`  |          510 |            $\times$ 3 |                    0.393 |              $\times$ 113 |      100.00 |            0.036 |              $\times$ 167 |      98.50 |
| `FastRealBoostBins("numba_cuda")` |           53 |           $\times$ 33 |                    0.257 |              $\times$ 173 |      100.00 |            0.018 |              $\times$ 334 |      98.41 |

|fit times along growing T|predict times (test) along growing T|
|-|-|
|<img src="/extras/fig_experiment_real_2001519960_20230626_mnist-b_time_fit.png"/>|<img src="/extras/fig_experiment_real_2001519960_20230626_mnist-b_time_predict_test.png"/>|

#### 'HaGRID-HFs (PALM, 10NPI)', T=2048: &nbsp;&nbsp; train data: (232 299 $\times$ 18 225) ~ 10<sup>9.6</sup>, &nbsp; test data: (77 433 $\times$ 18 225 $\rightarrow$ 2 048) ~ 10<sup>8.2</sup>
| classifier                        | fit time [s] | fit speedup $\approx$ | predict time [s]         | predict speedup $\approx$ | acc [%]     | predict time [s] | predict speedup $\approx$ | acc [%]    |
|:----------------------------------|-------------:|----------------------:|-------------------------:|--------------------------:|------------:|-----------------:|--------------------------:|-----------:|
|                                   |  **(train)** |           **(train)** |              **(train)** |               **(train)** | **(train)** |       **(test)** |                **(test)** | **(test)** |
| `HistGradientBoostingClassifier`  |        2 662 |          $\times$ 3.3 |                   13.667 |              $\times$ 1.0 |       98.71 |            4.493 |              $\times$ 1.0 |      98.65 |
| `FastRealBoostBins("numba_cuda")` |        8 908 |          $\times$ 1.0 |                    5.060 |              $\times$ 2.7 |      100.00 |            1.277 |              $\times$ 3.5 |      99.07 |

|fit times along growing T|predict times (test) along growing T|
|-|-|
|<img src="/extras/fig_experiment_real_1178284368_20230627_hagrid-hfs-10_time_fit.png"/>|<img src="/extras/fig_experiment_real_1178284368_20230627_hagrid-hfs-10_time_predict_test.png"/>|

More experiments on time performance can be carried out using the script `main_experimenter.py` with command line interface (see next section).

### Script for experiments: `main_experimenter.py` 
By executing `python main_experimenter.py -h` (or `--help`) one obtains help on the script arguments:
```bash
"FAST-REAL-BOOST-BINS": AN ENSEMBLE CLASSIFIER FOR FAST PREDICTIONS IMPLEMENTED IN PYTHON VIA NUMBA.JIT AND NUMBA.CUDA. [main_experimenter]
[for help use -h or --help switch]
CLASSIFIERS DEFINITIONS:
[(<class 'sklearn.ensemble._weight_boosting.AdaBoostClassifier'>, {'algorithm': 'SAMME.R'}, {'color': 'black'}),
 (<class 'sklearn.ensemble._gb.GradientBoostingClassifier'>, {'max_depth': 1}, {'color': 'green'}),
 (<class 'sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingClassifier'>, {'max_depth': 1, 'early_stopping': False}, {'color': 'orange'}),
 (<class 'frbb.FastRealBoostBins'>, {'fit_mode': 'numba_jit', 'decision_function_mode': 'numba_jit'}, {'color': 'blue'}),
 (<class 'frbb.FastRealBoostBins'>, {'fit_mode': 'numba_cuda', 'decision_function_mode': 'numba_cuda'}, {'color': 'red'})]
REAL DATA DEFINITIONS:
[('fddb-patches', 'read_data_fddb_patches', 'FDDB-PATCHES (3NPI)'),
 ('cifar-10', 'read_data_cifar_10', 'CIFAR-10 (AIRPLANE)'),
 ('mnist-b', 'read_data_mnist_b', 'MNIST-B (DIGIT 0)'),
 ('fddb-hfs-100', 'read_data_fddb_haar_npi_100', 'FDDB-HFs (100NPI)'),
 ('fddb-hfs-300', 'read_data_fddb_haar_npi_300', 'FDDB-HFs (300NPI)'),
 ('hagrid-hfs-10', 'read_data_hagrid_haar_npi_10', 'HaGRID-HFs (PALM, 10NPI)'),
 ('hagrid-hfs-30', 'read_data_hagrid_haar_npi_30', 'HaGRID-HFs (PALM, 30NPI)')]
RANDOM DATA SIZES DEFINABLE VIA -nmm (--NMM_MAGN_ORDERS) OPTION.
usage: main_experimenter.py [-h] [-cf CLFS_FLAGS [CLFS_FLAGS ...]] [-dk {real,random}] [-rdf REAL_DATA_FLAGS [REAL_DATA_FLAGS ...]]
                            [-rd {<class 'numpy.int8'>,<class 'numpy.uint8'>,<class 'numpy.int16'>,<class 'numpy.uint16'>,<class 'numpy.int32'>,<class 'numpy.uint32'>,<class 'numpy.int64'>,<class 'numpy.uint64'>,<class 'numpy.float32'>,<class 'numpy.float64'>}]
                            [-nmm NMM_MAGN_ORDERS [NMM_MAGN_ORDERS ...]] [-ts TS [TS ...]] [-bs BS [BS ...]] [-s SEED] [-p PLOTS] [-pan {T,B,n,m_train,m_test}]
                            [-pvn {acc_test,acc_train,time_fit,time_predict_train,time_predict_test} [{acc_test,acc_train,time_fit,time_predict_train,time_predict_test} ...]]

optional arguments:
  -h, --help            show this help message and exit
  -cf CLFS_FLAGS [CLFS_FLAGS ...], --CLFS_FLAGS CLFS_FLAGS [CLFS_FLAGS ...]
                        boolean flags (list) specifying which classifiers from the predefined set will participate in experiments (default: [True, False, False, True, True]) (attention: type them using spaces as separators)
  -dk {real,random}, --DATA_KIND {real,random}
                        kind of data on which to experiment (default: random)
  -rdf REAL_DATA_FLAGS [REAL_DATA_FLAGS ...], --REAL_DATA_FLAGS REAL_DATA_FLAGS [REAL_DATA_FLAGS ...]
                        boolean flags (list) specifying which data sets from the predefined set will participate in experiments on real data (default: [True, False, False, False, False, False, False]) (attention: type them
                        using spaces as separators)
  -rd {<class 'numpy.int8'>,<class 'numpy.uint8'>,<class 'numpy.int16'>,<class 'numpy.uint16'>,<class 'numpy.int32'>,<class 'numpy.uint32'>,<class 'numpy.int64'>,<class 'numpy.uint64'>,<class 'numpy.float32'>,<class 'numpy.float64'>}, --RANDOM_DTYPE {<class 'numpy.int8'>,<class 'numpy.uint8'>,<class 'numpy.int16'>,<class 'numpy.uint16'>,<class 'numpy.int32'>,<class 'numpy.uint32'>,<class 'numpy.int64'>,<class 'numpy.uint64'>,<class 'numpy.float32'>,<class 'numpy.float64'>}
                        dtype of input numpy arrays for experiments on random data (default: <class 'numpy.int8'>) (attention: please type it as e.g. 'np.uint8', 'np.float32', etc.)
  -nmm NMM_MAGN_ORDERS [NMM_MAGN_ORDERS ...], --NMM_MAGN_ORDERS NMM_MAGN_ORDERS [NMM_MAGN_ORDERS ...]
                        list of tuples represented as strings defining orders of magnitude for random input arrays in experiments on random data (default: [(3, 4, 4)]) (attention: type them using spaces as separators and with
                        each tuple in quotation marks)
  -ts TS [TS ...], --TS TS [TS ...]
                        ensemble sizes (list) to impose on each type of classifier in experiments (default: [16, 32, 64]) (attention: type them using spaces as separators)
  -bs BS [BS ...], --BS BS [BS ...]
                        bins counts (list) to impose on each type of classifier in experiments (default: [8]) (attention: type them using spaces as separators)
  -s SEED, --SEED SEED  randomization seed, (default: 0)
  -p PLOTS, --PLOTS PLOTS
                        boolean flag indicating if plots should be generated after experiments (default: False)
  -pan {T,B,n,m_train,m_test}, --PLOTS_ARG_NAME {T,B,n,m_train,m_test}
                        name of argument quantity to be placed on horizontal axis in plots (default: T)
  -pvn {acc_test,acc_train,time_fit,time_predict_train,time_predict_test} [{acc_test,acc_train,time_fit,time_predict_train,time_predict_test} ...], --PLOTS_VALUES_NAMES {acc_test,acc_train,time_fit,time_predict_train,time_predict_test} [{acc_test,acc_train,time_fit,time_predict_train,time_predict_test} ...]
                        names of value quantities to be placed on vertical axis in plots (default: ['acc_test', 'acc_train', 'time_fit', 'time_predict_train', 'time_predict_test']) (attention: type them using spaces as
                        separators)
```
#### Examples of `main_experimenter.py` usage
```bash
python main_experimenter.py
```
Running the script with no arguments (as shown above) defaults to execution of 3 experiments (for ensembles of sizes 16, 32, 64), 
where `AdaBoostClassifier` and two instances of `FastRealBoostBins` are compared on one random data set 
(10<sup>3</sup> features, train sample: 10<sup>4</sup>, test sample: 10<sup>4</sup>). <br/>
Example full output: [/extras/log_experiment_random_1752355477_20230812.txt](/extras/log_experiment_random_1752355477_20230812.txt).

```bash
python main_experimenter.py -dk random -cf 0 1 1 1 0 -nmm "(2, 4, 3)" "(3, 3, 4)" -t 32 64 128 -b 16
```
Execution of the line above leads to 6 experiments (2 random data sets times 3 ensemble sizes), where classifiers `GradientBoostingClassifier`, 
`HistGradientBoostingClassifier` and `FastRealBoostBins("numba_jit")` (note the 0/1 flags of classifiers indicated by `-cf` option) are compared. 
Sizes of random data sets are defined by `-nmm` option. The first one is defined as: 10<sup>2</sup> features with sample sizes 10<sup>4</sup> (train), 
and 10<sup>3</sup> (test). The second is defined as: 10<sup>3</sup> features with sample sizes 10<sup>3</sup> (train), and 10<sup>4</sup> (test).
Ensemble sizes are defined by `-t` option and the number of bins by `-b` (it also could have been a sequence of numbers, leading to more experiments).
Flags of classifiers can also be specified as `False`/`True` strings. We remark that `-dk random` switch (choosing the data kind) could have been skipped, as `random` is the default selection. <br/>
Example full output: [/extras/log_experiment_random_2426086665_20230813.txt](/extras/log_experiment_random_2426086665_20230813.txt).

```bash
python main_experimenter.py -dk real -cf 1 1 1 1 1 -rdf 1 0 0 0 0 0 0 -t 16 32 64 128 512 1024 -p True
```
This execution leads to 6 experiments (because of 6 ensemble sizes) where all classifiers are trained and tested on a real data set named 'FDDB-PATCHES (3NPI)'.
The `-p True` switch asks for plots to be produced (similar to the ones presented earlier), once the experiments are done. The plots shall be saved in the `/extras/` folder, both as eps and pdf files. <br/>
Example full output: [/extras/log_experiment_real_1903270360_20230625_fddb-patches.txt](/extras/log_experiment_real_1903270360_20230625_fddb-patches.txt).

### Applying `FastRealBoostBins` as an object detector
Owing to efficiency of `FastRealBoostBins`'s decision function, it can be applied even as an object detector working under the expensive regime of a traditional sliding window-based detection procedure.
By that we mean a procedure that scans densly a video frame (at multiple positions and scales) and requests a great number of predictions from a classifier - target or non-target? 
This number depends on frames resolution and other settings, but usually ranges from 10<sup>4</sup> to 10<sup>5</sup>.

To accomplish such an application, one should take advantage of the fact that at predict (detection) stage, it suffices to prepare only the *selected* features of multiple objects (windows to be checked)
for the classifier, once it has been trained. With such a subset of selected features, one can directly call a suitable private function, e.g. `_decision_function_numba_cuda_job_int16` to ask for predictions,
instead of `predict` (the latter expects all features to be passed). Moreover, with GPU/CUDA computations at disposal, the feature extraction can be done fast at GPU device side.

Using [FDDB](http://vis-www.cs.umass.edu/fddb) and [HaGRID](https://github.com/hukenovs/hagrid) data, coupled with Haar-like features (HFs), we trained `FastRealBoostBins` classifiers as detectors of *faces*
and *palm gestures*, respectively. To reduce memory transfers between host and device, constant pieces of information (e.g. coordinates of all windows to be checked, HFs related information) were prepared just
once and placed in device-side arrays prior to an actual video sequence. Below we present example snapshots (click them to see videos) and obtained efficiency measurements from two environments with different 
GPU devices: 1. GeForce RTX 3090 (contemporary, high-performance), 2. Quadro M4000M (older generation).
Full details of environment 1 in a former section. Full details of environment 2 given below.

Hardware environment 2: Intel(R) Xeon(R) CPU E3-1505M v5 @ 2.80GHz, 63.9 GB RAM, NVIDIA Quadro M4000M GPU. <br/>
Software environment 2: Windows 10, Python 3.9.7 [MSC v.1916 64 bit (AMD64)], numpy 1.20.0, numba 0.54.1, sklearn 1.0.2, cv2 4.5.5-dev, nvcc 11.6.

#### Sample videos
<table>    
   <tr>
      <td align="center">environment 1 (GeForce RTX 3090)<br/>64k windows, 2 detectors, T=1024 each</td>
      <td align="center">environment 2 (Quadro M4000M)<br/>22k windows, 2 detectors, T=1024 each</td>
      <td align="center">environment 1 (GeForce RTX 3090)<br/>64k windows, 2 detectors, T=2048 each</td>
   </tr>
   <tr>
      <td><br/><a href="https://github.com/pklesk/fast_rboost_bins/assets/23095311/bbcd05d0-24f6-49cf-be5e-210a71d2595c"><img src="/extras/screenshot_video_1_geforce_rtx_3090__1280_960.jpg"/></td>
      <td><br/><a href="https://github.com/pklesk/fast_rboost_bins/assets/23095311/df08ca75-2cc2-4608-bcbd-e7019134030c"><img src="/extras/screenshot_video_2_quadro_m4000m__1280_960.jpg"/></td>
      <td><br/><a href="https://github.com/pklesk/fast_rboost_bins/assets/23095311/efa212a5-88c7-4aa0-bc43-934d74410a1a"><img src="/extras/screenshot_video_3_geforce_rtx_3090__1280_960.jpg"/></td>
   </tr>
</table>         
<table>
   <tr>
      <td align="center">environment 1 (GeForce RTX 3090)<br/>64k windows, 1 detector, T=1024</td>
      <td align="center">environment 1 (GeForce RTX 3090)<br/>22k windows, 2 detectors, T=1024 each</td>
      <td align="center">environment 1 (GeForce RTX 3090)<br/>22k windows, 2 detectors, T=512 each</td>
   </tr>
   <tr>
      <td><br/><a href="https://github.com/pklesk/fast_rboost_bins/assets/23095311/a7fb77a8-0ed4-456e-9f0f-ed6aebd0e5ba"><img src="/extras/screenshot_video_4_geforce_rtx_3090__1280_960.jpg"/></td>
      <td><br/><a href="https://github.com/pklesk/fast_rboost_bins/assets/23095311/f61d494c-73b8-4565-bbf7-505fa49f20a7"><img src="/extras/screenshot_video_5_geforce_rtx_3090__1280_960.jpg"/></td>
      <td><br/><a href="https://github.com/pklesk/fast_rboost_bins/assets/23095311/39da2aed-c9c6-49ec-abfe-ca89787c5d85"><img src="/extras/screenshot_video_6_geforce_rtx_3090__1280_960.jpg"/></td>
   </tr>           
</table>

#### Environment 1 (GeForce RTX 3090): efficiency of object detectors (averages for 1000 frames)
| windows per frame | detectors | T     | HFs avg. time [ms] | FRBB avg. time [ms] | computations FPS | display FPS |
|------------------:|----------:|------:|-------------------:|--------------------:|-----------------:|------------:|
|            22 278 |         1 |   512 |               3.51 |                1.29 |           122.50 |       30.09 |
|            22 278 |         1 |  1024 |               3.73 |                1.38 |           118.82 |       30.09 |
|            22 278 |         1 |  2048 |               4.42 |                1.62 |           106.99 |       30.09 |
|            22 278 |         2 |   512 |               5.64 |                2.40 |           115.30 |       30.06 |
|            22 278 |         2 |  1024 |               6.07 |                2.60 |           107.03 |       30.09 |
|            22 278 |         2 |  2048 |               6.91 |                2.97 |            95.49 |       30.09 |
|            64 173 |         1 |   512 |               4.72 |                2.71 |            93.66 |       30.12 |
|            64 173 |         1 |  1024 |               5.44 |                2.99 |            87.18 |       30.12 |
|            64 173 |         1 |  2048 |               6.47 |                3.56 |            80.48 |       30.12 |
|            64 173 |         2 |   512 |               7.20 |                5.03 |            78.05 |       30.12 |
|            64 173 |         2 |  1024 |               8.39 |                5.50 |            68.89 |       30.12 |
|            64 173 |         2 |  2048 |              10.14 |                6.20 |            59.22 |       30.12 |

Please note very short response times of `FastRealBoostBins`. For example, 3.56 ms for 64k windows (query objects) using an ensemble of size 2048 (one detector case).
Note that this means computation of responses for a data array with the total of entries (sample size $\times$ features count) of order ~ 10<sup>8.1</sup>.

#### Environment 2 (Quadro M4000M): efficiency of object detectors (averages for 1000 frames)
| windows per frame | detectors | T     | HFs avg. time [ms] | FRBB avg. time [ms] | computations FPS | display FPS |
|------------------:|----------:|------:|-------------------:|--------------------:|-----------------:|------------:|
|            22 278 |         1 |   512 |              17.54 |                9.88 |            34.25 |       23.83 |
|            22 278 |         1 |  1024 |              31.79 |               11.36 |            22.11 |       16.86 |
|            22 278 |         1 |  2048 |              61.96 |               15.99 |            12.48 |       10.88 |
|            22 278 |         2 |   512 |              35.27 |               19.37 |            18.07 |       13.11 |
|            22 278 |         2 |  1024 |              63.84 |               23.17 |            11.40 |        9.28 |
|            22 278 |         2 |  2048 |             124.00 |               33.67 |             6.31 |        5.71 |

### Script for object detection: `main_detector` 
By executing `python main_detector.py -h` (or `--help`) one obtains help on the script arguments:
```bash
"FAST-REAL-BOOST-BINS": AN ENSEMBLE CLASSIFIER FOR FAST PREDICTIONS IMPLEMENTED IN PYTHON VIA NUMBA.JIT AND NUMBA.CUDA. [main_detector]
[for help use -h or --help switch]
usage: main_detector.py [-h] [-k {face,hand}] [-s S] [-p P] [-npi NPI] [-t T] [-b B] [-seed SEED] [-dhfsa] [-dhfss] [-rd] [-form] [-maom] [-adtom] [-ddiv]
                        [-ddivc {gpu_cuda,cpu_simple,cpu_parallel}] [-ddivpj DEMO_DETECT_IN_VIDEO_PARALLEL_JOBS] [-ddivvl] [-ddivvd]
                        [-ddivf DEMO_DETECT_IN_VIDEO_FRAMES] [-ddivmc] [-cv2vcci CV2_VIDEO_CAPTURE_CAMERA_INDEX] [-cv2oim] [-cv2nf] [-ds DETECTION_SCALES]
                        [-dwhm DETECTION_WINDOW_HEIGHT_MIN] [-dwwm DETECTION_WINDOW_WIDTH_MIN] [-dwg DETECTION_WINDOW_GROWTH] [-dwj DETECTION_WINDOW_JUMP]
                        [-ddt DETECTION_DECISION_THRESHOLD] [-dp {None,avg,nms}] [-mccn MC_CLFS_NAMES [MC_CLFS_NAMES ...]]
                        [-mcdt MC_DECISION_THRESHOLDS [MC_DECISION_THRESHOLDS ...]]

optional arguments:
  -h, --help            show this help message and exit
  -k {face,hand}, --KIND {face,hand}
                        detector kind (default: face)
  -s S, --S S           "scales" parameter of Haar-like features (default: 5)
  -p P, --P P           "positions" parameter of Haar-like features (default: 5)
  -npi NPI, --NPI NPI   "negatives per image" parameter, used in procedures generating data sets from images (default: 300 with -k set to face)
  -t T, --T T           number of boosting rounds, (default: 1024)
  -b B, --B B           numbet of bins, (default: 8)
  -seed SEED, --SEED SEED
                        randomization seed, (default: 0)
  -dhfsa, --DEMO_HAAR_FEATURES_ALL
                        turn on demo of all Haar-like features
  -dhfss, --DEMO_HAAR_FEATURES_SELECTED
                        turn on demo of selected Haar-like features
  -rd, --REGENERATE_DATA
                        turn on data regeneration
  -form, --FIT_OR_REFIT_MODEL
                        fit new or refit an existing model
  -maom, --MEASURE_ACCS_OF_MODEL
                        measure accuracies of a model
  -adtom, --ADJUST_DECISION_THRESHOLD_OF_MODEL
                        adjust decision threshold of a model (based on ROC for testing data)
  -ddiv, --DEMO_DETECT_IN_VIDEO
                        turn on demo of detection in video
  -ddivc {gpu_cuda,cpu_simple,cpu_parallel}, --DEMO_DETECT_IN_VIDEO_COMPUTATIONS {gpu_cuda,cpu_simple,cpu_parallel}
                        type of computations for demo of detection in video (default: gpu_cuda)
  -ddivpj DEMO_DETECT_IN_VIDEO_PARALLEL_JOBS, --DEMO_DETECT_IN_VIDEO_PARALLEL_JOBS DEMO_DETECT_IN_VIDEO_PARALLEL_JOBS
                        number of parallel jobs (only in case of cpu_parallel set for -ddivc) (default: 8)
  -ddivvl, --DEMO_DETECT_IN_VIDEO_VERBOSE_LOOP
                        turn on verbosity for main loop of detection in video
  -ddivvd, --DEMO_DETECT_IN_VIDEO_VERBOSE_DETECT
                        turn on detailed verbosity for detection in video
  -ddivf DEMO_DETECT_IN_VIDEO_FRAMES, --DEMO_DETECT_IN_VIDEO_FRAMES DEMO_DETECT_IN_VIDEO_FRAMES
                        limit overall detection in video to given number of frames
  -ddivmc, --DEMO_DETECT_IN_VIDEO_MULTIPLE_CLFS
                        turn on demo of detection in video with multiple classifiers (currently: face and hand detectors)
  -cv2vcci CV2_VIDEO_CAPTURE_CAMERA_INDEX, --CV2_VIDEO_CAPTURE_CAMERA_INDEX CV2_VIDEO_CAPTURE_CAMERA_INDEX
                        video camera index (default: 0)
  -cv2oim, --CV2_VIDEO_CAPTURE_OS_IS_MSWINDOWS
                        indicates that OS is MS Windows (for cv2 and directx purposes)
  -cv2nf, --CV2_VIDEO_CAPTURE_NO_FLIP
                        indicates that no frame flipping is wanted
  -ds DETECTION_SCALES, --DETECTION_SCALES DETECTION_SCALES
                        number of detection scales (default: 9)
  -dwhm DETECTION_WINDOW_HEIGHT_MIN, --DETECTION_WINDOW_HEIGHT_MIN DETECTION_WINDOW_HEIGHT_MIN
                        minimum height of detection window (default: 96)
  -dwwm DETECTION_WINDOW_WIDTH_MIN, --DETECTION_WINDOW_WIDTH_MIN DETECTION_WINDOW_WIDTH_MIN
                        minimum width of detection window (default: 96)
  -dwg DETECTION_WINDOW_GROWTH, --DETECTION_WINDOW_GROWTH DETECTION_WINDOW_GROWTH
                        growth factor of detection window (default: 1.2)
  -dwj DETECTION_WINDOW_JUMP, --DETECTION_WINDOW_JUMP DETECTION_WINDOW_JUMP
                        relative jump of detection window (default: 0.05)
  -ddt DETECTION_DECISION_THRESHOLD, --DETECTION_DECISION_THRESHOLD DETECTION_DECISION_THRESHOLD
                        decision threshold, can be set to None then classifier's internal threshold is used (default: None)
  -dp {None,avg,nms}, --DETECTION_POSTPROCESS {None,avg,nms}
                        type of detection postprocessing (default: avg)
  -mccn MC_CLFS_NAMES [MC_CLFS_NAMES ...], --MC_CLFS_NAMES MC_CLFS_NAMES [MC_CLFS_NAMES ...]
                        classifiers names (list) for detection with multiple classifiers (default: ['clf_frbb_face_n_18225_S_5_P_5_NPI_300_SEED_0_T_1024_B_8.bin',
                        'clf_frbb_hand_n_18225_S_5_P_5_NPI_30_SEED_0_T_1024_B_8.bin']) (attention: type them using spaces as separators)
  -mcdt MC_DECISION_THRESHOLDS [MC_DECISION_THRESHOLDS ...], --MC_DECISION_THRESHOLDS MC_DECISION_THRESHOLDS [MC_DECISION_THRESHOLDS ...]
                        decision thresholds (list) for detection with multiple classifiers, any can be set to None (default: [None, None]) (attention: type them using
                        spaces as separators)
```

#### Examples of `main_detector.py` usage
```bash
python main_detector.py -ddivmc
```
The line above executes a demonstration of detection in a video sequence captured from camera, using two default classifiers (two  instances of `FastRealBoostBins`, each being an ensemble of size 1024) trained to detect faces and palm gestures.
Default values are used for all other relevant settings (decision thresholds, detection procedure parameters, video camera selection, etc.). To quit the demonstration window, 'esc' key should be pressed.

```bash
python main_detector.py -ddivmc -mcdt 4.2 5.5
```
Execution of the line above runs the mentioned demonstration, but changes decision thresholds of the two classifiers from their internal defaults to manually imposed values.

```bash
python main_detector.py -ddivmc -ds 12 -dwhm 64 -dwwm 64
```
This execution leads to a heavier detection procedure that scans each frame using 12 scales for the sliding window, starting from its minimum size of 64 $\times$ 64.
This results in approximately 64k windows to be checked per frame (instead the default of 22k: 9 scales, starting from 96 $\times$ 96 window). 
Other detection procedure related options are ``-dwg`` and ``-dwj``, allowing to change the growth factor and the relative jump for the sliding window.

```bash
python main_detector.py -ddivmc -mccm clf_frbb_face_n_18225_S_5_P_5_NPI_300_SEED_0_T_2048_B_8.bin clf_frbb_hand_n_18225_S_5_P_5_NPI_30_SEED_0_T_2048_B_8.bin
```
The above example executes a detection demonstration using two specific classifiers (stored in folder `/models/`), being ensembles of size 2048, instead of the default ones.

```bash
python main_detector.py -rd -k face -npi 50 -s 4 -p 6
```
As an example of other functionalities, the line above generates or regenerates a data set (`-rd` option) meant for face detection based on FDDB images (see folder `/data_raw/fddb/`).
The switch `-npi 50` (negatives per image) asks for 50 negative examples to be sampled randomly from each image (note: positive examples - the targets - are extracted exactly 
in accordance with annotations).
The last fragment `-s 4 -p 6` specifies the parameterization related to Haar-like features, defining the number of scaling variants along each dimension and the size of grid with
anchoring points (see documentation of `haar.py` module for more details). The resulting generated data set (data arrays for training and testing) shall be pickled and stored 
as binary files in folder `/data/`.<br/>

Remark: currently, other possible selection for the kind of data (option `-k`) is `hand`, leading to generation of a data set based on HaGRID database (see folder `/data_raw/hagrid/` for instructions).

```bash
python main_detector.py -form -k face -npi 50 -s 4 -p 6 -t 512 -b 16
```
Once a data set is ready, one can ask to fit, or refit, a model to the data (`-form` option) as in the above example. The last fragment `-t 512 -b 16` indicates the wanted
parameters to be imposed on a `FastRealBoostBins` instance: 512 as the size of ensemble (equivalently - the number of boosting rounds) and 16 as the number of bins.

```bash
python main_detector.py -ddiv -k face -npi 50 -s 4 -p 6 -t 512 -b 16
```
Subsequently, when the fit is done, one can check how the obtained classifier works as an object detector e.g. with an execution as above.
The new option `-ddiv` is meant for demonstration of detection in video with a single classifier, in contrast to `-ddivmc` meant for multiple classifiers. <br/>
Remark: by default, all newly trained classifiers use the value of 0.0 as their decision threshold (attribute `decision_threshold_` in instances of `FastRealBoostBins` class).
To adjust the threshold to a possibly better value, based on ROC analysis and the precision measure, use `-adtom` option.

## License
This work is licensed under <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.

## Acknowledgments and credits
- [Numba](https://numba.pydata.org): a high-performance just-in-time Python compiler.
- [FDDB](http://vis-www.cs.umass.edu/fddb): Face Detection Data Set and Benchmark; (Jain and Learned-Miller, 2010): Technical Report UM-CS-2010-009, Dept. of Computer Science, University of Massachusetts, Amherst [[pdf]](http://vis-www.cs.umass.edu/fddb/fddb.pdf).
- [MNIST](https://ieeexplore.ieee.org/document/6296535): The MNIST Database of Handwritten Digit Images for Machine Learning Research; (Li, 2012): IEEE Signal Processing Magazine 29 (6).
- [HaGRID](https://github.com/hukenovs/hagrid): HAnd Gesture Recognition Image Dataset (Kapitanov, Makhlyarchuk, Kvanchiani and Nagaev, 2022) [[arXiv]](https://arxiv.org/abs/2206.08219).
