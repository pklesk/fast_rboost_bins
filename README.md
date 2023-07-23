[under developement]
# FastRealBoostBins: An ensemble classifier for fast predictions implemented in Python using numba.jit and numba.cuda
<table>
<tr>
    <td><img src="/extras/fig_experiment_real_1903270360_20230625_fddb-patches_time_predict_test.png"/></td>
    <td><br/><a href="https://github.com/pklesk/fast_rboost_bins/assets/23095311/5e390cfc-84e8-4281-82d3-91a0b72c9c36"><img src="/extras/video_quadro_screenshot.jpg"/></a></td>
</tr>
</table>

Taking advantage of [Numba](https://numba.pydata.org/) (a high-performance just-in-time Python compiler) 
we provide a fast operating implementation of a boosting algorithm
in which bins with logit transform values play the role of “weak learners”.

The software comes as a Python class compliant with [scikit-learn](https://scikit-learn.org) library.
It allows to choose between CPU and GPU computations for each of the two stages: fit and predict (decision function). 
The efficiency of implementation has been confirmed on large data sets where the total of array entries (sample
size $\times$ features count) was of order 10<sup>10</sup> at fit stage and 10<sup>8</sup> at predict stage.
In case of GPU-based fit, the main boosting loop is designed as five CUDA kernels responsible for: 
weights binning, computing logits, computing exponential errors, finding the error minimizer, and examples reweighting. 
The GPU-based predict is computed by a single CUDA kernel. We apply suitable reduction patterns and mutexes 
to carry out summations and 'argmin' operations. 

To test the predict stage performance we compare `FastRealBoostBins` against state-of-the-art classifiers from `sklearn.ensemble` 
using large data sets and focusing on response times. In an additional experiment, we make our
classifiers operate as object detectors under heavy computational load (over 60k queries per a video frame using ensembles of size 2048).

## Example usage
With `frbb.py` file (containing `FastRealBoostBins` class) included to some project, one can write e.g.:
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
        # checking classifiers    
        for clf in clfs:
            print(f"  CLF: {clf}...")
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
            print(f"    ACCs -> TRAIN {clf.score(X_train, y_train)}, TEST: {clf.score(X_test, y_test)}")            
            print(f"    TIMES [s] -> FIT: {t2_fit - t1_fit}, PREDICT (TRAIN): {t2_predict_train - t1_predict_train}, PREDICT (TEST): {t2_predict_test - t1_predict_test}")            
```
And produces the following output (actual observed times are machine dependent but their proportions should be roughly preserved):
```bash
DATA SHAPE (TRAIN AND TEST): 1000 x 1000
  CLF: AdaBoostClassifier(n_estimators=1024)...
    ACCs -> TRAIN 1.0, TEST: 0.491
    TIMES [s] -> FIT: 68.27675604820251, PREDICT (TRAIN): 0.9272050857543945, PREDICT (TEST): 0.9245297908782959
  CLF: FastRealBoostBins(T=1024, B=8, outliers_ratio=0.05, logit_max: 2.0, fit_mode='numba_jit', decision_function_mode='numba_jit')...
    ACCs -> TRAIN 1.0, TEST: 0.521
    TIMES [s] -> FIT: 9.538139343261719, PREDICT (TRAIN): 0.008235454559326172, PREDICT (TEST): 0.007825374603271484
  CLF: FastRealBoostBins(T=1024, B=8, outliers_ratio=0.05, logit_max: 2.0, fit_mode='numba_cuda', decision_function_mode='numba_cuda')...
    ACCs -> TRAIN 1.0, TEST: 0.521
    TIMES [s] -> FIT: 6.34455132484436, PREDICT (TRAIN): 0.0038661956787109375, PREDICT (TEST): 0.0037419795989990234
DATA SHAPE (TRAIN AND TEST): 10000 x 1000
  CLF: AdaBoostClassifier(n_estimators=1024)...
    ACCs -> TRAIN 0.8686, TEST: 0.4957
    TIMES [s] -> FIT: 852.4775321483612, PREDICT (TRAIN): 12.728276491165161, PREDICT (TEST): 12.74528455734253
  CLF: FastRealBoostBins(T=1024, B=8, outliers_ratio=0.05, logit_max: 2.0, fit_mode='numba_jit', decision_function_mode='numba_jit')...
    ACCs -> TRAIN 0.8535, TEST: 0.5054
    TIMES [s] -> FIT: 112.78886556625366, PREDICT (TRAIN): 0.11319923400878906, PREDICT (TEST): 0.1127011775970459
  CLF: FastRealBoostBins(T=1024, B=8, outliers_ratio=0.05, logit_max: 2.0, fit_mode='numba_cuda', decision_function_mode='numba_cuda')...
    ACCs -> TRAIN 0.8553, TEST: 0.5122
    TIMES [s] -> FIT: 16.35269331932068, PREDICT (TRAIN): 0.0788724422454834, PREDICT (TEST): 0.07758545875549316
```

## Constructor parameters
TODO

## Estimated attributes
TODO

## Selected experimental results

### Comparison against state-of-the-art classifier from `sklearn.ensemble`

In tables below we write for shortness `FastRealBoostBins('numba_jit')` which, in fact, stands for `FastRealBoostBins(fit_mode='numba_jit', decision_function_mode='numba_jit')`,
and `FastRealBoostBins('numba_cuda')` standing for `FastRealBoostBins(fit_mode='numba_cuda', decision_function_mode='numba_cuda')`.
To have approximately equal conditions for comparison (e.g. same number of weak learners, each learner based on 1 feature) we forced the following settings on classifiers
from `sklearn.ensemble`:
```python
AdaBoostClassifier(algorithm="SAMME.R", max_depth=1, n_estimators=T)
GradientBoostingClassifier(max_depth=1, n_estimators=T)
HistGradientBoostingClassifier(early_stopping=False, max_iter=T, max_bins=B)
```

Hardware environment: Intel(R) Core(TM) i7-10700 CPU @ 2.90GHz, 125.7 GB RAM, NVIDIA GeForce RTX 3090 GPU.<br/>
Software environment: Linux 5.15.0-71-generic, Python 3.8.10, GCC 9.4.0, numpy 1.22.3, numba 0.57.0, sklearn 1.0.2, cv2 4.6.0, nvcc 11.7.

#### 'FDDB-PATCHES (3NPI)', T=1024: &nbsp;&nbsp; train data: (11 775 $\times$ 3 072) ~ 10<sup>7.6</sup>, &nbsp; test data: (1 303 $\times$ 3 072 $\rightarrow$ 1 024) ~ 10<sup>6.1</sup>
| classifier                        | fit time [s] | fit speedup $\approx$ | predict time [s]         | predict speedup $\approx$ | acc [%]     | predict time [s] | predict speedup $\approx$ | acc [%]    |
|:----------------------------------|-------------:|----------------------:|-------------------------:|--------------------------:|------------:|-----------------:|--------------------------:|-----------:|
|                                   |  **(train)** |           **(train)** |              **(train)** |               **(train)** | **(train)** |       **(test)** |                **(test)** | **(test)** |
| `AdaBoostClassifier`              |        1 421 |            $\times$ 1 |                   33.285 |                $\times$ 1 |       99.94 |            2.702 |                $\times$ 1 |      89.56 |
| `GradientBoostingClassifier`      |        1 341 |            $\times$ 1 |                    0.191 |              $\times$ 174 |       94.04 |            0.013 |              $\times$ 208 |      90.18 |
| `HistGradientBoostingClassifier`  |           14 |          $\times$ 102 |                    0.288 |              $\times$ 116 |       93.17 |            0.030 |               $\times$ 90 |      90.02 |
| `FastRealBoostBins('numba_jit')`  |          395 |            $\times$ 4 |                    0.096 |              $\times$ 347 |       99.98 |            0.009 |              $\times$ 300 |      88.41 |
| `FastRealBoostBins('numba_cuda')` |           43 |           $\times$ 33 |                    0.068 |              $\times$ 489 |       99.97 |            0.003 |              $\times$ 901 |      88.33 |

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
| `FastRealBoostBins('numba_jit')`  |          510 |            $\times$ 3 |                    0.393 |              $\times$ 113 |      100.00 |            0.036 |              $\times$ 167 |      98.50 |
| `FastRealBoostBins('numba_cuda')` |           53 |           $\times$ 33 |                    0.257 |              $\times$ 173 |      100.00 |            0.018 |              $\times$ 334 |      98.41 |

|fit times along growing T|predict times (test) along growing T|
|-|-|
|<img src="/extras/fig_experiment_real_2001519960_20230626_mnist-b_time_fit.png"/>|<img src="/extras/fig_experiment_real_2001519960_20230626_mnist-b_time_predict_test.png"/>|

#### 'HaGRID-HFs (PALM, 10NPI)', T=2048: &nbsp;&nbsp; train data: (232 299 $\times$ 18 225) ~ 10<sup>9.6</sup>, &nbsp; test data: (77 433 $\times$ 18 225 $\rightarrow$ 2 048) ~ 10<sup>8.2</sup>
| classifier                        | fit time [s] | fit speedup $\approx$ | predict time [s]         | predict speedup $\approx$ | acc [%]     | predict time [s] | predict speedup $\approx$ | acc [%]    |
|:----------------------------------|-------------:|----------------------:|-------------------------:|--------------------------:|------------:|-----------------:|--------------------------:|-----------:|
|                                   |  **(train)** |           **(train)** |              **(train)** |               **(train)** | **(train)** |       **(test)** |                **(test)** | **(test)** |
| `HistGradientBoostingClassifier`  |        2 662 |          $\times$ 3.3 |                   13.667 |              $\times$ 1.0 |       98.71 |            4.493 |              $\times$ 1.0 |      98.65 |
| `FastRealBoostBins('numba_cuda')` |        8 908 |          $\times$ 1.0 |                    5.060 |              $\times$ 2.7 |      100.00 |            1.277 |              $\times$ 3.5 |      99.07 |

|fit times along growing T|predict times (test) along growing T|
|-|-|
|<img src="/extras/fig_experiment_real_1178284368_20230627_hagrid-hfs-10_time_fit.png"/>|<img src="/extras/fig_experiment_real_1178284368_20230627_hagrid-hfs-10_time_predict_test.png"/>|

## Script for experiments: `main_experimenter` 
By executing `python main_experimenter.py -h` (or `--help`) one obtains help on script arguments:
```bash
"FAST-REAL-BOOST-BINS": AN ENSEMBLE CLASSIFIER FOR FAST PREDICTIONS IMPLEMENTED IN PYTHON VIA NUMBA.JIT AND NUMBA.CUDA. [main_experimenter]
[for help use -h or --help switch]
REAL DATA DEFINITIONS:
[('fddb-patches', 'read_data_fddb_patches', 'FDDB-PATCHES (3NPI)'),
 ('cifar-10', 'read_data_cifar_10', 'CIFAR-10 (AIRPLANE)'),
 ('mnist-b', 'read_data_mnist_b', 'MNIST-B (DIGIT 0)'),
 ('fddb-hfs-100', 'read_data_fddb_haar_npi_100', 'FDDB-HFs (100NPI)'),
 ('fddb-hfs-300', 'read_data_fddb_haar_npi_300', 'FDDB-HFs (300NPI)'),
 ('hagrid-hfs-10', 'read_data_hagrid_haar_npi_10', 'HaGRID-HFs (PALM, 10NPI)'),
 ('hagrid-hfs-30', 'read_data_hagrid_haar_npi_30', 'HaGRID-HFs (PALM, 30NPI)')]
CLASSIFIERS DEFINITIONS:
[(<class 'sklearn.ensemble._weight_boosting.AdaBoostClassifier'>, {'algorithm': 'SAMME.R'}, {'color': 'black'}),
 (<class 'sklearn.ensemble._gb.GradientBoostingClassifier'>, {'max_depth': 1}, {'color': 'green'}),
 (<class 'sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingClassifier'>, {'max_depth': 1, 'early_stopping': False}, {'color': 'orange'}),
 (<class 'frbb.FastRealBoostBins'>, {'fit_mode': 'numba_jit', 'decision_function_mode': 'numba_jit'}, {'color': 'blue'}),
 (<class 'frbb.FastRealBoostBins'>, {'fit_mode': 'numba_cuda', 'decision_function_mode': 'numba_cuda'}, {'color': 'red'})]
usage: main_experimenter.py [-h] [-dk {real,random}] [-rdf REAL_DATA_FLAGS REAL_DATA_FLAGS REAL_DATA_FLAGS REAL_DATA_FLAGS REAL_DATA_FLAGS REAL_DATA_FLAGS REAL_DATA_FLAGS]
                            [-cf CLFS_FLAGS CLFS_FLAGS CLFS_FLAGS CLFS_FLAGS CLFS_FLAGS]
                            [-rd {<class 'numpy.int8'>,<class 'numpy.uint8'>,<class 'numpy.int16'>,<class 'numpy.uint16'>,<class 'numpy.int32'>,<class 'numpy.uint32'>,<class 'numpy.int64'>,<class 'numpy.uint64'>,<class 'numpy.float32'>,<class 'numpy.float64'>}]
                            [-nmm NMM_MAGN_ORDERS [NMM_MAGN_ORDERS ...]] [-ts TS [TS ...]] [-bs BS [BS ...]] [-s SEED] [-p PLOTS] [-pan {T,B,n,m_train,m_test}]
                            [-pvn {acc_test,acc_train,time_fit,time_predict_train,time_predict_test} [{acc_test,acc_train,time_fit,time_predict_train,time_predict_test} ...]]

optional arguments:
  -h, --help            show this help message and exit
  -dk {real,random}, --DATA_KIND {real,random}
                        kind of data on which to experiment (default: real)
  -rdf REAL_DATA_FLAGS REAL_DATA_FLAGS REAL_DATA_FLAGS REAL_DATA_FLAGS REAL_DATA_FLAGS REAL_DATA_FLAGS REAL_DATA_FLAGS, --REAL_DATA_FLAGS REAL_DATA_FLAGS REAL_DATA_FLAGS REAL_DATA_FLAGS REAL_DATA_FLAGS REAL_DATA_FLAGS REAL_DATA_FLAGS REAL_DATA_FLAGS
                        boolean flags (list) specifying which data sets from the predefined set will participate in experiments on real data (default: [False, False, False, False, False, False, False]) (attention: type them using
                        spaces as separators)
  -cf CLFS_FLAGS CLFS_FLAGS CLFS_FLAGS CLFS_FLAGS CLFS_FLAGS, --CLFS_FLAGS CLFS_FLAGS CLFS_FLAGS CLFS_FLAGS CLFS_FLAGS CLFS_FLAGS
                        boolean flags (list) specifying which classifiers from the predefined set will participate in experiments (default: [True, True, True, True, True]) (attention: type them using spaces as separators)
  -rd {<class 'numpy.int8'>,<class 'numpy.uint8'>,<class 'numpy.int16'>,<class 'numpy.uint16'>,<class 'numpy.int32'>,<class 'numpy.uint32'>,<class 'numpy.int64'>,<class 'numpy.uint64'>,<class 'numpy.float32'>,<class 'numpy.float64'>}, --RANDOM_DTYPE {<class 'numpy.int8'>,<class 'numpy.uint8'>,<class 'numpy.int16'>,<class 'numpy.uint16'>,<class 'numpy.int32'>,<class 'numpy.uint32'>,<class 'numpy.int64'>,<class 'numpy.uint64'>,<class 'numpy.float32'>,<class 'numpy.float64'>}
                        dtype of input numpy arrays for experiments on random data (default: <class 'numpy.int8'>) (attention: please type it as e.g. 'np.uint8', 'np.float32', etc.)
  -nmm NMM_MAGN_ORDERS [NMM_MAGN_ORDERS ...], --NMM_MAGN_ORDERS NMM_MAGN_ORDERS [NMM_MAGN_ORDERS ...]
                        list of tuples represented as strings defining orders of magnitude for random input arrays in experiments on random data (default: [(5, 3, 5)]) (attention: type them using spaces as separators and with each
                        tuple in quotation marks)
  -ts TS [TS ...], --TS TS [TS ...]
                        ensemble sizes (list) to impose on each type of classifier in experiments (default: [16, 32, 64, 128, 256, 512, 1024]) (attention: type them using spaces as separators)
  -bs BS [BS ...], --BS BS [BS ...]
                        bins counts (list) to impose on each type of classifier in experiments (default: [8]) (attention: type them using spaces as separators)
  -s SEED, --SEED SEED  randomization seed, (default: 0)
  -p PLOTS, --PLOTS PLOTS
                        boolean flag indicating if plots should be generated after experiments (default: False)
  -pan {T,B,n,m_train,m_test}, --PLOTS_ARG_NAME {T,B,n,m_train,m_test}
                        name of argument quantity to be placed on horizontal axis in plots (default: T)
  -pvn {acc_test,acc_train,time_fit,time_predict_train,time_predict_test} [{acc_test,acc_train,time_fit,time_predict_train,time_predict_test} ...], --PLOTS_VALUES_NAMES {acc_test,acc_train,time_fit,time_predict_train,time_predict_test} [{acc_test,acc_train,time_fit,time_predict_train,time_predict_test} ...]
                        names of value quantities to be placed on vertical axis in plots (default: ['acc_test', 'acc_train', 'time_fit', 'time_predict_train', 'time_predict_test']) (attention: type them using spaces as separators)
```

## Script for object detection: `main_detector` 
By executing `python main_detector.py -h` (or `--help`) one obtains help on script arguments:
```bash
"FAST-REAL-BOOST-BINS": AN ENSEMBLE CLASSIFIER FOR FAST PREDICTIONS IMPLEMENTED IN PYTHON VIA NUMBA.JIT AND NUMBA.CUDA. [main_detector]
[for help use -h or --help switch]
usage: main_detector.py [-h] [-k {face,hand}] [-s S] [-p P] [-npi NPI] [-t T] [-b B] [-seed SEED] [-dhfsa] [-dhfss] [-rd] [-form] [-maom] [-adtom] [-ddiv] [-ddivc {gpu_cuda,cpu_simple,cpu_parallel}]
                        [-ddivpj DEMO_DETECT_IN_VIDEO_PARALLEL_JOBS] [-ddivvl] [-ddivvd] [-ddivf DEMO_DETECT_IN_VIDEO_FRAMES] [-ddivmc] [-cv2vcci CV2_VIDEO_CAPTURE_CAMERA_INDEX] [-cv2iim] [-ds DETECTION_SCALES]
                        [-dwhm DETECTION_WINDOW_HEIGHT_MIN] [-dwwm DETECTION_WINDOW_WIDTH_MIN] [-dwg DETECTION_WINDOW_GROWTH] [-dwj DETECTION_WINDOW_JUMP] [-ddt DETECTION_DECISION_THRESHOLD] [-dp {None,avg,nms}]
                        [-mccn MC_CLFS_NAMES [MC_CLFS_NAMES ...]] [-mcdt MC_DECISION_THRESHOLDS [MC_DECISION_THRESHOLDS ...]]

optional arguments:
  -h, --help            show this help message and exit
  -k {face,hand}, --KIND {face,hand}
                        detector kind (default: face)
  -s S, --S S           'scales' parameter of Haar-like features (default: 5)
  -p P, --P P           'positions' parameter of Haar-like features (default: 5)
  -npi NPI, --NPI NPI   'negatives per image' parameter, used in procedures generating data sets from images (default: 300 with -k set to face)
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
                        number of parallel jobs (only in case of 'cpu_parallel' set for -ddivc) (default: 8)
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
  -cv2iim, --CV2_VIDEO_CAPTURE_IS_IT_MSWINDOWS
                        specify if OS is MS Windows (for cv2 and directx purposes)
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
                        classifiers names (list) for detection with multiple classifiers (default: ['clf_frbb_face_n_18225_S_5_P_5_NPI_300_SEED_0_T_1024_B_8.bin', 'clf_frbb_hand_n_18225_S_5_P_5_NPI_30_SEED_0_T_1024_B_8.bin'])
                        (attention: type them using spaces as separators)
  -mcdt MC_DECISION_THRESHOLDS [MC_DECISION_THRESHOLDS ...], --MC_DECISION_THRESHOLDS MC_DECISION_THRESHOLDS [MC_DECISION_THRESHOLDS ...]
                        decision thresholds (list) for detection with multiple classifiers, any can be set to None (default: [None, None]) (attention: type them using spaces as separators)
```

## License
This work is licensed under <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.

## Acknowledgments and credits
- [Numba](https://numba.pydata.org): a high-performance just-in-time Python compiler
- [FDDB](http://vis-www.cs.umass.edu/fddb): Face Detection Data Set and Benchmark; (Jain and Learned-Miller, 2010): Technical Report UM-CS-2010-009, Dept. of Computer Science, University of Massachusetts, Amherst [[pdf]](http://vis-www.cs.umass.edu/fddb/fddb.pdf)
- [HaGRID](https://github.com/hukenovs/hagrid): HAnd Gesture Recognition Image Dataset (Kapitanov, Makhlyarchuk, Kvanchiani and Nagaev, 2022): [[arXiv]](https://arxiv.org/abs/2206.08219)
