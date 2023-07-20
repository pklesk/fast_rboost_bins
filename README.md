[under developement]
# FastRealBoostBins: An ensemble classifier for fast predictions implemented in Python using numba.jit and numba.cuda
<table>
<tr>
    <td><img src="/extras/fig_experiment_real_1903270360_20230625_fddb-patches_time_predict_test.png"/></td>
    <td><br/><img src="/extras/video_quadro_screenshot.jpg"/></td>
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

## Simple example of usage
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

#### **'FDDB-PATCHES (3NPI)':** &nbsp;&nbsp; train data: (11 775 $\times$ 3 072) ~ 10<sup>7.6</sup>, &nbsp;&nbsp; test data: (1 303 $\times$ 3 072 $\rightarrow$ 1 024) ~ 10<sup>6.1</sup>
| classifier                        | fit time [s] | fit speedup $\approx$ | predict time [s]         | predict speedup $\approx$ | acc [%]     | predict time [s] | predict speedup $\approx$ | acc [%]    |
|:----------------------------------|-------------:|----------------------:|-------------------------:|--------------------------:|------------:|-----------------:|--------------------------:|-----------:|
|                                   |  **(train)** |           **(train)** |              **(train)** |               **(train)** | **(train)** |       **(test)** |                **(test)** | **(test)** |
| `AdaBoostClassifier`              |        1 421 |            $\times$ 1 |                   33.285 |                $\times$ 1 |       99.94 |            2.702 |                $\times$ 1 |      89.56 |
| `GradientBoostingClassifier`      |        1 341 |            $\times$ 1 |                    0.191 |              $\times$ 174 |       94.04 |            0.013 |              $\times$ 208 |      90.18 |
| `HistGradientBoostingClassifier`  |           14 |          $\times$ 102 |                    0.288 |              $\times$ 116 |       93.17 |            0.030 |               $\times$ 90 |      90.02 |
| `FastRealBoostBins('numba_jit')`  |          395 |            $\times$ 4 |                    0.096 |              $\times$ 347 |       99.98 |            0.009 |              $\times$ 300 |      88.41 |
| `FastRealBoostBins('numba_cuda')` |           43 |           $\times$ 33 |                    0.068 |              $\times$ 489 |       99.97 |            0.003 |              $\times$ 901 |      88.33 |

### Fit and predict times along growing ensemble sizes
|fit times|predict times|
|-|-|
|<img src="/extras/fig_experiment_real_1903270360_20230625_fddb-patches_time_fit.png"/>|<img src="/extras/fig_experiment_real_1903270360_20230625_fddb-patches_time_predict_test.png"/>|
|<img src="/extras/fig_experiment_real_2001519960_20230626_mnist-b_time_fit.png"/>|<img src="/extras/fig_experiment_real_2001519960_20230626_mnist-b_time_predict_test.png"/>|
|<img src="/extras/fig_experiment_real_1178284368_20230627_hagrid-hfs-10_time_fit.png"/>|<img src="/extras/fig_experiment_real_1178284368_20230627_hagrid-hfs-10_time_predict_test.png"/>|

## Script for experiments: `main_experimenter` 
TODO

## Script for object detection: `main_detector` 
By executing `python main_detector.py -h` (or `--help`) one obtains help on script arguments:
```bash
"FAST-REAL-BOOST-BINS": AN ENSEMBLE CLASSIFIER FOR FAST PREDICTIONS IMPLEMENTED IN PYTHON VIA NUMBA.JIT AND NUMBA.CUDA.
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
  -mcdt MC_DECISION_THRESHOLDS [MC_DECISION_THRESHOLDS ...], --MC_DECISION_THRESHOLDS MC_DECISION_THRESHOLDS [MC_DECISION_THRESHOLDS ...]
                        decision thresholds (list) for detection with multiple classifiers, any can be set to None (default: [None, None])
```

## License
This work is licensed under <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.

## Acknowledgments and credits
- [Numba](https://numba.pydata.org): a high-performance just-in-time Python compiler
- [FDDB](http://vis-www.cs.umass.edu/fddb): Face Detection Data Set and Benchmark; (Jain and Learned-Miller, 2010): Technical Report UM-CS-2010-009, Dept. of Computer Science, University of Massachusetts, Amherst [[pdf]](http://vis-www.cs.umass.edu/fddb/fddb.pdf)
- [HaGRID](https://github.com/hukenovs/hagrid): HAnd Gesture Recognition Image Dataset (Kapitanov, Makhlyarchuk, Kvanchiani and Nagaev, 2022): [[arXiv]](https://arxiv.org/abs/2206.08219)
