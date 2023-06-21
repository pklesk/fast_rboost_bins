# [under developement]
# FastRealBoostBins: An ensemble classifier for fast predictions implemented in Python using numba.jit and numba.cuda

Taking advantage of [Numba](https://numba.pydata.org/) (a high-performance just-in-time Python compiler) 
we provide a fast operating implementation 
of a boosting algorithm variant in which bins with logit transform values 
play the role of ``weak learners''. The implementation comes as a Python class compliant
with the scheme of [scikit-learn](https://scikit-learn.org) library. 

The software allows to choose between CPU and GPU computations for each of the two stages: fit and predict (decision function). 
The efficiency of implementation has been confirmed on large data sets where the total of array entries (sample size $\times$ features count) 
was of order $10^{10}$ at fit stage and $10^{8}$ at predict stage. In the case of GPU-based fit, the body of main boosting loop 
is designed as five CUDA kernels responsible for: weights binning, computing logit values, computing exponential errors, 
finding the error minimizer, and examples reweighting. The GPU-based predict is computed by a single CUDA kernel. 
We apply suitable reduction patterns to carry out summations and `argmin' operations. For reductions 
that spread beyond the scope of a single block of threads, we introduce appropriate mutex mechanisms.

To test the performance of the predict stage we compare FastRealBoostBins against three state-of-the-art ensemble classifiers from scikit-learn,
using several large data sets and focusing on the response time. In an additional experiment, we make our ensemble classifiers operate as object 
detectors under heavy computational load (e.g.~over $60$k queries per a video frame using ensembles of size $2048$).

## License
This work is licensed under <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.

## Acknowledgments and credits
- [Numba](https://numba.pydata.org): a high-performance just-in-time Python compiler
- [FDDB](http://vis-www.cs.umass.edu/fddb): Face Detection Data Set and Benchmark; (Jain and Learned-Miller, 2010): Technical Report UM-CS-2010-009, Dept. of Computer Science, University of Massachusetts, Amherst [[pdf]](http://vis-www.cs.umass.edu/fddb/fddb.pdf)
- [HaGRID](https://github.com/hukenovs/hagrid): HAnd Gesture Recognition Image Dataset (Kapitanov, Makhlyarchuk, Kvanchiani and Nagaev, 2022): [[arXiv]](https://arxiv.org/abs/2206.08219)
