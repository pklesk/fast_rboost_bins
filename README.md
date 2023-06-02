# FastRealBoostBins: A fast ensemble classifier implemented in Python using numpy, numba.jit and numba.cuda

Taking advantage of [Numba](https://numba.pydata.org/) (a high-performance Python compiler), we provide a fast implementation 
of a boosting algorithm variant in which bins with logit transform values play the role of ``weak learners''. 
The implementation comes as a Python class compliant with the scheme of \code{scikit-learn} library. The software allows to choose between CPU and GPU computations 
for each of the two stages: fit and predict. The efficiency of implementation has been confirmed on large 
data sets where the total of array entries (sample size $\times$ features count) was of 
order $10^{10}$ at fit stage and $10^{8}$ at predict stage. In the case of GPU-based fit, the body of 
main boosting loop is designed as five CUDA kernels responsible for: weights binning, 
computing logit values, computing exponential errors, finding the error minimizer, and examples reweighting.
The GPU-based predict is computed by a single CUDA kernel. 
We apply suitable reduction patterns to carry out summations and `argmin' operations. For reductions 
that spread beyond the scope of a single block of threads, we introduce appropriate mutex mechanisms. 
To test the performance of the predict stage, we make our ensemble classifiers operate as object 
detectors under heavy computational load (e.g.~over $20$k queries per a video frame using ensembles of size $1024$).


## Software description
TODO

### Contents and high-level functionality
TODO

### CUDA reductions (for summations and argmin operations)
TODO

### Mutex mechanisms
TODO

### GPU-based fit with numba.cuda
TODO

### Available variants of decision function
TODO
