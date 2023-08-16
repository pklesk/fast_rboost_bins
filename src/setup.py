from setuptools import setup

setup(
    name = "frbb",
    version = "1.0.0",
    description = "This module contains FastRealBoostBins class representing an ensemble classifier for fast predictions implemented using numba.jit and numba.cuda.",
    author = "Przemysław Klęsk",
    author_email = "pklesk@zut.edu.pl",
    url = "https://github.com/pklesk/fast_rboost_bins",
    py_modules = ["frbb"],
    install_requires = [
        "numpy",
        "numba",
        "scikit-learn"
    ],
    license = "GNU Affero General Public License v3",
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: OS Independent",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology", 
        "Intended Audience :: Education", 
        "Intended Audience :: Developers",        
        "Topic :: Software Development :: Libraries :: Python Modules"                
    ]
)