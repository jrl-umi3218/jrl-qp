jrl-qp (alpha version)
======================

[![License](https://img.shields.io/badge/License-BSD%203--Clause-green.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![CI](https://github.com/jrl-umi3218/jrl-qp/workflows/CI%20of%20jrl-qp/badge.svg?branch=master)](https://github.com/jrl-umi3218/jrl-qp/actions?query=workflow%3A%22CI+of+jrl-qp%22)
[![Documentation](https://img.shields.io/badge/doxygen-online-brightgreen?logo=read-the-docs&style=flat)](http://jrl-umi3218.github.io/jrl-qp/doxygen/HEAD/index.html)

This library offers tools and implementations to write, specialize and test QP solvers.

It comes (so far) with an implementation of the Goldfarb-Idnani dual solver described in the seminal paper *D. Goldfarb, A. Idnani, "A numerically stable dual method for solving strictly convex quadratic programs", Mathematical Programming 27 (1983) 1-33 *.

The implementation is done with [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page).


Installation
-------------
Compilation has been tested on Linux (gcc/clang) and Windows (Visual Studio).

### Dependencies

To compile you will need the following tools:

 * [Git](https://git-scm.com/)
 * [CMake](https://cmake.org/) >= 3.1.3
 * [doxygen](http://www.doxygen.org)
 * A compiler with C++17 support

jrl-qp has a single dependency:
 * [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) >= 3.2.8

If you have the following solvers, you can run benchmark comparisons with them :
 * [eigen-qld](https://github.com/jrl-umi3218/eigen-qld)
 * [eigen-quadprog](https://github.com/jrl-umi3218/eigen-quadprog)
 * [eigen-lssol](git@gite.lirmm.fr:multi-contact/eigen-lssol.git) (private repository)


This repository also uses [jrl-cmakemodules](https://github.com/jrl-umi3218/jrl-cmakemodules), and [google benchmark](https://github.com/google/benchmark) as submodules.

### Building from source on Linux

Follow the standard CMake build procedure:

```sh
git clone --recursive https://github.com/jrl-umi3218/jrl-qp
cd jrl-qp
mkdir build && cd build
cmake [options] ..
make && sudo make install
```

where the main options are:
 * `-DCMAKE_BUILD_TYPE=Release` Build in Release mode
 * `-DCMAKE_INSTALL_PREFIX=some/path/to/install` default is `/usr/local`


Tests
-----
Aside from hand-crafted and randomized tests, this repository can use the Maros and Meszaros QP collection ([bottom of this page](http://www.doc.ic.ac.uk/~im/)), that can also be found [here](https://github.com/YimingYAN/QP-Test-Problems) with a Matlab version of the problems.
To use this collection, simply specify its path in the CMake options.
