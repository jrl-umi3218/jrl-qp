/* Copyright 2020 CNRS-AIST JRL
 */

#pragma once
#include <jrl-qp/defs.h>

namespace jrlqp::test
{ 
  /** Check that all the matrix and vector dimensions of the folowing QP problem
    * are consistent:
    * min. .5 x^T G x + a^T x
    * s.t. bl <= C x <= bu
    *      xl <= x <= xu
    * where the variable bounds are optional.
    * When transposedC is true, the constraints read bl <= C^T x <= bu
    */
  void checkDimensions(const MatrixConstRef& G, const VectorConstRef& a,
                       const MatrixConstRef& C, const VectorConstRef& bl, const VectorConstRef& bu,
                       const VectorConstRef& xl, const VectorConstRef& xu,
                       bool transposedC = false);

  /** Check that all the matrix and vector dimensions of the folowing QP problem
    * are consistent:
    * min. .5 x^T G x + a^T x
    * s.t. bl <= C x <= bu
    *      xl <= x <= xu
    * where the variable bounds are optional. Further check that the solution 
    * vector x and the vector of Lagrange multipliers have the correct dimensions.
    * When transposedC is true, the constraints read bl <= C^T x <= bu
    */
  void checkDimensions(const VectorConstRef& x, const VectorConstRef& u,
                       const MatrixConstRef& G, const VectorConstRef& a,
                       const MatrixConstRef& C, const VectorConstRef& bl, const VectorConstRef& bu,
                       const VectorConstRef& xl, const VectorConstRef& xu,
                       bool transposedC = false);

  /** Check that all the matrix and vector dimensions of the folowing FP problem
    * are consistent:
    * find x
    * s.t. bl <= C x <= bu
    *      xl <= x <= xu
    * where the variable bounds are optional. Further check that the solution 
    * vector x and the vector of Lagrange multipliers have the correct dimensions.
    * When transposedC is true, the constraints read bl <= C^T x <= bu
    */
  void checkDimensions(const VectorConstRef& x, const VectorConstRef& u,
                       const MatrixConstRef& C, const VectorConstRef& bl, const VectorConstRef& bu,
                       const VectorConstRef& xl, const VectorConstRef& xu,
                       bool transposedC = false);

  bool testKKT(const VectorConstRef& x, const VectorConstRef& u,
               const MatrixConstRef& G, const VectorConstRef& a,
               const MatrixConstRef& C, const VectorConstRef& bl, const VectorConstRef& bu,
               const VectorConstRef& xl, const VectorConstRef& xu, 
               bool transposedC = false, double tau_p = 1e-6, double tau_d = 1e-6);

  bool testKKTStationarity(const VectorConstRef& x, const VectorConstRef& u,
               const MatrixConstRef& G, const VectorConstRef& a,
               const MatrixConstRef& C, const VectorConstRef& bl, const VectorConstRef& bu,
               const VectorConstRef& xl, const VectorConstRef& xu, 
               bool transposedC= false, double tau_d = 1e-6);

  bool testKKTFeasibility(const VectorConstRef& x, const VectorConstRef& u,
               const MatrixConstRef& C, const VectorConstRef& bl, const VectorConstRef& bu,
               const VectorConstRef& xl, const VectorConstRef& xu, 
               bool transposedC= false, double tau_p = 1e-6, double tau_d = 1e-6);
}