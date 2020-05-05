/* Copyright 2020 CNRS-AIST JRL
 */

#pragma once

#include <Eigen/Core>
#include <jrl-qp/api.h>

namespace jrlqp::test
{
  // Description of the characteristics of a least-square problem
  // min. 0.5 ||A x - b||^2
  // s.t. E x = f
  //      l <= C x <= u
  //      x^- <= x <= x^+
  struct JRLQP_DLLAPI ProblemCharacteristics
  {
    int nVar_;                       //Number of variables
    int nObj_;                       //Row size of A
    int nEq_ = 0;                    //Row size of E
    int nIneq_ = 0;                  //Row size of C
    int rankObj_;                    //Rank of A
    int nSharedRank_ = 0;            //rk(A)+nAct - rk([A;C_act]) where C_act is the matrix of active constraints and nAct its row size
    int nStrongActIneq_ = 0;         //Number of strongly active general inequality constraints
    int nWeakActIneq_ = 0;           //Number of weakly active general inequality constraints
    int nStrongActBounds_ = 0;       //Number of strongly active bounds
    int nWeakActBounds_ = 0;         //Number of weakly active bounds
    bool bounds_ = false;            //whether to use bounds or not
    bool doubleSidedIneq_ = false;   //whether to use double-sided general constraints or not
    bool strictlyFeasible_ = false;  //If true, the set of feasible points must not be reduced to a singleton (except if nEq == nVar_)

    ProblemCharacteristics(int nVar, int nObj)
      : nVar_(nVar), nObj_(nObj), rankObj_(nObj) {}
    ProblemCharacteristics(int nVar, int nObj, int nEq, int nIneq)
      : nVar_(nVar), nObj_(nObj), rankObj_(nObj), nEq_(nEq), nIneq_(nIneq) {}
    ProblemCharacteristics(int nVar, int nObj, int nEq, int nIneq, int rankObj, int nSharedRank, int nStrongActIneq, int nWeakActIneq, int nStrongActBounds, int nWeakActBounds, int bounds, int doubleSidedIneq, int strictlyFeasible)
      : nVar_(nVar), nObj_(nObj), nEq_(nEq), nIneq_(nIneq), rankObj_(rankObj), nSharedRank_(nSharedRank),
      nStrongActIneq_(nStrongActIneq), nWeakActIneq_(nWeakActIneq), nStrongActBounds_(nStrongActBounds),
      nWeakActBounds_(nWeakActBounds), bounds_(bounds), doubleSidedIneq_(doubleSidedIneq), strictlyFeasible_(strictlyFeasible) {}

    ProblemCharacteristics& nEq(int n) { nEq_ = n; return *this; }
    ProblemCharacteristics& nIneq(int n) { nIneq_ = n; return *this; }
    ProblemCharacteristics& rankObj(int n) { rankObj_ = n; return *this; }
    ProblemCharacteristics& nSharedRank(int n) { nSharedRank_ = n; return *this; }
    ProblemCharacteristics& nStrongActIneq(int n) { nStrongActIneq_ = n; return *this; }
    ProblemCharacteristics& nWeakActIneq(int n) { nWeakActIneq_ = n; return *this; }
    ProblemCharacteristics& nStrongActBounds(int n) { nStrongActBounds_ = n; return *this; }
    ProblemCharacteristics& nWeakActBounds(int n) { nWeakActBounds_ = n; return *this; }
    ProblemCharacteristics& bounds(bool b = true) { bounds_ = b; return *this; }
    ProblemCharacteristics& doubleSidedIneq(bool b = true) { doubleSidedIneq_ = b; return *this; }
    ProblemCharacteristics& strictlyFeasible(bool b = true) { strictlyFeasible_ = b; return *this; }

    void check() const;
  };

  struct JRLQP_DLLAPI RandomLeastSquare
  {
    Eigen::VectorXd x;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    Eigen::MatrixXd E;
    Eigen::VectorXd f;
    Eigen::VectorXd lambdaEq;
    Eigen::MatrixXd C;
    Eigen::VectorXd l;
    Eigen::VectorXd u;
    Eigen::VectorXd lambdaIneq;
    Eigen::VectorXd xl;
    Eigen::VectorXd xu;
    Eigen::VectorXd lambdaBnd;
    bool doubleSidedIneq;
    bool bounds;

    struct KKT
    {
      Eigen::VectorXd dL;
      Eigen::VectorXd eqViol;
      Eigen::VectorXd ineqViol;
      Eigen::VectorXd bndViol;
      Eigen::VectorXd ineqCompl;
      Eigen::VectorXd bndCompl;
    };

    KKT computeKKTValues() const;
    bool checkKKT() const;
    bool testKKT() const;
    bool dispKKT() const;

    void disp() const;
  };

  RandomLeastSquare JRLQP_DLLAPI randomProblem(const ProblemCharacteristics& characs);

}
