/* Copyright 2020 CNRS-AIST JRL
 */

#pragma once

#include <Eigen/Core>

#include <jrl-qp/DualSolver.h>

namespace jrlqp
{
  class JRLQP_DLLAPI GoldfarbIdnaniSolver: public DualSolver
  {
  public:
    GoldfarbIdnaniSolver();
    GoldfarbIdnaniSolver(int nbVar, int nbCstr, bool useBounds);

    TerminationStatus solve(MatrixRef G, const VectorConstRef& a, const MatrixConstRef& C,
      const VectorConstRef& bl, const VectorConstRef& bu, const VectorConstRef& xl, const VectorConstRef& xu);

  protected:
    struct Problem
    {
    public:
      Problem() : G(Eigen::Map<Eigen::MatrixXd>(0x0,0,0)), a(Eigen::VectorXd(0)), C(Eigen::MatrixXd(0, 0))
        , bl(Eigen::VectorXd(0)), bu(Eigen::VectorXd(0)), xl(Eigen::VectorXd(0)), xu(Eigen::VectorXd(0)){}
      
      // Map<const EigenObj> and thus Ref<const EigenObj> has its operator= deleted.
      // We thus need to delete as well for Problem. This is not an issue as a solver is not
      // meant to be copied. If it was needed to implement this operator, on would need to
      // use placement new tricks on the XXXConstRef members.
      Problem& operator=(const Problem&) = delete;
      MatrixRef G;
      VectorConstRef a;
      MatrixConstRef C;
      VectorConstRef bl;
      VectorConstRef bu;
      VectorConstRef xl;
      VectorConstRef xu;
    };

    virtual internal::InitTermination init_() override;
    virtual internal::ConstraintNormal selectViolatedConstraint_(const VectorConstRef& x) const override;
    virtual void computeStep_(VectorRef z, VectorRef r, const internal::ConstraintNormal& np) const override;
    virtual StepLenghth  computeStepLength_(const internal::ConstraintNormal& np, const VectorConstRef& x,
      const VectorConstRef& u, const VectorConstRef& z, const VectorConstRef& r) const override;
    virtual bool addConstraint_(const internal::ConstraintNormal& np) override;
    virtual bool removeConstraint_(int l) override;
    virtual void resize_(int nbVar, int nbCstr, bool useBounds) override;

    virtual void initActiveSet();

    void addInitialConstraint(const internal::ConstraintNormal& np);

    mutable internal::Workspace<> work_d_;
    internal::Workspace<> work_J_;
    internal::Workspace<> work_R_;
    Problem pb_;
  };

}
