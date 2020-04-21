/* Copyright 2020 CNRS-AIST JRL
 */

#pragma once

#include <Eigen/Core>

#include <jrl-qp/api.h>
#include <jrl-qp/defs.h>
#include <jrl-qp/SolverOptions.h>
#include <jrl-qp/internal/ActiveSet.h>
#include <jrl-qp/internal/ConstraintNormal.h>
#include <jrl-qp/internal/Workspace.h>
#include <jrl-qp/utils/Debug.h>
#include <jrl-qp/utils/Logger.h>


namespace jrlqp
{
  class JRLQP_DLLAPI DualSolver
  {
  public:
    DualSolver();
    DualSolver(int nbVar, int nbCstr, bool useBounds);

    void resize(int nbVar, int nbCstr, bool useBounds);

    void options(const SolverOptions& options);

    WConstVector solution() const;
    WConstVector multipliers() const;
    double objectiveValue() const;

  protected:
    struct StepLenghth { double t1; double t2; int l; };
    TerminationStatus solve();

    void init();
    internal::ConstraintNormal selectViolatedConstraint(const VectorConstRef& x) const;
    void computeStep(VectorRef z, VectorRef r, const internal::ConstraintNormal& np) const;
    StepLenghth computeStepLength(const internal::ConstraintNormal& np, const VectorConstRef& x, 
      const VectorConstRef& u, const VectorConstRef& z, const VectorConstRef& r) const;
    bool addConstraint(const internal::ConstraintNormal& np);
    bool removeConstraint(int l, VectorRef u);


    virtual void init_() = 0;
    virtual internal::ConstraintNormal selectViolatedConstraint_(const VectorConstRef& x) const = 0;
    virtual void computeStep_(VectorRef z, VectorRef r, const internal::ConstraintNormal& np) const = 0;
    virtual StepLenghth  computeStepLength_(const internal::ConstraintNormal& np, const VectorConstRef& x, 
      const VectorConstRef& u, const VectorConstRef& z, const VectorConstRef& r) const = 0;
    virtual bool addConstraint_(const internal::ConstraintNormal& np) = 0;
    virtual bool removeConstraint_(int l) = 0;
    virtual void resize_(int nbVar, int nbCstr, bool useBounds) = 0;

  private:
    void resize_p(int nbVar, int nbCstr, bool useBounds);

  protected:
    SolverOptions options_;
    utils::Logger log_;

    int nbVar_;
    internal::ActiveSet A_;

    double f_;
    internal::Workspace<> work_x_;
    internal::Workspace<> work_z_;
    mutable internal::Workspace<> work_u_;
    mutable internal::Workspace<> work_r_;
    mutable bool needToExpandMultipliers_;
  };

}