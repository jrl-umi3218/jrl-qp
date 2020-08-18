/* Copyright 2020 CNRS-AIST JRL */

#pragma once

#include <Eigen/Core>

#include <jrl-qp/api.h>
#include <jrl-qp/defs.h>
#include <jrl-qp/SolverOptions.h>
#include <jrl-qp/internal/ActiveSet.h>
#include <jrl-qp/internal/ConstraintNormal.h>
#include <jrl-qp/internal/TerminationType.h>
#include <jrl-qp/internal/Workspace.h>
#include <jrl-qp/utils/debug.h>
#include <jrl-qp/utils/Logger.h>


namespace jrl::qp
{
  /** Base class for dual QP solver. It implements the general logic of the
    * Goldfarb-Idnani paper, and relies on call to virtual functions to do the
    * actual work, depending on the specificities of the problem.
    */
  class JRLQP_DLLAPI DualSolver
  {
  public:
    DualSolver();
    /** Pre-allocate the data for a problem with \p nbVar variables, \p nbCstr
      * (general) constraints, and bounds if \p useBounds is \a true.*/
    DualSolver(int nbVar, int nbCstr, bool useBounds);

    virtual ~DualSolver() = default;

    /** Resize the data for a problem with \p nbVar variables, \p nbCstr
      * (general) constraints, and bounds if \p useBounds is \a true.
      *
      * \internal Call resize_ first, then resize_p.
      */
    void resize(int nbVar, int nbCstr, bool useBounds);

    /** Specify the solver options.*/
    void options(const SolverOptions& options);

    /** Get the solution.*/
    WConstVector solution() const;
    /** Get the Lagrange multipliers at the solution.*/
    WConstVector multipliers() const;
    /** Get the objective value at the solution.*/
    double objectiveValue() const;

  protected:
    struct StepLenghth { double t1; double t2; int l; };

    /** Call to the solving routines. Need to be initiated by the derived class. */
    TerminationStatus solve();
    /** Finalize a function call by logging relevant information.*/
    TerminationStatus terminate(TerminationStatus status);

    /** Initialize the problem data. In particular compute the initial primal-dual point
      * and perform the initial decompositions.
      */
    internal::InitTermination init();
    /** Select a violated constraint and return it's normal \p n+.*/
    internal::ConstraintNormal selectViolatedConstraint(const VectorConstRef& x) const;
    /** Compute a primal step z and dual step r, given \p n+*/
    void computeStep(VectorRef z, VectorRef r, const internal::ConstraintNormal& np) const;
    /** Compute a step length and update x and u given the data n+, z and r.*/
    StepLenghth computeStepLength(const internal::ConstraintNormal& np, const VectorConstRef& x, 
      const VectorConstRef& u, const VectorConstRef& z, const VectorConstRef& r) const;
    /** Add a constraint to the active set and update the computation data accordingly.*/
    bool addConstraint(const internal::ConstraintNormal& np);
    /** Remove the l-th active constraint from the active set and update the
      * computation data accordingly.
      */
    bool removeConstraint(int l, VectorRef u);


    /** Compute the initial iterate, the corresponding objective value and
      * initialize any relevant data of the derived class.
      */
    virtual internal::InitTermination init_() = 0;
    /** Select the violated constraint to be considered for the current iteration.*/
    virtual internal::ConstraintNormal selectViolatedConstraint_(const VectorConstRef& x) const = 0;
    /** Compute a primal step z and dual step r, given \p n+*/
    virtual void computeStep_(VectorRef z, VectorRef r, const internal::ConstraintNormal& np) const = 0;
    /** Compute a step length and update x and u given the data n+, z and r.*/
    virtual StepLenghth  computeStepLength_(const internal::ConstraintNormal& np, const VectorConstRef& x,
      const VectorConstRef& u, const VectorConstRef& z, const VectorConstRef& r) const = 0;
    /** Add a constraint to the active set and update the computation data accordingly.*/
    virtual bool addConstraint_(const internal::ConstraintNormal& np) = 0;
    /** Remove the l-th active constraint from the active set and update the
      * computation data accordingly.
      */
    virtual bool removeConstraint_(int l) = 0;
    /** Resize the data managed by the derived class.*/
    virtual void resize_(int nbVar, int nbCstr, bool useBounds) = 0;

  private:
    /** Resize the data managed by this class.*/
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
