/* Copyright 2020 CNRS-AIST JRL */

#pragma once

#include <Eigen/Core>

#include <jrl-qp/SolverOptions.h>
#include <jrl-qp/api.h>
#include <jrl-qp/defs.h>
#include <jrl-qp/internal/ActiveSet.h>
#include <jrl-qp/internal/SelectedConstraint.h>
#include <jrl-qp/internal/TerminationType.h>
#include <jrl-qp/internal/Workspace.h>
#include <jrl-qp/utils/Logger.h>
#include <jrl-qp/utils/debug.h>

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
  void options(const SolverOptions & options);

  /** Get the solution.*/
  WConstVector solution() const;
  /** Get the Lagrange multipliers at the solution.*/
  WConstVector multipliers() const;
  /** Get the objective value at the solution.*/
  double objectiveValue() const;
  /** Get the number of active-set iterations used to find the solution*/
  int iterations() const;

  /** Get the active set at the solution.
   *
   * The i-th element corresponds to the activation status of the i-th constraint
   * with the general constraints first, followed by the bound constraints.
   */
  const std::vector<ActivationStatus> & activeSet() const;

  /** Reset the active set.*/
  void resetActiveSet();

protected:
  struct StepLength
  {
    double t1;
    double t2;
    int l;
  };

  /** Call to the solving routines. Need to be initiated by the derived class. */
  TerminationStatus solve();
  /** Finalize a function call by logging relevant information.*/
  TerminationStatus terminate(TerminationStatus status);

  /** Initialize the problem data. In particular compute the initial primal-dual point
   * and perform the initial decompositions.
   */
  internal::InitTermination init();
  /** Select a violated constraint and return it as a description of \p n+.*/
  internal::SelectedConstraint selectViolatedConstraint(const VectorConstRef & x) const;
  /** Compute a primal step z and dual step r, given \p n+*/
  void computeStep(VectorRef z, VectorRef r, const internal::SelectedConstraint & sc) const;
  /** Compute a step length and update x and u given the data n+, z and r.*/
  StepLength computeStepLength(const internal::SelectedConstraint & sc,
                               const VectorConstRef & x,
                               const VectorConstRef & u,
                               const VectorConstRef & z,
                               const VectorConstRef & r) const;
  /** Add a constraint to the active set and update the computation data accordingly.*/
  bool addConstraint(const internal::SelectedConstraint & sc);
  /** Remove the l-th active constraint from the active set and update the
   * computation data accordingly.
   */
  bool removeConstraint(int l, VectorRef u);
  /** Compute the dot product between the vector corresponding to sc and z.*/
  virtual double dot(const internal::SelectedConstraint & sc, const VectorConstRef & z);

  /** Compute the initial iterate, the corresponding objective value and
   * initialize any relevant data of the derived class.
   */
  virtual internal::InitTermination init_() = 0;
  /** Select the violated constraint to be considered for the current iteration.*/
  virtual internal::SelectedConstraint selectViolatedConstraint_(const VectorConstRef & x) const = 0;
  /** Compute a primal step z and dual step r, given \p n+*/
  virtual void computeStep_(VectorRef z, VectorRef r, const internal::SelectedConstraint & sc) const = 0;
  /** Compute a step length and update x and u given the data n+, z and r.*/
  virtual StepLength computeStepLength_(const internal::SelectedConstraint & sc,
                                        const VectorConstRef & x,
                                        const VectorConstRef & u,
                                        const VectorConstRef & z,
                                        const VectorConstRef & r) const = 0;
  /** Add a constraint to the active set and update the computation data accordingly.*/
  virtual bool addConstraint_(const internal::SelectedConstraint & sc) = 0;
  /** Remove the l-th active constraint from the active set and update the
   * computation data accordingly.
   */
  virtual bool removeConstraint_(int l) = 0;
  /** Compute the dot product between the vector corresponding to sc and z.*/
  virtual double dot_(const internal::SelectedConstraint & sc, const VectorConstRef & z) = 0;
  /** Resize the data managed by the derived class.*/
  virtual void resize_(int nbVar, int nbCstr, bool useBounds) = 0;

private:
  /** Resize the data managed by this class.*/
  void resize_p(int nbVar, int nbCstr, bool useBounds);

protected:
  SolverOptions options_;
  utils::Logger log_;

  int it_; // number of iterations

  int nbVar_;
  internal::ActiveSet A_;

  double f_;
  internal::Workspace<> work_x_;
  internal::Workspace<> work_z_;
  mutable internal::Workspace<> work_u_;
  mutable internal::Workspace<> work_r_;
  mutable bool needToExpandMultipliers_;
};

} // namespace jrl::qp
