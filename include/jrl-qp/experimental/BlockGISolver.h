/* Copyright 2020 CNRS-AIST JRL */

#pragma once

#include <vector>

#include <Eigen/Core>

#include <jrl-qp/DualSolver.h>
#include <jrl-qp/structured/StructuredC.h>
#include <jrl-qp/structured/StructuredG.h>
#include <jrl-qp/structured/StructuredJ.h>
#include <jrl-qp/structured/StructuredQR.h>

namespace jrl::qp::experimental
{
/** Implementation of the Goldfarb-Idnani dual QP solver.*/
class JRLQP_DLLAPI BlockGISolver : public DualSolver
{
public:
  BlockGISolver();
  /** Pre-allocate the data for a problem with \p nbVar variables, \p nbCstr
   * (general) constraints, and bounds if \p useBounds is \a true.*/
  BlockGISolver(int nbVar, int nbCstr, bool useBounds);

  virtual ~BlockGISolver() = default;

  /** Solve the problem
   *  min. 0.5 x^T G x + a^T x
   *  s.t. bl <= C^T x <= bu
   *       xl <=  x <= xu
   */
  TerminationStatus solve(const structured::StructuredG & G,
                          const VectorConstRef & a,
                          const structured::StructuredC & C,
                          const VectorConstRef & bl,
                          const VectorConstRef & bu,
                          const VectorConstRef & xl,
                          const VectorConstRef & xu,
                          const std::vector<ActivationStatus> & as = {});

protected:
  /** Structure to gather the problem definition. */
  struct Problem
  {
  public:
    Problem()
    : G(), a(Eigen::VectorXd(0)), C(), bl(Eigen::VectorXd(0)), bu(Eigen::VectorXd(0)), xl(Eigen::VectorXd(0)),
      xu(Eigen::VectorXd(0))
    {
    }

    // Map<const EigenObj> and thus Ref<const EigenObj> has its operator= deleted.
    // We thus need to delete as well for Problem. This is not an issue as a solver is not
    // meant to be copied. If it was needed to implement this operator, on would need to
    // use placement new tricks on the XXXConstRef members.
    Problem & operator=(const Problem &) = delete;
    structured::StructuredG G;
    VectorConstRef a;
    structured::StructuredC C;
    VectorConstRef bl;
    VectorConstRef bu;
    VectorConstRef xl;
    VectorConstRef xu;
    std::vector<ActivationStatus> as;
  };

  internal::InitTermination init_() override;
  internal::SelectedConstraint selectViolatedConstraint_(const VectorConstRef & x) const override;
  void computeStep_(VectorRef z, VectorRef r, const internal::SelectedConstraint & sc) const override;
  StepLength computeStepLength_(const internal::SelectedConstraint & sc,
                                const VectorConstRef & x,
                                const VectorConstRef & u,
                                const VectorConstRef & z,
                                const VectorConstRef & r) const override;
  bool addConstraint_(const internal::SelectedConstraint & sc) override;
  bool removeConstraint_(int l) override;
  double dot_(const internal::SelectedConstraint & sc, const VectorConstRef & z) override;
  void resize_(int nbVar, int nbCstr, bool useBounds) override;

  virtual internal::TerminationType processInitialActiveSet();
  virtual internal::TerminationType initializeComputationData();
  virtual internal::TerminationType initializePrimalDualPoints();

  mutable internal::Workspace<> work_d_;
  structured::StructuredJ J_;
  structured::StructuredQR QR_;
  // internal::Workspace<> work_tmp_; // for multiplication by Householder transform
  // internal::Workspace<> work_hCoeffs_; // for initial QR decomposition
  mutable internal::Workspace<> work_cx_;
  internal::Workspace<> work_bact_;
  Problem pb_;
};

} // namespace jrl::qp::experimental
