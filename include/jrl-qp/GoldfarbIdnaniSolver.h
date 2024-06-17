/* Copyright 2020 CNRS-AIST JRL */

#pragma once

#include <Eigen/Core>

#include <jrl-qp/DualSolver.h>

namespace jrl::qp
{
/** Implementation of the Goldfarb-Idnani dual QP solver.*/
class JRLQP_DLLAPI GoldfarbIdnaniSolver : public DualSolver
{
public:
  GoldfarbIdnaniSolver();
  /** Pre-allocate the data for a problem with \p nbVar variables, \p nbCstr
   * (general) constraints, and bounds if \p useBounds is \a true.*/
  GoldfarbIdnaniSolver(int nbVar, int nbCstr, bool useBounds);

  virtual ~GoldfarbIdnaniSolver() = default;

  /** Solve the problem
   *  min. 0.5 x^T G x + a^T x
   *  s.t. bl <= C^T x <= bu
   *       xl <=  x <= xu
   */
  TerminationStatus solve(MatrixRef G,
                          const VectorConstRef & a,
                          const MatrixConstRef & C,
                          const VectorConstRef & bl,
                          const VectorConstRef & bu,
                          const VectorConstRef & xl,
                          const VectorConstRef & xu);

  /** This is used to set the precomputed R factor in the QR decomposition of the initially
   * active constraints. Should be used when options.gFactorization is GFactorization::L_TINV_Q.
   * Use only if you know what you are doing.
   */
  void setPrecomputedR(MatrixConstRef precompR);

protected:
  /** Structure to gather the problem definition. */
  struct Problem
  {
  public:
    Problem()
    : G(Eigen::Map<Eigen::MatrixXd>(0x0, 0, 0)), a(Eigen::VectorXd(0)), C(Eigen::MatrixXd(0, 0)),
      bl(Eigen::VectorXd(0)), bu(Eigen::VectorXd(0)), xl(Eigen::VectorXd(0)), xu(Eigen::VectorXd(0))
    {
    }

    // Map<const EigenObj> and thus Ref<const EigenObj> has its operator= deleted.
    // We thus need to delete as well for Problem. This is not an issue as a solver is not
    // meant to be copied. If it was needed to implement this operator, on would need to
    // use placement new tricks on the XXXConstRef members.
    Problem & operator=(const Problem &) = delete;
    MatrixRef G;
    VectorConstRef a;
    MatrixConstRef C;
    VectorConstRef bl;
    VectorConstRef bu;
    VectorConstRef xl;
    VectorConstRef xu;
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

  virtual void initActiveSet();

  void addInitialConstraint(const internal::SelectedConstraint & sc);

  internal::TerminationType processMatrixG();
  internal::TerminationType processInitialActiveSetWithEqualityOnly();
  internal::TerminationType initializeComputationData();
  internal::TerminationType initializePrimalDualPoints();

  mutable internal::Workspace<> work_d_;
  internal::Workspace<> work_J_;
  internal::Workspace<> work_R_;
  internal::Workspace<> work_tmp_;
  internal::Workspace<> work_hCoeffs_; // for initial QR decomposition
  internal::Workspace<> work_bact_;
  Problem pb_;
};

} // namespace jrl::qp
