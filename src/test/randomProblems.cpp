/* Copyright 2020 CNRS-AIST JRL */

#include <algorithm>
#include <iostream>

#include <Eigen/QR>

#include <jrl-qp/test/randomMatrices.h>
#include <jrl-qp/test/randomProblems.h>

using namespace Eigen;

namespace jrl::qp::test
{
RandomLeastSquare randomProblem(const ProblemCharacteristics & characs)
{
  characs.check();
  auto [nVar, nObj, nEq, nIneq, rankObj, nSharedRank, nStrongActIneq, nWeakActIneq, nStrongActBounds, nWeakActBounds,
        bounds, doubleSidedIneq, strictlyFeasible] = characs;

  int nstrongGeneral = nEq + nStrongActIneq;
  int nstrong = nstrongGeneral + nStrongActBounds; //<=nVar_
  int nBounds = bounds ? nVar : 0;

  RandomLeastSquare pb;

  // 1 - We generate the matrix A, the matrix of strongly active constraints a non-zero vector
  // [u;v] such that A^T u + Ca^T v = 0
  // If there are bounds, the last lines of Ca will be [I 0]. For now we consider the active
  // bounds to be on the first variables.
  MatrixXd Ca;
  VectorXd reducedMultipliers(nObj + nstrong);
  int colsTot = nObj + nstrong;
  if(nObj == 0)
  {
    pb.A.resize(0, nVar);
    Ca = randn(nstrong, nVar);
    reducedMultipliers.setZero();
  }
  else if(nstrong == 0)
  {
    pb.A = randn(nObj, nVar, rankObj);
    Ca.resize(0, nVar);
    reducedMultipliers.setZero();
  }
  else if(colsTot > nVar)
  {
    int rankTot = rankObj + nstrong - nSharedRank;
    if(rankTot > nVar) rankTot = nVar;
    std::tie(pb.A, Ca) = randDependent(nVar, nObj, rankObj, nstrong, nstrong, rankTot);
    Ca.bottomRows(nStrongActBounds).setIdentity();
    // [A^T Ca^T] has more than nVar_ columns: some are necessarily linearly dependent on others
    // We use a rank-revealing QR to separate a set of linearly independent columns from the other
    MatrixXd M(nVar, nObj + nstrong);
    M << pb.A.transpose(), Ca.transpose();
    ColPivHouseholderQR<MatrixXd> qr(M);
    auto R1 = qr.matrixQR().topLeftCorner(rankTot, rankTot).template triangularView<Upper>();
    auto R2 = qr.matrixQR().topRightCorner(rankTot, colsTot - rankTot);
    auto u = reducedMultipliers.head(rankTot);
    auto v = reducedMultipliers.tail(colsTot - rankTot);
    v.setRandom();
    // u = -inv(R1)*R2*v
    u.noalias() = -R2 * v;
    R1.solveInPlace(u);
    qr.colsPermutation().applyThisOnTheLeft(reducedMultipliers);
    // std::cout << "Reduced = \n" << reducedMultipliers.transpose() << std::endl;
    // std::cout << "M = \n" << M.transpose() << std::endl;
    // std::cout << (M * reducedMultipliers).transpose() << std::endl;
  }
  else
  {
    int rankTot = rankObj + nstrong - nSharedRank;
    if(rankTot == nVar) // We can't have this case or the only solution will be 0
      rankTot = nVar - 1;
    std::tie(pb.A, Ca) = randDependent(nVar, nObj, rankObj, nstrong, nstrong, rankTot);
    Ca.bottomRows(nStrongActBounds).setIdentity();
    // [A^T Ca^T] has at most rank nVar_-1 so that its null space is not empty
    MatrixXd M(nObj + nstrong, nVar);
    M << pb.A, Ca;
    ColPivHouseholderQR<MatrixXd> qr(M);
    MatrixXd Q = qr.matrixQ();
    auto N = Q.rightCols(nVar - rankTot); // nulspace of [A^T Ca^T]
    reducedMultipliers.noalias() = N * VectorXd::Random(nVar - rankTot);
    // std::cout << "Reduced = \n" << reducedMultipliers.transpose() << std::endl;
    // std::cout << "M = \n" << M << std::endl;
    // std::cout << (M.transpose() * reducedMultipliers).transpose() << std::endl;
  }

  // 2 - In case of simple bounds, we need to make sure of the Lagrange multipliers sign
  if(!doubleSidedIneq)
  {
    auto mult = reducedMultipliers.segment(nObj + nEq, nStrongActIneq);
    auto Ci = Ca.middleRows(nEq, nStrongActIneq);
    for(Eigen::Index i = 0; i < nStrongActIneq; ++i)
    {
      if(mult[i] < 0)
      {
        mult[i] = -mult[i];
        Ci.row(i) = -Ci.row(i);
      }
    }
  }

  // 3 - Creating final data variables and starting to populate them
  pb.x.resize(nVar);
  pb.b.resize(nObj);
  pb.E = Ca.topRows(nEq);
  pb.f.resize(nEq);
  pb.lambdaEq = reducedMultipliers.segment(nObj, nEq);
  pb.C.resize(nIneq, nVar);
  pb.l.resize(nIneq);
  pb.u.resize(nIneq);
  if(!doubleSidedIneq) // If constraints are not double-sided, they have the form Cx <= u
    pb.l.setConstant(-std::numeric_limits<double>::infinity());
  pb.lambdaIneq.resize(nIneq);
  pb.C.topRows(nStrongActIneq) = Ca.middleRows(nEq, nStrongActIneq);
  pb.lambdaIneq.head(nStrongActIneq) = reducedMultipliers.segment(nObj + nEq, nStrongActIneq);
  pb.lambdaIneq.tail(nIneq - nStrongActIneq).setZero();
  pb.xl.resize(nBounds);
  pb.xu.resize(nBounds);
  pb.lambdaBnd.resize(nBounds);
  pb.lambdaBnd.head(nStrongActBounds) = reducedMultipliers.tail(nStrongActBounds);
  pb.lambdaBnd.tail(nBounds - nStrongActBounds).setZero();
  pb.bounds = bounds;

  // 4 - Now we add weakly active and inactive constraints
  if(nWeakActIneq <= nstrong)
  {
    Eigen::MatrixXd Q = randOrtho(nstrong);
    auto Q1 = Q.topRows(nWeakActIneq);
    if(strictlyFeasible)
    {
      // We want the normal vectors of the weakly active constraints to be in the positive cone of
      // normal vectors of the strongly active constraints.
      // We change all normal vectors of strongly active constraints as if they were all active at
      // their upper bounds, and take a positive combination of them. At this stage, the resulting
      // weak constraints will be active at their upper bounds.
      auto mult = reducedMultipliers.tail(nstrong);
      pb.C.middleRows(nStrongActIneq, nWeakActIneq).noalias() = Q1.cwiseAbs() * mult.cwiseSign() * Ca;
    }
    else
      pb.C.middleRows(nStrongActIneq, nWeakActIneq).noalias() = Q1 * Ca;
  }
  else
  {
    Eigen::MatrixXd Q = randOrtho(nWeakActIneq);
    auto Q1 = Q.leftCols(nstrong);
    if(strictlyFeasible)
    {
      // See description above.
      auto mult = reducedMultipliers.tail(nstrong);
      pb.C.middleRows(nStrongActIneq, nWeakActIneq).noalias() = Q1.cwiseAbs() * mult.cwiseSign() * Ca;
    }
    else
      pb.C.middleRows(nStrongActIneq, nWeakActIneq).noalias() = Q1 * Ca;
  }
  pb.C.bottomRows(nIneq - nStrongActIneq - nWeakActIneq) = randn(nIneq - nStrongActIneq - nWeakActIneq, nVar);

  // 5 - We choose the value of the solution and of the vectors so that KKT conditions are satisfied
  pb.x.setRandom();
  // if (nObj > 0)
  pb.b = pb.A * pb.x - reducedMultipliers.head(nObj);
  // else
  //  b.resize(0);
  pb.f = pb.E * pb.x;
  pb.u.noalias() = pb.C * pb.x;
  if(doubleSidedIneq)
  {
    pb.l.noalias() = pb.C * pb.x;
    Eigen::VectorXd rl = Eigen::VectorXd::Random(nIneq).cwiseAbs();
    Eigen::VectorXd ru = Eigen::VectorXd::Random(nIneq).cwiseAbs();

    for(int i = 0; i < nStrongActIneq; ++i)
    {
      if(pb.lambdaIneq[i] > 0) // Upper bound is active
        pb.l[i] -= rl[i];
      else // Lower bound is active
        pb.u[i] += ru[i];
    }
    for(int i = nStrongActIneq; i < nStrongActIneq + nWeakActIneq; ++i)
    {
      // we randomly choose to activate at the upper or lower bound, changing the sign of the
      // constraint if needed.
      if(rl[i] > ru[i]) // This arbitrary criterion has a 50-50 chance to pass
        pb.l[i] -= rl[i]; // We consider the constraint active at its upper bound
      else
      {
        pb.C.row(i) = -pb.C.row(i);
        pb.u[i] = -pb.u[i] + ru[i];
      }
    }
    int inact = nIneq - nStrongActIneq - nWeakActIneq;
    pb.l.tail(inact) -= rl.tail(inact);
    pb.u.tail(inact) += ru.tail(inact);
  }
  else
  {
    int inact = nIneq - nStrongActIneq - nWeakActIneq;
    pb.u.tail(inact) += Eigen::VectorXd::Random(inact).cwiseAbs();
  }
  if(bounds)
  {
    Eigen::VectorXd r = Eigen::VectorXd::Random(nVar);
    pb.xl = pb.x;
    pb.xu = pb.x;
    for(int i = 0; i < nStrongActBounds; ++i)
    {
      if(pb.lambdaBnd[i] > 0) // Upper bound is active
        pb.xl[i] -= std::abs(r[i]);
      else // Lower bound is active
        pb.xu[i] += std::abs(r[i]);
    }
    for(int i = nStrongActBounds; i < nStrongActBounds + nWeakActBounds; ++i)
    {
      if(r[i] > 0) // We decide that the upper bound is active
        pb.xl[i] -= r[i];
      else // We decide that the lower bound is active
        pb.xu[i] -= r[i]; // we want +std::abs, and r[i] < 0, hence the -=
    }
    int inact = nVar - nStrongActBounds - nWeakActBounds;
    pb.xl.tail(inact) -= Eigen::VectorXd::Random(inact).cwiseAbs();
    pb.xu.tail(inact) += Eigen::VectorXd::Random(inact).cwiseAbs();
  }

  // 6 - Up to now, the constraints/bounds have been order as strongly acive, weakly active
  // and inactive. We randomize this order using Fisher–Yates shuffle algorithm.
  for(int i = nIneq - 1; i > 0; --i)
  {
    int j = effolkronium::random_static::get(0, i);
    pb.C.row(i).swap(pb.C.row(j));
    std::swap(pb.u[i], pb.u[j]);
    std::swap(pb.lambdaIneq[i], pb.lambdaIneq[j]);
    if(doubleSidedIneq) std::swap(pb.l[i], pb.l[j]);
  }
  if(bounds)
  {
    for(int i = nVar - 1; i > 0; --i)
    {
      int j = effolkronium::random_static::get(0, i);
      pb.A.col(i).swap(pb.A.col(j));
      pb.C.col(i).swap(pb.C.col(j));
      pb.E.col(i).swap(pb.E.col(j));
      std::swap(pb.xl[i], pb.xl[j]);
      std::swap(pb.xu[i], pb.xu[j]);
      std::swap(pb.lambdaBnd[i], pb.lambdaBnd[j]);
      std::swap(pb.x[i], pb.x[j]);
    }
  }

  return pb;
}

void ProblemCharacteristics::check() const
{
  assert(nVar_ >= 0 && nObj_ >= 0 && nEq_ >= 0 && nIneq_ >= 0 && rankObj_ >= 0 && nStrongActIneq_ >= 0
         && nWeakActIneq_ >= 0 && nStrongActBounds_ >= 0 && nWeakActBounds_ >= 0);
  assert(nVar_ >= nObj_);
  assert(nVar_ >= nEq_);
  assert(nStrongActIneq_ + nWeakActIneq_ <= nIneq_);
  assert((bounds_ && nStrongActBounds_ + nWeakActBounds_ <= nVar_)
         || (!bounds_ && nStrongActBounds_ == 0 && nWeakActBounds_ == 0));
  assert(nEq_ + nStrongActIneq_ + nStrongActBounds_ <= nVar_);
  assert(rankObj_ <= nObj_);
  assert(nSharedRank_ <= rankObj_);
}

template<typename Derived>
void disp(const std::string & name, const MatrixBase<Derived> & M)
{
  if(M.rows() != 0 && M.cols() != 0)
  {
    if(M.cols() > 1)
      std::cout << name << "= \n" << M << std::endl;
    else
      std::cout << name << "= \n" << M.transpose() << std::endl;
  }
}

void RandomLeastSquare::disp() const
{
  std::cout << "Output\n";

  std::cout << "x = \n" << x.transpose() << std::endl;
  std::cout << "A = \n" << A << std::endl;
  std::cout << "b = \n" << b.transpose() << std::endl;
  std::cout << "E = \n" << E << std::endl;
  std::cout << "f = \n" << f.transpose() << std::endl;
  std::cout << "C = \n" << C << std::endl;
  std::cout << "l = \n" << l.transpose() << std::endl;
  std::cout << "u = \n" << u.transpose() << std::endl;
  std::cout << "xl = \n" << xl.transpose() << std::endl;
  std::cout << "xu = \n" << xu.transpose() << std::endl;
  std::cout << "lambdaEq = \n" << lambdaEq.transpose() << std::endl;
  std::cout << "lambdaIneq = \n" << lambdaIneq.transpose() << std::endl;
  std::cout << "lambdaBnd = \n" << lambdaBnd.transpose() << std::endl;
}
} // namespace jrl::qp::test
