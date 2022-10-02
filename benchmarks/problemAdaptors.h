/* Copyright 2020 CNRS-AIST JRL
 */

#pragma once

#include <jrl-qp/test/problems.h>

namespace jrl::qp::test
{
struct GIPb : public QPProblem<>
{
  GIPb() = default;
  GIPb(const QPProblem<true> & pb);
};

struct EigenQuadprogPb
{
  EigenQuadprogPb() = default;
  EigenQuadprogPb(const QPProblem<true> & pb);

  Eigen::MatrixXd Q;
  Eigen::VectorXd c;
  Eigen::MatrixXd Aeq;
  Eigen::VectorXd beq;
  Eigen::MatrixXd Aineq;
  Eigen::VectorXd bineq;
};

struct EiQuadprogPb
{
  EiQuadprogPb() = default;
  EiQuadprogPb(const QPProblem<true> & pb);

  Eigen::MatrixXd G;
  Eigen::VectorXd g0;
  Eigen::MatrixXd CE;
  Eigen::VectorXd ce0;
  Eigen::MatrixXd CI;
  Eigen::VectorXd ci0;
};

struct LssolPb
{
  LssolPb() = default;
  LssolPb(const QPProblem<true> & pb);

  Eigen::MatrixXd Q;
  Eigen::VectorXd p;
  Eigen::MatrixXd C;
  Eigen::VectorXd l;
  Eigen::VectorXd u;
};

struct ProxSuitePb
{
  ProxSuitePb() = default;
  ProxSuitePb(const QPProblem<true> & pb);

  Eigen::MatrixXd H;
  Eigen::VectorXd g;
  Eigen::MatrixXd A;
  Eigen::VectorXd b;
  Eigen::MatrixXd C;
  Eigen::VectorXd l;
  Eigen::VectorXd u;
};

struct QLDPb
{
  QLDPb() = default;
  QLDPb(const QPProblem<true> & pb);

  Eigen::MatrixXd Q;
  Eigen::VectorXd c;
  Eigen::MatrixXd A;
  Eigen::VectorXd b;
  Eigen::VectorXd xl;
  Eigen::VectorXd xu;
};
} // namespace jrl::qp::test