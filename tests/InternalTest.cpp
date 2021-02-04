/* Copyright 2020-2021 CNRS-AIST JRL */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

#include <jrl-qp/internal/OrthonormalSequence.h>
#include <jrl-qp/internal/SingleNZSegmentVector.h>
#include <jrl-qp/internal/Workspace.h>

#include <Eigen/LU>

using namespace Eigen;
using namespace jrl::qp;
using namespace jrl::qp::internal;

TEST_CASE("Workspace change ld")
{
  Workspace w(10, 10);
  MatrixXd M(4, 4);
  M << 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16;

  auto W1 = w.asMatrix(4, 4, 8);
  W1 = M;

  w.changeLd(4, 4, 8, 5);
  auto W2 = w.asMatrix(4, 4, 5);
  FAST_CHECK_EQ(M, W2);

  w.changeLd(4, 4, 5, 10);
  auto W3 = w.asMatrix(4, 4, 10);
  FAST_CHECK_EQ(M, W3);
}

TEST_CASE("ElemOrthonormalSequence Householder")
{
  // Multiple householder transform
  {
    ElemOrthonormalSequence H(OSeqType::Householder, 8, 3);
    VectorXd v1 = VectorXd::Random(8);
    VectorXd v2 = VectorXd::Random(7);
    VectorXd v3 = VectorXd::Random(6);
    double tau1, tau2, tau3, beta;
    v1.makeHouseholderInPlace(tau1, beta);
    v2.makeHouseholderInPlace(tau2, beta);
    v3.makeHouseholderInPlace(tau3, beta);
    H.add(v1.tail(7), tau1);
    H.add(v2.tail(6), tau2);
    H.add(v3.tail(5), tau3);

    // Apply to full vector"
    {
      VectorXd u = VectorXd::Random(8);
      VectorXd v = u;
      VectorXd Hu = u;
      VectorXd w(8);
      Hu.tail(6).applyHouseholderOnTheLeft(v3.tail(5), tau3, w.data());
      Hu.tail(7).applyHouseholderOnTheLeft(v2.tail(6), tau2, w.data());
      Hu.applyHouseholderOnTheLeft(v1.tail(7), tau1, w.data());

      H.applyToTheLeft(u, 0, 8);

      FAST_CHECK_UNARY(u.isApprox(Hu, 1e-8));

      MatrixXd Q = H.toDense();
      FAST_CHECK_UNARY(u.isApprox(Q * v, 1e-8));
      FAST_CHECK_EQ(u.norm(), doctest::Approx(v.norm()));

      u = v;
      H.applyTransposeToTheLeft(u, 0, 8);
      FAST_CHECK_UNARY(u.isApprox(Q.transpose() * v, 1e-8));
      FAST_CHECK_EQ(u.norm(), doctest::Approx(v.norm()));
    }

    // Apply to partial vector
    {
      // u = [0 0 x x x 0 0 0]
      VectorXd u = VectorXd::Zero(8);
      u.segment(2, 3).setRandom();
      VectorXd v = u;

      MatrixXd Q = H.toDense();
      for(int i = 0; i < 3; ++i)
      {
        for(int j = 0; j < 4; ++j)
        {
          u = v; // restore initial value
          H.applyToTheLeft(u, i, 8 - i - j);

          FAST_CHECK_UNARY(u.isApprox(Q * v, 1e-8));
          FAST_CHECK_EQ(u.norm(), doctest::Approx(v.norm()));

          u = v; // restore initial value
          H.applyTransposeToTheLeft(u, i, 8 - i - j);
          FAST_CHECK_UNARY(u.isApprox(Q.transpose() * v, 1e-8));
          FAST_CHECK_EQ(u.norm(), doctest::Approx(v.norm()));
        }
      }
    }
  }

  // Single householder transform
  {
    ElemOrthonormalSequence H(
        OSeqType::Householder, 8,
        3); // We keep 3 on purpose to test the use case where preallocation is larger than needed.
    VectorXd v1 = VectorXd::Random(8);
    double tau1, beta;
    v1.makeHouseholderInPlace(tau1, beta);
    H.add(v1.tail(7), tau1);

    // Apply to full vector
    {
      VectorXd u = VectorXd::Random(8);
      VectorXd v = u;
      VectorXd Hu = u;
      VectorXd w(8);
      Hu.applyHouseholderOnTheLeft(v1.tail(7), tau1, w.data());

      H.applyToTheLeft(u, 0, 8);

      FAST_CHECK_UNARY(u.isApprox(Hu, 1e-8));

      MatrixXd Q = H.toDense();
      FAST_CHECK_UNARY(u.isApprox(Q * v, 1e-8));
      FAST_CHECK_EQ(u.norm(), doctest::Approx(v.norm()));

      u = v;
      H.applyTransposeToTheLeft(u, 0, 8);
      FAST_CHECK_UNARY(u.isApprox(Q.transpose() * v, 1e-8));
      FAST_CHECK_EQ(u.norm(), doctest::Approx(v.norm()));
    }

    // Apply to partial vector
    {
      // u = [0 0 x x x 0 0 0]
      VectorXd u = VectorXd::Zero(8);
      u.segment(2, 3).setRandom();
      VectorXd v = u;

      MatrixXd Q = H.toDense();
      for(int i = 0; i < 3; ++i)
      {
        for(int j = 0; j < 4; ++j)
        {
          u = v; // restore initial value
          H.applyToTheLeft(u, i, 8 - i - j);

          FAST_CHECK_UNARY(u.isApprox(Q * v, 1e-8));
          FAST_CHECK_EQ(u.norm(), doctest::Approx(v.norm()));

          u = v; // restore initial value
          H.applyTransposeToTheLeft(u, i, 8 - i - j);
          FAST_CHECK_UNARY(u.isApprox(Q.transpose() * v, 1e-8));
          FAST_CHECK_EQ(u.norm(), doctest::Approx(v.norm()));
        }
      }
    }
  }
}

TEST_CASE("ElemOrthonormalSequence Givens")
{
  Givens G1, G2, G3, G4, G5;
  G1.makeGivens(1, 2);
  G2.makeGivens(3, 4);
  G3.makeGivens(5, 6);
  G4.makeGivens(7, 8);
  G5.makeGivens(9, 10);
  ElemOrthonormalSequence H(OSeqType::Givens, 8, 5);
  H.add(G1);
  H.add(G2);
  H.add(G3);
  H.add(G4);
  H.add(G5);

  //apply to full vector
  {
    VectorXd u = VectorXd::Random(8);
    VectorXd v = u;
    VectorXd Hu = u;
    Hu.applyOnTheLeft(4, 5, G5);
    Hu.applyOnTheLeft(3, 4, G4);
    Hu.applyOnTheLeft(2, 3, G3);
    Hu.applyOnTheLeft(1, 2, G2);
    Hu.applyOnTheLeft(0, 1, G1);

    H.applyToTheLeft(u, 0, 8);
    FAST_CHECK_UNARY(u.isApprox(Hu, 1e-8));

    MatrixXd Q = H.toDense();
    FAST_CHECK_UNARY(u.isApprox(Q * v, 1e-8));

    u = v;
    H.applyTransposeToTheLeft(u, 0, 8);
    FAST_CHECK_UNARY(u.isApprox(Q.transpose() * v, 1e-8));
  }

  // Apply to partial vector
  {
    // u = [0 0 x x x 0 0 0]
    VectorXd u = VectorXd::Zero(8);
    u.segment(2, 3).setRandom();
    VectorXd v = u;

    MatrixXd Q = H.toDense();
    for(int i = 0; i < 3; ++i)
    {
      for(int j = 0; j < 4; ++j)
      {
        u = v; // restore initial value
        H.applyToTheLeft(u, i, 8 - i - j);

        FAST_CHECK_UNARY(u.isApprox(Q * v, 1e-8));

        u = v; // restore initial value
        H.applyTransposeToTheLeft(u, i, 8 - i - j);
        FAST_CHECK_UNARY(u.isApprox(Q.transpose() * v, 1e-8));
      }
    }
  }
  }

TEST_CASE("OrthonormalSequence")
{
  OrthonormalSequence H(16);
  VectorXd v1 = VectorXd::Random(6);
  VectorXd v2 = VectorXd::Random(5);
  VectorXd v3 = VectorXd::Random(4);
  double tau1, tau2, tau3, beta;
  v1.makeHouseholderInPlace(tau1, beta);
  v2.makeHouseholderInPlace(tau2, beta);
  v3.makeHouseholderInPlace(tau3, beta);
  // 6x6 Transformation starting at index 2
  H.prepare(OSeqType::Householder, 6, 3);
  H.add(2, v1.tail(5), tau1);
  H.add(3, v2.tail(4), tau2);
  H.add(4, v3.tail(3), tau3);

  Givens G1, G2, G3, G4, G5;
  G1.makeGivens(1, 2);
  G2.makeGivens(3, 4);
  G3.makeGivens(5, 6);
  G4.makeGivens(7, 8);
  G5.makeGivens(9, 10);

  // 6x6 Transformation starting at index 5
  H.prepare(OSeqType::Givens, 6, 5);
  H.add(5, G1);
  H.add(6, G2);
  H.add(7, G3);
  H.add(8, G4);
  H.add(9, G5);

  // 4x4 Transformation starting at index 12
  VectorXd v4 = VectorXd::Random(4);
  double tau4;
  v4.makeHouseholderInPlace(tau4, beta);
  H.prepare(OSeqType::Householder, 4, 1);
  H.add(12, v4.tail(3), tau4);

  // 7x7 Transformation starting at index 1
  VectorXd v5 = VectorXd::Random(7);
  double tau5;
  v5.makeHouseholderInPlace(tau5, beta);
  H.prepare(OSeqType::Householder, 7, 1);
  H.add(1, v5.tail(6), tau5);

  MatrixXd Q = MatrixXd::Identity(16, 16);
  ElemOrthonormalSequence H1(OSeqType::Householder, 6, 3);
  H1.add(v1.tail(5), tau1);
  H1.add(v2.tail(4), tau2);
  H1.add(v3.tail(3), tau3);
  Q.middleCols(2, 6) *= H1.toDense();
  ElemOrthonormalSequence H2(OSeqType::Givens, 6, 5);
  H2.add(G1);
  H2.add(G2);
  H2.add(G3);
  H2.add(G4);
  H2.add(G5);
  Q.middleCols(5, 6) *= H2.toDense();
  ElemOrthonormalSequence H3(OSeqType::Householder, 4, 1);
  H3.add(v4.tail(3), tau4);
  Q.middleCols(12, 4) *= H3.toDense();
  ElemOrthonormalSequence H4(OSeqType::Householder, 7, 1);
  H4.add(v5.tail(6), tau5);
  Q.middleCols(1, 7) *= H4.toDense();

  // Test on full vector
  {
    VectorXd u = VectorXd::Random(16);
    VectorXd v = u;

    H.applyToTheLeft(u);
    FAST_CHECK_UNARY(u.isApprox(Q * v, 1e-8));
    FAST_CHECK_EQ(u.norm(), doctest::Approx(v.norm()));

    u = v;
    H.applyTransposeToTheLeft(u);
    FAST_CHECK_UNARY(u.isApprox(Q.transpose() * v, 1e-8));
    FAST_CHECK_EQ(u.norm(), doctest::Approx(v.norm()));
  }

  // Test on partial vector
  {
    VectorXd r = VectorXd::Random(4);
    for(int i=0; i<=12; ++i)
    {
      SingleNZSegmentVector u(r, i, 16);
      VectorXd v(16);
      u.toFullVector(v);
      VectorXd w(16);

      H.applyToTheLeft(w, u);
      FAST_CHECK_UNARY(w.isApprox(Q * v, 1e-8));
      FAST_CHECK_EQ(w.norm(), doctest::Approx(v.norm()));

      H.applyTransposeToTheLeft(w, u);
      FAST_CHECK_UNARY(w.isApprox(Q.transpose() * v, 1e-8));
      FAST_CHECK_EQ(w.norm(), doctest::Approx(v.norm()));
    }
  }
}