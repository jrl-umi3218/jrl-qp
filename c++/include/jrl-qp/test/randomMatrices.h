/* Copyright 2020 CNRS-AIST JRL
 */

#pragma once

#include <Eigen/Core>

#include <3rd-party/effolkronium/random.hpp>


namespace jrlqp::test
{
  /** Functor returning number following a normal distribution.*/
  template<typename Scalar>
  struct scalar_normal_random_op
  {
    scalar_normal_random_op(double mean = 0, double stddev = 1)
      : d_(mean, stddev) {}

    inline const Scalar operator() () const { return effolkronium::random_static::get(d_); }

    mutable std::normal_distribution<Scalar> d_;
  };

  /** Generate a random vector with elements following a normal distribution.
    * \param size Size of the vector.
    * \param mean Mean parameter of the normal distribution.
    * \param stddev Standard deviation parameter of the normal distribution.
    */
  inline auto randnVec(Eigen::Index size, double mean = 0, double stddev = 1)
  {
    return Eigen::VectorXd::NullaryExpr(size, scalar_normal_random_op<double>(mean, stddev));
  }

  /** Generate a random matrix with elements following a normal distribution.
    * \param rows Number of rows of the matrix.
    * \param cols Number of columns of the matrix.
    * \param mean Mean parameter of the normal distribution.
    * \param stddev Standard deviation parameter of the normal distribution.
    */
  inline auto randnMat(Eigen::Index rows, Eigen::Index cols, double mean = 0, double stddev = 1)
  {
    return Eigen::MatrixXd::NullaryExpr(rows, cols, scalar_normal_random_op<double>(mean, stddev));
  }

  /** Generate a random unit vector with uniform distribution on the unit sphere.
    * \param size Size of the vector.
    */
  inline Eigen::VectorXd randUnitVec(Eigen::Index size)
  {
    Eigen::VectorXd r = randnVec(size);
    r.normalize();
    return r;
  }

  /** Generate a random orthogonal matrix with Haar distribution.
    * \param size Number of rows and columns of the matrix.
    * \param special If \a true, the determinant of the generated matrix is 1
    * (i.e the matrix belongs to SO(size)).
    * Otherwise it is 1 or -1 with 50% probability each (it belongs to O(size))
    */
  inline Eigen::MatrixXd randOrtho(Eigen::Index size, bool special = false)
  {
    assert(size >= 0);
    Eigen::VectorXd buffv(size);
    Eigen::VectorXd buffw(size);
    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(size, size);

    if (size == 0)
      return Q;

    bool positiveDet;
    if (effolkronium::random_static::get<bool>())
    {
      Q(size - 1, size - 1) = 1;
      positiveDet = true;
    }
    else
    {
      Q(size - 1, size - 1) = -1;
      positiveDet = false;
    }

    for (Eigen::Index i = 2; i <= size; ++i)
    {
      auto v = buffv.head(i);
      auto w = buffw.head(i);

      // Random unit vector
      v = randUnitVec(i);

      // compute (v+sgn(v_1)*e_1)/||v+sgn(v_1)*e_1||
      bool positive = v[0] >= 0;
      if (positive)
      {
        v[0] += 1;
        if (i % 2 == 0)
          positiveDet = !positiveDet;
      }
      else
      {
        v[0] -= 1;
        positiveDet = !positiveDet;
      }
      v.normalize();

      Eigen::MatrixXd H;
      if (positive)
        H = -(Eigen::MatrixXd::Identity(i, i) - 2 * v * v.transpose());
      else
        H = Eigen::MatrixXd::Identity(i, i) - 2 * v * v.transpose();
      // Apply the Householder transformation H = -sgn(v_1)*(I-2*vv^T) : Q = H*Q
      w = Q.bottomRightCorner(i, i).transpose() * v;
      if (positive)
      {
        Q(size - i, size - i) = -1;
        Q.bottomRightCorner(i - 1, i - 1) *= -1;
        Q.bottomRightCorner(i, i).noalias() += 2 * v * w.transpose();
      }
      else
      {
        Q.bottomRightCorner(i, i).noalias() -= 2 * v * w.transpose();
      }
    }

    if (special && !positiveDet)
      Q.col(0) *= -1;

    return Q;
  }

  /** Generate a random matrix with a specified rank.
    *
    * Coefficients of the matrix have empirically a normal distribution with a
    * mean of 0 and a standard deviation of 1.
    *
    * \param rows Number of rows
    * \param cols Number of cols
    * \param rank Rank of the matrix. Must be lower or equal to both \p rows and
    * \p cols. If let to is default value or set negative, the minimum of \p rows
    * and \p cols is taken instead.
    *
    * \internal We generate two random orthogonal matrices U and V with Haar
    * distribution and a rectangular matrix S with elements to 0 everywhere but for
    * the \a rank first diagonal that are set to random values uniformly distributed
    * of [-s;s] with s = sqrt((3. * rows * cols) / rank). We returns U*S*V.
    * The scaling s comes from the empirical observation that the above technique
    * used with non-zero elements of S taken uniformly from [-1;1] yields matrices
    * whose elements are approximately following a normal distribution with mean 0
    * and variance equal to rank/(3*rows*cols).
    */
  inline Eigen::MatrixXd randn(Eigen::Index rows, Eigen::Index cols, Eigen::Index rank = -1)
  {
    assert(rank <= rows && rank <= cols && "Invalid rank");
    Eigen::Index p = std::min(rows, cols);
    if (rank < 0)
      rank = p;

    if (rank == 0)
      return Eigen::MatrixXd::Zero(rows, cols);

    if (rank == p)
      return randnMat(rows, cols);

    Eigen::VectorXd s = Eigen::VectorXd::Zero(p);
    s.head(rank).setRandom();
    // set the correct variance
    s.head(rank) *= std::sqrt((3. * rows * cols) / rank);

    Eigen::MatrixXd M(rows, cols);

    if (rows <= cols)
    {
      M.leftCols(rows).noalias() = randOrtho(rows) * s.asDiagonal();
      M.rightCols(cols - rows).setZero();
      return M * randOrtho(cols);
    }
    else
    {
      M.topRows(cols).noalias() = s.asDiagonal() * randOrtho(cols);
      M.bottomRows(rows - cols).setZero();
      return randOrtho(rows) * M;
    }
  }

  /** Returns two matrices A and B with specified sizes and ranks and such that [A;B] has a given rank
    *
    * \param cols Column size of A and B.
    * \param rowsA Row size of A.
    * \param rankA Rank of A.
    * \param rowsB Row size of B.
    * \param rankA Rank of B.
    * \param rankAB Rank of [AB].
    */
  inline std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> randDependent(Eigen::Index cols, Eigen::Index rowsA, Eigen::Index rankA, Eigen::Index rowsB, Eigen::Index rankB, Eigen::Index rankAB)
  {
    assert(rankA <= rowsA && rankA <= cols);
    assert(rankB <= rowsB && rankB <= cols);
    assert(rankAB >= rankA && rankAB >= rankB && rankAB <= rankA + rankB && rankAB <= cols);

    Eigen::MatrixXd M = randn(rankA + rankB, cols, rankAB);
    Eigen::MatrixXd A(rowsA, cols);
    Eigen::MatrixXd B(rowsB, cols);

    if (rankA == rowsA)
      A = M.topRows(rankA);
    else
      A.noalias() = randOrtho(rowsA).leftCols(rankA) * M.topRows(rankA);

    if (rankB == rowsB)
      B = M.bottomRows(rankB);
    else
      B.noalias() = randOrtho(rowsB).leftCols(rankB) * M.bottomRows(rankB);

    return std::make_tuple(A, B);
  }
}
