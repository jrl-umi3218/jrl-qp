/* Copyright 2020 CNRS-AIST JRL */

#pragma once

#include <Eigen/Dense>
#include <iosfwd>
#include <jrl-qp/internal/meta.h>

namespace jrl::qp::utils
{
/** Check if class \t T can be converted to Eigen::Ref<const Eigen::MatrixXd> */
template<typename T>
inline constexpr bool is_convertible_to_eigen_ref_v = std::is_convertible_v<T, Eigen::Ref<const Eigen::MatrixXd>>;

/** A small utility class to write Eigen matrices in a stream with a matlab-readable format.
 *
 * Example of use:
 * Eigen::MatrixXd M = Eigen::MatrixXd::Random(5,6);
 * std::cout << (toMatlab)M << std::endl;
 *
 * Inspired from a code by Nicolas Mansard.
 */
class toMatlab
{
public:
  toMatlab(const Eigen::Ref<const Eigen::MatrixXd> & M) : mat(M) {}

  template<typename Derived, typename std::enable_if<!is_convertible_to_eigen_ref_v<Derived>, int>::type = 0>
  toMatlab(const Eigen::EigenBase<Derived> & M) : tmp(M), mat(tmp)
  {
  }

private:
  Eigen::MatrixXd tmp;
  const Eigen::Ref<const Eigen::MatrixXd> mat;

  friend std::ostream & operator<<(std::ostream &, const toMatlab &);
};

inline std::ostream & operator<<(std::ostream & o, const toMatlab & tom)
{
  if(tom.mat.cols() == 1)
  {
    Eigen::IOFormat fmt(Eigen::StreamPrecision, 0, "; ", ";", "", "", "[", "]");
    o << tom.mat.transpose().format(fmt);
  }
  else
  {
    Eigen::IOFormat fmt(Eigen::StreamPrecision, 0, ", ", ";\n", "", "", "[", "]");
    o << tom.mat.format(fmt);
  }
  return o;
}
} // namespace jrl::qp::utils
