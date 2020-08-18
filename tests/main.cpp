/* Copyright 2020 CNRS-AIST JRL */

#include <iostream>
#include <chrono>
#include <vector>

#include <Eigen/Core>
#include <Eigen/QR>

#include <jrl-qp/utils/debug.h>
#include <jrl-qp/utils/Logger.h>

using namespace Eigen;
using namespace jrl::qp;


int main()
{
  MatrixXd A = MatrixXd::Random(3, 5);
  
  utils::Logger l(std::cout, "qp");
  l.setFlag(0x00000001);
  l.startIter(0);
  l.log(0x00000001, "test", 3);
  l.log(0x00000010, "test2", 4);
  l.log(0x00000001, "A", A);

  LOG_NEW_ITER(l,1);
  l.log(1, "n", 5, "A", A, "r", 3.14);

  auto s = l.subLog("choice");
  s.startIter(0);
  s.log(1, "b", Vector3d(3, -1, 2));

  int c = 2;
  double d;

  LOG(l, 1, A, c);
  LOG_COMMENT(l, 1, "This is a comment");
  DBG_COMMENT(l, 1, "This is another comment");

  std::uint32_t u = 1;
  std::cout << ~u << std::endl;
  std::cout << ~decltype(u)(0) << std::endl;
}
