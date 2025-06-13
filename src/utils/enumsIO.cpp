#include <jrl-qp/utils/enumsIO.h>

#include <iostream>

std::ostream & operator<<(std::ostream & os, jrl::qp::TerminationStatus s)
{
  switch(s)
  {
    case jrl::qp::TerminationStatus::SUCCESS:
      os << "success";
      return os;
    case jrl::qp::TerminationStatus::INCONSISTENT_INPUT:
      os << "inconsistent inputs";
      return os;
    case jrl::qp::TerminationStatus::NON_POS_HESSIAN:
      os << "non positive hessian matrix";
      return os;
    case jrl::qp::TerminationStatus::INFEASIBLE:
      os << "infeasible";
      return os;
    case jrl::qp::TerminationStatus::MAX_ITER_REACHED:
      os << "maximum iteration reached";
      return os;
    case jrl::qp::TerminationStatus::LINEAR_DEPENDENCY_DETECTED:
      os << "linear dependency detected";
      return os;
    case jrl::qp::TerminationStatus::OVERCONSTRAINED_PROBLEM:
      os << "overconstrained problem";
      return os;
    default:
      os << "unknown failure (" << static_cast<int>(s) << ")";
      return os;
  }
}
