/* Copyright 2020 CNRS-AIST JRL */

#include <jrl-qp/utils/Logger.h>

namespace jrl::qp::utils
{
Logger & Logger::setFlag(std::uint32_t flag, bool add)
{
  if(add)
    flags_ |= flag;
  else
    flags_ &= ~flag;
  return *this;
}

Logger & Logger::setOutputStream(std::ostream & os)
{
  os_ = &os;
  return *this;
}

void Logger::startIter(int i)
{
  iter_ = i;
  log(~decltype(flags_)(0), "it", i); // ~decltype(flags_)(0): number of the same type as flags, with all bits to 1
}

Logger Logger::subLog(const std::string & name) const
{
  return {*os_, name_ + "(" + std::to_string(iter_ + 1) + ")." + name, flags_};
}
} // namespace jrl::qp::utils