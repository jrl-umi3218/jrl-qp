/* Copyright 2020 CNRS-AIST JRL */

#pragma once

#include <jrl-qp/api.h>
#include <jrl-qp/enums.h>

namespace jrl::qp::internal
{
  /** A utility class to indicate the termination status of a call to a function
    * or sub-function.
    *
    * Meant to be derived and give a strong type to TerminationStatus values.
    */
  class TerminationType
  {
  public:
    TerminationType(TerminationStatus status) : status_(status) {}

    operator bool() const { return status_ == TerminationStatus::SUCCESS; }
    operator TerminationStatus() const { return status_; }

    TerminationStatus status() const { return status_; }

  private:
    TerminationStatus status_;
  };

  class InitTermination : public TerminationType
  {
    using TerminationType::TerminationType;
  };
}