/* Copyright 2020-2021 CNRS-AIST JRL */

#pragma once

#include <iosfwd>

#include <jrl-qp/api.h>
#include <jrl-qp/enums.h>

JRLQP_DLLAPI std::ostream & operator<<(std::ostream & os, jrl::qp::TerminationStatus s);
