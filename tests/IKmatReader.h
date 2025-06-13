/* Copyright 2020-2021 CNRS-AIST JRL */

#pragma once

#include <string>

#include <Eigen/Core>

Eigen::MatrixXd readMat(const std::string & filename);
std::tuple<Eigen::MatrixXd,
           Eigen::VectorXd,
           Eigen::MatrixXd,
           Eigen::VectorXd,
           Eigen::MatrixXd,
           Eigen::VectorXd,
           Eigen::VectorXd,
           Eigen::VectorXd>
    readIKPbFile(const std::string & filename);
