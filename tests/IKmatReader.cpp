/* Copyright 2020-2021 CNRS-AIST JRL */

#include <fstream>
#include <vector>

#include "IKmatReader.h"

using namespace Eigen;

Eigen::MatrixXd readMat(const std::string & filename)
{
  std::ifstream aif(filename);
  std::vector<VectorXd> rows;
  if(aif.is_open())
  {
    std::string line;
    std::vector<double> row;
    while(std::getline(aif, line))
    {
      std::istringstream iss(line);
      row.clear();
      std::copy(std::istream_iterator<double>(iss), std::istream_iterator<double>(), std::back_inserter(row));
      rows.push_back(Map<VectorXd>(row.data(), row.size()));
    }
  }
  MatrixXd M(rows.size(), rows.size()?rows.front().size():0);
  for(size_t i=0; i<rows.size(); ++i)
  {
    M.row(i) = rows[i];
  }

  return M;
}
