/* Copyright 2020-2021 CNRS-AIST JRL */

#include <fstream>
#include <vector>

#include "IKmatReader.h"

using namespace Eigen;

Eigen::VectorXd readVecLine(const std::string & line, std::vector<double> & buf)
{
  std::istringstream iss(line);
  buf.clear();
  std::copy(std::istream_iterator<double>(iss), std::istream_iterator<double>(), std::back_inserter(buf));
  return Map<VectorXd>(buf.data(), buf.size());
}

int readIntLine(std::ifstream & aif)
{
  std::string line;
  if(std::getline(aif, line))
  {
    std::istringstream iss(line);
    int i;
    iss >> i;
    if(!iss.fail()) return i;
  }

  throw std::runtime_error("Unable to read int");
}

bool isDataName(const std::string & line, const std::string & name)
{
  if(line[0] == '=')
  {
    return line.compare(5, name.size(), name) == 0;
  }
  else
    return false;
}

void readDataNameLine(std::ifstream & aif, const std::string & name)
{
  std::string line;
  if(std::getline(aif, line))
  {
    if(isDataName(line, name)) return;
  }

  throw std::runtime_error("Unable to read data name");
}

Eigen::MatrixXd readMat(std::ifstream & aif, int nbRows = -1, bool forceVec = false)
{
  std::string line;
  std::vector<double> row;
  if(nbRows < 0)
  {
    std::vector<VectorXd> rows;
    while(std::getline(aif, line)) rows.push_back(readVecLine(line, row));
    MatrixXd M(rows.size(), rows.size() ? rows.front().size() : 0);
    for(size_t i = 0; i < rows.size(); ++i) M.row(i) = rows[i];
    return M;
  }
  else if(nbRows == 0)
  {
    if(std::getline(aif, line))
      return MatrixXd(0, forceVec ? 1 : 0);
    else
      throw std::runtime_error("Unable to empty matrix row");
  }
  else
  {
    MatrixXd M;
    if(std::getline(aif, line))
    {
      auto r = readVecLine(line, row);
      M.resize(nbRows, r.size());
      M.row(0) = r;
    }
    else
      throw std::runtime_error("Unable to matrix row");
    for(int i = 1; i < nbRows; ++i)
    {
      if(std::getline(aif, line))
        M.row(i) = readVecLine(line, row);
      else
        throw std::runtime_error("Unable to matrix row");
    }
    return M;
  }
}

Eigen::MatrixXd readMat(const std::string & filename)
{
  std::ifstream aif(filename);
  if(aif.is_open())
  {
    return readMat(aif);
  }

  throw std::runtime_error("Unable to open file " + filename);
}

std::tuple<Eigen::MatrixXd,
           Eigen::VectorXd,
           Eigen::MatrixXd,
           Eigen::VectorXd,
           Eigen::MatrixXd,
           Eigen::VectorXd,
           Eigen::VectorXd,
           Eigen::VectorXd>
    readIKPbFile(const std::string & filename)
{
  const std::string field[] = {"dim_var", "dim_eq", "dim_ineq", "Q", "c", "A", "b", "C", "d", "x_min", "x_max"};
  std::ifstream aif(filename);
  std::vector<VectorXd> rows;
  if(aif.is_open())
  {
    readDataNameLine(aif, "dim_var");
    int n = readIntLine(aif);
    readDataNameLine(aif, "dim_eq");
    int me = readIntLine(aif);
    readDataNameLine(aif, "dim_ineq");
    int mi = readIntLine(aif);
    readDataNameLine(aif, "Q");
    Eigen::MatrixXd Q = readMat(aif, n);
    readDataNameLine(aif, "c");
    Eigen::VectorXd c = readMat(aif, n);
    readDataNameLine(aif, "A");
    Eigen::MatrixXd A = readMat(aif, me);
    readDataNameLine(aif, "b");
    Eigen::VectorXd b = readMat(aif, me, true);
    readDataNameLine(aif, "C");
    Eigen::MatrixXd C = readMat(aif, mi);
    readDataNameLine(aif, "d");
    Eigen::VectorXd d = readMat(aif, mi, true);
    readDataNameLine(aif, "x_min");
    Eigen::VectorXd xl = readMat(aif, n, true);
    readDataNameLine(aif, "x_max");
    Eigen::VectorXd xu = readMat(aif, n, true);
    return std::make_tuple(Q, c, A, b, C, d, xl, xu);
  }
  throw std::runtime_error("Unable to open file " + filename);
}