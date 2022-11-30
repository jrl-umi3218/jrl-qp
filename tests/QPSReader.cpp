/* Copyright 2020 CNRS-AIST JRL */

#include <algorithm>
#include <cctype>
#include <exception>
#include <fstream>
#include <optional>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "QPSReader.h"

#define THROW(message, context)          \
  {                                      \
    std::stringstream msg;               \
    msg << message << context << "\n";   \
    throw std::runtime_error(msg.str()); \
  }

using namespace Eigen;

namespace
{
using namespace jrl::qp::test;

const char * lineName[] = {"normal", "name",    "rows",   "columns", "rhs",  "ranges",
                           "bounds", "quadobj", "endata", "empty",   "other"};

QPSReader::RowType rowType(char c)
{
  switch(std::tolower(c))
  {
    case 'e':
      return QPSReader::RowType::E;
    case 'l':
      return QPSReader::RowType::L;
    case 'g':
      return QPSReader::RowType::G;
    case 'n':
      return QPSReader::RowType::N;
    default:
      return QPSReader::RowType::UNKNOWN;
  }
}

QPSReader::BndType bndType(const std::string s)
{
  if(s == "LO")
    return QPSReader::BndType::LO;
  else if(s == "UP")
    return QPSReader::BndType::UP;
  else if(s == "FX")
    return QPSReader::BndType::FX;
  else if(s == "FR")
    return QPSReader::BndType::FR;
  else if(s == "MI")
    return QPSReader::BndType::MI;
  else if(s == "PL")
    return QPSReader::BndType::PL;
  else
    return QPSReader::BndType::UNKNOWN;
}

std::ostream & operator<<(std::ostream & os, const QPSReader::Context & c)
{
  os << "(line " << c.line << ", "
     << "section " << lineName[static_cast<int>(c.section)] << ")\n";
  return os;
}

void toLower(std::string & s)
{
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
}

QPSReader::LineType lineType(const std::string & line)
{
  if(line.empty())
  {
    return QPSReader::LineType::EMPTY;
  }
  else if(line[0] == ' ')
  {
    return QPSReader::LineType::NORMAL;
  }
  else
  {
    std::istringstream is(line);
    std::string s;
    is >> s;
    toLower(s);

    for(int i = 1; i < 9; ++i)
    {
      if(s == lineName[i]) return QPSReader::LineType(i);
    }
  }
  return QPSReader::LineType::OTHER;
}

std::string parseName(const std::string & line, const QPSReader::Context & c)
{
  std::istringstream is(line);
  std::string head, name;
  is >> head >> name;
  if(is.fail()) THROW("Failed to read name ", c);
  return name;
}

std::pair<QPSReader::RowType, std::string> parseRow(const std::string & line, const QPSReader::Context c)
{
  std::istringstream is(line);

  char t;
  is >> t;
  if(is.fail()) THROW("Failed to read row type ", c);
  auto type = rowType(t);
  if(type == QPSReader::RowType::UNKNOWN) THROW("Unknown row type ", c);

  std::string name;
  is >> name;
  if(is.fail()) THROW("Failed to read row name ", c);

  return {type, name};
}

struct NamedValue
{
  std::string name;
  double value;
};

std::optional<NamedValue> readNamedValue(std::istream & is, const QPSReader::Context & c, bool second)
{
  NamedValue r;
  is >> r.name;
  if(is.eof() && second) return {};
  if(is.fail()) THROW("Failed to read " << (second ? "second" : "first") << " name ", c);

  is >> r.value;
  if(is.fail()) THROW("Failed to read " << (second ? "second" : "first") << " value ", c);
  return r;
}

struct ParsedValueLine
{
  std::string name;
  NamedValue val1;
  std::optional<NamedValue> val2;
};

ParsedValueLine parseValueLine(const std::string & line, const QPSReader::Context & c)
{
  std::istringstream is(line);
  ParsedValueLine pvl;

  is >> pvl.name;
  if(is.fail()) THROW("Failed to read " << lineName[static_cast<int>(c.line)] << " name ", c);

  pvl.val1 = *readNamedValue(is, c, false);
  pvl.val2 = readNamedValue(is, c, true);

  return pvl;
}
} // namespace

namespace jrl::qp::test
{
QPSReader::QPSReader(bool fullObjMat) : fullObjMat(fullObjMat) {}

std::pair<QPProblem<>, ProblemProperties> QPSReader::read(const std::string & filename)
{
  std::ifstream file(filename);
  if(!file.is_open()) THROW("Unable to opern file " << filename << " ", context);

  std::string line;

  // parsing file
  while(std::getline(file, line))
  {
    ++context.line;
    LineType lt = lineType(line);
    if(lt == LineType::OTHER || lt == LineType::EMPTY) continue;

    if(lt == LineType::NORMAL)
      processLine(line, lt);
    else
    {
      context.section = lt;
      if(lt == LineType::NAME) problemName = parseName(line, context);
    }
  }

  // populating data
  int n = static_cast<int>(mapCol.size());
  QPProblem<> qp;
  ProblemProperties properties;
  qp.G = MatrixXd::Zero(n, n);
  qp.a = VectorXd::Zero(n);
  qp.C = MatrixXd::Zero(nRows, n);
  qp.l = VectorXd::Zero(nRows);
  qp.u = VectorXd::Zero(nRows);
  qp.xl = VectorXd::Zero(n);
  qp.xu = VectorXd::Constant(n, bigBnd);
  qp.objCst = objCst;
  properties.nbEq = 0;

  for(const auto & t : GVal) qp.G(std::get<0>(t), std::get<1>(t)) = std::get<2>(t);
  if(fullObjMat)
    qp.G.template triangularView<StrictlyUpper>() = qp.G.template triangularView<StrictlyLower>().transpose();
  for(const auto & p : aVal) qp.a[p.first] = p.second;
  for(const auto & t : CVal) qp.C(std::get<0>(t), std::get<1>(t)) = std::get<2>(t);
  for(const auto & r : mapRow)
  {
    auto [i, type] = r.second;
    switch(type)
    {
      case RowType::E:
        qp.l[i] = qp.u[i] = 0;
        ++properties.nbEq;
        break;
      case RowType::L:
        qp.l[i] = -bigBnd;
        qp.u[i] = 0;
        break;
      case RowType::G:
        qp.l[i] = 0;
        qp.u[i] = +bigBnd;
        break;
      default:
        break;
    }
  }
  for(const auto & b : bVal)
  {
    auto [i, v] = b.first;
    switch(b.second)
    {
      case RowType::E:
        qp.l[i] = qp.u[i] = v;
        break;
      case RowType::L:
        qp.l[i] = -bigBnd;
        qp.u[i] = v;
        break;
      case RowType::G:
        qp.l[i] = v;
        qp.u[i] = +bigBnd;
        break;
      default:
        assert(false);
        break;
    }
  }
  for(const auto & r : rVal)
  {
    auto [i, v] = r.first;
    switch(r.second)
    {
      case RowType::E:
        if(v >= 0)
          qp.u[i] += v;
        else
          qp.l[i] += v;
        break;
      case RowType::L:
        qp.l[i] = qp.u[i] - std::abs(v);
        break;
      case RowType::G:
        qp.u[i] = qp.l[i] + std::abs(v);
        break;
      default:
        assert(false);
        break;
    }
  }
  properties.hasFixedVariables = false;
  for(const auto & x : xVal)
  {
    auto [i, v] = x.first;
    switch(x.second)
    {
      case BndType::LO:
        qp.xl[i] = v;
        break;
      case BndType::UP:
        qp.xu[i] = v;
        break;
      case BndType::FX:
        qp.xl[i] = qp.xu[i] = v;
        properties.hasFixedVariables = true;
        break;
      case BndType::FR:
        qp.xl[i] = -bigBnd;
        qp.xu[i] = bigBnd;
        break;
      case BndType::MI:
        qp.xl[i] = -bigBnd;
        break;
      case BndType::PL:
        qp.xu[i] = bigBnd;
        break;
      default:
        assert(false);
        break;
    }
  }

  properties.nbVar = n;
  properties.nbCstr = nRows;
  properties.useBounds = (qp.xl.array() > -bigBnd).any() || (qp.xu.array() < bigBnd).any();

  return {qp, properties};
}

void QPSReader::processLine(const std::string & line, LineType /*type*/)
{
  switch(context.section)
  {
    case LineType::NAME:
      THROW("We shouldn't be in a NAME section ", context);
    case LineType::ROWS:
      readRow(line);
      break;
    case LineType::COLUMNS:
      readColumn(line);
      break;
    case LineType::RHS:
      readRHS(line);
      break;
    case LineType::RANGES:
      readRanges(line);
      break;
    case LineType::BOUNDS:
      readBounds(line);
      break;
    case LineType::QUADOBJ:
      readQuadObj(line);
      break;
    case LineType::ENDATA:
      THROW("We shouldn't be in a ENDATA section ", context);
    default:
      break;
  }
}

void QPSReader::readRow(const std::string & line)
{
  auto [type, name] = parseRow(line, context);
  if(mapRow.find(name) != mapRow.end()) THROW("Duplicate row name", context);

  if(type == RowType::UNKNOWN)
    THROW("Unknown row type ", context)
  else if(type == RowType::N)
  {
    if(objWasRead) THROW("We don't handle \"no restriction\" rows ", context);
    objWasRead = true;
    mapRow[name] = {-1, type};
  }
  else
  {
    mapRow[name] = {nRows, type};
    ++nRows;
  }
}

void QPSReader::readColumn(const std::string & line)
{
  auto [colName, val1, val2] = parseValueLine(line, context);

  // If this is a new column, add it to mapCol, otherwise retrieve its index
  int cIdx;
  if(auto it = mapCol.find(colName); it != mapCol.end())
  {
    cIdx = it->second;
  }
  else
  {
    cIdx = static_cast<int>(mapCol.size());
    mapCol[colName] = cIdx;
  }

  // Add values
  addValueToColumn(cIdx, val1.name, val1.value);
  if(val2) addValueToColumn(cIdx, val2->name, val2->value);
}

void QPSReader::addValueToColumn(int cIdx, const std::string & rowName, double val)
{
  auto [rIdx, rType] = mapRow.at(rowName);
  if(rType == RowType::N)
    aVal.push_back({cIdx, val});
  else
    CVal.push_back({rIdx, cIdx, val});
}

void QPSReader::readRHS(const std::string & line)
{
  auto [name, val1, val2] = parseValueLine(line, context);
  if(rhsName.empty())
    rhsName = name;
  else if(rhsName != name)
    THROW("Attempting to use different RHS name. I don't know what this means ", context);

  addValueToRHS(val1.name, val1.value);
  if(val2) addValueToRHS(val2->name, val2->value);
}

void QPSReader::addValueToRHS(const std::string & rowName, double val)
{
  auto [rIdx, rType] = mapRow.at(rowName);
  if(rType == RowType::N)
    objCst = -val; // minus because the rhs in on the wrong side
  else
    bVal.push_back({{rIdx, val}, rType});
}

void QPSReader::readRanges(const std::string & line)
{
  auto [name, val1, val2] = parseValueLine(line, context);
  if(rangeName.empty())
    rangeName = name;
  else if(rangeName != name)
    THROW("Attempting to use different range name. I don't know what this means ", context);

  addValueToRanges(val1.name, val1.value);
  if(val2) addValueToRanges(val2->name, val2->value);
}

void QPSReader::addValueToRanges(const std::string & rowName, double val)
{
  auto [rIdx, rType] = mapRow.at(rowName);
  if(rType == RowType::N) THROW("Attempting to add range on a N row ", context);
  rVal.push_back({{rIdx, val}, rType});
}

void QPSReader::readBounds(const std::string & line)
{
  std::istringstream is(line);
  std::string stype;
  is >> stype;
  if(is.fail()) THROW("Unable to read bound type", context);
  auto type = bndType(stype);
  if(type == BndType::UNKNOWN) THROW("Unknown bound type", context);

  std::string bndName;
  is >> bndName;
  if(is.fail()) THROW("Unable to read bound name", context);

  if(type == BndType::FR)
  {
    std::string colName;
    is >> colName;
    if(is.fail()) THROW("Unable to read column name", context);
    int cIdx = mapCol.at(colName);
    xVal.push_back({{cIdx, std::numeric_limits<double>::infinity()}, type});
  }
  else
  {
    auto [colName, val] = *readNamedValue(is, context, false);
    int cIdx = mapCol.at(colName);
    xVal.push_back({{cIdx, val}, type});
  }
}

void QPSReader::readQuadObj(const std::string & line)
{
  std::istringstream is(line);
  auto [colName, val1, val2] = parseValueLine(line, context);
  int cIdx = mapCol.at(colName);
  int rIdx = mapCol.at(val1.name);
  GVal.push_back({rIdx, cIdx, val1.value});
  if(val2)
  {
    rIdx = mapCol.at(val2->name);
    GVal.push_back({rIdx, cIdx, val2->value});
  }
}
} // namespace jrl::qp::test
