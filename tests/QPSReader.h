/* Copyright 2020 CNRS-AIST JRL
 */

#pragma once

#include <iosfwd>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>

#include <jrl-qp/test/problems.h>

namespace jrl::qp::test
{
  struct ProblemProperties
  {
    int nbVar;
    int nbCstr;
    int nbEq;
    bool useBounds;
    bool hasFixedVariables;
  };

  /** A reader for the QPS format, a simple extension over the MPS format of IBM.
    * Reference of MPS format: http://lpsolve.sourceforge.net/5.5/mps-format.htm
    * Reference of QPS format: https://github.com/YimingYAN/QP-Test-Problems
    */
  class QPSReader
  {
  public:
    enum class LineType
    {
      NORMAL,
      NAME,
      ROWS,
      COLUMNS,
      RHS,
      RANGES,
      BOUNDS,
      QUADOBJ,
      ENDATA,
      EMPTY,
      OTHER
    };

    enum class RowType 
    { 
      E, 
      L, 
      G, 
      N, 
      UNKNOWN 
    };

    // We don't consider the binary and integer variables
    enum class BndType
    {
      LO,
      UP,
      FX,
      FR,
      MI,
      PL,
      UNKNOWN
    };

    struct Context
    {
      int line;
      LineType section;
    };


    QPSReader(bool fullObjMat = false);

    std::pair<QPProblem<>, ProblemProperties> read(const std::string& filename);

  private:
    using vectorVal = std::pair<int, double>;
    using matrixVal = std::tuple<int, int, double>;

    void reset();
    void processLine(const std::string& line, LineType type);
    // Update mapRow from a line of type ROWS
    void readRow(const std::string& line);
    // Update mapCol from a line of type COLUMNS
    void readColumn(const std::string& line);
    void addValueToColumn(int cIdx, const std::string& rowName, double val);
    void readRHS(const std::string& line);
    void addValueToRHS(const std::string& rowName, double val);
    void readRanges(const std::string& line);
    void addValueToRanges(const std::string& rowName, double val);
    void readBounds(const std::string& line);
    void readQuadObj(const std::string& line);

    bool fullObjMat = false; // If true, both lower and upper part of the objective matrix are filled.
    double bigBnd = std::numeric_limits<double>::infinity();

    std::string problemName = "";
    Context context = { 0, LineType::OTHER };
    // A map from a row name to an pair (index, row type)
    std::unordered_map<std::string, std::pair<int, RowType> > mapRow = {};
    std::unordered_map<std::string, int> mapCol = {};
    std::string rhsName = "";
    std::string rangeName = "";
    std::vector<matrixVal> CVal = {};
    std::vector<matrixVal> GVal = {};
    std::vector<vectorVal> aVal = {};
    std::vector<std::pair<vectorVal, RowType> > bVal = {};
    std::vector<std::pair<vectorVal, RowType> > rVal = {};
    std::vector<std::pair<vectorVal, BndType> > xVal = {};
    double objCst = 0;
    int nRows = 0;
    bool objWasRead = false;
  };
}