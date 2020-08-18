/* Copyright 2020 CNRS-AIST JRL
 */

#pragma once

#include <iosfwd>
#include <string>

#include <jrl-qp/api.h>
#include <jrl-qp/defs.h>
#include <jrl-qp/internal/meta.h>
#include <jrl-qp/utils/toMatlab.h>

namespace jrl::qp::utils
{
  /** Class for logging data in a Matlab-compatible way, with a filter based on
    * a system of binary flags.
    */
  class JRLQP_DLLAPI Logger
  {
  public:
    /** \param os Stream where the data will be logged.
      * \param name Name of the Matlab struct that will record the data.
      * \param flags Flag for which the logger will effectively output something.
      */
    Logger(std::ostream& os, std::string name, std::uint32_t flags = 0)
      : flags_(flags), iter_(-1), name_(name), os_(&os) {}

    /** Add or remove one or several (aggregated) flags to filter the data to log.
      * If \p add is \a true, \p flag is added with a "or". If \a false, it is
      * removed with a "and not".
      */
    Logger& setFlag(std::uint32_t flag, bool add = true);

    /** Set the output stream.*/
    Logger& setOutputStream(std::ostream& os);

    /** Log a comment \p c, if \p flag pass the filter.*/
    void comment(std::uint32_t flag, std::string_view c) const;
    /** Indicate that a new iteration of the solver is starting.*/
    void startIter(int i);
    /** Log a given number of values.
      *
      * \param flag Flag(s) for which the values should be logged.
      * \param args Sequence of pairs (name, value).
      */
    template<typename... Args>
    void log(std::uint32_t flag, Args&&... args) const;

    /** Return a sub-logger. It will act as a struct with data for the current
      * iteration of its parent log.
      */
    Logger subLog(const std::string& name) const;

    /** Current iteration number. */
    int iter() const { return iter_; }

  private:
    /** Output the start of log line corresponding to
      *  - logging some general data if \p b is \c false
      *  - logging data for the current iteration if \p b is \c true.
      */
    template<bool b>
    std::ostream& logIter() const;

    template<bool b, typename T, typename... Args>
    void log_(const std::string& valName, const T& val, Args&&... args) const;

    template<bool b, typename T>
    void log_(const T&) const;

    template<bool>
    void log_() const {}

    template<bool b, typename Derived>
    void logVal_(const std::string& valName, const Eigen::EigenBase<Derived>& M) const;

    template<bool b, typename Other, typename std::enable_if<(!internal::derives_from<Other, Eigen::EigenBase>()), int>::type=0>
    void logVal_(const std::string& valName, const Other& val) const;

    std::uint32_t flags_;
    int iter_;
    std::string name_;
    std::ostream* os_;
  };


  template<>
  inline std::ostream& Logger::logIter<true>() const
  {
    *os_ << name_ << "(" << iter_ + 1 << ")";
    return *os_;
  }

  template<>
  inline std::ostream& Logger::logIter<false>() const
  {
    *os_ << name_ << "Data";
    return *os_;
  }


  template<typename... Args>
  inline void Logger::log(std::uint32_t flag, Args&&... args) const
  {
    if (flag & flags_)
    {
      if (flag & constant::noIterationFlag)
        log_<false>(std::forward<Args>(args)...);
      else
        log_<true>(std::forward<Args>(args)...);
    }
  }

  template<bool b, typename T, typename... Args>
  inline void Logger::log_(const std::string& valName, const T& val, Args&&... args) const
  {
    logVal_<b>(valName, val);
    log_<b>(std::forward<Args>(args)...);
  }

  template<bool, typename T>
  inline void Logger::log_(const T&) const
  {
    static_assert(internal::always_false<T>::value && "incorrect number of arguments.");
  }

  template<bool b, typename Derived>
  inline void Logger::logVal_(const std::string& valName, const Eigen::EigenBase<Derived>& M) const
  {
    logIter<b>() << "." << valName << " = " << (toMatlab)M.const_derived() << ";\n";
  }

  template<bool b, typename Other, typename std::enable_if<(!internal::derives_from<Other, Eigen::EigenBase>()), int>::type>
  inline void Logger::logVal_(const std::string& valName, const Other& val) const
  {
    logIter<b>() << "." << valName << " = " << val << ";\n";
  }
}