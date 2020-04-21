/* Copyright 2020 CNRS-AIST JRL
 */

#pragma once

#include <iosfwd>
#include <string>

#include <jrl-qp/api.h>
#include <jrl-qp/defs.h>
#include <jrl-qp/internal/meta.h>
#include <jrl-qp/utils/toMatlab.h>

namespace jrlqp::utils
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
    Logger(std::ostream& os, const std::string& name, std::uint32_t flags = 0)
      : flags_(flags), iter_(-1), name_(name), os_(&os) {}

    /***/
    Logger& setFlag(std::uint32_t flag, bool add = true);

    Logger& setOutputStream(std::ostream& os);

    void comment(std::uint32_t flag, const std::string& c) const;
    void startIter(int i);
    template<typename... Args>
    void log(std::uint32_t flag, Args&&... args) const;

    Logger subLog(const std::string& name) const;

  private:
    template<bool b>
    std::ostream& logIter() const;

    template<typename Bool, typename T, typename... Args>
    void log_(const Bool& b, const std::string& valName, const T& val, Args&&... args) const;

    template<typename Bool, typename T>
    void log_(const Bool& b, const T&) const;

    template<typename Bool>
    void log_(const Bool& b) const {}

    template<typename Bool, typename Derived>
    void logVal_(const Bool&, const std::string& valName, const Eigen::EigenBase<Derived>& M) const;

    template<typename Bool, typename Other, typename std::enable_if<(!internal::derives_from<Other, Eigen::EigenBase>()), int>::type=0>
    void logVal_(const Bool&, const std::string& valName, const Other& val) const;

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
        log_(std::false_type{}, std::forward<Args>(args)...);
      else
        log_(std::true_type{}, std::forward<Args>(args)...);
    }
  }

  template<typename Bool, typename T, typename... Args>
  inline void Logger::log_(const Bool& b, const std::string& valName, const T& val, Args&&... args) const
  {
    logVal_(b, valName, val);
    log_(b, std::forward<Args>(args)...);
  }

  template<typename Bool, typename T>
  inline void Logger::log_(const Bool&, const T&) const
  {
    static_assert(internal::always_false<T>::value && "incorrect number of arguments.");
  }

  template<typename Bool, typename Derived>
  inline void Logger::logVal_(const Bool&, const std::string& valName, const Eigen::EigenBase<Derived>& M) const
  {
    logIter<Bool::value>() << "." << valName << " = " << (toMatlab)M.const_derived() << ";\n";
  }

  template<typename Bool, typename Other, typename std::enable_if<(!internal::derives_from<Other, Eigen::EigenBase>()), int>::type>
  inline void Logger::logVal_(const Bool&, const std::string& valName, const Other& val) const
  {
    logIter<Bool::value>() << "." << valName << " = " << val << ";\n";
  }
}