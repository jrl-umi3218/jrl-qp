/* Copyright 2020 CNRS-AIST JRL
 */

#pragma once

#include <iosfwd>
#include <string>

#include <jrl-qp/api.h>
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
      : flags_(flags), iter_(-1), name_(name), os_(os) {}

    /***/
    Logger& setFlag(std::uint32_t flag, bool add = true);

    void comment(std::uint32_t flag, const std::string& c) const;
    void startIter(int i);
    template<typename... Args>
    void log(std::uint32_t flag, Args&&... args) const;

    Logger subLog(const std::string& name) const;

  private:

    std::ostream& logIter() const { os_ << name_ << "(" << iter_+1 << ")"; return os_; }

    template<typename T, typename... Args>
    void log_(const std::string& valName, const T& val, Args&&... args) const;

    template<typename T>
    void log_(const T&) const;

    void log_() const {}

    template<typename Derived>
    void logVal_(const std::string& valName, const Eigen::DenseBase<Derived>& M) const;

    template<typename Other, typename std::enable_if<(!internal::derives_from<Other, Eigen::EigenBase>()), int>::type=0>
    void logVal_(const std::string& valName, const Other& val) const;

    std::uint32_t flags_;
    int iter_;
    std::string name_;
    std::ostream& os_;
  };



  template<typename... Args>
  inline void Logger::log(std::uint32_t flag, Args&&... args) const
  {
    if (flag & flags_)
      log_(std::forward<Args>(args)...);
  }

  template<typename T, typename... Args>
  inline void Logger::log_(const std::string& valName, const T& val, Args&&... args) const
  {
    logVal_(valName, val);
    log_(std::forward<Args>(args)...);
  }

  template<typename T>
  inline void Logger::log_(const T&) const
  {
    static_assert(internal::always_false<T>::value && "incorrect number of arguments.");
  }

  template<typename Derived>
  inline void Logger::logVal_(const std::string& valName, const Eigen::DenseBase<Derived>& M) const
  {
    logIter() << "." << valName << " = " << (toMatlab)M << ";\n";
  }

  template<typename Other, typename std::enable_if<(!internal::derives_from<Other, Eigen::EigenBase>()), int>::type>
  inline void Logger::logVal_(const std::string& valName, const Other& val) const
  {
    logIter() << "." << valName << " = " << val << ";\n";
  }
}