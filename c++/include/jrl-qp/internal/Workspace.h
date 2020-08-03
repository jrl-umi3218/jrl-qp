/* Copyright 2020 CNRS-AIST JRL
 */

#pragma once

#include <Eigen/Core>

#include <jrl-qp/api.h>

namespace jrlqp::internal
{
  /** Dummy struct to disambiguate call to non-const methods.*/
  struct NotConst {};

  /** Class providing a memory buffer and the ablity to see it as a matrix or vector.*/
  template<typename Scalar = double>
  class JRLQP_DLLAPI Workspace
  {
  public:
    /** Default constructor, providing a buffer of size 0.*/
    Workspace() : buffer_(0) {}
    /** Constructing a workspace with a buffer of a specific size.
      *
      * \param size. Size of the buffer
      */
    Workspace(int size) : buffer_(size) {}
    /** Constructing a workspace able to contain a \p row by \p col matrix.*/
    Workspace(int rows, int cols) : Workspace(rows* cols) {}

    /** Resize to the given size.
      *
      * \param size The new size.
      * \param fit If \a true, force memory reallocation even if when the previous
      * buffer was large enough. If \a false, only reallocate if needed.
      */
    void resize(int size, bool fit = false)
    {
      if (size > buffer_.size() || (size<buffer_.size() && fit))
      {
        buffer_.resize(size);
      }
    }

    /** Get the size of the buffer.*/
    int size() const { return static_cast<int>(buffer_.size()); }

    /** Shortcut for resize(rows*cols, fit). */
    void resize(int rows, int cols, bool fit = false)
    {
      resize(rows * cols, fit);
    }

    /** Return the buffer as a Eigen::Vector-like object of the required size. 
      * Non const version.
      */
    auto asVector(int size, NotConst = {})
    {
      assert(size <= buffer_.size());
      return buffer_.head(size);
    }

    /** Return the buffer as a Eigen::Vector-like object of the required size.
      * Const version.
      */
    auto asVector(int size) const
    {
      assert(size <= buffer_.size());
      return buffer_.head(size);
    }

    /** Return the buffer as a Eigen::Matrix-like object of the required sizes.
      * Non const version.
      *
      * \param rows Number of rows.
      * \param cols Number of cols.
      * \param ld Leading dimension of the matrix.
      * 
      * \note The leading dimension parameter can be important when the matrix
      * size is meant to change, so that each columns start with the same 
      * elements.
      */
    auto asMatrix(int rows, int cols, int ld, NotConst = {})
    {
      assert(ld * cols <= buffer_.size());
      assert(ld >= rows);
      return Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>, 0, Eigen::Stride<Eigen::Dynamic, 1> >
        (buffer_.data(), rows, cols, Eigen::Stride<Eigen::Dynamic,1>(ld,1));
    }

    /** Return the buffer as a Eigen::Matrix-like object of the required sizes.
      * Const version.
      * See non-const version for more details
      */
    auto asMatrix(int rows, int cols, int ld) const
    {
      assert(ld * cols <= buffer_.size());
      assert(ld >= rows);
      return Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>, 0, Eigen::Stride<Eigen::Dynamic, 1> >
        (buffer_.data(), rows, cols, Eigen::Stride<Eigen::Dynamic, 1>(ld, 1));
    }

    /** Set all elements to the buffer to zero.*/
    void setZero()
    {
      buffer_.setZero();
    }

  private:
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> buffer_;
  };
}

namespace jrlqp
{
  /** Type of a Eigen::Vector-like object from Workspace*/
  using WVector = decltype(internal::Workspace<double>().asVector(0));
  /** Type of a const Eigen::Vector-like object from Workspace*/
  using WConstVector = decltype(std::add_const_t<internal::Workspace<double>>().asVector(0));
}