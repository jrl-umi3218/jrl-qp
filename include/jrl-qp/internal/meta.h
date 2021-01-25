/* Copyright 2020 CNRS-AIST JRL */

#pragma once

#include <type_traits>

namespace jrl::qp::internal
{
/** An helper struct used by derives_from.*/
template<template<typename...> class Base>
struct is_base
{
  /** Accept any class derived from Base<T...>.*/
  template<typename... T>
  static std::true_type check(Base<T...> const volatile &);
  /** Fallback function that will be used for type not deriving from Base<T...>. */
  static std::false_type check(...);
};

/** Check if class \t T derives from the templated class \t Base.
 *
 * This relies on jrl::qp::internal::is_base::check: if T derives from \t Base,
 * the overload returning \a std::true_type will be selected, otherwise, it
 * will be the one returning \a std::false_type.
 *
 * Adapted from https://stackoverflow.com/a/5998303/11611648
 */
template<typename T, template<typename...> class Base>
constexpr bool derives_from()
{
  return decltype(is_base<Base>::check(std::declval<const T &>()))::value;
}

/** Check if class \t T derives from the non-templated class \t Base
 * This returns \a false if \t Base is not a class.
 */
template<typename T, typename Base>
constexpr bool derives_from()
{
  return std::is_base_of_v<Base, T>;
}

/** Used to enable a function for a list of types.
 *
 * To have a function work for T equal or deriving from any B1, B2, ... or Bk
 * where Bi are types.
 * Use as template<typename T, enable_for_t<T,B1, B2, ..., ..., Bk>=0>
 */
template<typename T, typename... Base>
using enable_for_t = std::enable_if_t<(... || (std::is_same_v<T, Base> || derives_from<T, Base>())), int>;

/** Used to enable a function for a list of templated classes.
 *
 * To have a function work for T equal or deriving from any B1, B2, ... or Bk
 * where Bi are a templated classes.
 * Use as template<typename T, enable_for_templated_t<T,B1, B2, ..., ..., Bk>=0>
 */
template<typename T, template<typename...> class... Base>
using enable_for_templated_t = std::enable_if_t<(... || derives_from<T, Base>()), int>;

/** A sink, whose value is always true. */
template<typename T>
class always_true : public std::true_type
{
};

/** A sink, whose value is always false. */
template<typename T>
class always_false : public std::false_type
{
};

/** Identity functor.
 * 
 * Taken from https://stackoverflow.com/a/15202612
 */
struct identity
{
  template<typename T>
  constexpr auto operator()(T && v) const noexcept -> decltype(std::forward<T>(v))
  {
    return std::forward<T>(v);
  }
};

/** Functor to convert an enumeration to its undelying type.*/
struct to_underlying_type
{
  template<typename T>
  constexpr auto operator()(const T & v) const noexcept
  {
    static_assert(std::is_enum_v<T>, "This is needed before C++20.");
    return static_cast<std::underlying_type_t<T>>(v);
  }
};

/** Helper functor that takes an argument and
 *  - returns it as is, if it is not an enum
 *  - cast it as its underlying type if it is one
 */
template<typename T>
using cast_as_underlying_if_enum = std::conditional_t<std::is_enum_v<T>, to_underlying_type, identity>;

} // namespace jrl::qp::internal
