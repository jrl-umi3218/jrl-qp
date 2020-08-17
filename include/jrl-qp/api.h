/* Copyright 2020 CNRS-AIST JRL
*/

#pragma once

# if defined _WIN32 || defined __CYGWIN__
// On Microsoft Windows, use dllimport and dllexport to tag symbols.
#  define JRLQP_DLLIMPORT __declspec(dllimport)
#  define JRLQP_DLLEXPORT __declspec(dllexport)
#  define JRLQP_DLLLOCAL
# else
// On Linux, for GCC >= 4, tag symbols using GCC extension.
#  if __GNUC__ >= 4
#   define JRLQP_DLLIMPORT __attribute__ ((visibility("default")))
#   define JRLQP_DLLEXPORT __attribute__ ((visibility("default")))
#   define JRLQP_DLLLOCAL  __attribute__ ((visibility("hidden")))
#  else
// Otherwise (GCC < 4 or another compiler is used), export everything.
#   define JRLQP_DLLIMPORT
#   define JRLQP_DLLEXPORT
#   define JRLQP_DLLLOCAL
#  endif // __GNUC__ >= 4
# endif // defined _WIN32 || defined __CYGWIN__

# ifdef JRLQP_STATIC
// If one is using the library statically, get rid of
// extra information.
#  define JRLQP_DLLAPI
#  define JRLQP_LOCAL
# else
// Depending on whether one is building or using the
// library define DLLAPI to import or export.
#  ifdef JRLQP_EXPORTS
#   define JRLQP_DLLAPI JRLQP_DLLEXPORT
#  else
#   define JRLQP_DLLAPI JRLQP_DLLIMPORT
#  endif // JRLQP_EXPORTS
#  define JRLQP_LOCAL JRLQP_DLLLOCAL
# endif // JRLQP_STATIC
