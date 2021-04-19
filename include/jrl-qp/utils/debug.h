/* Copyright 2020 CNRS-AIST JRL */

#pragma once

#ifndef NO_LOG
inline constexpr bool NO_LOG_ = 0;
#else
inline constexpr bool NO_LOG_ = 1;
#endif

#ifdef NDEBUG
inline constexpr bool DEBUG_OUTPUT = 0;
#else
inline constexpr bool DEBUG_OUTPUT = 1;
#endif

#define JRLQP_PP_ID(x) x

#define JRLQP_PP_APPLY(macro, ...) JRLQP_PP_ID(macro(__VA_ARGS__))

/** Count the number of argument passed to it.*/
#define JRLQP_PP_NARG(...) JRLQP_PP_ID(JRLQP_PP_NARG_(__VA_ARGS__, JRLQP_PP_RSEQ_N()))
#define JRLQP_PP_NARG_(...) JRLQP_PP_ID(JRLQP_PP_ARG_N(__VA_ARGS__))
#define JRLQP_PP_ARG_N(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, N, ...) N
#define JRLQP_PP_RSEQ_N() 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0

#define JRLQP_CHOOSE_AUTO_NAME_START(count) JRLQP_AUTO_NAME_ARG##count

#define JRLQP_AUTO_NAME_ARG(...) \
  JRLQP_PP_ID(JRLQP_PP_APPLY(JRLQP_CHOOSE_AUTO_NAME_START, JRLQP_PP_NARG(__VA_ARGS__))(__VA_ARGS__))

#define JRLQP_AUTO_NAME_ARG1(x) #x, x
#define JRLQP_AUTO_NAME_ARG2(x, ...) #x, x, JRLQP_PP_ID(JRLQP_AUTO_NAME_ARG1(__VA_ARGS__))
#define JRLQP_AUTO_NAME_ARG3(x, ...) #x, x, JRLQP_PP_ID(JRLQP_AUTO_NAME_ARG2(__VA_ARGS__))
#define JRLQP_AUTO_NAME_ARG4(x, ...) #x, x, JRLQP_PP_ID(JRLQP_AUTO_NAME_ARG3(__VA_ARGS__))
#define JRLQP_AUTO_NAME_ARG5(x, ...) #x, x, JRLQP_PP_ID(JRLQP_AUTO_NAME_ARG4(__VA_ARGS__))
#define JRLQP_AUTO_NAME_ARG6(x, ...) #x, x, JRLQP_PP_ID(JRLQP_AUTO_NAME_ARG5(__VA_ARGS__))
#define JRLQP_AUTO_NAME_ARG7(x, ...) #x, x, JRLQP_PP_ID(JRLQP_AUTO_NAME_ARG6(__VA_ARGS__))
#define JRLQP_AUTO_NAME_ARG8(x, ...) #x, x, JRLQP_PP_ID(JRLQP_AUTO_NAME_ARG7(__VA_ARGS__))
#define JRLQP_AUTO_NAME_ARG9(x, ...) #x, x, JRLQP_PP_ID(JRLQP_AUTO_NAME_ARG8(__VA_ARGS__))
#define JRLQP_AUTO_NAME_ARG10(x, ...) #x, x, JRLQP_PP_ID(JRLQP_AUTO_NAME_ARG9(__VA_ARGS__))

#define JRLQP_ENABLE_LOG_(macro, ...)        \
  do                                   \
  {                                    \
    if(!NO_LOG_)                       \
    {                                  \
      JRLQP_PP_ID(macro(__VA_ARGS__)); \
    }                                  \
  } while(0)
#define JRLQP_ENABLE_DEBUG_(macro, ...)      \
  do                                   \
  {                                    \
    if(DEBUG_OUTPUT)                   \
    {                                  \
      JRLQP_PP_ID(macro(__VA_ARGS__)); \
    }                                  \
  } while(0)

#define JRLQP_LOG_(logger, flags, ...) \
  JRLQP_PP_ID(logger.log(static_cast<uint32_t>(flags), JRLQP_PP_ID(JRLQP_AUTO_NAME_ARG(__VA_ARGS__))))
#define JRLQP_LOG_AS_(logger, flags, ...) JRLQP_PP_ID(logger.log(static_cast<uint32_t>(flags), __VA_ARGS__))
#define JRLQP_LOG(...) JRLQP_PP_ID(JRLQP_ENABLE_LOG_(JRLQP_LOG_, __VA_ARGS__))
#define JRLQP_DBG(...) JRLQP_PP_ID(JRLQP_ENABLE_DEBUG_(JRLQP_LOG_, __VA_ARGS__))
#define JRLQP_LOG_AS(...) JRLQP_PP_ID(JRLQP_ENABLE_LOG_(JRLQP_LOG_AS_, __VA_ARGS__))
#define JRLQP_DBG_AS(...) JRLQP_PP_ID(JRLQP_ENABLE_DEBUG_(JRLQP_LOG_AS_, __VA_ARGS__))

#define JRLQP_LOG_COMMENT(logger, flag, ...) \
  JRLQP_PP_ID(JRLQP_ENABLE_LOG_(logger.comment, static_cast<uint32_t>(flag), __VA_ARGS__))
#define JRLQP_DBG_COMMENT(logger, flag, ...) \
  JRLQP_PP_ID(JRLQP_ENABLE_DEBUG_(logger.comment, static_cast<uint32_t>(flag), __VA_ARGS__))

#define JRLQP_LOG_NEW_ITER(logger, it) JRLQP_ENABLE_LOG_(logger.startIter, it)
#define JRLQP_DBG_NEW_ITER(logger, it) JRLQP_ENABLE_DEBUG_(logger.startIter, it)

#define JRLQP_LOG_RESET(logger) JRLQP_ENABLE_LOG_(logger.startIter, -1)
#define JRLQP_DBG_RESET(logger) JRLQP_ENABLE_DEBUG_(logger.startIter, -1)

#define JRLQP_DEBUG_ONLY(expr) \
  do                           \
  {                            \
    if(DEBUG_OUTPUT)           \
    {                          \
      expr;                    \
    }                          \
  } while(0)
