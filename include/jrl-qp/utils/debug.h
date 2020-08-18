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

#define PP_ID(x) x

#define PP_APPLY(macro, ...) PP_ID(macro(__VA_ARGS__))

/** Count the number of argument passed to it.*/
#define PP_NARG(...) PP_ID(PP_NARG_(__VA_ARGS__, PP_RSEQ_N()))
#define PP_NARG_(...) PP_ID(PP_ARG_N(__VA_ARGS__))
#define PP_ARG_N(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, N, ...) N
#define PP_RSEQ_N() 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0

#define CHOOSE_AUTO_NAME_START(count) AUTO_NAME_ARG##count

#define AUTO_NAME_ARG(...) PP_ID(PP_APPLY(CHOOSE_AUTO_NAME_START, PP_NARG(__VA_ARGS__))(__VA_ARGS__))

#define AUTO_NAME_ARG1(x) #x, x
#define AUTO_NAME_ARG2(x, ...) #x, x, PP_ID(AUTO_NAME_ARG1(__VA_ARGS__))
#define AUTO_NAME_ARG3(x, ...) #x, x, PP_ID(AUTO_NAME_ARG2(__VA_ARGS__))
#define AUTO_NAME_ARG4(x, ...) #x, x, PP_ID(AUTO_NAME_ARG3(__VA_ARGS__))
#define AUTO_NAME_ARG5(x, ...) #x, x, PP_ID(AUTO_NAME_ARG4(__VA_ARGS__))
#define AUTO_NAME_ARG6(x, ...) #x, x, PP_ID(AUTO_NAME_ARG5(__VA_ARGS__))
#define AUTO_NAME_ARG7(x, ...) #x, x, PP_ID(AUTO_NAME_ARG6(__VA_ARGS__))
#define AUTO_NAME_ARG8(x, ...) #x, x, PP_ID(AUTO_NAME_ARG7(__VA_ARGS__))
#define AUTO_NAME_ARG9(x, ...) #x, x, PP_ID(AUTO_NAME_ARG8(__VA_ARGS__))
#define AUTO_NAME_ARG10(x, ...) #x, x, PP_ID(AUTO_NAME_ARG9(__VA_ARGS__))

#define ENABLE_LOG_(macro, ...)  \
  do                             \
  {                              \
    if(!NO_LOG_)                 \
    {                            \
      PP_ID(macro(__VA_ARGS__)); \
    }                            \
  } while(0)
#define ENABLE_DEBUG_(macro, ...) \
  do                              \
  {                               \
    if(DEBUG_OUTPUT)              \
    {                             \
      PP_ID(macro(__VA_ARGS__));  \
    }                             \
  } while(0)

#define LOG_(logger, flags, ...) PP_ID(logger.log(static_cast<uint32_t>(flags), PP_ID(AUTO_NAME_ARG(__VA_ARGS__))))
#define LOG_AS_(logger, flags, ...) PP_ID(logger.log(static_cast<uint32_t>(flags), __VA_ARGS__))
#define LOG(...) PP_ID(ENABLE_LOG_(LOG_, __VA_ARGS__))
#define DBG(...) PP_ID(ENABLE_DEBUG_(LOG_, __VA_ARGS__))
#define LOG_AS(...) PP_ID(ENABLE_LOG_(LOG_AS_, __VA_ARGS__))
#define DBG_AS(...) PP_ID(ENABLE_DEBUG_(LOG_AS_, __VA_ARGS__))

#define LOG_COMMENT(logger, flag, s) ENABLE_LOG_(logger.comment, static_cast<uint32_t>(flag), s)
#define DBG_COMMENT(logger, flag, s) ENABLE_DEBUG_(logger.comment, static_cast<uint32_t>(flag), s)

#define LOG_NEW_ITER(logger, it) ENABLE_LOG_(logger.startIter, it)
#define DBG_NEW_ITER(logger, it) ENABLE_DEBUG_(logger.startIter, it)

#define LOG_RESET(logger) ENABLE_LOG_(logger.startIter, -1)
#define DBG_RESET(logger) ENABLE_DEBUG_(logger.startIter, -1)

#define DEBUG_ONLY(expr) \
  do                     \
  {                      \
    if(DEBUG_OUTPUT)     \
    {                    \
      expr;              \
    }                    \
  } while(0)
