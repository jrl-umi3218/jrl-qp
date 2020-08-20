#include <jrl-qp/internal/memoryChecks.h>

namespace jrl::qp::internal
{
bool is_malloc_allowed_impl(bool update, bool new_value = false)
{
  static bool value = true;
  if(update == 1) value = new_value;
  return value;
}

/** Check if dynamic allocation is allowed in Eigen operations. */
void check_that_malloc_is_allowed()
{
  eigen_assert(is_malloc_allowed_impl(false));
}

/** Allow or disallow dynamic allocation in Eigen operations. */
bool set_is_malloc_allowed(bool allow)
{
  is_malloc_allowed_impl(true, allow);
  return Eigen::internal::set_is_malloc_allowed(allow);
}
} // namespace jrl::qp::internal