/* Copyright 2020 CNRS-AIST JRL
 */

#pragma once

static inline void testSizes(benchmark::internal::Benchmark* b) {
  std::vector sizes = { 5,10,20,30,50,70,100,150,200,250,300,400,500 };
  for (auto i: sizes)
    b->Args({ i });
}

#define MAT_BENCHMARK(name) BENCHMARK(name)->Apply(testSizes)->Unit(benchmark::kMicrosecond)