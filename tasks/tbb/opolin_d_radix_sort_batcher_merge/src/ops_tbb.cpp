#include "tbb/opolin_d_radix_sort_batcher_merge/include/ops_tbb.hpp"

#include <tbb/tbb.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

bool opolin_d_radix_batcher_sort_tbb::RadixBatcherSortTaskTbb::PreProcessingImpl() {
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + size_);
  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<int>(output_size, 0);
  return true;
}

bool opolin_d_radix_batcher_sort_tbb::RadixBatcherSortTaskTbb::ValidationImpl() {
  // Check equality of counts elements
  size_ = static_cast<int>(task_data->inputs_count[0]);
  if (size_ <= 0 || task_data->inputs.empty() || task_data->outputs.empty()) {
    return false;
  }
  if (task_data->inputs[0] == nullptr || task_data->outputs[0] == nullptr) {
    return false;
  }
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool opolin_d_radix_batcher_sort_tbb::RadixBatcherSortTaskTbb::RunImpl() {
  output_ = input_;
  BatcherMergeRadixSort(output_);
  return true;
}

bool opolin_d_radix_batcher_sort_tbb::RadixBatcherSortTaskTbb::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<int*>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}

void opolin_d_radix_batcher_sort_tbb::RadixSort(std::vector<int>& a, int l, int r) {
  if (r - l < 2) {
    return;
  }
  int maxv = 0;
  for (int i = l; i < r; ++i) {
    maxv = std::max(maxv, std::abs(a[i]));
  }
  int digits = (maxv == 0 ? 1 : static_cast<int>(std::log10(maxv)) + 1);
  std::vector<int> buf(r - l);
  const int base = 10;
  for (int d = 0, exp = 1; d < digits; ++d, exp *= base) {
    std::array<int, base> cnt{};
    for (int i = l; i < r; ++i) {
      cnt[(std::abs(a[i]) / exp) % base]++;
    }
    for (int i = 1; i < base; ++i) {
      cnt[i] += cnt[i - 1];
    }
    for (int i = r; i-- > l;) {
      int idx = (std::abs(a[i]) / exp) % base;
      buf[--cnt[idx]] = a[i];
    }
    for (int i = l, j = 0; i < r; ++i, ++j) {
      a[i] = buf[j];
    }
  }
}

void opolin_d_radix_batcher_sort_tbb::OddEvenMerge(std::vector<int>& a, int l, int m, int r) {
  int n = r - l;
  std::vector<int> tmp(n);
  int i = l, j = m, k = 0;
  while (i < m && j < r) {
    tmp[k++] = (a[i] <= a[j] ? a[i++] : a[j++]);
  }
  while (i < m) {
    tmp[k++] = a[i++];
  }
  while (j < r) {
    tmp[k++] = a[j++];
  }
  for (int t = 1; t + 1 < n; t += 2) {
    if (tmp[t] > tmp[t + 1]) {
      std::swap(tmp[t], tmp[t + 1]);
    }
  }
  std::copy(tmp.begin(), tmp.end(), a.begin() + l);
}

void opolin_d_radix_batcher_sort_tbb::BatcherMergeRadixSort(std::vector<int>& a) {
  int n = static_cast<int>(a.size());
  if (n < 2) {
    return;
  }
  int threads = tbb::this_task_arena::max_concurrency();
  std::vector<int> bounds(threads + 1);
  for (int t = 0; t <= threads; ++t) {
    bounds[t] = (n * t) / threads;
  }
  tbb::parallel_for(tbb::blocked_range<int>(0, threads), [&](auto& range) {
    for (int t = range.begin(); t < range.end(); ++t) {
      RadixSort(a, bounds[t], bounds[t + 1]);
    }
  });
  int segs = threads;
  while (segs > 1) {
    int pairs = segs / 2;
    tbb::parallel_for(0, pairs, [&](int p) {
      int l = bounds[2 * p];
      int m = bounds[2 * p + 1];
      int r = bounds[2 * p + 2];
      OddEvenMerge(a, l, m, r);
    });
    for (int i = 0; i < pairs; ++i) {
      bounds[i] = bounds[2 * i];
    }
    if (segs % 2 == 1) {
      bounds[pairs] = bounds[segs];
    }
    segs = pairs + (segs % 2);
    bounds[segs] = n;
  }
}