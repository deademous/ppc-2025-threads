#include "tbb/opolin_d_radix_sort_batcher_merge/include/ops_tbb.hpp"

#include <tbb/tbb.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
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

void opolin_d_radix_batcher_sort_tbb::SortByDigit(std::vector<int>& vec) {
  size_t n = vec.size();
  if (n < 2) {
    return;
  }
  std::vector<uint32_t> uns_vec(n);
  for (size_t i = 0; i < n; ++i) {
    uns_vec[i] = static_cast<uint32_t>(vec[i]) ^ 0x80000000u;
  }
  RadixSort(uns_vec);
  for (size_t i = 0; i < n; ++i) {
    vec[i] = static_cast<int>(uns_vec[i] ^ 0x80000000u);
  }
}

void RadixSort(std::vector<uint32_t>& vec) {
  const size_t n = vec.size();
  std::vector<uint32_t> buf(n);
  for (int shift = 0; shift < 32; shift += 8) {
    int cnt[256] = {};
    for (size_t i = 0; i < vec.size(); ++i) {
      cnt[(vec[i] >> shift) & 255]++;
    }
    for (int i = 1; i < 256; ++i) {
      cnt[i] += cnt[i-1];
    }
    for (int i = int(n) - 1; i >= 0; --i) {
      uint32_t byte = (vec[i] >> shift) & 255u;
      buf[--cnt[byte]] = vec[i];
    }
    vec.swap(buf);
  }
}

void opolin_d_radix_batcher_sort_tbb::OddEvenMerge(std::vector<int>& vec, int left, int n, int step) {
  int m = 2 * step;
  if (m < n) {
    tbb::parallel_invoke(
      [&]{ OddEvenMerge(vec, left, n, m); },
      [&]{ OddEvenMerge(vec, left + step, n, m); }
    );
    for (int i = left + step; i + step < left + n; i += m) {
      if (vec[i] > vec[i + step]) {
        std::swap(vec[i], vec[i + step]);
      }
    }
  } else {
    if (vec[left] > vec[left + step]) {
      std::swap(vec[left], vec[left + step]);
    }
  }
}

void opolin_d_radix_batcher_sort_tbb::OddEvenMergeSort(std::vector<int>& vec, int left, int n) {
  if (n > 1) {
    int m = n / 2;
    tbb::parallel_invoke(
      [&]{ OddEvenMergeSort(vec, left, m); },
      [&]{ OddEvenMergeSort(vec, left + m, m); }
    );
    OddEvenMerge(vec, left, n, 1);
  }
}

void opolin_d_radix_batcher_sort_tbb::BatcherMergeRadixSort(std::vector<int>& vec) {
  int n = static_cast<int>(vec.size());
  if (n < 2) {
    return;
  }
  int threads = tbb::this_task_arena::max_concurrency();
  int chunk = std::max(1, (n + threads - 1) / threads);
  tbb::parallel_for(
    tbb::blocked_range<int>(0, n, chunk),
    [&](const tbb::blocked_range<int>& range) {
      int left = range.begin();
      right = std::min(range.end(), n) - 1;
      std::vector<int> sub(vec.begin() + left, vec.begin() + right + 1);
      SortByDigit(sub);
      std::copy(sub.begin(), sub.end(), vec.begin() + left);
    }
  );
  OddEvenMergeSort(vec, 0, n);
}
