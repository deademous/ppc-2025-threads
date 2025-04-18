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

void opolin_d_radix_batcher_sort_tbb::SortByDigit(std::vector<int>& vec) {
  if (vec.empty()) {
    return;
  }
  int max_val = *std::max_element(vec.begin(), vec.end());
  for (int exp = 1; max_val / exp > 0; exp *= 10) {
    std::vector<int> output(vec.size());
    std::vector<int> count(10, 0);
    for (int num : vec) {
      int index = (num / exp) % 10;
      count[index]++;
    }
    for (int i = 1; i < 10; i++) {
      count[i] += count[i - 1];
    }
    for (int i = vec.size() - 1; i >= 0; i--) {
      int index = (vec[i] / exp) % 10;
      output[--count[index]] = vec[i];
    }
    vec.swap(output);
  }
}

void opolin_d_radix_batcher_sort_tbb::RadixSort(std::vector<int>& a, int l, int r) {
  if (l >= r) {
    return;
  }
  std::vector<int> positive;
  std::vector<int> negative;
  for (int i = l; i <= r; ++i) {
    if (a[i] >= 0) {
      positive.push_back(a[i]);
    } else {
      negative.push_back(-a[i]);
    }
  }
  tbb::parallel_invoke([&]() { SortByDigit(positive); }, [&]() { SortByDigit(negative); });
  std::reverse(negative.begin(), negative.end());
  size_t idx = l;
  for (int num : negative) {
    a[idx++] = -num;
  }
  for (int num : positive) {
    a[idx++] = num;
  }
}

void opolin_d_radix_batcher_sort_tbb::OddEvenMerge(std::vector<int>& a, int l, int r) {
  int n = r - l + 1;
  if (n <= 1) {
    return;
  }
  int m = (l + r) / 2;
  tbb::parallel_invoke([&] { OddEvenMerge(a, l, m); }, [&] { OddEvenMerge(a, m + 1, r); });
  for (int i = l + 1; i + 1 <= r; i += 2) {
    if (a[i] > a[i + 1]) {
      std::swap(a[i], a[i + 1]);
    }
  }
}

void opolin_d_radix_batcher_sort_tbb::BatcherMergeRadixSort(std::vector<int>& a) {
  int n = a.size();
  if (n <= 1) {
    return;
  }
  int num_threads = tbb::this_task_arena::max_concurrency();
  int chunk_size = std::max(1, (n + num_threads - 1) / num_threads);
  tbb::parallel_for(tbb::blocked_range<int>(0, n, chunk_size), [&](const tbb::blocked_range<int>& r) {
    int start = r.begin();
    int end = std::min(r.end(), n) - 1;
    if (start <= end) {
      RadixSort(a, start, end);
    }
  });
  OddEvenMerge(a, 0, n - 1);
}
