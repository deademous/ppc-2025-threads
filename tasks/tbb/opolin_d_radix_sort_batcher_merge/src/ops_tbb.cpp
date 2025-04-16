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

void opolin_d_radix_batcher_sort_tbb::RadixSort(std::vector<int>& arr, int exp) {
  const std::size_t n = arr.size();
  std::vector<int> output(n);
  std::vector<int> count(10, 0);
  tbb::mutex count_mutex;

  tbb::parallel_for(tbb::blocked_range<std::size_t>(0, n), [&](const tbb::blocked_range<std::size_t>& r) {
    std::vector<int> local_count(10, 0);
    for (std::size_t i = r.begin(); i != r.end(); ++i) {
      local_count[(arr[i] / exp) % 10]++;
    }
    tbb::mutex::scoped_lock lock(count_mutex);
    for (int i = 0; i < 10; ++i) {
      count[i] += local_count[i];
    }
  });

  // Accumulate the counts
  for (int i = 1; i < 10; i++) {
    count[i] += count[i - 1];
  }
  // Sort based on current digit
  tbb::parallel_for(tbb::blocked_range<std::size_t>(0, n), [&](const tbb::blocked_range<std::size_t>& r) {
    for (std::size_t i = r.end(); i-- > r.begin();) {
      int index = (arr[i] / exp) % 10;
      tbb::mutex::scoped_lock lock(count_mutex);
      output[--count[index]] = arr[i];
    }
  });

  tbb::parallel_for(tbb::blocked_range<std::size_t>(0, n), [&](const tbb::blocked_range<std::size_t>& r) {
    for (std::size_t i = r.begin(); i != r.end(); ++i) {
      arr[i] = output[i];
    }
  });
}

void opolin_d_radix_batcher_sort_tbb::parallelOddEvenMerge(std::vector<int>& arr, int lo, int hi) {
  int n = hi - lo + 1;
  int r = 1;
  while (r < n) {
    r = r << 1;
  }
  std::vector<int> b(n);
  for (int i = 0; i < n; i++) {
    b[i] = arr[lo + i];
  }
  for (int s = 1; s <= r; s = s * 2) {
    int d = s * 2;
    tbb::parallel_for(0, n / d + (n % d != 0), [&](int i) {
      int i_d = i * d;
      int l = i_d;
      int m = std::min(i_d + s, n);
      int h = std::min(i_d + d, n);
      int p = l;
      int q = m;
      int t = 0;
      std::vector<int> tmp(h - l);
      while (p < m && q < h) {
        if (b[p] <= b[q]) {
          tmp[t++] = b[p++];
        } else {
          tmp[t++] = b[q++];
        }
      }
      while (p < m) {
        tmp[t++] = b[p++];
      }
      while (q < h) {
        tmp[t++] = b[q++];
      }
      for (int i = 0; i < t; i++) {
        b[l + i] = tmp[i];
      }
    });
  }
  for (int i = 0; i < n; i++) {
    arr[lo + i] = b[i];
  }
}

void opolin_d_radix_batcher_sort_tbb::BatcherMergeRadixSort(std::vector<int>& arr) {
  size_t n = arr.size();
  if (n <= 1) {
    return;
  }

  int max_val = 0;
  for (int val : arr) {
    max_val = std::max(max_val, std::abs(val));
  }
  int num_digits = (max_val == 0) ? 1 : static_cast<int>(std::log10(max_val)) + 1;
  for (int digit = 0; digit < num_digits; ++digit) {
    int exp = static_cast<int>(std::pow(10, digit));
    RadixSort(arr, exp);
  }
  parallelOddEvenMerge(arr, 0, n - 1);
}