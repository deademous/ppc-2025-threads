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
  int exp = 1;
  while (max_val / exp > 0) {
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
    vec = output;
    exp *= 10;
  }
}

void opolin_d_radix_batcher_sort_tbb::RadixSort(std::vector<int>& a, int l, int r) {
  if (l >= r) return;
  std::vector<int> positive;
  std::vector<int> negative;
  for (int i = l; i <= r; ++i) {
    if (a[i] >= 0) {
      positive.push_back(a[i]);
    } else {
      negative.push_back(-a[i]);
    }
  }
  tbb::parallel_invoke(
    [&]() { SortByDigit(positive); },
    [&]() { SortByDigit(negative); }
  );
  std::reverse(negative.begin(), negative.end());
  size_t idx = l;
  for (int num : negative) {
    a[idx++] = -num;
  }
  for (int num : positive) {
    a[idx++] = num;
  }
}

void opolin_d_radix_batcher_sort_tbb::OddEvenMerge(std::vector<int>& a, int l, int m, int r) {
  int n = r - l + 1;
  if (n <= 1) {
    return;
  }
  if (n == 2) {
    if (a[l] > a[r]) {
      std::swap(a[l], a[r]);
    }
    return;
  }

  int mid = (l + r) / 2;
  std::vector<int> even, odd;
  for (int i = l; i <= r; i += 2) {
    even.push_back(a[i]);
    if (i + 1 <= r) odd.push_back(a[i + 1]);
  }
  tbb::parallel_invoke(
    [&]() { OddEvenMerge(a, l, (l + mid) / 2, mid); },
    [&]() { OddEvenMerge(a, mid + 1, (mid + 1 + r) / 2, r); }
  );
  tbb::parallel_for(tbb::blocked_range<int>(l, mid + 1), [&](const tbb::blocked_range<int>& range) {
    for (int i = range.begin(); i < range.end(); ++i) {
      int j = i + (mid - l) + 1;
      if (j <= r && a[i] > a[j]) {
        std::swap(a[i], a[j]);
      }
    }
  });
}

void opolin_d_radix_batcher_sort_tbb::BatcherMergeRadixSort(std::vector<int>& a) {
  int n = a.size();
  if (n <= 1) {
    return;
  }
  int num_threads = tbb::task_scheduler_init::default_num_threads();
  int chunk_size = std::max(1, n / num_threads);

  tbb::parallel_for(tbb::blocked_range<int>(0, n, chunk_size), [&](const tbb::blocked_range<int>& r) {
    int start = r.begin();
    int end = std::min(r.end() - 1, n - 1);
    if (start <= end) {
      RadixSort(a, start, end);
    }
  });
  for (int step = chunk_size; step < n; step *= 2) {
    tbb::parallel_for(tbb::blocked_range<int>(0, n, 2 * step), [&](const tbb::blocked_range<int>& r) {
      for (int i = r.begin(); i < r.end(); i += 2 * step) {
        int left = i;
        int mid = i + step - 1;
        int right = std::min(i + 2 * step - 1, n - 1);
        if (mid < n - 1) {
          OddEvenMerge(a, left, mid, right);
        }
      }
    });
  }
}