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
  if (vec.size() <= 1) {
    return;
  }
  std::vector<uint32_t> uns_vec(vec.size());
  for (std::size_t i = 0; i < vec.size(); i++) {
    uns_vec[i] = static_cast<uint32_t>(vec[i]) ^ 0x80000000u;
  }
  std::vector<uint32_t> buf(vec.size());
  for (int shift = 0; shift < 32; shift += 8) {
    int cnt[256] = {};
    for (std::size_t i = 0; i < uns_vec.size(); i++) {
      cnt[(uns_vec[i] >> shift) & 255]++;
    }
    for (std::size_t i = 1; i < 256; i++) {
      cnt[i] += cnt[i - 1];
    }
    for (std::size_t i = uns_vec.size(); i-- > 0;) {
      uint32_t byte = (uns_vec[i] >> shift) & 255u;
      buf[cnt[byte] - 1] = uns_vec[i];
      cnt[byte]--;
    }
    uns_vec.swap(buf);
  }
  for (std::size_t i = 0; i < vec.size(); i++) {
    vec[i] = static_cast<int>(uns_vec[i] ^ 0x80000000u);
  }
}

void opolin_d_radix_batcher_sort_tbb::OddEvenMerge(std::vector<int>& vec, int left, int n, int step, int size) {
  if (step <= 0 || n <= 1) {
    return;
  }
  int m = 2 * step;
  if (m < n) {
    tbb::parallel_invoke([&] { OddEvenMerge(vec, left, n, m, size); },
                         [&] { OddEvenMerge(vec, left + step, n, m, size); });
  }
  if (step < n) {
    int end_i = left + n - step;
    for (int i = left; i < end_i; i += m) {
      if (i + step < size) {
        if (vec[i] > vec[i + step]) {
          std::swap(vec[i], vec[i + step]);
        }
      }
    }
    for (int i = left + step; i < end_i; i += m) {
      if (i + step < size) {
        if (vec[i] > vec[i + step]) {
          std::swap(vec[i], vec[i + step]);
        }
      }
    }
  }
}

void opolin_d_radix_batcher_sort_tbb::OddEvenMergeSort(std::vector<int>& vec, int left, int n, int size) {
  if (n > 1) {
    int m = (n + 1) / 2;
    tbb::parallel_invoke([&] { OddEvenMergeSort(vec, left, m, size); },
                         [&] { OddEvenMergeSort(vec, left + m, n - m, size); });
    OddEvenMerge(vec, left, n, 1, size);
  }
}

void opolin_d_radix_batcher_sort_tbb::BatcherMergeRadixSort(std::vector<int>& vec) {
  int n = static_cast<int>(vec.size());
  if (n < 2) {
    return;
  }
  int threads = tbb::this_task_arena::max_concurrency();
  int chunk_size = std::max(1, (n + threads - 1) / threads);
  tbb::parallel_for(tbb::blocked_range<int>(0, n, chunk_size), [&](const tbb::blocked_range<int>& range) {
    int left = range.begin();
    int right = std::min(range.end(), n);
    std::vector<int> sub(vec.begin() + left, vec.begin() + right);
    SortByDigit(sub);
    std::copy(sub.begin(), sub.end(), vec.begin() + left);
  });
  int current_run_size = chunk_size;
  while (current_run_size < n) {
    int merge_segment_size = 2 * current_run_size;
    tbb::parallel_for(tbb::blocked_range<int>(0, n, merge_segment_size), [&](const tbb::blocked_range<int>& range) {
      int left = range.begin();
      int right = std::min(left + merge_segment_size, n);
      int segment_length = right - left;
      if (segment_length > 1) {
        OddEvenMergeSort(vec, left, segment_length, n);
      }
    });
    if (merge_segment_size > n && current_run_size < n) {
      current_run_size = n;
    } else {
      current_run_size = merge_segment_size;
    }
  }
}
