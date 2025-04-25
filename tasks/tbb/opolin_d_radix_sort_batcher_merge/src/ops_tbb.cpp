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
  size_t n = vec.size();
  std::vector<uint32_t> uns_vec(n);
  std::vector<uint32_t> buf(n);
  uint32_t sign_mask = 0x80000000u;
  for (size_t i = 0; i < n; i++) {
    uns_vec[i] = static_cast<uint32_t>(vec[i]) ^ sign_mask;
  }
  for (int shift = 0; shift < 32; shift += 8) {
    std::vector<size_t> cnt(256, 0);
    for (size_t i = 0; i < uns_vec.size(); i++) {
      cnt[(uns_vec[i] >> shift) & 255]++;
    }
    for (size_t i = 1; i < 256; i++) {
      cnt[i] += cnt[i - 1];
    }
    for (size_t i = uns_vec.size(); i-- > 0;) {
      uint32_t byte = (uns_vec[i] >> shift) & 255u;
      buf[cnt[byte] - 1] = uns_vec[i];
      cnt[byte]--;
    }
    uns_vec.swap(buf);
  }
  for (std::size_t i = 0; i < vec.size(); i++) {
    vec[i] = static_cast<int>(uns_vec[i] ^ sign_mask);
  }
}

void opolin_d_radix_batcher_sort_tbb::OddEvenMerge(std::vector<int>& vec, int left, int n, int step) {
  if (step <= 0 || n <= 1) {
    return;
  }
  int m = 2 * step;
  if (m < n) {
    tbb::parallel_invoke([&] { OddEvenMerge(vec, left, n, m); }, [&] { OddEvenMerge(vec, left + step, n, m); });
  }
  if (step < n) {
    int upper_bound = left + n - step;
    tbb::parallel_invoke(
        [&] {
          if (left < upper_bound) {
            tbb::parallel_for(
                tbb::blocked_range<int>(left, upper_bound, m),
                [&](const tbb::blocked_range<int>& r) {
                  for (int i = r.begin(); i < r.end(); ++i) {
                    if (vec[i] > vec[i + step]) {
                      std::swap(vec[i], vec[i + step]);
                    }
                  }
                },
                tbb::simple_partitioner{});
          }
        },
        [&] {
          if (left + step < upper_bound) {
            tbb::parallel_for(
                tbb::blocked_range<int>(left + step, upper_bound, m),
                [&](const tbb::blocked_range<int>& r) {
                  for (int i = r.begin(); i < r.end(); ++i) {
                    if (vec[i] > vec[i + step]) {
                      std::swap(vec[i], vec[i + step]);
                    }
                  }
                },
                tbb::simple_partitioner{});
          }
        });
  }
}

void opolin_d_radix_batcher_sort_tbb::OddEvenMergeSort(std::vector<int>& vec, int left, int n) {
  if (n > 1) {
    int m = (n + 1) / 2;
    tbb::parallel_invoke([&] { OddEvenMergeSort(vec, left, m); }, [&] { OddEvenMergeSort(vec, left + m, n - m); });
    OddEvenMerge(vec, left, n, 1);
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
    int current_n = range.end() - left;
    if (current_n > 0) {
      std::vector<int> sub_vec(vec.begin() + left, vec.begin() + range.end());
      SortByDigit(sub_vec);
      std::copy(sub_vec.begin(), sub_vec.end(), vec.begin() + left);
    }
  });
  OddEvenMergeSort(vec, 0, n);
}
