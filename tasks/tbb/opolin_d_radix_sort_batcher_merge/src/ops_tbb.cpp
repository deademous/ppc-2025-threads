#include "tbb/opolin_d_radix_sort_batcher_merge/include/ops_tbb.hpp"

#include <tbb/tbb.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
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

void opolin_d_radix_batcher_sort_tbb::CompareSwap(std::vector<int>& vec, size_t i, size_t j) {
  if (vec[i] > vec[j]) {
    std::swap(vec[i], vec[j]);
  }
}

void opolin_d_radix_batcher_sort_tbb::OddEvenMergeStep(std::vector<int>& vec, size_t dist) {
  size_t n = vec.size();
  tbb::parallel_for(tbb::blocked_range<size_t>(0, n - dist),
      [&](const tbb::blocked_range<size_t>& range) {
      for (size_t i = range.begin(); i < range.end(); ++i) {
           CompareSwap(vec, i, i + dist);
      }
  });
}

void opolin_d_radix_batcher_sort_tbb::BatcherMergeNetwork(std::vector<int>& vec) {
  size_t n = vec.size();
  if (n <= 1) {
    return;
  }
  for (size_t p = 1; p < n; p <<= 1) {
    for (size_t k = p; k >= 1; k >>= 1) {
      tbb::parallel_for(tbb::blocked_range<size_t>(0, n - k),
      [&](const tbb::blocked_range<size_t>& r) {
        for (size_t j = r.begin(); j < r.end(); ++j) {
          bool in_same_block = ((j / (p * 2)) == ((j + k) / (p * 2)));
          if (in_same_block && ((j & p) == 0)) {
            CompareSwap(vec, j, j + k);
          }
        }
      });
    }
  }
}

void opolin_d_radix_batcher_sort_tbb::BatcherMergeRadixSort(std::vector<int>& vec) {
  size_t n = vec.size();
  if (n <= 1) {
    return;
  }
  int max_threads = tbb::this_task_arena::max_concurrency();
  size_t num_threads = static_cast<size_t>(max_threads);
  num_threads = std::min(num_threads, n);

  size_t chunk_size = n / num_threads;
  size_t remainder = n % num_threads;

  tbb::parallel_for(
    tbb::blocked_range<size_t>(0, num_threads),
    [&](const tbb::blocked_range<size_t>& r) {
      for (size_t i = r.begin(); i < r.end(); ++i) {
        size_t start = i * chunk_size + std::min(i, remainder);
        size_t end = start + chunk_size + (i < remainder ? 1 : 0);
        if (start < end) {
          std::vector<int> chunk(vec.begin() + start, vec.begin() + end);
          SortByDigit(chunk);
          std::copy(chunk.begin(), chunk.end(), vec.begin() + start);
        }
      }
    });
  BatcherMergeNetwork(vec);
}