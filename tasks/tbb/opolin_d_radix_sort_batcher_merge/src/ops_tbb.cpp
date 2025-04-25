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

void opolin_d_radix_batcher_sort_tbb::BatcherMergeRadixSort(std::vector<int>& vec) {
  const size_t n = vec.size();
  if (n <= 1) return;
  const int max_threads = tbb::this_task_arena::max_concurrency();
  const size_t block_size = std::max<size_t>(1, n / max_threads);
  tbb::parallel_for(tbb::blocked_range<size_t>(0, n, block_size), [&](const tbb::blocked_range<size_t>& r) {
    auto start = vec.begin() + r.begin();
    auto end = vec.begin() + r.end();
    std::vector<int> block(start, end);
    SortByDigit(block);
    std::copy(block.begin(), block.end(), start);
  });
  for (size_t step = block_size; step < n; step *= 2) {
    const size_t pair_step = step * 2;
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n / pair_step + 1), [&](const tbb::blocked_range<size_t>& r) {
      for (size_t i = r.begin(); i < r.end(); ++i) {
        size_t left = i * pair_step;
        size_t mid = left + step;
        size_t right = std::min(left + pair_step, n);
        if (mid < right) {
          OddEvenMerge(vec, left, right - left);
        }
      }
    });
  }
}

void opolin_d_radix_batcher_sort_tbb::OddEvenMerge(std::vector<int>& vec, size_t low, size_t n) {
  if (n <= 1) return;
  size_t mid = n / 2;
  size_t even_size = mid + (n % 2);
  size_t odd_size = mid;
  tbb::parallel_invoke([&] { OddEvenMerge(vec, low, even_size); },
                       [&] { OddEvenMerge(vec, low + even_size, odd_size); });
  tbb::parallel_for(tbb::blocked_range<size_t>(0, mid), [&](const tbb::blocked_range<size_t>& r) {
    for (size_t i = r.begin(); i < r.end(); ++i) {
      size_t idx_even = low + 2 * i;
      size_t idx_odd = idx_even + 1;
      if (idx_odd < low + n && vec[idx_even] > vec[idx_odd]) {
        std::swap(vec[idx_even], vec[idx_odd]);
      }
    }
  });
}