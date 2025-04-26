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
  if (vec.empty()) {
    return;
  }
  std::vector<uint32_t> keys(vec.size());
  tbb::parallel_for(tbb::blocked_range<size_t>(0, vec.size()), [&](const tbb::blocked_range<size_t>& r) {
    for (size_t i = r.begin(); i < r.end(); ++i) {
      keys[i] = ConvertKey(vec[i]);
    }
  });
  std::vector<uint32_t> buf(vec.size());
  for (int shift = 0; shift < 32; shift += 8) {
    std::vector<size_t> count(256, 0);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, keys.size()), [&](const tbb::blocked_range<size_t>& r) {
      for (size_t i = r.begin(); i < r.end(); ++i) {
        ++count[(keys[i] >> shift) & 0xFF];
      }
    });
    for (size_t i = 1; i < 256; ++i) {
      count[i] += count[i - 1];
    }
    for (size_t i = keys.size(); i-- > 0;) {
      uint8_t byte = (keys[i] >> shift) & 0xFF;
      buf[--count[byte]] = keys[i];
    }
    keys.swap(buf);
  }

  tbb::parallel_for(tbb::blocked_range<size_t>(0, vec.size()), [&](const tbb::blocked_range<size_t>& r) {
    for (size_t i = r.begin(); i < r.end(); ++i) {
      vec[i] = static_cast<int>(keys[i] ^ ((keys[i] >> 31) >> 1));
    }
  });
}

uint32_t opolin_d_radix_batcher_sort_tbb::ConvertKey(int num) {
  return static_cast<uint32_t>(num) ^ (static_cast<uint32_t>(num >> 31) >> 1);
}

void opolin_d_radix_batcher_sort_tbb::BatcherMerge(std::vector<int>& arr, size_t low, size_t n) {
  if (n <= 1) {
    return;
  }
  size_t mid = n/2;
  tbb::parallel_invoke(
    [&]{ BatcherMerge(arr, low, mid); },
    [&]{ BatcherMerge(arr, low + mid, n - mid); });
  size_t step = (n + 1)/2;
  tbb::parallel_for(tbb::blocked_range<size_t>(0, step),
  [&](const tbb::blocked_range<size_t>& r) {
    for (size_t i = r.begin(); i < r.end(); ++i) {
      size_t idx1 = low + i * 2;
      size_t idx2 = idx1 + step;
      if (idx2 < low + n && arr[idx1] > arr[idx2]) {
        std::swap(arr[idx1], arr[idx2]);
      }
    }
  });
}

void opolin_d_radix_batcher_sort_tbb::BatcherMergeRadixSort(std::vector<int>& vec) {
  const size_t n = vec.size();
  if (n <= 1) {
    return;
  }
  const size_t block_size = std::max<size_t>(1, n / tbb::this_task_arena::max_concurrency());
  tbb::parallel_for(tbb::blocked_range<size_t>(0, n, block_size),
  [&](const tbb::blocked_range<size_t>& r) {
    auto start = vec.begin() + r.begin();
    auto end = vec.begin() + r.end();
    std::vector<int> block(start, end);
    SortByDigit(block);
    std::copy(block.begin(), block.end(), start);
  });
  for (size_t k = 2; k < 2 * n; k *= 2) {
    for (size_t j = k / 2; j > 0; j /= 2) {
      tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
      [&](const tbb::blocked_range<size_t>& r) {
        for (size_t i = r.begin(); i < r.end(); ++i) {
          size_t l = i ^ j;
          if (l > i && l < n) {
            if ((i & k) == 0 && vec[i] > vec[l]) {
              std::swap(vec[i], vec[l]);
            }
            if ((i & k) != 0 && vec[i] < vec[l]) {
              std::swap(vec[i], vec[l]);
            }
          }
        }
      });
    }
  }
}
