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
  std::vector<uint32_t> keys(size_);
  tbb::parallel_for(tbb::blocked_range<size_t>(0, size_), [&](const tbb::blocked_range<size_t>& r) {
    for (size_t i = r.begin(); i < r.end(); ++i) {
      keys[i] = ConvertIntToKey(input_[i]);
    }
  });
  RadixSort(keys);
  output_.resize(size_);
  tbb::parallel_for(tbb::blocked_range<size_t>(0, size_), [&](const tbb::blocked_range<size_t>& r) {
    for (size_t i = r.begin(); i < r.end(); ++i) {
      output_[i] = ConvertKeyToInt(keys[i]);
    }
  });
  BatcherOddEvenMerge(output_, 0, static_cast<int>(output_.size()));
  return true;
}

bool opolin_d_radix_batcher_sort_tbb::RadixBatcherSortTaskTbb::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<int*>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}

uint32_t opolin_d_radix_batcher_sort_tbb::ConvertIntToKey(int num) { return static_cast<uint32_t>(num) ^ 0x80000000U; }

int opolin_d_radix_batcher_sort_tbb::ConvertKeyToInt(uint32_t key) { return static_cast<int>(key ^ 0x80000000U); }

void opolin_d_radix_batcher_sort_tbb::RadixSort(std::vector<uint32_t>& keys) {
  size_t n = keys.size();
  if (n <= 1) {
    return;
  }
  const int bits_per_pass = 8;
  const int num_passes = sizeof(uint32_t);
  const int radix = 1 << bits_per_pass;
  std::vector<uint32_t> output_keys(n);
  for (int pass = 0; pass < num_passes; ++pass) {
    std::vector<std::atomic<size_t>> count(radix);
    int shift = pass * bits_per_pass;
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n), [&](const tbb::blocked_range<size_t>& r) {
      for (size_t i = r.begin(); i < r.end(); ++i) {
        auto byte = static_cast<uint8_t>((keys[i] >> shift) & (radix - 1));
        count[byte].fetch_add(1, std::memory_order_relaxed);
      }
    });
    size_t current_sum = 0;
    for (int j = 0; j < radix; ++j) {
      size_t cnt = count[j].load(std::memory_order_relaxed);
      count[j] = current_sum;
      current_sum += cnt;
    }
    for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
      auto byte = static_cast<uint8_t>((keys[i] >> shift) & (radix - 1));
      size_t dest_index = --count[byte];
      output_keys[dest_index] = keys[i];
    }
    keys.swap(output_keys);
  }
}

void opolin_d_radix_batcher_sort_tbb::BatcherOddEvenMerge(std::vector<int>& arr, int low, int high) {
  if (high - low <= 1) {
    return;
  }
  int mid = (low + high) / 2;

  tbb::parallel_invoke([&] { BatcherOddEvenMerge(arr, low, mid); }, [&] { BatcherOddEvenMerge(arr, mid, high); });

  tbb::parallel_for(tbb::blocked_range<int>(low, mid), [&](const tbb::blocked_range<int>& r) {
    for (int i = r.begin(); i < r.end(); ++i) {
      if (arr[i] > arr[i + mid - low]) {
        std::swap(arr[i], arr[i + mid - low]);
      }
    }
  });
}