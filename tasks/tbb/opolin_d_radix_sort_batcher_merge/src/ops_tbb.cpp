#include "tbb/opolin_d_radix_sort_batcher_merge/include/ops_tbb.hpp"

#include <tbb/tbb.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
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
  RadixSort(output_);
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

void opolin_d_radix_batcher_sort_tbb::RadixSort(std::vector<int>& data) {
  size_t n = data.size();
  if (n <= 1) {
    return;
  }
  std::vector<uint32_t> keys(n);
  tbb::parallel_for(tbb::blocked_range<size_t>(0, n), [&](const tbb::blocked_range<size_t>& r) {
    for (size_t i = r.begin(); i < r.end(); ++i) {
      keys[i] = ConvertIntToKey(data[i]);
    }
  });
  const int radix = 256;
  std::vector<uint32_t> output_keys(n);
  for (int pass = 0; pass < 4; ++pass) {
    std::vector<std::atomic<size_t>> count(radix);
    for (auto& c : count) {
      c.store(0, std::memory_order_relaxed);
    }
    int shift = pass * 8;
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n), [&](const tbb::blocked_range<size_t>& r) {
      for (size_t i = r.begin(); i < r.end(); ++i) {
        auto byte = static_cast<uint8_t>((keys[i] >> shift) & 0xFF);
        count[byte].fetch_add(1, std::memory_order_relaxed);
      }
    });
    std::vector<size_t> count_prefix(radix);
    count_prefix[0] = count[0].load(std::memory_order_relaxed);
    for (int j = 1; j < radix; ++j) {
      count_prefix[j] = count_prefix[j - 1] + count[j].load(std::memory_order_relaxed);
    }
    for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
      auto byte = static_cast<uint8_t>((keys[i] >> shift) & 0xFF);
      size_t index = --count_prefix[byte];
      output_keys[index] = keys[i];
    }
    keys.swap(output_keys);
  }
  tbb::parallel_for(tbb::blocked_range<size_t>(0, n), [&](const tbb::blocked_range<size_t>& r) {
    for (size_t i = r.begin(); i < r.end(); ++i) {
      data[i] = ConvertKeyToInt(keys[i]);
    }
  });
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