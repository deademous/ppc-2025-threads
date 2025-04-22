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
  tbb::parallel_for(tbb::blocked_range<size_t>(0, vec.size()), [&](const tbb::blocked_range<size_t>& r) {
    for (size_t i = r.begin(); i != r.end(); ++i) {
      uns_vec[i] = static_cast<uint32_t>(vec[i]) ^ 0x80000000u;
    }
  });
  std::vector<uint32_t> buffer(vec.size());
  for (int shift = 0; shift < 32; shift += 8) {
    std::vector<size_t> counts(256, 0);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, vec.size()), [&](const tbb::blocked_range<size_t>& r) {
      std::vector<size_t> local_counts(256, 0);
      for (size_t i = r.begin(); i != r.end(); ++i) {
        local_counts[(uns_vec[i] >> shift) & 0xFF]++;
      }
      for (size_t j = 0; j < 256; ++j) {
        counts[j] += local_counts[j];
      }
    });
    for (size_t i = 1; i < 256; ++i) {
      counts[i] += counts[i - 1];
    }
    tbb::parallel_for(tbb::blocked_range<size_t>(0, vec.size()), [&](const tbb::blocked_range<size_t>& r) {
      for (size_t i = r.end(); i-- > r.begin();) {
        uint32_t byte = (uns_vec[i] >> shift) & 0xFF;
        buffer[--counts[byte]] = uns_vec[i];
      }
    });
    uns_vec.swap(buffer);
  }

  tbb::parallel_for(tbb::blocked_range<size_t>(0, vec.size()), [&](const tbb::blocked_range<size_t>& r) {
    for (size_t i = r.begin(); i != r.end(); ++i) {
      vec[i] = static_cast<int>(uns_vec[i] ^ 0x80000000u);
    }
  });
}

void opolin_d_radix_batcher_sort_tbb::OddEvenMerge(std::vector<int>& vec, int start, int mid, int end) {
  if (end - start <= 1) {
    return;
  }
  int n = end - start;
  tbb::parallel_invoke([&] { OddEvenMerge(vec, start, (start + mid) / 2, mid); }, [&] { OddEvenMerge(vec, mid, (mid + end) / 2, end); });
  for (int step = 1; step < n; step *= 2) {
    tbb::parallel_for(tbb::blocked_range<int>(0, n / (2 * step)), [&](const tbb::blocked_range<int>& r) {
      for (int i = r.begin(); i != r.end(); ++i) {
        int left = start + 2 * step * i;
        int right = std::min(left + 2 * step, end);
        for (int j = left + step; j < right; j += step) {
          if (j - step >= start && vec[j - step] > vec[j]) {
            std::swap(vec[j - step], vec[j]);
          }
        }
      }
    });
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
    std::vector<int> temp(vec.begin() + range.begin(), vec.begin() + std::min(range.end(), n));
    SortByDigit(temp);
    std::copy(temp.begin(), temp.end(), vec.begin() + range.begin());
  });
  int step = chunk_size;
  while (step < n) {
    tbb::parallel_for(tbb::blocked_range<int>(0, n, 2 * step), [&](const tbb::blocked_range<int>& r) {
      for (int left = r.begin(); left < r.end(); left += 2 * step) {
        int mid = std::min(left + step, n);
        int right = std::min(left + 2 * step, n);
        if (mid < right) {
          OddEvenMerge(vec, left, mid, right);
        }
      }
    });
    step *= 2;
  }
}