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

void opolin_d_radix_batcher_sort_tbb::CompExch(int &a, int &b) {
  if (a > b) {
    std::swap(a, b);
  }
}

void opolin_d_radix_batcher_sort_tbb::BatcherSortNetwork(std::vector<int> &arr, int l, int r) {
  if (l >= r) return;
  int n = r - l + 1;
  for (int p = 1; p < n; p += p) {
    for (int k = p; k > 0; k /= 2) {
      for (int j = k % p; j + k < n; j += (k + k)) {
        int i_range_end = n - j - k;
        if (i_range_end > 0) {
          tbb::parallel_for(tbb::blocked_range<int>(0, i_range_end), [&](const tbb::blocked_range<int>& range_i) {
            for (int i = range_i.begin(); i < range_i.end(); ++i) {
              int idx1 = l + j + i;
              int idx2 = idx1 + k;
              if (idx2 < (int)arr.size()) {
                if (((i / (2 * p)) == ((i + k) / (2 * p)))) {
                  CompExch(arr[idx1], arr[idx2]);
                }
              }
            }
          });
        }
      }
    }
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
  BatcherSortNetwork(vec, 0, n - 1);
}