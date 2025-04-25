#include "tbb/opolin_d_radix_sort_batcher_merge/include/ops_tbb.hpp"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/task_arena.h>
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

void opolin_d_radix_batcher_sort_tbb::MergeBlocksStep(std::pair<int*, int>& left, std::pair<int*, int>& right) {
  std::inplace_merge(left.first, right.first, right.first + right.second);
  left.second += right.second;
}

void opolin_d_radix_batcher_sort_tbb::ParallelBatcherMergeBlocks(std::vector<int>& arr, int num_threads) {
  const int sz = static_cast<int>(arr.size());
  if (sz <= 1) {
    return;
  }
  const int bsz = sz / num_threads;
  const int bex = sz % num_threads;
  std::vector<std::pair<int*, int>> vb(num_threads);
  int current_pos = 0;
  for (int i = 0; i < num_threads; ++i) {
    int current_bsz = bsz + (i < bex ? 1 : 0);
    vb[i] = std::make_pair(arr.data() + current_pos, current_bsz);
    current_pos += current_bsz;
  }
  for (int step = 1; step < num_threads; step *= 2) {
    const int merge_block_size = 2 * step;
    tbb::parallel_for(tbb::blocked_range<int>(0, num_threads, merge_block_size),
                      [&](const tbb::blocked_range<int>& range) {
                        for (int i = range.begin(); i < range.end(); i += merge_block_size) {
                          if (i + step < num_threads) {
                            MergeBlocksStep(vb[i], vb[i + step]);
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
  int chunk_size = (n + threads - 1) / threads;
  chunk_size = std::max(1, std::min(chunk_size, n));
  tbb::parallel_for(tbb::blocked_range<int>(0, n, chunk_size), [&](const tbb::blocked_range<int>& range) {
    int left = range.begin();
    int current_n = range.end() - left;
    if (current_n > 0) {
      std::vector<int> sub_vec(vec.begin() + left, vec.begin() + range.end());
      SortByDigit(sub_vec);
      std::copy(sub_vec.begin(), sub_vec.end(), vec.begin() + left);
    }
  });
  int num_chunks = (n + chunk_size - 1) / chunk_size;
  ParallelBatcherMergeBlocks(vec, num_chunks);
}