#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace opolin_d_radix_batcher_sort_tbb {
uint32_t ConvertIntToKey(int num);
int ConvertKeyToInt(uint32_t key);
void ParallelRadixSortInternal(std::vector<uint32_t>& keys);
void ParallelBatcherOddEvenMergeInternal(std::vector<int>& arr, size_t low, size_t high, size_t total_size);

class RadixBatcherSortTaskTbb : public ppc::core::Task {
 public:
  explicit RadixBatcherSortTaskTbb(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_;
  std::vector<int> output_;
  int size_;
};
}  // namespace opolin_d_radix_batcher_sort_tbb
