#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace opolin_d_radix_batcher_sort_tbb {
void BatcherMergeRadixSort(std::vector<int>& vec);
void OddEvenMergeStep(std::vector<int>& vec, size_t dist);
void BatcherMergeNetwork(std::vector<int>& vec);
void CompareSwap(std::vector<int>& vec, size_t i, size_t j);
void SortByDigit(std::vector<int>& vec);

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
