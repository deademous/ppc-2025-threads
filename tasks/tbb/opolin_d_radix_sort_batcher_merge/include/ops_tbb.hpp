#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace opolin_d_radix_batcher_sort_tbb {
void BatcherMergeRadixSort(std::vector<int>& a);
void RadixSort(std::vector<int>& a, int l, int r);
void OddEvenMerge(std::vector<int>& a, int l, int m, int r);
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
