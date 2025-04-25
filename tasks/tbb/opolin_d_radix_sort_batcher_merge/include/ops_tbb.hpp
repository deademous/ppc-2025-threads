#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace opolin_d_radix_batcher_sort_tbb {
void BatcherMergeRadixSort(std::vector<int>& vec);
void BatcherMerge(std::vector<int>& arr, size_t l, size_t m, size_t r);
void SortByDigit(std::vector<int>& vec);
uint32_t ConvertKey(int num);

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
