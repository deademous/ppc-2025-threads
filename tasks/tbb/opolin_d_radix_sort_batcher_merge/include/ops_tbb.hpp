#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace opolin_d_radix_batcher_sort_tbb {
void BatcherMergeRadixSort(std::vector<int> &vec);
void MergeBlocksStep(std::pair<int *, int> &left, std::pair<int *, int> &right);
void ParallelBatcherMergeBlocks(std::vector<int> &arr, int num_threads);
void SortByDigit(std::vector<int> &vec);

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
