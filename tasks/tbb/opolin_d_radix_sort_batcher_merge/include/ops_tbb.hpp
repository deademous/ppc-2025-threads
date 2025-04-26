#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace opolin_d_radix_batcher_sort_tbb {
uint32_t ConvertIntToKey(int num);
int ConvertKeyToInt(uint32_t key);
void RadixSort(std::vector<uint32_t>& keys);
void BatcherOddEvenMerge(std::vector<int>& arr, int low, int high);

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
  size_t size_;
};
}  // namespace opolin_d_radix_batcher_sort_tbb
