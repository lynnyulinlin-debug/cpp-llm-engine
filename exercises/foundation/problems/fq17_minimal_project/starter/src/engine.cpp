#include "engine.h"

#include <cassert>
#include <utility>

Workspace::Workspace(std::size_t size)
    : data_(size ? std::make_unique<int[]>(size) : nullptr), size_(size) {}

Workspace::Workspace(Workspace&& other) noexcept
    : data_(std::move(other.data_)), size_(other.size_) {
  other.size_ = 0;
}

Workspace& Workspace::operator=(Workspace&& other) noexcept {
  if (this != &other) {
    data_ = std::move(other.data_);
    size_ = other.size_;
    other.size_ = 0;
  }
  return *this;
}

Workspace::~Workspace() = default;

int& Workspace::at(std::size_t index) {
  assert(index < size_);
  return data_[index];
}

std::size_t Workspace::size() const { return size_; }

EngineStep::EngineStep(std::size_t workspace_size) : workspace_(workspace_size) {}

int EngineStep::run(int input) {
  workspace_.at(0) = input;
  return workspace_.at(0) + 1;
}
