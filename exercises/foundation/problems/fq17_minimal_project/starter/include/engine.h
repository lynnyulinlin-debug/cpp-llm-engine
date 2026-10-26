#pragma once

#include <cstddef>
#include <memory>

class Workspace {
 public:
  explicit Workspace(std::size_t size);

  Workspace(const Workspace&) = delete;
  Workspace& operator=(const Workspace&) = delete;
  Workspace(Workspace&&) noexcept;
  Workspace& operator=(Workspace&&) noexcept;
  ~Workspace();

  int& at(std::size_t index);
  std::size_t size() const;

 private:
  std::unique_ptr<int[]> data_;
  std::size_t size_ = 0;
};

class EngineStep {
 public:
  explicit EngineStep(std::size_t workspace_size);
  int run(int input);

 private:
  Workspace workspace_;
};
