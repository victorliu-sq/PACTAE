#ifndef STOPWATCH_H
#define STOPWATCH_H
#include "utils/utils.h"

class StopWatch {
private:
  uint64_t start_;
  uint64_t end_;
  bool running_;

public:
  explicit StopWatch(bool start_now = true): running_(start_now) {
    if (start_now) {
      Start();
    }
  }

  void Start() {
    start_ = getNanoSecond();
  }

  void Stop() {
    end_ = getNanoSecond();
  }

  auto GetEclapsedMs() -> double {
    return (end_ - start_) / 1e6;
  }

  auto GetEclapsedNs() -> ns_t {
    return end_ - start_;
  }
};

#endif //STOPWATCH_H
