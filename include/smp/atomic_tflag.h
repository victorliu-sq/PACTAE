#ifndef ATOMIC_BIT_H
#define ATOMIC_BIT_H

// Terminate Flag Value
enum class TFlagValue : int {
  NotFinished = 0,
  FinishedByCpu = 1,
  FinishedByGpu = 2
};

// Atomic Termiante Flag
class AtomicTFlag {
public:
  AtomicTFlag() {
    flag_.store(TFlagValue::NotFinished);
  }

  inline auto TrySetTFlag(TFlagValue val) -> bool {
    TFlagValue expected = TFlagValue::NotFinished;
    return flag_.compare_exchange_strong(expected, val);
  }

  inline TFlagValue GetFlag() const {
    return flag_.load();
  }

  inline void Reset() {
    flag_.store(TFlagValue::NotFinished);
  }

private:
  std::atomic<TFlagValue> flag_;
};

#endif //ATOMIC_BIT_H
