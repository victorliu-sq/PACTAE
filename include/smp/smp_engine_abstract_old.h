#ifndef SMP_ENGINE_ABSTRACT_H
#define SMP_ENGINE_ABSTRACT_H
#include <utils/timer.h>

class OldAbsSmpEngine {
public:
  virtual ~OldAbsSmpEngine() = default;

  void FindStableMatching() {
    PROFILE_SCOPE("Total Time");

    if (!this->IsPerfect()) {
      {
        PROFILE_SCOPE("Total Init Time");
        this->InitProc();
      }

      this->CoreProc();

      this->PostProc();
    } else {
      std::cout << "Perfect matching identified; skipping execution steps." << std::endl;
    }
  }

protected:
  inline virtual auto IsPerfect() const -> bool = 0;

  virtual void InitProc() = 0;

  virtual void CoreProc() = 0;

  virtual void PostProc() = 0;
};

#endif //SMP_ENGINE_ABSTRACT_H
