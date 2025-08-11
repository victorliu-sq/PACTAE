#ifndef SMP_INIT_H
#define SMP_INIT_H

#include <string>
#include <thread>
#include <utils/utils.h>
#include <vector>

namespace bamboosmp {
  struct SmpObjInit {
    // Memory Management
    HVector<int> flatten_pref_lists_m_vec;
    HVector<int> flatten_pref_lists_w_vec;

    // Access
    int *flatten_pref_lists_m;
    int *flatten_pref_lists_w;

    // ----- New factory: flat (1D) path with no re-flattening -----
    static auto CreateFromFlatLists(const FlatPreferenceList &flat_m,
                                    const FlatPreferenceList &flat_w,
                                    int n) -> UPtr<SmpObjInit> {
      return UPtr<SmpObjInit>(new SmpObjInit(flat_m, flat_w, n));
    }

    // Initialize pointers immediately after vectors created
    void InitPointers() {
      flatten_pref_lists_m = RawPtr(flatten_pref_lists_m_vec);
      flatten_pref_lists_w = RawPtr(flatten_pref_lists_w_vec);
    }

  private:
    // New path: no flattening, just take the flat vectors directly
    SmpObjInit(const FlatPreferenceList &flat_m,
                   const FlatPreferenceList &flat_w,
                   int /*n*/)
      : flatten_pref_lists_m_vec(flat_m),
        flatten_pref_lists_w_vec(flat_w) {
      this->InitPointers();
    }
  };
} // namespace bamboosmp


#endif //SMP_INIT_H
