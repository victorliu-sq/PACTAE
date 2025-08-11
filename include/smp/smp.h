#pragma once
#include <string>
#include <thread>
#include <utils/utils.h>
#include <vector>

namespace bamboosmp {
  struct SmpObj {
    // Memory Management
    Vector<int> flatten_pref_lists_m_vec;
    Vector<int> flatten_pref_lists_w_vec;

    // Access
    int *flatten_pref_lists_m;
    int *flatten_pref_lists_w;

    // static auto CreateFromJson(const String &filepath) -> UPtr<SmpObj>;

    // ----- Existing factory: keeps 2D path for compatibility -----
    static auto CreateFromPrefLists(const PreferenceLists &pl_m,
                                    const PreferenceLists &pl_w,
                                    int n) -> UPtr<SmpObj>;

    // ----- New factory: flat (1D) path with no re-flattening -----
    static auto CreateFromFlatLists(const FlatPreferenceList &flat_m,
                                    const FlatPreferenceList &flat_w,
                                    int n) -> UPtr<SmpObj> {
      return UPtr<SmpObj>(new SmpObj(flat_m, flat_w, n));
    }

    // Initialize pointers immediately after vectors created
    void InitPointers() {
      flatten_pref_lists_m = RawPtr(flatten_pref_lists_m_vec);
      flatten_pref_lists_w = RawPtr(flatten_pref_lists_w_vec);
    }

  private:
    SmpObj(const PreferenceLists &pl_m, const PreferenceLists &pl_w, int n);

    // New path: no flattening, just take the flat vectors directly
    SmpObj(const FlatPreferenceList &flat_m,
                   const FlatPreferenceList &flat_w,
                   int /*n*/)
      : flatten_pref_lists_m_vec(flat_m),
        flatten_pref_lists_w_vec(flat_w) {
      this->InitPointers();
    }

    static void FlattenRowHost(const PreferenceLists &pls, Vector<int> &flat_pl, int row, int n);

    static auto ParallelFlattenHost(const PreferenceLists &pl, int n) -> Vector<int>;
  };
} // namespace bamboosmp
