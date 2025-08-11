#ifndef SMP_ENGINE_MULTIGPU_H
#define SMP_ENGINE_MULTIGPU_H
#include <utils/array_view.h>
#include <utils/stream.h>

#include "smp.h"
#include "smp_engine_abstract_old.h"
#include "atomic_tflag.h"

namespace bamboosmp {
  class SmpEngineMultiGpu : public OldAbsSmpEngine {
  public:
    SmpEngineMultiGpu(SmpObj &smp, int size, int num_gpus);

    ~SmpEngineMultiGpu();

    auto GetStableMarriage() const -> std::vector<int>;

    /* For Tests */
    inline void RunInitProcOnlyForTest() {
      this->InitProc();
    }

    auto RetrievePRMatrixForTest() -> Vector<PRNode>;

    auto RetrieveRankMatrixForTest() -> Vector<int>;

  protected:
    inline auto IsPerfect() const -> bool override {
      return is_perfect_;
    }

    void InitProc() override;

    void CoreProc() override;

    void PostProc() override;

  private:
    const SmpObj &smp_;
    int n_;
    int num_gpus_;

    bool is_perfect_;
    HVector<int> stable_matching_;

    Vector<CudaStream> gpu_streams_;
    Vector<DVector<int> > multigpu_device_pref_lists_w_;
    Vector<DVector<int> > multigpu_device_pref_lists_m_;
    Vector<DVector<int> > multigpu_device_rank_mtx_w_;
    Vector<DVector<PRNode> > multigpu_device_prnodes_m_;

    SPtr<AtomicTFlag> tflag_;
    Vector<DVector<int> > multigpu_device_next_proposed_w_;
    Vector<DVector<int> > multigpu_device_partner_rank_;

    Vector<CudaStream> monitor_streams_;
    HVector<int> final_partner_ranks_;
    // HVector<PRNode> host_prnodes_m_;
    // HVector<PRNode> host_prnodes_m_;

    HVector<PRNode> host_prnodes_m_;
    PRNode *host_prnodes_m_ptr_;
    HVector<int> host_next_proposed_w_;

    // helper methods for InitProc
    void EnablePeerAccess();

    void H2DAsync();

    void InitMatriceAsync();

    // helper methods for Gpu-side Core Computation
    void DoWorkOnGpu(SPtr<AtomicTFlag> tflag);

    // helper methods for Cpu-side Core Computation
    void DoWorkOnCpu(SPtr<AtomicTFlag> tflag);

    void MonitorProceture(int &unmatched_id,
                          int &unmatched_num,
                          HVector<int> &host_partner_rank,
                          const Vector<CudaStream> &monitor_streams);

    void PartnerRanksD2H(HVector<int> &host_partner_rank,
                         const Vector<CudaStream> &monitor_streams);

    void PRNodesD2H(PRNode *host_prnodes_m_ptr,
                    const Vector<CudaStream> &monitor_streams);

    void LAProcedure(int m,
                     ArrayView<PRNode> host_prnode_m_view,
                     ArrayView<int> host_partner_ranks_view,
                     ArrayView<int> host_next_proposed_w_view,
                     ArrayView<int> pref_lists_w_view);


    // Destructor
    void DisablePeerAccess();
  };
}

#endif //SMP_ENGINE_MULTIGPU_H
