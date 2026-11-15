# 给定 Contract：gdn_prefill

本实验只将以下内容作为已给定的任务事实。它复制自 `schema_template/examples/gdn_prefill/root.yaml` 的 Contract 区域，保留原外部要求；来源文件的 Design 与 dispatch 不作为本次设计答案。

原材料的排版或历史字段写法不作为当前语法模板，书写以正式 schema_docs 为准。硬件资料占位或其他不足可以照实记为输入问题，不伪造资料，也不要求本实验补齐全部硬件调查。继续能独立完成的设计判断；不能推进的部分保留实际疑问。

```yaml
contract:
  algorithm_requirement:
    constant:
      HQ: {value: 4, meaning: Query and key head count.}
      HV: {value: 32, meaning: Value and state head count.}
      DK: {value: 128, meaning: Key dimension.}
      DV: {value: 128, meaning: Value dimension.}
      epsilon: {value: 1.0e-10, meaning: Additive stabilizer for linear decay.}
    input:
      q: {meaning: Unnormalized queries., shape: [token: T, head: HQ, dk: DK], dtype: bf16}
      k: {meaning: Pre-normalized keys., shape: [token: T, head: HQ, dk: DK], dtype: bf16, range: "BF16 result of torch.nn.functional.normalize(p=2.0, dim=-1) on the BF16 generator output."}
      v: {meaning: Values., shape: [token: T, head: HV, dv: DV], dtype: bf16}
      alpha: {meaning: Linear decay factors., shape: [token: T, head: HV], dtype: fp32, range: "0 <= alpha < 1"}
      beta: {meaning: Delta-update strengths., shape: [token: T, head: HV], dtype: fp32, range: "0 <= beta < 1"}
      initial_state: {meaning: Explicit incoming value-by-key state., shape: [batch: B, head: HV, dv: DV, dk: DK], dtype: fp32, range: "All zeros or the final state of an independent 32-token reference prefix per sequence under the same input contract."}
      cu_seqlens: {meaning: Packed sequence boundaries., shape: [boundary: B+1], dtype: int64, range: "Starts at 0, strictly increases, ends at T; 1<=B<=16 and T<=16384."}
      scale: {meaning: Mandatory Python output multiplier., shape: [], dtype: fp64, range: "Python binary64 1.0/math.sqrt(DK), equal to 0.08838834764831843."}
      output: {meaning: Preallocated output buffer., shape: [token: T, head: HV, dv: DV], dtype: bf16, range: "Arbitrary contents including NaN; previous contents ignored."}
      output_state: {meaning: Preallocated final-state buffer., shape: [batch: B, head: HV, dv: DV, dk: DK], dtype: fp32, range: "Arbitrary contents including NaN; previous contents ignored."}
    output:
      O: {meaning: First returned tensor aliasing and fully overwriting output., shape: [token: T, head: HV, dv: DV], dtype: bf16}
      S_final: {meaning: Second returned tensor aliasing and fully overwriting output_state., shape: [batch: B, head: HV, dv: DV, dk: DK], dtype: fp32}
    compute:
      - name: Decay and update the state before inclusive readout
        math: |
          All tensors are contiguous CUDA tensors. Before key normalization, q/k/v rows have independent
          Normal(0,0.05) centers and Uniform(center-0.25,center+0.25) elements rounded to BF16.
          Independently for each 0<=b<B and 0<=h<HV, let hq=floor(h/(HV/HQ)),
          L_b=cu_seqlens[b+1]-cu_seqlens[b]>0 and State(-1)=initial_state[b,h].
          For 0<=t<L_b, x=cu_seqlens[b]+t, 0<=j<DV and 0<=d<DK:
            D=(alpha[x,h]+epsilon)*State(t-1).
            delta[j]=beta[x,h]*(v[x,h,j]-sum_d D[j,d]*k[x,hq,d]).
            State(t)[j,d]=D[j,d]+delta[j]*k[x,hq,d].
            O[x,h,j]=bf16(scale*sum_d State(t)[j,d]*q[x,hq,d]).
          S_final[b,h]=State(L_b-1); only output and output_state are mutated.
          Timing includes Model.forward and its int64-to-int32 boundary conversion; input generation,
          normalization, continuation-prefix evaluation, allocation and NaN fill are outside timing.
        symbol:
          State: {meaning: Updated reference state after local token t., shape: [batch: B, head: HV, dv: DV, dk: DK], dtype: fp32}
    golden_reference:
      source: bcacdwk/KernelSchema@01fff90d714abc6e84904ea9c44523ba66d34a0e:kernel_wiki/aka-four-problems-20261019/gdn_prefill/reference.py::Model.forward
      tolerance:
        O: "atol=0.01, rtol=0.01; every element of both the returned tensor and mutated_inputs.output."
        S_final: "atol=0.005, rtol=0.001; every element of both the returned tensor and mutated_inputs.output_state."
      gap: The oracle evaluates the recurrence with FP32 block_size=32 inversion and log-space decay; that blocked evaluation defines the floating-point comparison values.
  hardware_specification:
    device: NVIDIA L20D
    arch: SM103
    execution:
      warp: 32 SIMT threads within one CTA.
      CTA: Warps co-resident on one SM, sharing SMEM and barriers.
      cluster: CTAs co-scheduled within one GPC, with cluster synchronization.
    feature:
      SMEM: CTA-lifetime shared storage, addressable by cluster peers while its owner remains alive.
      TMEM: Explicitly allocated Tensor Core storage, separate from SMEM.
      register: Thread-private registers backed by the SM register file.
      asynchronous_execution: Asynchronous transfers and MMA require their primitive-specific completion and visibility guarantees.
    stack:
      language: CuTe DSL 4.6.1
      toolchain: "CuTe DSL 4.6.1 bundled compiler; torch 2.9.0+cu130; cuda-python 13.1.1. CUDA compiler and driver versions were not recorded separately."
    wiki:
      hardware: "<wiki-hardware@pending:L20D/SM103.md>"
      programming: {language: "<wiki-language@pending:CuTe-DSL/4.6.1/>", virtual_isa: "<wiki-virtual-isa@pending:PTX/version/>", machine_isa: "<wiki-machine-isa@pending:SASS/SM103/>"}
      microbench: "<wiki-microbench@pending:L20D/>"
      sota: {kernel: "<wiki-sota-kernel@pending:L20D/>", schema: "<wiki-sota-schema@pending:L20D/>"}
      profile: "<wiki-profile@pending:NVIDIA/>"
```

