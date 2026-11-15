# 给定 Contract：qsa_prefill

本实验只将以下内容作为已给定的任务事实。它复制自 `schema_template/examples/qsa_prefill/root.yaml` 的 Contract 区域，保留原外部要求；来源文件的 Design 与 dispatch 不作为本次设计答案。

原材料的排版或历史字段写法不作为当前语法模板，书写以正式 schema_docs 为准。硬件资料占位或其他不足可以照实记为输入问题，不伪造资料，也不要求本实验补齐全部硬件调查。继续能独立完成的设计判断；不能推进的部分保留实际疑问。

```yaml
contract:
  algorithm_requirement:
    constant:
      Rows: {value: 4096, meaning: CP-local query rows.}
      QueryHeads: {value: 24, meaning: Query head count.}
      KVHeads: {value: 2, meaning: KV head count.}
      D: {value: 256, meaning: Query and value dimension.}
      PageSize: {value: 128, meaning: Tokens per physical cache page.}
      CachePages: {value: 2626, meaning: Physical cache page count.}
      TableWidth: {value: 2054, meaning: Logical page-table width.}
      SelectedWidth: {value: 2051, meaning: Selection slots before the valid-count word.}
      GroupSize: {value: "QueryHeads/KVHeads", meaning: Query heads sharing one KV head.}
      scale: {value: "1/sqrt(D)", meaning: Attention score scale.}
    input:
      q: {meaning: Query rows., shape: [row: Rows, head: QueryHeads, d: D], dtype: bf16}
      k_cache: {meaning: Paged keys., shape: [page: CachePages, page_token: PageSize, kv_head: KVHeads, d: D], dtype: bf16, range: "Mapped pages are finite; other storage may contain arbitrary BF16 bits, including NaN and infinity."}
      v_cache: {meaning: Paged values sharing the interleaved K allocation and strides., shape: [page: CachePages, page_token: PageSize, kv_head: KVHeads, d: D], dtype: bf16, range: "Mapped pages are finite; other storage may contain arbitrary BF16 bits, including NaN and infinity."}
      logical_indices:
        meaning: Selected logical token IDs followed by the valid count.
        shape: [row: Rows, packed: "SelectedWidth+1"]
        dtype: int32
        range: |
          One invocation-wide CP rank rho is in {0,1}; c_r=floor((15*8192+rho*Rows+r+1)/4), t_r=(r+1)%4.
          The first 2048 entries are 512 quads 4*g+[0,1,2,3] with distinct 0<=g<c_r.
          Entries 2048+l equal 4*c_r+l for 0<=l<t_r; remaining selection slots are -1; the last word is 2048+t_r.
      block_table: {meaning: Request-to-physical-page lookup., shape: [request: 1, logical_page: TableWidth], dtype: int32, range: "block_table[0,p]=1352+p for 0<=p<1024; -1 otherwise."}
      token_to_req: {meaning: Request owning each row., shape: [row: Rows], dtype: int32, range: [0, 0]}
      use_prefill_config: {meaning: Ignored configuration hint., shape: [], dtype: bool}
      output_gate: {meaning: Elementwise sigmoid logits., shape: [row: Rows, head: QueryHeads, d: D], dtype: bf16}
      out: {meaning: Optional destination whose initial contents are not read., shape: [row: Rows, head: QueryHeads, d: D], dtype: bf16, range: "Arbitrary BF16 bits, including NaN and infinity.", default: "allocate empty_like(q)"}
    output:
      O: {meaning: Result written into out and returned as that same tensor., shape: [row: Rows, head: QueryHeads, d: D], dtype: bf16}
    compute:
      - name: Resolve the supplied selection and apply gated attention
        math: |
          K/V share storage with element strides [2157376,512,256,1]; v_cache starts 65536 elements after k_cache.
          For each independent row r and head h, v=floor(h/GroupSize), J(r)={0,...,logical_indices[r,SelectedWidth]-1}.
          For j in J(r), k=logical_indices[r,j], page=block_table[0,floor(k/PageSize)], offset=k%PageSize.
          K_j=k_cache[page,offset,v,:], V_j=v_cache[page,offset,v,:]; score[j]=scale*sum_{d=0}^{D-1} q[r,h,d]*K_j[d].
          a=max_{j in J(r)} score[j]; z=sum_{j in J(r)} exp(score[j]-a).
          A[r,h,d]=bf16(sum_{j in J(r)} (exp(score[j]-a)/z)*V_j[d]).
          O[r,h,d]=bf16(A[r,h,d]/(1+exp(-output_gate[r,h,d]))).
          The supplied selection encodes causality; add no further mask. Write every out element and leave other inputs unchanged.
          The timed callable includes selection preprocessing, attention, gating and output allocation when needed; input construction is excluded.
        symbol:
          J: {meaning: Valid selection-slot indices of a row., shape: not_applicable, dtype: not_applicable}
          A: {meaning: Attention result rounded before gating., shape: "ref:O.shape", dtype: bf16}
    golden_reference:
      source: bcacdwk/KernelSchema@01fff90d714abc6e84904ea9c44523ba66d34a0e:kernel_wiki/aka-four-problems-20261019/qsa_prefill/reference.py::Model.forward
      tolerance: {O: "atol=3.2e-5, rtol=0.02; check both the returned tensor and mutated out."}
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
      language: CuTe DSL 4.6.1 (nvidia-cutlass-dsl).
      toolchain: "CuTe DSL 4.6.1 bundled compiler; CUDA 13.0; cuda-python 13.1.1; torch 2.9.0+cu130."
    wiki:
      hardware: "<wiki-hardware@pending:L20D/SM103.md>"
      programming: {language: "<wiki-language@pending:CuTe-DSL/4.6.1/>", virtual_isa: "<wiki-virtual-isa@pending:PTX/pending/>", machine_isa: "<wiki-machine-isa@pending:SASS/SM103/>"}
      microbench: "<wiki-microbench@pending:L20D/>"
      sota: {kernel: "<wiki-sota-kernel@pending:L20D/>", schema: "<wiki-sota-schema@pending:L20D/>"}
      profile: "<wiki-profile@pending:NVIDIA/>"
```

