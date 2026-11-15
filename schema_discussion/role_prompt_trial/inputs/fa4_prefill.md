# 给定 Contract：fa4_prefill

本实验只将以下内容作为已给定的任务事实。它复制自 `schema_template/examples/fa4_prefill/root.yaml` 的 Contract 区域，保留原外部要求；来源文件的 Design 与 dispatch 不作为本次设计答案。

原材料的排版或历史字段写法不作为当前语法模板，书写以正式 schema_docs 为准。硬件资料占位或其他不足可以照实记为输入问题，不伪造资料，也不要求本实验补齐全部硬件调查。继续能独立完成的设计判断；不能推进的部分保留实际疑问。

```yaml
contract:
  algorithm_requirement:
    constant:
      Hq: {value: 16, meaning: Query head count.}
      Hkv: {value: 1, meaning: KV head count.}
      D: {value: 256, meaning: Query and value channel count.}
      Page: {value: 64, meaning: Tokens per physical KV page.}
    input:
      query: {meaning: Concatenated request queries., shape: [{token: TotalQ}, {head: Hq}, {channel: D}], dtype: fp8_e4m3fn}
      kv_cache: {meaning: HND cache with keys then values on the kind axis., shape: [{page: Pages}, {kind: 2}, {head: Hkv}, {token: Page}, {channel: D}], dtype: fp8_e4m3fn}
      workspace_buffer: {meaning: Caller scratch zeroed before first isolated use., shape: [{byte: WorkspaceBytes}], dtype: uint8}
      block_tables: {meaning: Logical page to physical page mapping., shape: [{request: BatchCount}, {page_slot: TableColumns}], dtype: int32, range: 'Used entries in [0,Pages); unused entries are -1.'}
      seq_lens: {meaning: Used KV lengths., shape: [{request: BatchCount}], dtype: int32, range: Positive.}
      max_q_len: {meaning: Maximum request query length., shape: [], dtype: python_int}
      max_kv_len: {meaning: Maximum used KV length., shape: [], dtype: python_int}
      bmm1_scale: {meaning: Composite QK scale including quantization scales., shape: [], dtype: fp64, range: [0.0625, 0.0625]}
      bmm2_scale: {meaning: Composite output scale., shape: [], dtype: fp64, range: [1.0, 1.0]}
      batch_size: {meaning: ABI request count., shape: [], dtype: python_int}
      cum_seq_lens_q: {meaning: Query prefix boundaries., shape: [{boundary: 'BatchCount+1'}], dtype: int32}
      cum_seq_lens_kv: {meaning: Prefix counts of allocated pages rather than tokens., shape: [{boundary: 'BatchCount+1'}], dtype: int32}
      window_left: {meaning: Unrestricted causal history sentinel., shape: [], dtype: python_int, range: [-1, -1]}
      sinks: {meaning: Required disabled sink argument., shape: [], dtype: python_none, range: Python None.}
      o_sf_scale: {meaning: Required disabled output quantization argument., shape: [], dtype: python_none, range: Python None.}
      out: {meaning: Caller destination whose initial contents are ignored., shape: [{token: TotalQ}, {head: Hq}, {channel: D}], dtype: bf16, range: 'Any BF16 bit pattern, including inf and NaN.'}
    output:
      output: {meaning: Returned alias of the updated input out., shape: [{token: TotalQ}, {head: Hq}, {channel: D}], dtype: bf16}
      out_after: {meaning: In-place contents of input out., shape: [{token: TotalQ}, {head: Hq}, {channel: D}], dtype: bf16}
    compute:
    - name: Interpret the callable and pages
      math: |
        python_int is a Python integer; fp64 scalars are Python floats; python_none is the singleton Python None ABI type, not a tensor.
        Tensors are contiguous in the stated axis order on one CUDA device. batch_size=BatchCount.
        cum_seq_lens_q[0]=cum_seq_lens_kv[0]=0; cum_seq_lens_q[BatchCount]=TotalQ.
        Qlen(b)=cum_seq_lens_q[b+1]-cum_seq_lens_q[b]; L(b)=seq_lens[b]; 0<Qlen(b)<=L(b).
        max_q_len=max_b Qlen(b); max_kv_len=max_b L(b); Pages=sum_b ceil(L(b)/Page).
        cum_seq_lens_kv[b+1]-cum_seq_lens_kv[b]=ceil(L(b)/Page)<=TableColumns.
        For 0<=j<L(b), Kref[b,j,d]=kv_cache[block_tables[b,floor(j/Page)],0,0,j%Page,d], with Vref using kind=1.
        The complete standalone callable is timed. Only out is an observable input mutation; workspace_buffer is scratch.
      symbol:
        Qlen: {meaning: Query length function of request b., shape: not_applicable, dtype: not_applicable}
        L: {meaning: Used KV length function of request b., shape: not_applicable, dtype: not_applicable}
        Kref: {meaning: Logical keys on each request's valid domain., shape: [{request: BatchCount}, {key: max_kv_len}, {channel: D}], dtype: fp8_e4m3fn}
        Vref: {meaning: Logical values on each request's valid domain., shape: 'ref:Kref.shape', dtype: fp8_e4m3fn}
    - name: Right-aligned causal attention
      math: |
        For each b, 0<=q<Qlen(b), 0<=h<Hq and 0<=d<D, let a=cum_seq_lens_q[b]+q.
        J(b,q)={j in integers | 0<=j<L(b) and j<=L(b)-Qlen(b)+q}.
        score[j]=bmm1_scale*sum_{d=0}^{D-1} query[a,h,d]*Kref[b,j,d] for j in J(b,q).
        p[j]=exp(score[j]-max_k score[k])/sum_{k in J(b,q)} exp(score[k]-max_u score[u]), with all maxima over J(b,q).
        out_after[a,h,d]=bf16(bmm2_scale*sum_{j in J(b,q)} p[j]*Vref[b,j,d]); output is the same tensor object as out.
      symbol:
        J: {meaning: Visible key set function of request and query., shape: not_applicable, dtype: not_applicable}
    golden_reference:
      source: bcacdwk/KernelSchema@01fff90d714abc6e84904ea9c44523ba66d34a0e:kernel_wiki/aka-four-problems-20261019/fa4_prefill/reference.py::Model.forward
      tolerance: {output: 'atol=0.06, rtol=0.04', out_after: 'atol=0.06, rtol=0.04'}
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
      language: CuTe DSL, nvidia-cutlass-dsl==4.6.1
      toolchain: nvidia-cutlass-dsl 4.6.1 bundled compiler; torch 2.9.0+cu130; cuda-python 13.1.1; CUDA compiler and driver versions were not recorded.
    wiki:
      hardware: "<wiki-hardware@pending:L20D/SM103.md>"
      programming: {language: '<wiki-language@pending:CuTe-DSL/4.6.1/>', virtual_isa: '<wiki-virtual-isa@pending:PTX/pending/>', machine_isa: '<wiki-machine-isa@pending:SASS/SM103/>'}
      microbench: "<wiki-microbench@pending:L20D/>"
      sota: {kernel: '<wiki-sota-kernel@pending:L20D/>', schema: '<wiki-sota-schema@pending:L20D/>'}
      profile: "<wiki-profile@pending:NVIDIA/>"
```

