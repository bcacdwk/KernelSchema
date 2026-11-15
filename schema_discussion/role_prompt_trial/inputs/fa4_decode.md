# 给定 Contract：fa4_decode

本实验只将以下内容作为已给定的任务事实。它复制自 `schema_template/examples/fa4_decode/root.yaml` 的 Contract 区域，保留原外部要求；来源文件的 Design 与 dispatch 不作为本次设计答案。

原材料的排版或历史字段写法不作为当前语法模板，书写以正式 schema_docs 为准。硬件资料占位或其他不足可以照实记为输入问题，不伪造资料，也不要求本实验补齐全部硬件调查。继续能独立完成的设计判断；不能推进的部分保留实际疑问。

```yaml
contract:
  algorithm_requirement:
    constant:
      HeadCount: {value: 16, meaning: Query head count.}
      KVHeadCount: {value: 1, meaning: KV head count.}
      D: {value: 256, meaning: 'Query, key and value dimension.'}
      PageSize: {value: 64, meaning: Tokens per cache page.}
      TableWidth: {value: 16434, meaning: Block-table entries per request.}
    input:
      query: {meaning: Query tail, shape: [{query: TotalQuery}, {head: HeadCount}, {d: D}], dtype: fp8_e4m3fn}
      kv_cache: {meaning: HND cache with key at selector 0 and value at selector 1, shape: [{page: PageCount}, {selector: 2}, {kv_head: KVHeadCount}, {page_token: PageSize}, {d: D}],
        dtype: fp8_e4m3fn}
      workspace_buffer: {meaning: Scratch zeroed before first isolated use; not a semantic output., shape: [{byte: WorkspaceBytes}], dtype: uint8}
      block_tables: {meaning: Logical-to-physical page map., shape: [{batch: BatchCount}, {page_index: TableWidth}], dtype: int32, range: -1 <= block_tables < PageCount; valid pages
          are nonnegative}
      seq_lens: {meaning: KV lengths including the query tails., shape: [{batch: BatchCount}], dtype: int32, range: q_len_per_req <= seq_lens <= PageSize*TableWidth}
      max_seq_len: {meaning: Maximum entry of seq_lens, shape: [], dtype: python_int}
      bmm1_scale: {meaning: Composite score scale including captured quantization scales., shape: [], dtype: fp64, range: [0.0625, 0.0625]}
      bmm2_scale: {meaning: Composite value-output scale, shape: [], dtype: fp64, range: [1.0, 1.0]}
      window_left: {meaning: Full-attention sentinel, shape: [], dtype: python_int, range: [-1, -1]}
      sinks: {meaning: Required None sentinel; sinks disabled., shape: [], dtype: python_none, range: Python None only}
      o_sf_scale: {meaning: Required None sentinel; output scaling tensor disabled., shape: [], dtype: python_none, range: Python None only}
      out: {meaning: Destination; incoming contents are ignored., shape: [{query: TotalQuery}, {head: HeadCount}, {d: D}], dtype: bf16, range: Any BF16 bit pattern; never read before
          overwrite.}
      q_len_per_req: {meaning: Number of query-tail tokens per request, shape: [], dtype: python_int, range: "q_len_per_req in {1, 4}"}
    output:
      output: {meaning: 'Returned tensor, aliasing out.', shape: [{query: TotalQuery}, {head: HeadCount}, {d: D}], dtype: bf16}
      out_after: {meaning: In-place replacement of input out; identical to output., shape: [{query: TotalQuery}, {head: HeadCount}, {d: D}], dtype: bf16}
    compute:
    - name: Interpret the captured callable boundary
      math: |
        python_int denotes a Python integer; fp64 scalar arguments are Python binary64 floats.
        python_none denotes the Python None singleton and carries no numeric payload.
        TotalQuery=BatchCount*q_len_per_req; max_seq_len=max_b seq_lens[b]; L_b=seq_lens[b].
        All tensors are contiguous on one CUDA device; sinks=None and o_sf_scale=None.
        For 0<=b<BatchCount, the first ceil(L_b/PageSize) block-table entries are valid pages.
        For 0<=k<L_b and 0<=d<D, let page=block_tables[b,floor(k/PageSize)]:
          K[b,k,d]=kv_cache[page,0,0,k%PageSize,d]; V[b,k,d]=kv_cache[page,1,0,k%PageSize,d].
        The timed boundary includes any partial-result combination; cache construction is outside it.
      symbol:
        K: {meaning: Logical keys after page lookup, shape: [{batch: BatchCount}, {kv: max_seq_len}, {d: D}], dtype: fp8_e4m3fn}
        V: {meaning: Logical values after page lookup, shape: 'ref:K.shape', dtype: fp8_e4m3fn}
    - name: Attend to the right-aligned query tail
      math: |
        For independent b,t,h with 0<=b<BatchCount, 0<=t<q_len_per_req, 0<=h<HeadCount:
          visible_key(b,t)={k in integers | 0<=k<=L_b-q_len_per_req+t}.
          score[b,t,h,k]=bmm1_scale*sum_{d=0}^{D-1} query[b*q_len_per_req+t,h,d]*K[b,k,d].
          m=max_{k in visible_key(b,t)} score[b,t,h,k]; Z=sum_{k in visible_key(b,t)} exp(score[b,t,h,k]-m).
          probability[b,t,h,k]=exp(score[b,t,h,k]-m)/Z on visible_key(b,t), and zero otherwise.
          output[b*q_len_per_req+t,h,d]=bf16(bmm2_scale*sum_{k in visible_key(b,t)} probability[b,t,h,k]*V[b,k,d]).
        All visible sets are nonempty. out_after=output; return the same tensor object as out; no LSE is returned.
      symbol:
        visible_key: {meaning: Valid causal keys for one tail token, shape: not_applicable, dtype: not_applicable}
        score: {meaning: Scaled query-key score on valid logical keys, shape: [{batch: BatchCount}, {query_tail: q_len_per_req}, {head: HeadCount}, {kv: max_seq_len}], dtype: fp32}
        probability: {meaning: Softmax probability on valid logical keys, shape: 'ref:score.shape', dtype: fp32}
    golden_reference:
      source: bcacdwk/KernelSchema@01fff90d714abc6e84904ea9c44523ba66d34a0e:kernel_wiki/aka-four-problems-20261019/fa4_decode/reference.py::Model.forward
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
    stack: {language: 'CuTe DSL, nvidia-cutlass-dsl 4.6.1', toolchain: 'nvidia-cutlass-dsl 4.6.1 compiler; PyTorch 2.9.0+cu130; cuda-python 13.1.1; CUDA compiler and driver versions were not recorded.'}
    wiki:
      hardware: <wiki-hardware@pending:L20D/SM103.md>
      programming: {language: '<wiki-language@pending:CuTe-DSL/4.6.1/>', virtual_isa: '<wiki-virtual-isa@pending:PTX/pending/>', machine_isa: '<wiki-machine-isa@pending:SASS/SM103/>'}
      microbench: <wiki-microbench@pending:L20D/>
      sota: {kernel: '<wiki-sota-kernel@pending:L20D/>', schema: '<wiki-sota-schema@pending:L20D/>'}
      profile: <wiki-profile@pending:NVIDIA/>
```

