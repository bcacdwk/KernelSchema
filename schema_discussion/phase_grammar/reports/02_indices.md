# 共享索引：保留能力，收窄登记义务

**建议保留共享索引，名称优先用 `index`。本次没有找到必须新增任意值域、列表域、stride 或 mapping 字段的反例。** 真正值得修订的是填写边界：单角色遍历 `worktile` 也应允许命名域；局部 `for` 的变量传给 `view` 形参，不应仅因此被强制再次登记为公共域。后一点是对现行填写条件的建议澄清，不声称严格按现有文字已经可以省略所有这些声明。

六份根稿全文已读，另定向检查九份分支及七个固定源码文件的相关片段。172 份有设计材料的文件只做过域形式定位扫描，不能据此声称审阅了整树；五份无创作 scaffold、43 个未开始节点不作为设计样本。FMHA、MoE 使用 corpus 指定的当前 working 根稿。旧 `repeat/prologue/steady/epilogue` 和旧动词只按其关系理解，不按字段改名判成表达能力不足。

| 算子 | 根稿域条目 | 主要索引关系 | 分支压力点与判断 |
|---|---:|---|---|
| [GDN root](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gdn/root.yaml:443) | 3 | prep 的条件 `a`、头组 `r`；chain 的 chunk `c` | segmented 有降序 SegB 合成和条件重跑；合成已由数学定义，`Run` 用 `if` 足够 |
| [GEMM root](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gemm/root.yaml:266) | 5 | 工作 `t`、pair `q`、连续 `n`、预取 `p`、片段 `d` | wide_recovery_full 的四次提交共用 YAML alias，但索引和资源实例仍按 phase 分开 |
| [MLA root](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_mla/root.yaml:364) | 3 | 动态工作 `w`、依工作变化的 `t`、连续 `n` | bootstrap 只需 `t`；root 的 terminal Next 仍是事件，不是额外 worktile |
| [Sparse root](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_sparse/root.yaml:382) | 1 | 分片 chunk `c`，实际 ID 由数学 gather 映射给出 | tc_packed 的运行时 union 保留重复项；序号域已经足够 |
| [FMHA root](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_fmha/root.yaml:462) | 2 | key group `g`、局部两半 `z` | 奇偶 warpgroup、局部 prefix 和各行选择性 suffix 均可表达；`z` 可局部化 |
| [MoE root](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/root.yaml:617) | 9 | gather 的工作 `t`；FC1/FC2 各自的 `t/i/g/c` | 单角色 FC2 仍可保留连续槽编号；内部片段循环和空 copy-group 尾部须区分 |

根稿合计 23 个域条目，都是本地 `tile_pipeline`，所以这里的 raw/effective 字段出现量同为 23；不是运行迭代数，也不是独立数学变量总数。分支 alias 不算作者重复输入：例如 GEMM 四个 phase 共享 `wide_pipe`，不能把四份展开后的域表当四次手写。公共计数见 [census](../data/census.md)，本报告的路径、状态、52 个原文证据片段及源码 SHA256 见 [02_indices.json](../data/02_indices.json)。

## 七个关键判断

**I01 — 连续版本编号有实质用途，不能只留下工作分配或槽位。**（`existing_structure`，涉及 resource）

GEMM 的 producer、issuer、epilogue 独立推进，三槽操作数与两槽产品分别用 `n%3`、`n%2`；复用等待连到 `n+3`、`n+2`。其 [源码](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_gemm/fp8gg_umma.cu:423)把 `pj` 放在 work 循环之外，并在每个 pair 后增加。MLA 更不能用一个固定乘数代替：[`t: [0,N(w)]` 与 flatten](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_mla/root.yaml:364)处理每份工作不同的 key 数，[源码](../sources/kernel/extract_mla/solution.py:859)也分别维护 CTA 本地 item 序号和跨 item 的 tile 计数。

三个层次分工清楚：`work_schedule` 给出本 worker 得到哪些工作；公共索引给出这些工作及内部轮次的共同身份；`view.slot_index` 决定某个版本放在哪个物理槽。相同槽可以先后放很多版本，相同工作内也可有不同动作进度。仅把它们都写成 `t` 或 `%depth` 会丢失接续关系。保留现有 `worktile + 整数域 + flatten`；不要对动态 `worktile` 预先取总长，也不要让域声明隐含领取或重置。

**I02 — 稀疏列表的值和发生序号应分开；无需任意集合域。**（`other_layer`，涉及 resource）

Sparse root 明确要求[重复 ID 保留重数](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_sparse/root.yaml:106)。tc_packed 的[数学定义](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_sparse/tc_packed.yaml:43)把 pool/id、各 query 的重数和运行时排列放在 `UnionKey/UnionMask`，而流水线只遍历 `c: [0,C(grp)]`。源码确实给[同一 query 的重复 key 另开条目](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_sparse/dsv4_kernel.cu:2364)，再用 [upos](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_sparse/dsv4_kernel.cu:2453)把逻辑序号映射到两段物理存储。

现有表达的尝试已经成立：`k=c*KC+u` 是列表位置，`UnionKey[k]` 才是实际 pool/id。两个条目即使读同一物理行，也可有不同的 `k`；若反而直接拿 ID 当动作身份，还可能合并本应保留的重复发生。MoE 的 [`rho/RoutePos/TileExpert`](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/root.yaml:224)也是同类分层。数据映射保留在数学定义，工作集合保留在调度，`describe` 简短绑定；没有必要再建一套 `domain.values/filter/map`。

**I03 — 局部变量与旧式条件域可以直接减掉。**（`existing_structure`）

FMHA root 的 `z` 只用于单个 Main 的两半循环，改成 `for(z=0:2)` 即可；两个动作仍带 `(g,z)`，不是合并成同一个事件。[源码两半循环](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_fmha/source/candidate/fmha_tcg.cuh:4698)也只在该协作角色内部，并在覆盖相同 ScoreHalf 区域前结束前半读取。GDN prep 的 [`r`](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gdn/root.yaml:533)可写局部范围，`a:[0,Active]` 改成 `if(Active==1)`；只出现一次的保存/排空动作随之去掉假参数 `a`。

MoE root 的两份 `c:[0,2]` 未被作为循环使用，片段动作已经[显式写 `(t,0)` 与 `(t,1)`](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/root.yaml:821)。保留实参和四个 epilogue role，无需为这两个常量单独开域。这些是具体冗余，不支持删除共享索引概念。

删除条件伪索引不等于删除无 payload 事件。MoE 单角色 FC2 的 [`CopyGroup(t,s)`](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/fused_big_mono.yaml:382)在 `s>=I/BK` 仍提交空组；其事件序列维持 wait-group 的意义。有效 operand 轮次和额外协议发生次数可以不同，不能把二者强压成同一个域。

**I04 — 奇偶分工和独立推进不要求增加 stride 或 role scope。**（`existing_structure`，涉及 resource）

FMHA [fused_d128_group](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_fmha/fused_d128_group.yaml:325)明确写出 `g=cLo+w+2*j`：Even 的 `j` 长度为 `ceil_div(Ng,2)`，Odd 为 `floor(Ng/2)`；跨角色最大值、分母和 PV 分别有边。[源码](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_fmha/source/candidate/fmha_tcg.cuh:3656)按 `grp+=2` 推进，[最大值交接](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_fmha/source/candidate/fmha_tcg.cuh:3733)发生在两条独立 QK 进度之间。

这句局部 `j→g` 映射本身已经是合理、简短的语义。若希望直接使用公共 `g`，也能在两角色的 `for(g)` 下分别加奇偶 `if`，把前驱统一写成另一角色的 `g-1`。不产生锁步，`Ng=1` 时 Odd 仍执行其准备/收尾，只是不产生组内动作。下面草图只是展示已有表达力，不建议为统一索引名批量重写完整分支。

**I05 — 物理别名、YAML alias 与真实事件身份不能互相替代。**（`existing_structure`，涉及 resource、milestone）

MLA 的 [`Item(w)` 与 `Next(w)`](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_mla/root.yaml:457)分别映射 `w%2`、`(w+1)%2`。因此写 `Next(w+1)` 会覆盖 Item(w) 所在的槽，必须保留[四角色都已 latch Item(w) 的条件](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_mla/root.yaml:626)。`Next(w)` 的参数表示当前工作发起的下一次发布；成功时它承载下一份工作，失败时仍发布 terminal 并完成。源码也确实[先发布 terminal、arrive，后退出](../sources/kernel/extract_mla/solution.py:1449)。不能把这次发布改名为“下一工作存在”或依靠槽号隐去轮次。

MoE 的同一个 `Accumulator(t)` 有[四个独立的 quadrant release](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/root.yaml:814)，动作身份还包括 role，复用方必须等待所有真实读者。GEMM 的四份 [`wide_pipe` alias](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gemm/wide_recovery_full.yaml:454)则按 phase 建立新实例。现行身份规则已经覆盖这些关系；增加“通用变量域”既无帮助，也不能修复源码缺失的读完保证。

**I06 — 应补清“什么时候登记”，尤其是 view 形参与单角色 worktile。**（`underspecified_rule`，涉及 resource）

现行[填写条件](../sources/rules/5.design.tile_pipeline.md:70)主要列“多个 role 共用，或资源视图用到它”。但 MoE [gather](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/root.yaml:617)只有一个 role、没有登记 resource，却需要 `t: worktile` 遍历静态步长分配的工作；[源码](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_nvfp4_moe.cu:1189)确实是一条跨 packet 的工作循环。这应作为明确允许条件，不要求虚构第二个 role 或把工作序列重复展开成整数公式。

反方向，[view 规则](../sources/rules/5.design.tile_pipeline.md:362)又明确形参不建立循环、不捕获 role 当前变量。GEMM 的 `Fragment(n,d): slot_index: d%2` 中，形式参数 `d` 可以改叫 `x`，意义不变；不能因此就要求再声明一个公共 `x` 域。它接收本次动作传入的实际版本，局部 `for(d=0:...)` 可以提供这个版本。当前“视图用到就登记”的强读法会把这种函数形参也当成公共循环变量，产生不必要的重复登记。

建议把规则说成三点：公共索引统一**需要共同引用的轮次身份或边界**，也可以登记单角色所需的 `worktile/flatten`；**局部变量作为 view 实参不单独触发公共域登记**，形参只参与槽/区域替换；所有动态端点必须在使用处已知并就绪。Sparse 的 `C(grp)` 由 union build 的[发布动作](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_sparse/tc_packed.yaml:377)产生，不能因为域表写在前面就提前读取。必要时先在 role 中等待该发布，再进入 `if/for`。这些都使用现有动作与依赖，不需要新值域形式。

`N/Hoist/GroupWidth/Full(t)` 等普通参数或判定值仍需有定义及读取保证，但不因此成为 domain。共享索引也不必是每个 role 都直接绑定的同名 `for` 变量：FMHA 的公共 `g` 可以通过各角色的局部 `j` 映射得到。

**I07 — 部分长说明是可压短的算法语义，不是藏了共享索引缺口。**（`concise_semantics`，涉及 resource、milestone）

GDN segmented 的 [`seed`](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gdn/segmented.yaml:626)调用上游已定义的降序合成；[数学定义](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gdn/segmented.yaml:131)与[源码](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_gdn/gdn_kernel.cu:2265)都保留先 `z-1..1`、后 `0` 的数值顺序。它没有每个 segment 的独立发布或可复用边界，一句“Compose SeedC in the order defined by segments, then publish Rh”就能承接；无需给每个 scalar loop 建公共索引。若以后确有逐步事件需求，可以用升序发生序号和 `z-1-j` 访问映射表达降序。

FMHA fscan 的 [`prefix`](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_fmha/tcg_d128_fscan_cp_direct_narrow.yaml:546)保留四件关键事即可：先等 GroupMax；按上游顺序扫 bounded prefix；接续 MaxState；这段工作可与已发起的 K/V 传输重叠。[源码](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_fmha/source/candidate/fmha_tcg.cuh:4398)支持这一分层，16 项寄存器批次的容量由 PrefixBatch 记录，不必把 16 次标量加载逐项搬进 pipe。

review_candidate 的 segmented scan 也明确[各行独立选择后缀](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_fmha/tcg_d128_segmented_tma_xpose_narrow.yaml:566)，[源码](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_fmha/source/candidate/fmha_tc.cuh:1489)是已知范围内的 `for/if` 和数学 recurrence。这不是所有 role 共同遍历一个筛选后集合；不应强制做成公共域，也不能据此要求 pipe 支持源码中的所有控制结构。

与此同时，必须保留已命名动作之间不可推导的等待。MLA 的 descriptor 复用和 MoE 的条件 Drain 是结构关系，不能只留在长 `describe`。GDN 的 mini-chunk race、FMHA 候选的[空 split staging race](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_fmha/tcg_d128_segmented_tma_xpose_narrow.yaml:692)是另外的来源/草稿问题；新增域、换名字或扩大动作都不会生成缺失的保证。

## 三个最小草图

以下均为建议片段，不是完整 schema，也没有进行整个动作迁移的合法性认证。保持现行字段名，便于单独评估表达；名称建议见后表。完整片段及省略条件同时保存在 JSON。

**1. FMHA root：共享 `g`，局部 `z`。** 原有准备、K/scale 就绪、其余每组动作及收尾继续保留；这里只展示两半子图。新动作的 done 使用现行完成含义。

```yaml
domain:
  g: [cLo, "cHi+1"]
role:
  MAIN:
    info:
      member: "warp[0:4]"
      using_verb:
        matmul:
          ScoreHalf:
            execution_mode: async_unordered
            describe: "Compute half z of group g into TensorMemory.ScoreHalf from Query, Key(g) and their declared scales; done makes the half readable."
        load:
          Score:
            execution_mode: serial
            describe: "Read the completed half into the corresponding Score register region; all half readers finish before the shared ScoreHalf region is reused."
    pipe:
      - "for(g)":
          - "for(z=0:2)":
              - "matmul:ScoreHalf(g,z)"
              - "load:Score(g,z)"
    waiting:
      prepared:
        - "{issue load:Score(g,z)} ← [{done matmul:ScoreHalf(g,z)}]"
```

**2. FMHA 奇偶组：同一 `g`，各 role 自己选择轮次。** 只展示最大值交接；QK readout 已先产生该角色的 GroupMax，其余概率、分母、PV、初始化与收尾仍在原位置。`done ← done` 本身不保护提前读取，`describe` 中真实的逐行 acquire 仍需保留。

```yaml
domain:
  g: [cLo, "cHi+1"]
role:
  Even:
    info:
      member: "warp[0:4]"
      using_verb:
        compute:
          MaxState: &max_step
            execution_mode: async_unordered
            describe: "Each row waits for the preceding group's MaxTransfer before reading it, except at cLo where it uses the initial state; apply normalize and publish MaxState(g) to Exchange.MaxTransfer(g). Done covers all row publications without a producer-side join."
    pipe:
      - "for(g)":
          - "if((g-cLo)%2 == 0)":
              - "compute:MaxState(g)"
    waiting:
      prepared:
        - "{done compute:MaxState(g)} ← [{done Odd::compute:MaxState(g-1)}]"
  Odd:
    info:
      member: "warp[4:8]"
      using_verb:
        compute:
          MaxState: *max_step
    pipe:
      - "for(g)":
          - "if((g-cLo)%2 == 1)":
              - "compute:MaxState(g)"
    waiting:
      prepared:
        - "{done compute:MaxState(g)} ← [{done Even::compute:MaxState(g-1)}]"
```

首项的 `g-1` 在声明窗口之外，不产生前驱；其后所有实际目标都有另一角色的前驱。这里没有要求两角色同时绑定同一个 `g`。完整改写若采用这种编号，需要一致迁移所有动作实参与复用边，不能只改循环头。

**3. MLA：保留 flatten 和 descriptor 的真实代次。** 其他角色的 request/latch/等待继续保留，分别迁移为 `signal:Next`、`load:Item` 等现行动作。这个片段不使用动态工作的 `count/last`。

```yaml
domain:
  w: worktile
  t: [0, "N(w)"]
  n: {flatten: [w, t]}
resource:
  Geometry:
    depth: 2
    location: SMEM
    shape: [field: 10]
    dtype: int32
    total_size: 80
    view:
      "Item(w)": {slot_index: "w%2"}
      "Next(w)": {slot_index: "(w+1)%2"}
role:
  Queue:
    info:
      member: "warp[10:11]"
      using_verb:
        compute:
          Next:
            execution_mode: serial
            describe: "After Softmax's request, claim work by atomic_queue and release-publish its geometry or the terminal descriptor to Geometry.Next(w); the terminal publication has a done event and creates no next worktile."
    pipe:
      - "for(w)":
          - "compute:Next(w)"
    waiting:
      allowed:
        - "{issue compute:Next(w)} ← [{done Softmax::signal:Next(w)}]"
      released:
        - "{issue compute:Next(w+1)} ← [{done Softmax::load:Item(w)}, {done Correction::load:Item(w)}, {done MMA::load:Item(w)}, {done Loader::load:Item(w)}]"
```

80 字节仅为 2 槽 × 10 个 int32 × 1 个 CTA 的 Geometry 账。`Next(w)` 与 `Item(w+1)` 可在成功领取时描述同一份几何数据，但两次不同动作的身份、完成事件以及失败终止的情形仍然分开。

## 名称与职责

| 现名 | 建议 | 理由与其他候选 |
|---|---|---|
| domain | **index** | 直接指出所登记的是有序逻辑索引及边界。`range` 只像区间，覆盖不了 worktile 序列及 flatten 的身份；`iteration` 像执行步骤，又与 work_schedule.iteration 重叠；`domain` 数学上成立，但容易被理解成任意集合或整个契约域。`shared_index` 更明确但较长，标题可补“共享” |
| resource | **storage**，可独立选择是否改 | 覆盖片上容量、区域、缓冲与原地寄存器状态。`buffer` 很惯用，适合环与 staging，但对 R/MaxState 等原地状态略窄；`allocation` 容易被误读为分配动作或分配器策略；`resource` 容易让人期待引擎、barrier 等任意资源 |
| milestone | **保留 milestone** | 对已有事件组的对外成立点是惯用含义。`event_group` 更字面但更长；`event` 容易和基础 issue/done、CUDA event 对象混淆；`checkpoint` 暗示保存/恢复状态；`sync` 暗示声明会产生同步 |

三个概念各管一件事：storage 是 worker 内的片上存储账，index 是需要公共解释的逻辑编号，milestone 是跨 phase 使用的已有事件组别名。它们不应兼任通用变量表、数学数据视图定义、工作分配器或同步原语。

## 定向分支与核验范围

下列分支均有实际设计材料；范围为相关片段的语义核对，不是整分支重新批准。精确读取行段在 JSON。

| 分支 | corpus 状态 | 本次关注 |
|---|---|---|
| [GDN segmented](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gdn/segmented.yaml:119) | approved | segment 局部坐标、降序合成、Run/Count |
| [GEMM wide_recovery_full](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gemm/wide_recovery_full.yaml:150) | approved | 四次提交、alias、独立索引和资源实例 |
| [MLA bootstrap](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_mla/bootstrap.yaml:81) | approved | 去除动态续领后仅剩局部 key 序列 |
| [Sparse tc_packed](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_sparse/tc_packed.yaml:27) | approved | 全文；runtime union、重复项、分角色进度 |
| [FMHA fused_d128_group](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_fmha/fused_d128_group.yaml:236) | approved | 奇偶组、stride、三类不同交接 |
| [FMHA fscan cp direct narrow](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_fmha/tcg_d128_fscan_cp_direct_narrow.yaml:546) | approved | 局部 prefix 与后续 attention 两段工作 |
| [FMHA segmented tma xpose narrow](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_fmha/tcg_d128_segmented_tma_xpose_narrow.yaml:548) | review_candidate | 各行选择后缀；空 split 来源限制单列 |
| [FMHA fscan cp direct wide](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_fmha/tcg_d128_fscan_cp_direct_wide.yaml:617) | needs_authoring | 四个 combine 组各自求和再交接 |
| [MoE fused_big_mono](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/fused_big_mono.yaml:330) | approved | 单角色、跨工作槽编号、局部片段、空提交尾部 |

复用了用户指定的 [SCHEMA_VERBOSITY](../../../schema_extract/archive/six-operators/20261109-methodology-review/SCHEMA_VERBOSITY.md) 和 [SCHEMA_DISCUSSION](../../../schema_extract/archive/20261109-ten-operator-text-audit/SCHEMA_DISCUSSION.md) 及旧审计定位材料；关键结论已回看 YAML 和源码，没有读取其他新代理报告。

[复核脚本](../scripts/02_indices.py)检查了六 root、九分支的固定哈希和 52 段证据原文，给源码证据附 SHA256，并解析了三个草图的 YAML。没有运行 GPU、继续提取、修改正式文档或验收这些不完整草图为完整 schema。“未发现缺口”只限上述材料与尝试，不是对所有合理 kernel 设计的完备性证明。
