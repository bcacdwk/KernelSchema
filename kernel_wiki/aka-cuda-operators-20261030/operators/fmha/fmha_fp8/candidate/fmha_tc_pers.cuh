// r19 PERSISTENT-FMTC body (included inside `if constexpr (PERS)` of
// fmha_tc_kernel; NOT a standalone translation unit). Port of the r18k MT
// persistent design — every mechanism there was measured bitwise-correct
// and hang-free across the fixed+domain+synthetic matrix (dv_dd1cb264f1d2,
// dv_65215c015cc2):
//   * scheduling (r19c): STATIC BALANCED PARTITION — the host partitions
//     all valid aids into per-worker contiguous segments (LPT upper bound,
//     then minimal-cap best-fit-decreasing; weight ntl+3 = boundary cost;
//     sim calibrated exact against dv_a7efac710e94 frontiers: c6 makespan
//     214→189, c0 26→19 = max-item = optimal). The loader walks its own
//     read-only segment (no atomics/counter → graph-replay safe by
//     construction); aid broadcast + cumulative consumer pacing (mma_qa_p
//     started-count; sm_read_ctr = 8 warp-reads per assignment; release
//     fences both directions) are the MT r18k contract. MT keeps the
//     dynamic ticket pool (its variance tax is per-MEGATILE and real —
//     static binding lost c1 0.911→1.005; fmtc tiles are cost-uniform, so
//     LLN makes static binding safe and its makespan beats greedy pull).
//   * ALL barrier slot/parity expressions ride the GLOBAL tile counter Ja
//     (worker-cumulative; identical in every role because tc_aid_geom is
//     pure): qk_done/score_released/probability_ready/output_rescaled are
//     NBUF-slotted by J%NBUF, k/v_loaded+tx NSTAGE-slotted by J%NSTAGE,
//     output_ready 2-slotted by J&1 (softmax corr consumer) — parities
//     (J/NBUF)&1 etc. exactly as the shipped per-CTA forms with j -> J.
//     r19fix: the LOADER's release/guard waits instead ride pv_done[]
//     (NSTAGE-slotted PV-retire commits on the PERS page): the shipped
//     2-slot forms are cadence-safe per-CTA but alias +1 phase at PERS
//     assignment boundaries (drain-tail overshoot; r19diag dv_6e58bb8fe9db
//     c0/c6 hangs) — a parity window that overshot by one can never reopen.
//   * q_loaded ALTERNATES between sh.q_loaded and the PERS-page qld_b
//     (slot qa&1, phase (qa>>1)&1) + arrive throttle (mma_qa_p >= qa-1):
//     single-slot parity can alias when two arrives land before a wait
//     (tiny-assignment chains) — structurally excluded by construction.
//   * Q double buffer (q2, PLANES==1 only; D256 has no smem headroom and
//     keeps ONE buffer, guarded instead): loader Q(qa) overwrite guard runs
//     from qa>=2 only (qa==1 writes the fresh q2 slot — MT r18k form) and
//     waits pv_done(Ja-NSTAGE) [overshoot-free: the slot's next completion
//     PV(Ja) is gated behind this loader's own q_loaded arrive; PV retire
//     implies every QK of assignment qa-2 retired] with a qk_done(Ja-1)
//     tiny-Ja fallback; the D256 single-buffer refill waits qk_done(Ja-1)
//     directly (its slot's next completion QK(Ja-1+NBUF) needs loader-gated
//     k_loaded AND q_loaded(qa) -> no aliasing; d4/c6-validated).
//   * o_drained (count 4, PERS page): the epilogue's 4 warps arrive after
//     their LAST O tmem_ld_wait; mma's first PV of assignment qa>=1 waits
//     parity (qa-1)&1 before the accumulate=false O overwrite.
//   * ALWAYS-group-1 epilogue (r18j): group 0 never enters it, so its next
//     assignment m=0 keeps the score_released -> QK chain flowing; when
//     group 1 did not own the last tile (odd ntl) it takes the final state
//     from state_m2/state_d behind probability_ready(J_last) — bit-
//     identical inv/base2 (mechanism proven in MT r18d/e/j).
// Scope: S==1 direct out/lse only (host-gated); AB hooks/TCW-in-loop are
// deliberately absent (production forms only); FMHA_WB stamps mirror the
// MT slot map.

  uint8_t* const pers_page_base =
      smem_raw + ((sizeof(Shared<TD, NSTAGE>) + 1023u) & ~(size_t)1023u);
  uint8_t* const q2 = (PLANES == 1) ? pers_page_base : nullptr;
  uint8_t* const pers_flags =
      (PLANES == 1) ? (pers_page_base + TM * 128) : pers_page_base;
  uint64_t* const o_drained = reinterpret_cast<uint64_t*>(pers_flags);
  uint64_t* const qld_b = reinterpret_cast<uint64_t*>(pers_flags + 16);
  volatile int* const mma_qa_p =
      reinterpret_cast<volatile int*>(pers_flags + 32);
  volatile int* const pull_seq =
      reinterpret_cast<volatile int*>(pers_flags + 40);
  volatile int* const pull_aid =
      reinterpret_cast<volatile int*>(pers_flags + 44);
  volatile int* const sm_read_ctr =
      reinterpret_cast<volatile int*>(pers_flags + 48);
  // r19fix: NSTAGE-slotted PV-retire signal for the LOADER's release/guard
  // waits (page offset +64, count 1, umma_commit'd by the mma warp next to
  // each output_ready[Jpv&1] commit). fmtc's production output_ready has
  // only TWO slots (Jpv&1) — cadence-safe per-CTA, but a 2-slot parity
  // window can never reopen after a +1-phase overshoot, and PERS assignment
  // boundaries structurally overshoot: while the loader stalls in the pull
  // pacing / Q guard, the mma's pvlag drain tail completes PV(Ja-2)/PV(Ja-1)
  // past the wait's target phase (r19diag dv_6e58bb8fe9db: c0 44/44 double-
  // pull workers stuck in the qa=1 guard, single-pull 104/104 complete; c6
  // 148/148 stuck at the boundary tl=0 release wait; d4 64/64 single-alloc
  // complete). With NSTAGE slots the target PV(wN)'s slot-next completion
  // is PV(wN+NSTAGE) = PV(Jtl) — gated behind THIS loader iteration's own
  // k/v issue (release) or q_loaded(qa) arrive (guard) → unreachable →
  // overshoot-free (the MT r18k argument; MT's output_ready is natively
  // NSTAGE-slotted, and its shipped history documents the two-wait/2-slot
  // form as FATAL — simmt.py repro at nmt=6). PV(wN) retirement implies
  // QK(wN) retirement (issue order + tcgen05.commit semantics), so ONE
  // wait frees both k and v of stage wN%NSTAGE.
  uint64_t* const pv_done = reinterpret_cast<uint64_t*>(pers_flags + 64);
  int* const ticket_ctr = const_cast<int*>(p.pers_list);
  const int* const pool_aids = p.pers_list + 1;
  const int nworkers = (int)gridDim.x;
  // r19c: STATIC balanced partition (fmtc): pers_list = [off_0..off_nw][aids];
  // this worker walks its own contiguous segment in heavy-first order.
  // Read-only -> graph-replay safe by construction (no counter, no atomics).
  const int* const seg_beg =
      p.pers_static ? (p.pers_list + (nworkers + 1) + p.pers_list[blockIdx.x])
                    : nullptr;
  const int seg_len =
      p.pers_static ? (p.pers_list[blockIdx.x + 1] - p.pers_list[blockIdx.x])
                    : 0;

  if (tid == 0) {
    *mma_qa_p = 0;
    *pull_seq = 0;
    *pull_aid = -1;
    *sm_read_ctr = 0;
    mbar_init(&sh.q_loaded, 96);
    mbar_init(qld_b, 96);
    mbar_init(&sh.pv_last, 1);
    mbar_init(o_drained, 4);   // group-1 epilogue (4 warps)
#pragma unroll
    for (int s = 0; s < NBUF; ++s) mbar_init(&sh.output_rescaled[s], 4);
#pragma unroll
    for (int s = 0; s < NSTAGE; ++s) {
      mbar_init(&sh.k_loaded[s], p.use_tma ? 1 : 96);
      mbar_init(&sh.v_loaded[s], p.use_tma ? 1 : 96);
      mbar_init(&sh.k_tx[s], 1);
      mbar_init(&sh.v_tx[s], 1);
    }
#pragma unroll
    for (int s = 0; s < NBUF; ++s) {
      mbar_init(&sh.qk_done[s], 1);
      mbar_init(&sh.score_released[s], 4);
      mbar_init(&sh.probability_ready[s], 4);
    }
#pragma unroll
    for (int s = 0; s < 2; ++s) {
      mbar_init(&sh.output_ready[s], 1);
    }
#pragma unroll
    for (int s = 0; s < NSTAGE; ++s) mbar_init(pv_done + s, 1);   // r19fix
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (warp == 8) {
    tmem_alloc(&sh.tmem_base, TMEM_COLS);
    tmem_relinq();
  }
  __syncthreads();
  const uint32_t tb = sh.tmem_base;
#ifdef FMHA_WB
  if (tid == 0) WB(8, (unsigned)clock64());
#endif

  if (warp >= 9) {
    // ------------------------- loaders (persistent) -----------------------
    setmaxnreg_dec<96>();
    const int ltid = tid - 288;
    int Ja = 0, qa = 0;
    for (;;) {
      int aid;
      if (p.pers_static) {
        // r19e: DIRECT segment binding (r18j machinery class) — every role
        // reads its own read-only segment; NO pull_aid/pull_seq broadcast,
        // NO pacing spins, NO sm_read_ctr atomics. r19d measured that
        // machinery at ~2us per assignment (c7 static-via-broadcast 1.02
        // vs r18j direct-bound 0.961). Uniform read: all 96 loader threads
        // see the same value (read-only global, same address).
        aid = (qa < seg_len) ? seg_beg[qa] : -1;
      } else {
        if (ltid == 0) {
          if (qa >= 1) {
            while (*mma_qa_p < qa || *sm_read_ctr < 8 * qa) {
            }
          }
          const int t = atomicAdd(ticket_ctr, 1);
          if (t >= p.pers_kmax) {
            if (t == p.pers_kmax + nworkers - 1) {
              *ticket_ctr = 0;   // self-clean for the next graph replay
              __threadfence();
            }
            aid = -1;
          } else {
            aid = pool_aids[t];
          }
          *pull_aid = aid;
          __threadfence_block();
          *pull_seq = qa + 1;
        }
        asm volatile("bar.sync 1, 96;" ::: "memory");
        aid = *pull_aid;
      }
      if (aid < 0) break;
      TCAssign A;
      if (!tc_aid_geom(p, aid, A)) { ++qa; continue; }   // host-guaranteed valid
      const int ntl = A.ntl;
      const int32_t* tab = p.table + (int64_t)A.b * p.Tstride;
      const uint8_t* kb = p.k + (int64_t)A.g * p.k_sh;
      const uint8_t* vb = p.v + (int64_t)A.g * p.v_sh;
      const int plog = p.page_log2, pmask = p.page - 1;
      uint8_t* const qb = (PLANES == 1 && (qa & 1)) ? q2 : sh.q[0];
#ifdef FMHA_WB
      if (ltid == 0) WB(19, (unsigned)aid);
#endif
      // Q-buffer overwrite guard (r19fix; see the pv_done comment above).
      // PLANES==2 (D256 single buffer): every qa>=1 refill waits QK(Ja-1)
      // retirement directly — the slot's next completion QK(Ja-1+NBUF) needs
      // loader-issued k_loaded AND q_loaded(qa), both after this point ->
      // overshoot-free (form validated empirically: d4 64/64 complete, c6
      // guard-pass stamps set on all 148 workers).
      // PLANES==1 (double buffer): qa==1 writes the FRESH q2 slot -> the
      // guard starts at qa>=2 (MT r18k form; the original qa>=1 2-slot
      // output_ready wait was the c0 hang: drain-tail PV(Ja-2) overshot the
      // target phase by +1 while the loader sat in the pull pacing). Target
      // PV(Ja-NSTAGE) on pv_done: slot-next = PV(Ja) = first PV of THIS
      // assignment, gated behind this loader's own q_loaded(qa) arrive ->
      // overshoot-free. PV(Ja-NSTAGE) retire implies QK(Ja-2) retire: PV(X)
      // issues at step X+pvlag, QK(Ja-2) at step Ja-2, and within a step QK
      // precedes PV — the implication needs pvlag >= NSTAGE-2, true for all
      // DEFAULT PLANES==1 configs (NSTAGE=3/pvlag=2, NSTAGE=4/pvlag=2;
      // force_lag1+NSTAGE=4 is the lone exception — debug knob, unused by
      // batteries/eval). The guarded slot's last reader is
      // QK(Ja-ntl_prev-1) <= QK(Ja-2) since ntl_prev >= 1.
      // Tiny-Ja fallback: QK(Ja-1) (loader-gated slot-next as above).
      if (PLANES == 2) {
        if (qa >= 1) {
          const int Jt = Ja - 1;
          mbar_wait(&sh.qk_done[Jt % NBUF], (Jt / NBUF) & 1);
        }
      } else if (qa >= 2) {
        if (Ja >= NSTAGE + 1) {
          const int Xg = Ja - NSTAGE;
          mbar_wait(pv_done + (Xg % NSTAGE), (Xg / NSTAGE) & 1);
        } else {
          const int Jt = Ja - 1;
          mbar_wait(&sh.qk_done[Jt % NBUF], (Jt / NBUF) & 1);
        }
      }
#ifdef FMHA_WB
      if (ltid == 0) WB(20, (unsigned)qa);
#endif
      for (int idx = ltid; idx < TM * CPR * PLANES; idx += 96) {
        const int pl = idx / (TM * CPR);
        const int rem = idx - pl * (TM * CPR);
        const int row = rem / CPR;
        const int col = (rem - row * CPR) * 16;
        int rg = A.r0 + row;
        if (rg > A.R - 1) rg = A.R - 1;
        const int t = rg / p.ratio;
        const int h = A.g * p.ratio + (rg - t * p.ratio);
        uint8_t* const qpl = (PLANES == 1) ? qb : sh.q[pl];
        tc_cp_async16(qpl + swz128(row, col),
                      p.q + (int64_t)(A.q0 + t) * p.q_tok_stride
                          + (int64_t)h * DV + pl * 128 + col);
      }
      // Arrive throttle: at most one pending arrive per q_loaded slot.
      if (qa >= 2) {
        while (*mma_qa_p < qa - 1) {
        }
      }
      cp_async_mbar_arrive((qa & 1) ? qld_b : &sh.q_loaded);
      if (p.use_tma) {
#pragma unroll 1
        for (int tl = 0; tl < ntl; ++tl) {
          const int Jtl = Ja + tl;
          const int stage = Jtl % NSTAGE;
          const int n0t = A.lo + tl * TN;
          const int nseg = TN >> plog;
          const int tab_lo = n0t >> plog;
#ifdef FMHA_WB
          if (ltid == 0) WB(18, (unsigned)Jtl);
#endif
          int pg_reg = 0;
          if (ltid < nseg)
            pg_reg = __ldg(tab + min(tab_lo + ltid, p.Tstride - 1));
          if (Jtl >= NSTAGE) {
            const int wN = Jtl - NSTAGE;
            // ONE release wait on pv_done (r19fix, MT r18k form): the target
            // PV(wN)'s slot-next completion is PV(wN+NSTAGE) = PV(Jtl),
            // gated behind THIS iteration's own k/v arrive below ->
            // overshoot-free. The production two-wait form (qk_done[wN%NBUF]
            // + output_ready[wN&1]) aliases +1 phase at PERS boundaries:
            // the drain tail completes PV(wN+2) while the loader stalls in
            // pull/guard (c6 hang: 148/148 stuck at boundary tl=0). PV(wN)
            // retire implies QK(wN) retire -> k AND v of the stage are free.
            mbar_wait(pv_done + (wN % NSTAGE), (wN / NSTAGE) & 1);
          }
          if (ltid < nseg) sh.tab_cache[ltid] = pg_reg;
          asm volatile("bar.sync 1, 96;" ::: "memory");
          if (ltid == 0) {
            mbar_arrive_expect_tx(&sh.k_tx[stage], PLANES * TN * ROWB);
            mbar_arrive_expect_tx(&sh.v_tx[stage], PLANES * TN * ROWB);
          }
          if constexpr (PLANES == 1) {
            asm volatile("bar.sync 1, 96;" ::: "memory");   // arms visible
            if (ltid < nseg) {
              const int seg_b = p.page * ROWB;
              const int pg = sh.tab_cache[ltid];
              tma_4d(sh.k[stage][0] + ltid * seg_b, &p.kmap, 0, 0, pg, A.g,
                     &sh.k_tx[stage]);
              tma_4d(sh.v[stage][0] + ltid * seg_b, &p.vmap, 0, 0, pg, A.g,
                     &sh.v_tx[stage]);
            }
          } else {
            if (ltid == 0) {
              const int seg_b = p.page * ROWB;
              for (int pl = 0; pl < PLANES; ++pl)
                for (int sg = 0; sg < nseg; ++sg) {
                  const int pg = sh.tab_cache[sg];
                  tma_4d(sh.k[stage][pl] + sg * seg_b, &p.kmap, pl * 128, 0,
                         pg, A.g, &sh.k_tx[stage]);
                  tma_4d(sh.v[stage][pl] + sg * seg_b, &p.vmap, pl * 128, 0,
                         pg, A.g, &sh.v_tx[stage]);
                }
            }
          }
          if (ltid == 0) {
            mbar_arrive(&sh.k_loaded[stage]);
            mbar_arrive(&sh.v_loaded[stage]);
          }
        }
      } else {
#pragma unroll 1
        for (int tl = 0; tl < ntl; ++tl) {
          const int Jtl = Ja + tl;
          const int stage = Jtl % NSTAGE;
          const int n0t = A.lo + tl * TN;
          int posmax = n0t + TN - 1;
          if (posmax > A.hi - 1) posmax = A.hi - 1;
          const int tab_lo = n0t >> plog;
          const int nent = (posmax >> plog) - tab_lo + 1;
          int pg_reg = 0;
          if (ltid < nent) pg_reg = __ldg(tab + tab_lo + ltid);
          if (Jtl >= NSTAGE) {
            const int wN = Jtl - NSTAGE;
            // ONE release wait on pv_done (r19fix, MT r18k form): the target
            // PV(wN)'s slot-next completion is PV(wN+NSTAGE) = PV(Jtl),
            // gated behind THIS iteration's own k/v arrive below ->
            // overshoot-free. The production two-wait form (qk_done[wN%NBUF]
            // + output_ready[wN&1]) aliases +1 phase at PERS boundaries:
            // the drain tail completes PV(wN+2) while the loader stalls in
            // pull/guard (c6 hang: 148/148 stuck at boundary tl=0). PV(wN)
            // retire implies QK(wN) retire -> k AND v of the stage are free.
            mbar_wait(pv_done + (wN % NSTAGE), (wN / NSTAGE) & 1);
          }
          if (ltid < nent) sh.tab_cache[ltid] = pg_reg;
          asm volatile("bar.sync 1, 96;" ::: "memory");
#pragma unroll 4
          for (int idx = ltid; idx < TN * CPR * PLANES; idx += 96) {
            const int pl = idx / (TN * CPR);
            const int rem = idx - pl * (TN * CPR);
            const int row = rem / CPR;
            const int col = (rem - row * CPR) * 16;
            int pos = n0t + row;
            if (pos > A.hi - 1) pos = A.hi - 1;
            const int64_t poff = (int64_t)sh.tab_cache[(pos >> plog) - tab_lo];
            const int off = pos & pmask;
            tc_cp_async16(sh.k[stage][pl] + swz128(row, col),
                          kb + poff * p.k_sp + (int64_t)off * p.k_sn
                              + pl * 128 + col);
            tc_cp_async16(sh.v[stage][pl] + swz128(row, col),
                          vb + poff * p.v_sp + (int64_t)off * p.v_sn
                              + pl * 128 + col);
          }
          cp_async_mbar_arrive(&sh.k_loaded[stage]);
          cp_async_mbar_arrive(&sh.v_loaded[stage]);
        }
      }
      Ja += ntl;
      ++qa;
    }
  } else if (warp == 8) {
    // ------------------------ mma issue (persistent) ----------------------
    setmaxnreg_dec<96>();
    if (lane == 0) {
      const uint32_t id_qk = idesc_f8(TM, TN, false);
      const uint32_t id_pv = idesc_f8(TM, NPV, true);
      int Ja = 0, qa = 0;
      for (;;) {
        int aid;
        if (p.pers_static) {
          aid = (qa < seg_len) ? seg_beg[qa] : -1;   // r19e direct binding
        } else {
          while (*pull_seq < qa + 1) {
          }
          aid = *pull_aid;
        }
        if (aid < 0) break;
        TCAssign A;
        if (!tc_aid_geom(p, aid, A)) {
          // Host guarantees validity; defensive skip must still advance the
          // pacing flag or the loader's next pull would spin forever.
          __threadfence_block();
          *mma_qa_p = qa + 1;
          ++qa;
          continue;
        }
        const int nloc = A.ntl;
#ifdef FMHA_WB
        WB(5, (unsigned)aid);
#endif
        mbar_wait((qa & 1) ? qld_b : &sh.q_loaded, (qa >> 1) & 1);
        __threadfence_block();   // release: aid read precedes the flag
        *mma_qa_p = qa + 1;
#ifdef FMHA_WB
        WB(28, (unsigned)qa);
        if (qa == 0) WB(9, (unsigned)clock64());
#endif
        const uint8_t* const qbase =
            (PLANES == 1 && (qa & 1)) ? q2 : (const uint8_t*)sh.q[0];
#pragma unroll 1
        for (int j = 0; j < nloc + pvlag; ++j) {
          const int J = Ja + j;
#ifdef FMHA_WB
          WB(4, (unsigned)J);
#endif
          if (j < nloc) {
            const int sb = J % NBUF;
            const int stage = J % NSTAGE;
            mbar_wait(&sh.k_loaded[stage], (J / NSTAGE) & 1);
            if (p.use_tma) mbar_wait(&sh.k_tx[stage], (J / NSTAGE) & 1);
            if (J >= NBUF) {
              mbar_wait(&sh.score_released[sb], (J / NBUF - 1) & 1);
            }
            const uint32_t dmem = tb + sb * 128;
#pragma unroll
            for (int pl = 0; pl < PLANES; ++pl) {
              uint64_t ad = make_desc(2, 16, 8 * 128,
                                      qbase + (PLANES == 1 ? 0 : pl * (TM * 128)));
              uint64_t bd = tma64 ? make_desc(4, 0, 8 * 64, sh.k[stage][pl])
                                  : make_desc(2, 16, 8 * 128, sh.k[stage][pl]);
#pragma unroll
              for (int step = 0; step < KSTEPS; ++step) {
                umma_f8_ss(dmem, ad, bd, id_qk, pl != 0 || step != 0);
                if (step != KSTEPS - 1) { desc_add(ad, 32); desc_add(bd, 32); }
              }
            }
            umma_commit(&sh.qk_done[sb]);
#ifdef FMHA_WB
            if (J == 0) WB(10, (unsigned)clock64());
#endif
          }
          if (j >= pvlag) {
            const int pv = j - pvlag;
            const int Jpv = Ja + pv;
            const int pb = Jpv % NBUF;
            const int pvstage = Jpv % NSTAGE;
            // O-handoff gate: first PV of assignment qa>=1 may not touch O
            // before the previous epilogue finished READING it.
            if (pv == 0 && qa >= 1) mbar_wait(o_drained, (qa - 1) & 1);
            mbar_wait(&sh.v_loaded[pvstage], (Jpv / NSTAGE) & 1);
            if (p.use_tma) mbar_wait(&sh.v_tx[pvstage], (Jpv / NSTAGE) & 1);
            mbar_wait(&sh.probability_ready[pb], (Jpv / NBUF) & 1);
            if (pv >= 1) {
              mbar_wait(&sh.output_rescaled[Jpv % NBUF], (Jpv / NBUF) & 1);
            }
            const uint32_t pa = tb + pb * 128 + 32;
#pragma unroll
            for (int pl = 0; pl < PLANES; ++pl) {
              uint64_t vd = tma64 ? make_desc(4, 0, 8 * 64, sh.v[pvstage][pl])
                                  : make_desc(2, 0, 8 * 128, sh.v[pvstage][pl]);
              const uint32_t om = tb + O_TMEM_V + pl * 128;
#pragma unroll
              for (int step = 0; step < 4; ++step) {
                umma_f8_ts(om, pa + step * 8, vd, id_pv,
                           pv != 0 || step != 0);
                if (step != 3) desc_add(vd, tma64 ? 32 * 64 : 32 * 128);
              }
            }
            umma_commit(&sh.output_ready[Jpv & 1]);
            // r19fix: loader-facing NSTAGE-slotted PV-retire signal (see the
            // pv_done declaration comment). Both commits track the SAME
            // tensor-pipe prefix; the extra tcgen05.commit is one issue slot.
            umma_commit(pv_done + (Jpv % NSTAGE));
            if (pv == nloc - 1) umma_commit(&sh.pv_last);
          }
        }
        Ja += nloc;
        ++qa;
      }
#ifdef FMHA_WB
      WB(6, (unsigned)Ja);
      WB(11, (unsigned)clock64());
#endif
    }
  } else {
    // ---------------------- softmax groups (persistent) -------------------
    setmaxnreg_inc<200>();
    const int grp = warp >> 2;
    const int r = (warp & 3) * 32 + lane;
    const uint32_t st_lane = ((uint32_t)(warp & 3) * 32) << 16;
    const uint32_t oaddr = tb + O_TMEM_V + st_lane;
    int Ja = 0, qa = 0;
    for (;;) {
      int aid;
      if (p.pers_static) {
        aid = (qa < seg_len) ? seg_beg[qa] : -1;   // r19e direct binding
      } else {
        while (*pull_seq < qa + 1) {
        }
        aid = *pull_aid;
        __syncwarp(0xffffffffu);
        __threadfence_block();   // release: aid read precedes the counter
        if (lane == 0) atomicAdd((int*)sm_read_ctr, 1);
      }
      if (aid < 0) break;
      TCAssign A;
      if (!tc_aid_geom(p, aid, A)) { ++qa; continue; }
      const int nloc = A.ntl;
#ifdef FMHA_WB
      if (warp == 0 && lane == 0) WB(22, (unsigned)aid);
#endif
      const int rg = A.r0 + r;
      const int tok = rg / p.ratio;
      const int vis_i = A.kvlen - A.qlen + tok;
      const bool row_ok = (rg < A.R);
      float m2r = -INFINITY;   // per-assignment online-softmax state
      float dsum = 0.f;
      // r19j: the r19g group-0 o_drained entry gate is REMOVED (same
      // rationale as MT): ~2us/boundary cost ate the static wins (eval c7
      // 0.9811 vs 0.942-0.961 class; c6 ~4-5us), and the corruption it was
      // added for is the PARTITION-level tiny-chain regime, quarantined in
      // r19i independently of this gate. fmtc static (c6 BFD 3.46 alloc/w,
      // c0) was bitwise-clean WITHOUT the gate across r19c/d/e (5+ jobs);
      // the group-0-publish vs group-1-state-read window keeps its ~1-2us
      // cadence margin (publish needs the full qk_done(Ja_next) chain:
      // loader guard+fill+arrive -> mma QK issue+retire -> S read).
      // Fallback if ever needed: state_free barrier (group 1 arrives after
      // its state read, before pv_last; group 0 waits it at entry).
#pragma unroll 1
      for (int tl = grp; tl < nloc; tl += 2) {
        const int Jm = Ja + tl;
        const int sb = Jm % NBUF;
#ifdef FMHA_WB
        if (lane == 0) {
          if (warp == 0) WB(21, (unsigned)Jm);
          if (warp == 4) WB(25, (unsigned)Jm);
        }
#endif
        mbar_wait(&sh.qk_done[sb], (Jm / NBUF) & 1);
        uint32_t sc[128];
        tmem_ld32(sc, tb + sb * 128);
        tmem_ld_wait();
#pragma unroll
        for (int seg = 1; seg < 4; ++seg)
          tmem_ld32(sc + seg * 32, tb + sb * 128 + seg * 32);
        const int n0t = A.lo + tl * TN;
        int c_lo = 0, c_hi = 128;
        if (!row_ok) {
          c_hi = 0;
        } else {
          int ub = A.hi;
          if (p.causal) ub = min(ub, vis_i + 1);
          c_hi = min(128, ub - n0t);
          if (p.window_left >= 0) c_lo = max(0, vis_i - p.window_left - n0t);
          if (c_hi < c_lo) c_hi = c_lo;
        }
        const unsigned span = (unsigned)(c_hi - c_lo);
        float mx0 = -INFINITY, mx1 = -INFINITY, mx2 = -INFINITY, mx3 = -INFINITY;
        if (span >= 128) {
#pragma unroll
          for (int i = 0; i < 32; ++i) mx0 = fmaxf(mx0, __uint_as_float(sc[i]));
        } else {
#pragma unroll
          for (int i = 0; i < 32; ++i) {
            const bool in = (unsigned)(i - c_lo) < span;
            float raw = __uint_as_float(sc[i]);
            raw = in ? raw : -INFINITY;
            sc[i] = __float_as_uint(raw);
            mx0 = fmaxf(mx0, raw);
          }
        }
        tmem_ld_wait();
        // Publish ALL lanes' loads before lane0's release-arrive (v3a).
        __syncwarp(0xffffffffu);
        if (lane == 0) mbar_arrive(&sh.score_released[sb]);
        if (span >= 128) {
#pragma unroll
          for (int i = 32; i < 64; ++i) mx1 = fmaxf(mx1, __uint_as_float(sc[i]));
#pragma unroll
          for (int i = 64; i < 96; ++i) mx2 = fmaxf(mx2, __uint_as_float(sc[i]));
#pragma unroll
          for (int i = 96; i < 128; ++i) mx3 = fmaxf(mx3, __uint_as_float(sc[i]));
        } else {
#pragma unroll
          for (int i = 32; i < 64; ++i) {
            const bool in = (unsigned)(i - c_lo) < span;
            float raw = __uint_as_float(sc[i]);
            raw = in ? raw : -INFINITY;
            sc[i] = __float_as_uint(raw);
            mx1 = fmaxf(mx1, raw);
          }
#pragma unroll
          for (int i = 64; i < 96; ++i) {
            const bool in = (unsigned)(i - c_lo) < span;
            float raw = __uint_as_float(sc[i]);
            raw = in ? raw : -INFINITY;
            sc[i] = __float_as_uint(raw);
            mx2 = fmaxf(mx2, raw);
          }
#pragma unroll
          for (int i = 96; i < 128; ++i) {
            const bool in = (unsigned)(i - c_lo) < span;
            float raw = __uint_as_float(sc[i]);
            raw = in ? raw : -INFINITY;
            sc[i] = __float_as_uint(raw);
            mx3 = fmaxf(mx3, raw);
          }
        }
        if (tl >= 1) {
          // State handoff (LOCAL gate; GLOBAL Jm-1 slot/parity).
          mbar_wait(&sh.probability_ready[(Jm - 1) % NBUF],
                    ((Jm - 1) / NBUF) & 1);
          m2r = sh.state_m2[r];
          dsum = sh.state_d[r];
        }
        const float tmaxr = fmaxf(fmaxf(mx0, mx1), fmaxf(mx2, mx3));
        const float qs = p.qk_scale;
        const bool rebase = p.force_corr ||
                            (((tmaxr - m2r) * qs) > REBASE_MARGIN);
        const float mnr = rebase ? tmaxr : m2r;
        const float mbase = (mnr == -INFINITY) ? 0.f : mnr;
        const float nmus8 = PLOG2 - mbase * qs;
        const float accs = rebase ? tc_ex2((m2r - mnr) * qs) : 1.0f;
        uint32_t pw[32];
        uint32_t ps2[4] = {0u, 0u, 0u, 0u};
        const uint32_t qs2 = f16x2_dup(qs);
        const uint32_t nm2 = f16x2_dup(nmus8);
#pragma unroll
        for (int widx = 0; widx < 32; ++widx) {
          const uint32_t sa = cvt_f16x2_sat(__uint_as_float(sc[4 * widx + 0]),
                                            __uint_as_float(sc[4 * widx + 1]));
          const uint32_t sbw = cvt_f16x2_sat(__uint_as_float(sc[4 * widx + 2]),
                                             __uint_as_float(sc[4 * widx + 3]));
          const uint32_t pa = ex2_f16x2(hfma2_u32(sa, qs2, nm2));
          const uint32_t pbw = ex2_f16x2(hfma2_u32(sbw, qs2, nm2));
          ps2[widx & 3] = hadd2_u32(ps2[widx & 3], pa);
          ps2[widx & 3] = hadd2_u32(ps2[widx & 3], pbw);
          pw[widx] = pack_e4m3x4(pa, pbw);
        }
        const uint32_t pst = hadd2_u32(hadd2_u32(ps2[0], ps2[1]),
                                       hadd2_u32(ps2[2], ps2[3]));
        const float psum = f16x2_lo(pst) + f16x2_hi(pst);
        dsum = dsum * accs + psum;
        m2r = mnr;
        // Publish state BEFORE the probability_ready arrive (release).
        sh.state_m2[r] = m2r;
        sh.state_d[r] = dsum;
        tmem_st32(tb + sb * 128 + 32 + st_lane, pw);
        tmem_st_wait();
        __syncwarp(0xffffffffu);
        if (lane == 0) mbar_arrive(&sh.probability_ready[sb]);
#ifdef FMHA_WB
        if (warp == 0 && lane == 0) WB(26, (unsigned)Jm);
#endif
        {
          if (tl >= 1) {
            const bool do_corr = __ballot_sync(0xffffffffu, rebase);
            mbar_wait(&sh.output_ready[(Jm - 1) & 1], ((Jm - 1) >> 1) & 1);
            if (do_corr) {
              constexpr int NB = DV / 32;
              constexpr int BB = NB < 4 ? NB : 4;
              uint32_t vals[BB * 32];
#pragma unroll
              for (int b0 = 0; b0 < NB; b0 += BB) {
#pragma unroll
                for (int jj = 0; jj < BB; ++jj)
                  tmem_ld32(vals + jj * 32, oaddr + (b0 + jj) * 32);
                tmem_ld_wait();
#pragma unroll
                for (int i = 0; i < BB * 32; ++i)
                  vals[i] = __float_as_uint(__uint_as_float(vals[i]) * accs);
#pragma unroll
                for (int jj = 0; jj < BB; ++jj)
                  tmem_st32(oaddr + (b0 + jj) * 32, vals + jj * 32);
                tmem_st_wait();
              }
            }
            __syncwarp(0xffffffffu);
          }
          if (lane == 0) mbar_arrive(&sh.output_rescaled[Jm % NBUF]);
        }
      }
      // ---- ALWAYS-group-1 epilogue (r18j mechanism; S==1 direct out/lse) ----
      if (grp == 1) {
        const bool owner = ((nloc & 1) == 0);   // even nloc -> group1 owned last
        float m2r_e = m2r;
        float dsum_e = dsum;
        if (!owner) {
          const int Jl = Ja + nloc - 1;   // global index of the last tile
          mbar_wait(&sh.probability_ready[Jl % NBUF], (Jl / NBUF) & 1);
          m2r_e = sh.state_m2[r];
          dsum_e = sh.state_d[r];
        }
#ifdef FMHA_WB
        if (lane == 0) WB(23, (unsigned)qa);
#endif
        mbar_wait(&sh.pv_last, qa & 1);
#ifdef FMHA_WB
        if (lane == 0) WB(24, (unsigned)qa);
#endif
        const int t = rg / p.ratio;
        const int h = A.g * p.ratio + (rg - t * p.ratio);
        float base2 = 0.f;
        float inv = 0.f;
        if (row_ok) {
          base2 = m2r_e * p.qk_scale + log2f(dsum_e) - PLOG2;
          inv = p.bmm2 / dsum_e;
          if (p.has_sink) {
            const float sk = p.sinks[h] * 1.4426950408889634f;
            const float tot = fmaxf(base2, sk)
                              + log2f(1.f + exp2f(-fabsf(base2 - sk)));
            inv *= exp2f(base2 - tot);
            base2 = tot;
          }
          if (p.has_lse) p.lse[(int64_t)(A.q0 + t) * p.H + h] = base2;
        }
        uint16_t* orow =
            p.out + ((int64_t)(A.q0 + min(t, A.qlen - 1)) * p.H
                     + min(h, p.H - 1)) * DV;
        const bool wide = ((reinterpret_cast<uintptr_t>(p.out) & 15u) == 0u);
        constexpr int NBo = DV / 32;
        constexpr int BBo = NBo < 4 ? NBo : 4;
#pragma unroll
        for (int b0 = 0; b0 < NBo; b0 += BBo) {
          uint32_t vals[BBo * 32];
#pragma unroll
          for (int jj = 0; jj < BBo; ++jj)
            tmem_ld32(vals + jj * 32, oaddr + (b0 + jj) * 32);
          tmem_ld_wait();
          // O reads done -> release tmem to the next assignment's PV(pv==0)
          // overwrite; the global stores overlap onward.
          if (b0 + BBo >= NBo && lane == 0) mbar_arrive(o_drained);
          if (row_ok) {
            if (wide) {
#pragma unroll
              for (int pr = 0; pr < BBo * 16; pr += 4) {
                uint4 pk4;
                uint32_t* w4 = reinterpret_cast<uint32_t*>(&pk4);
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                  const float x0 = __uint_as_float(vals[2 * (pr + k)]) * inv;
                  const float x1 =
                      __uint_as_float(vals[2 * (pr + k) + 1]) * inv;
                  const __nv_bfloat162 pk =
                      __float22bfloat162_rn(make_float2(x0, x1));
                  w4[k] = *reinterpret_cast<const uint32_t*>(&pk);
                }
                *reinterpret_cast<uint4*>(orow + b0 * 32 + 2 * pr) = pk4;
              }
            } else {
#pragma unroll
              for (int pr = 0; pr < BBo * 16; ++pr) {
                const float x0 = __uint_as_float(vals[2 * pr]) * inv;
                const float x1 = __uint_as_float(vals[2 * pr + 1]) * inv;
                __nv_bfloat162 pk =
                    __float22bfloat162_rn(make_float2(x0, x1));
                *reinterpret_cast<uint32_t*>(orow + b0 * 32 + 2 * pr) =
                    *reinterpret_cast<uint32_t*>(&pk);
              }
            }
          }
        }
      }
      Ja += nloc;
      ++qa;
    }
  }

  TCW(37 + warp, 1);
  if (p.exit_mode != 2) {
    __syncthreads();
    TCW(50 + warp, 1);
#ifdef FMHA_WB
    if (warp == 9 && lane == 0) WB(12, (unsigned)clock64());
    if (tid == 0) {
      WB(13, (unsigned)clock64());
      WB(14, (unsigned)tc_wb_gt());
    }
#endif
  }
  if (warp == 8) {
    if (p.exit_mode == 0) {
      tmem_dealloc(tb, TMEM_COLS);
      TCW(62, 1);
    } else {
      TCW(61, 1);
    }
  }
