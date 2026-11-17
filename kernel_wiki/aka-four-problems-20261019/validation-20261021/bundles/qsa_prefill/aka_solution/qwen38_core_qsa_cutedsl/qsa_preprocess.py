"""Row-block union dedup preprocess kernel (native CuTeDSL).

One CTA per row block of ``br`` consecutive query rows.

Phase 1 walks every (row, column) quad of the block's packed
``logical_indices`` rows straight from global memory (128-bit vector loads,
software-pipelined one quad ahead so the LDG latency hides under the previous
quad's hash work), applies reference.py's validity rules, and deduplicates
valid tokens at compression-group granularity in an open-addressing hash table
whose slot holds the entry itself:

* ``sKey[h]``      : ``((req << 17) | (idx >> 2)) + 1`` (0 = empty), created
  by one atomic_cas;
* ``sLo[h]/sHi[h]``: 40-bit nibble pack; bit ``4*r + l`` records that block
  row r selected token ``4*g + l``.  Every inserter (creator or not) sets its
  bits with atomic_or, so there is no claim/publish window and no lost-update
  race.  Distinct-key slot collisions walk ``h+1`` (load factor <= ~0.32).

Phase 2 compacts occupied slots into the workspace with a deterministic
per-thread range + warp-prefix scan (one SMEM atomic per warp instead of one
per entry): entry order follows slot order, so the compaction adds no
run-to-run variance of its own.  Output is (req, g, nib_lo, nib_hi) per entry
plus a per-block count.  Invalid selections never set bits, so the main
kernel masks them out of every softmax.

Domain note: req < 2^14 and g < 2^17 hold for every legal draw (single block
table row; sequence_length = 131072); rows whose req would overflow the key
field are treated as invalid (they cannot occur in-domain).
"""
from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.typing import Int32

_LOG_SLOTS = 14  # 16384 slots; entry bound 5130 -> load factor <= 0.32
_SLOT_MASK = (1 << _LOG_SLOTS) - 1
_G_BITS = 17


@cute.jit
def _hash_key(x: Int32) -> Int32:
    h = x * Int32(0x2545F491)
    h = h ^ (h >> 15)
    h = h * Int32(0x27220A95)
    return (h ^ (h >> 13)) & Int32(_SLOT_MASK)


@cute.jit
def _shfl32(value: Int32, src_lane: Int32) -> Int32:
    return Int32(cute.arch.shuffle_sync(value, src_lane, mask_and_clamp=31))


class QSAPreprocessUnion:
    def __init__(self, br: int, sel_width: int, page_size: int, bt_width: int,
                 threads: int = 512):
        self.br = br
        self.sel_width = sel_width
        self.page_size = page_size
        self.bt_width = bt_width
        self.maxe = br * (sel_width // 4 + 1)  # union entry bound per block
        self.threads = threads
        assert self.maxe < (1 << _LOG_SLOTS), "hash table too small"
        assert (1 << _LOG_SLOTS) % threads == 0, "scan range must divide"

    @cute.jit
    def __call__(self, mLI, mBT, mT2R, mWS, mCount, rows: Int32, bt_rows: Int32,
                 num_pages: Int32, stream):
        nblocks = (rows + self.br - 1) // self.br
        self.kernel(mLI, mBT, mT2R, mWS, mCount, rows, bt_rows, num_pages).launch(
            grid=(nblocks, 1, 1), block=[self.threads, 1, 1],
            cluster=(1, 1, 1), stream=stream, min_blocks_per_mp=1)

    @cute.kernel
    def kernel(self, mLI, mBT, mT2R, mWS, mCount, rows: Int32, bt_rows: Int32,
               num_pages: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        blk, _, _ = cute.arch.block_idx()
        br, sel_width, page = self.br, self.sel_width, self.page_size
        bt_width, maxe = self.bt_width, self.maxe
        row0 = blk * br
        thr = self.threads
        packed_w = sel_width + 1  # packed row width incl. valid_count column
        lane = tidx % 32
        warp = tidx // 32

        @cute.struct
        class SharedStorage:
            s_key: cute.struct.MemRange[Int32, 1 << _LOG_SLOTS]
            s_lo: cute.struct.MemRange[Int32, 1 << _LOG_SLOTS]
            s_hi: cute.struct.MemRange[Int32, 1 << _LOG_SLOTS]
            s_vc: cute.struct.MemRange[Int32, br]
            s_bt: cute.struct.MemRange[Int32, bt_width]
            s_req: cute.struct.MemRange[Int32, br]
            counter: Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        key_ptr = storage.s_key.data_ptr()
        lo_ptr = storage.s_lo.data_ptr()
        hi_ptr = storage.s_hi.data_ptr()
        vc_ptr = storage.s_vc.data_ptr()
        bt_ptr = storage.s_bt.data_ptr()
        req_ptr = storage.s_req.data_ptr()
        cnt_ptr = storage.counter.ptr

        # ---- phase 0: small per-row inputs + table clear ----
        # (the packed index rows themselves are consumed straight from gmem in
        # phase 1 with vectorized pipelined loads: staging them through SMEM
        # only added a serial LDG->STS chain and 82KB of SMEM.)
        if tidx < br:
            grow = row0 + tidx
            v = Int32(-1)
            if grow < rows:
                v = Int32(mLI[grow, sel_width])   # valid_count column
            (vc_ptr + tidx).store(v)
            v = Int32(-1)
            if grow < rows:
                v = Int32(mT2R[grow])
            (req_ptr + tidx).store(v)
        for i in cutlass.range((bt_width + thr - 1) // thr):
            p = i * thr + tidx
            if p < bt_width:
                v = Int32(-1)
                if bt_rows > 0:
                    v = Int32(mBT[0, p])
                (bt_ptr + p).store(v)
        for i in cutlass.range((1 << _LOG_SLOTS) // thr, unroll_full=True):
            s = tidx + thr * i
            (key_ptr + s).store(Int32(0))
            (lo_ptr + s).store(Int32(0))
            (hi_ptr + s).store(Int32(0))
        if tidx == 0:
            cnt_ptr.store(Int32(0))
        cute.arch.sync_threads()

        # ---- phase 1: validity-filter + hash insert ----
        # Quad-packed: the contract packs each selected group as 4 consecutive
        # columns (4*g + l), so one quad = one hash entry with up to 4 lane
        # bits -> ~4x fewer CAS/OR atomics.  A per-lane fallback keeps the
        # kernel exact for any (out-of-pattern but legal) column packing.
        # Each quad's 4 words are one 16B-aligned vector load (row stride is
        # 2052 words = 513 x 16B, c0 is a multiple of 16B), software
        # pipelined one quad ahead.
        quads_row = sel_width // 4 + 1          # 513 for sel_width=2051
        total_q = br * quads_row
        n_iter = (total_q + thr - 1) // thr

        i0, i1, i2, i3 = self._load_quad(mLI, row0, rows, quads_row,
                                         packed_w, tidx, thr, tidx)
        for w in cutlass.range(n_iter):
            e = w * thr + tidx
            j0, j1, j2, j3 = self._load_quad(mLI, row0, rows, quads_row,
                                             packed_w, e + thr, thr, tidx)
            if e < total_q:
                r = e // quads_row
                q = e - r * quads_row
                grow = row0 + r
                if grow < rows:
                    c0 = q * 4
                    vc = (vc_ptr + r).load()
                    req = (req_ptr + r).load()
                    req_ok = (req >= 0) and (req < bt_rows)
                    req_ok = req_ok and (req < (1 << (31 - _G_BITS)))
                    idx0 = i0
                    idx1 = i1
                    idx2 = i2
                    idx3 = i3
                    v0 = req_ok and (c0 < vc) and (idx0 >= 0)
                    v1 = req_ok and (c0 + 1 < vc) and (idx1 >= 0)
                    v2 = req_ok and (c0 + 2 < vc) and (idx2 >= 0)
                    v3 = req_ok and (c0 + 3 < vc) and (idx3 >= 0)
                    anyv = v0 or v1 or v2 or v3
                    if anyv:
                        # quad group = first valid lane's group; conforming
                        # packs have all four lanes in one group.
                        gq = cutlass.select_(v0, idx0,
                                             cutlass.select_(v1, idx1,
                                                             cutlass.select_(v2, idx2, idx3)))
                        gq = gq >> 2
                        mis = Int32(0)
                        mis = mis + cutlass.select_(v0 and ((idx0 >> 2) != gq), Int32(1), Int32(0))
                        mis = mis + cutlass.select_(v1 and ((idx1 >> 2) != gq), Int32(1), Int32(0))
                        mis = mis + cutlass.select_(v2 and ((idx2 >> 2) != gq), Int32(1), Int32(0))
                        mis = mis + cutlass.select_(v3 and ((idx3 >> 2) != gq), Int32(1), Int32(0))
                        consistent = mis == 0
                        lpage = (gq << 2) // page
                        page_ok = lpage < bt_width
                        if page_ok:
                            phys = Int32(-1)
                            if req == 0:
                                phys = (bt_ptr + lpage).load()
                            else:
                                phys = Int32(mBT[req, lpage])
                            page_ok = (phys >= 0) and (phys < num_pages)
                        if consistent and page_ok:
                            m = Int32(0)
                            mh = Int32(0)
                            sh4 = r * 4
                            if v0:
                                b = sh4 + (idx0 & 3)
                                m = cutlass.select_(b < 32, m | (Int32(1) << b), m)
                                mh = cutlass.select_(b >= 32, mh | (Int32(1) << (b - 32)), mh)
                            if v1:
                                b = sh4 + (idx1 & 3)
                                m = cutlass.select_(b < 32, m | (Int32(1) << b), m)
                                mh = cutlass.select_(b >= 32, mh | (Int32(1) << (b - 32)), mh)
                            if v2:
                                b = sh4 + (idx2 & 3)
                                m = cutlass.select_(b < 32, m | (Int32(1) << b), m)
                                mh = cutlass.select_(b >= 32, mh | (Int32(1) << (b - 32)), mh)
                            if v3:
                                b = sh4 + (idx3 & 3)
                                m = cutlass.select_(b < 32, m | (Int32(1) << b), m)
                                mh = cutlass.select_(b >= 32, mh | (Int32(1) << (b - 32)), mh)
                            keyval = ((req << _G_BITS) | gq) + Int32(1)
                            h = _hash_key(keyval)
                            placed = False
                            while not placed:
                                old = cute.arch.atomic_cas(
                                    key_ptr + h, cmp=Int32(0), val=keyval,
                                    sem="relaxed", scope="cta")
                                if (old == 0) or (old == keyval):
                                    cute.arch.atomic_or(lo_ptr + h, m,
                                                        sem="relaxed", scope="cta")
                                    cute.arch.atomic_or(hi_ptr + h, mh,
                                                        sem="relaxed", scope="cta")
                                    placed = True
                                else:
                                    h = (h + Int32(1)) & Int32(_SLOT_MASK)
                        else:
                            # rare/never in-domain: mixed-group quad or bad page
                            # -> exact per-lane fallback
                            self._insert_lane(key_ptr, lo_ptr, hi_ptr, bt_ptr,
                                              mBT, req, idx0, v0, r, page,
                                              bt_width, num_pages)
                            self._insert_lane(key_ptr, lo_ptr, hi_ptr, bt_ptr,
                                              mBT, req, idx1, v1, r, page,
                                              bt_width, num_pages)
                            self._insert_lane(key_ptr, lo_ptr, hi_ptr, bt_ptr,
                                              mBT, req, idx2, v2, r, page,
                                              bt_width, num_pages)
                            self._insert_lane(key_ptr, lo_ptr, hi_ptr, bt_ptr,
                                              mBT, req, idx3, v3, r, page,
                                              bt_width, num_pages)
            i0 = j0
            i1 = j1
            i2 = j2
            i3 = j3

        cute.arch.sync_threads()

        # ---- phase 2: deterministic warp-prefix compaction ----
        # Each thread owns a contiguous slot range; per-range occupancy runs
        # through a warp-exclusive scan and ONE SMEM atomic per warp, then
        # entries are written to the workspace in slot order.
        NS = (1 << _LOG_SLOTS) // thr           # slots per thread (constexpr)
        base_out = (blk * maxe) * 4
        occ = Int32(0)
        for j in cutlass.range_constexpr(NS):
            s = tidx * NS + j
            k = (key_ptr + s).load()
            occ = occ + cutlass.select_(k != 0, Int32(1), Int32(0))
        x = occ
        for d in cutlass.range_constexpr(5):
            dd = 1 << d
            y = _shfl32(x, lane - dd)
            x = x + cutlass.select_(lane >= dd, y, Int32(0))
        excl = x - occ
        wtot = _shfl32(x, Int32(31))
        wbase = Int32(0)
        if lane == 0:
            wbase = cute.arch.atomic_add(cnt_ptr, wtot, sem="relaxed",
                                         scope="cta")
        wbase = _shfl32(wbase, Int32(0))
        rank = wbase + excl
        for j in cutlass.range_constexpr(NS):
            s = tidx * NS + j
            keyval = (key_ptr + s).load()
            if keyval != 0:
                k = keyval - Int32(1)
                mWS[base_out + rank * 4 + 0] = k >> _G_BITS
                mWS[base_out + rank * 4 + 1] = k & Int32((1 << _G_BITS) - 1)
                mWS[base_out + rank * 4 + 2] = (lo_ptr + s).load()
                mWS[base_out + rank * 4 + 3] = (hi_ptr + s).load()
                rank = rank + Int32(1)
        cute.arch.sync_threads()
        if tidx == 0:
            mCount[blk] = cnt_ptr.load()

    @cute.jit
    def _load_quad(self, mLI, row0, rows, quads_row, packed_w, e, thr, tidx):
        """Vectorized 16B load of quad ``e``'s four packed index words.

        Out-of-range or beyond-``rows`` quads return (-1,-1,-1,-1) so the
        validity logic marks every lane invalid without branching in the
        consumer.  The address is always in-tensor for the CTA's row block
        (the trailing partial block reads its own staged rows only via the
        grow < rows guard; the load itself is skipped then).
        """
        total_q = self.br * quads_row
        r = e // quads_row
        q = e - r * quads_row
        grow = row0 + r
        ok = (e < total_q) and (grow < rows)
        i0 = Int32(-1)
        i1 = Int32(-1)
        i2 = Int32(-1)
        i3 = Int32(-1)
        if ok:
            # 16B-aligned: row stride 2052 words = 513 x 4, q*4 words offset
            p = mLI.iterator + (grow * packed_w + q * 4)
            p16 = cute.make_ptr(p.dtype, p.toint(), cute.AddressSpace.gmem,
                                assumed_align=16)
            rQ = cute.make_rmem_tensor((4,), Int32)
            gQ = cute.make_tensor(p16, cute.make_layout((4,)))
            cute.autovec_copy(gQ, rQ)
            i0 = Int32(rQ[0])
            i1 = Int32(rQ[1])
            i2 = Int32(rQ[2])
            i3 = Int32(rQ[3])
        return i0, i1, i2, i3

    @cute.jit
    def _insert_lane(self, key_ptr, lo_ptr, hi_ptr, bt_ptr, mBT, req, idx, v,
                     r, page, bt_width, num_pages):
        """Exact single-token insert fallback (non-conforming quad packing)."""
        if v:
            lpage = idx // page
            ok = lpage < bt_width
            if ok:
                phys = Int32(-1)
                if req == 0:
                    phys = (bt_ptr + lpage).load()
                else:
                    phys = Int32(mBT[req, lpage])
                ok = (phys >= 0) and (phys < num_pages)
            if ok:
                g = idx >> 2
                sh = r * 4 + (idx & 3)
                keyval = ((req << _G_BITS) | g) + Int32(1)
                h = _hash_key(keyval)
                placed = False
                while not placed:
                    old = cute.arch.atomic_cas(
                        key_ptr + h, cmp=Int32(0), val=keyval,
                        sem="relaxed", scope="cta")
                    if (old == 0) or (old == keyval):
                        if sh < 32:
                            cute.arch.atomic_or(lo_ptr + h, Int32(1) << sh,
                                                sem="relaxed", scope="cta")
                        else:
                            cute.arch.atomic_or(hi_ptr + h, Int32(1) << (sh - 32),
                                                sem="relaxed", scope="cta")
                        placed = True
                    else:
                        h = (h + Int32(1)) & Int32(_SLOT_MASK)
