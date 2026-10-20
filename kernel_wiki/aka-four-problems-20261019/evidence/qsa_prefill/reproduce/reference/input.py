from __future__ import annotations

import base64
import hashlib
import json
import zlib

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Fitted step-15 selection law
#
# ``index_pattern='fitted_step15'`` synthesises the compression-group ids that
# the timed shapes score.  The law below was fitted to two lossless production
# step-15 CP-rank captures and stores *distribution statistics only*: relative
# recency offsets with per-offset selection frequencies, a frequency-binned
# row-to-row persistence chain, a per-row active-count law, sink/page budgets
# and a column-order law.  It stores no production page id, group id, hotspot
# identity or tensor value; the four position-anchored sink groups are
# re-placed at synthetic offsets derived from the caller seed on every draw.
# ---------------------------------------------------------------------------
_FITTED_LAW = json.loads(zlib.decompress(base64.b85decode('c-rk;*=`*-68#rF_gs+0J-=cEh9_|{#vroCPLPMd|32pw*=+W5H%STNKxpYM>?T>PjYaa*{o}zreD}x0k9alH!;e2ceD`3d*Xx!mJ-Y?}<flYm9sc9`PI|#k)}`*AAuMFXv()P~w11hvcr#bGgqE{Uir=(qb1iUgH^gLW72GsF-6dj~YleB?VHvYOPoLTI)cKK3b4{ltn&g(IB>(01Ah$ggyXnM(N`9pl=0X>F5;I6VUeN>Zc5S3%wYqNeRc$&^os89B{2@~=S)+8J$(?MEXM_M|%g`zG&-pz$bDbQLDyEBL>C&d$A?{30g@vw^p!Dg5^-Z4aDP%BATSqI~F_|@&+=WM<U@3=+F#Cm){p$q#di@<PX%^tlh5#&geM&p$gZ)i&^>gB+7rOA5YmZ^>x{a-RM;K=va;4r+c{L6h+~uK<i0xsrY#nBZI!L#R9^^#^=FpB?+IpBCetP@q7smU>z}3CJe7i3xF$Oe~PlzTN1hTM{@;4>76hB4Ga2cvCZYm;kIgAT|LZO8XPv1F?0y`B%@Caj~$(z=O$BdgYti2TXlrez#lJYH&dPg7O)%-Y=GWjA?#!@3NC@m8{WrmB_JH*2wGBgqOQz4_7w6H10#P+~lwRD+GbDaUaD8(QuLn#6qH;1tiB>5>6+)i;yXe=AlvIrbYzAi^Rg=I+&=mi5XjLGpmp|->QfD**SImX+xB22*~(2g*bEr2>{@m5<efK@{I7W_lgJ%%bACj2tH<zOx;O9H*(*%}2<a+orV2GkiAgK?xYqgUL3lnQU~(&0dcm<J##hc0yFsP7DQhbsUdit-^B5R&s0px;s`rR1uPk+c%v_3@;OT)!lY)&*3D!p1}i$ufY9(b@=>Ku)+HxON0l=nu38k6;?j2Z|7K@{#HTK166#F&td19l|pN%lX-I1T-+qg$Bop#=Qy3LMjp0yqw{39Wz8pZSo!uOyoBCY#0VRm=~z^pvGcco`g})92E!St;;&_D%q_BGIX#6ECmc6f@+jN&QT5+%H*>yPZDKMWycUfPF*rWF8l@>BU#E72EOXjOW1m@?NQy8d|qg;Koo*yhzDN+mhC>uVoqdn02gSS5DMDo&<d5fs905kRBp5+rH1|0q`;blqSRzY;*@9T#k5hE>n5$5WYv!(N!lCOi%-$=pTPpZLq>=iN{CN)xM;#B)o%>j*-MX#8rbM~(6^8JK0cQQZwoVm7I`PCG+FG;q-C(0!=Gqu8hx3zX=C(XRm;aRb_jVndOmV1(A~j^Cp7?<Rc;VfAk(P7417|RKXn>|kh(xLjO*gmB+We;XR23-i5ltEmstjBrV@qu!c_{n0bsRTtL;H!tyF3aH)^uRj74RcEIkkUaG|a$VW)|1K>f6*yNGAzF=`)~Ku1b-XGa0%Bc2uwZezU4H+nxh5Xhmx6?mE#XB^0#&NtIHWu05IC&8$kFbawdyulDT4Tn+c3DlyB)XwOXvRa3b1o8|1NM|Bi3I14_UX=4rj^?t0?TQ#wa|oX94%Veia<mHFH>+BK0R{}z(Z^W!?x)nZ((on2wmI!+Wk?=w3k?<Wib1!iq-n*G<EC?)>Nu<F(U^q}))%XK!uA3fli?x~T63yDR+Udj!3hhq6ul`PPjD24V&4v~xjobfn6c8z`Vu4s$16Hz)9GRS0`1VrH#F6huKA*|Q)@Rr*&U#RM7@HpQy1>4ETjw+sj?T&U?_}WPtffzfgtGDJk*pL6`?oLv_4A6OXOgyY&oI9mg%rl+;J~brrD|W=T)G_wvz7CQr?`>y-tY<Qqg|YX~MjqSPW9=1BZ6Ocj1J>c_r)qA>D-a)&HUi)KgtnoAs&)*PQGuChZYAlhvOKR3x)l<hY(IbtZ7B?)zxW$hphVon|U5SmenZdI~pp6J_P4v&uDdmZmW1@!N{8PKQnx@t$+t<Xe0wx^!Rec(v|0HQ&S_K}+YgqwAk!^8M3&m)*RtFQ@Gux4ckr?Q%+=>-4${Jzk=5U2<u|`<~X%_h@gR$lf`Ydq;YI3&$IrGR<?jX+~<>x8?KpyXkf8s`k>xx~`qfnV#*T<ji3k`xeG0#ujbASA$Hu`XswH%%<1l60>S`|CAm2wL5NYd2BY@PfMw+HmR=bz<Ts%)gQdd!}Rd)504*zdid^v(eSr+h34;`+fI##oVFA%wpUl>VN*Ard3Iaw2c2@wi`z|msd;SI_803o?MAnZ?hC|e4sF|Vsg-fQaJ$~;r<Z+8v+H}Na@L=!bEZ8R8*OHwZIQ*gf9X4<p|xQ2e$dxc!+i~LgI>Nx@Uh+V`ps(#vGq{r1h1o*`4{H>k;d;zU#;cZF_CXG$ZYA$Y+{;qxHd!D!P2UnLLXn1N^G;+cBZSf=efF4IDelnr!>`BI<-04cG<Zu6W8lW_f1uvq3sBz1L?M?zH!uDw2jJrx6$O--{VH({Z8Jq%<abW+2P;!zGUF2=|0IdS=t_<tUbGF_hK8-=ua2vnc+ZgJ^Rd@n5midIjlx!2LE5_nRUZ8t@2HJ(n@CT&KkeyzZ+(E+v?b^5z-bmBhS~;GflpsQ{dYu=K07?%W~E=t@N$8`A+DX9<?>sEG4ukF?M|q0|m9MFpS2wXJa4Db^kc)|Nfq9l8B9-+TJYPnHR1}DGZzns!5^KX}V^FDYP+)j&|`BPM?$JX>Dt9=4m>#9y@rOSvs{%-!vt(m^2NxGJ6<v_qCt%&Fl4gy<V@^_Cloo>*sfmUmm`D^9El}B}VVz|4#P(zfAakdJ|Y^gUN7HCu2<Z7Nx>xeF<53-$-`Rqu6qf?G#etDTX;&BO$X9-FrJF@$hCRk$^vxMlH=Lh-108cr2m3Xn9E`@#N?fLWr52a-Rm}rDXjO#hZ}6|9*OS_xQtqK7M}t!^htqe|`Hu3<cz#;8f4P7QR{#j>?a3zkGQ7@7te0fBfwe*x;uJ{r>6Q&+qZowy$44y?fMezde5V_3fu$u(WxD`XtXfCExm7c@()bgLm#0kvGI{;oWj)lJC)Ix%E6=;=8ZqUPo&z&o%Vj*LUM8>$$tl_w1DCCOI(W;q&{SKm78ZK;6$De+R+l>(}YQz5;ijcEuNH!@t;5{t|ljWx8`ugz+}Gh`9swoHjc`x*iOF60+A$=TWYDAklHW&g(hC8r+x1frW4@sN1eVysPSMM0zUia=r>C)>oeBSj8(?jbYaq;krJ_JX_ae++nlvx?E#D9q;nZT|C}z;kEO^Z|rp6{-zYMtxqIf256=5#JKN6neYU(bXv2)#j`MJ*($*XCoKa;+UK+2(aP7fFbMUE9(FGn-yC!arK=Ga+>8n3DFzSfY*>WmcxXKGSJGeZ=j4X_JfLiv;_iTAFV8yD3XYtXbX6M|p6`cU^AMlMjC<7e%DA3@xh{T@SgMMa{1Z{w=@JHep)=@?h&jSC_nJwWgivEW=-EscUZ)teWgd(^53%hUUfYqiYqlPs#{1527?I-7*VN~Ly3>;Z-R{qYbEAW#hw(&oWOZ1sYMwh{L-T-Bss2!H%pDnvCnSL1N>zQgG8wl_XuaXAvy5&REnM{nqmNfKMxF`E4L5^-zwJwIDnI6gLXMV9FL{Dn`tvXMLX*)qdcDG^|2Z$jEqOu_Ezua^Z;YVsfRk*T0(!j!#SSgF-28kTHJWgaNFM}nF<huk=t|BN7W)(^b>4sr&%OkO;_70kFwhrbCE+HE<S8a%#t@|AZn}=9MkS9*oP}e>xe-7oc0pGW(B-3|f-01`AB}{5oDw;a6Q4xzq159A_8k;6Y73E~2r|(FH*~C1KN!72@vcc1_0S@5;<j)$7pFMAM06D~jEZ|Cs8?>&7El-Ukc%KjAuy0pCL<HZelgXv6Oc_iAOMY!aH4g+YK}r3B~@tjC0)t{gO$Gfh<*swqm-#5)+DZ=e9{O6w5JBKv&r`pOM(+2OG2=0aP#qRv<))LB_?%qKFh8+yqpL~%Zd??9EPquQm*({tr;OPm(h8JPG$;?5qg<YgNW4*igHzoRh(l8#IRA}MDuEKE257SiBt><QdAsZrb<QvXl;|9BCs*u$Xd#J!n%9~VPkcvm_)3(b5PWw<)WGRW2l|CS5JjhAX$a!xZyda7u_}%(qMHtK&Yco<_ZhfwW8t&3E7r^<equ9?kLM9#|jXH?&ZuiQw_mQJvWTZ8L$<JtVf5TX0>y%NBrodxXGd?jGl~~fyGgwr<P4+dZMH)tNK8>A_45`h+UI9`-A$V88N^<Q$OX6e;QYWi2#athoZxneTB|?H69p^zwXs?otj{Bkyvb=n9>pmjSaq<fN)X^)l!~3H&T0f@Cvt+O1Lr;q!qM1Gq=Pla(6?y+B3246`Gr(eoS{vd-R^Ef>>xjRa|`$Nz~^|EIaa|yOfbh7N#Q+G=-2^3V6@VeNnV4*Y9*@X$Bq4>L8)wiK8o()ReVPDpA(c`l3s~`bz{9v8(7_cEYz1bS}&^{@|J#235+s5@f!NJ;Pn1R;9IQ8kn0?Ho^9&Y7_wv^Juv}q!o7NL#|K&dQ@Lg#hiC2ESeI&dv*JPDujW#WH*zlX5q-B8zU@=3Sn2cvERGWRP%Vi?m{bNiUDkto2Q(|ttyhcJi8ucM<Kk((Bh;30mHpR%(@s*$aQCthhl*~1@d7ENTlBBMwY9vbmoqM|E_JwC<nzv4g3i8cZHca8XRb%5}I)IL=}l}i%rLzY6?Zt*%RK1GY}n|typ;GrfGUgWL_N}W>OtDQWNxsqhQJt^Bigzw9^2z6r#Qd?l#I8Iz$a-R25rDC883xF)EDPU%^0FYZo|lm%?+v@j>Gg!-OOWcQzwbW}0YbIHi#f&mh&pbM+tIScQBCCOf1aK*u5KGZpvmG@*=J={wzGG~yUWVw~BBRVdD+f@=QE8;eY%B5QOD&oNzw5^W*MSTj)bnGKb+f+Yj8Iv2qm8fjKX!x#yY^f$xd6H_TbMKHSUWHeyvd5gQO%&21I>2H}62R*8?LPf&|@Rm81mT4w3XB5P|$uMPNQe;?EqeF<Y9&c>1kamL@JHn(LW0sE*y>^9SJG#HeEN@Hp80Rgb+N@CLBd%?aSRV1cdm79J|9d~a+;s$^Z`V1LzT12|_mmsZ`&&G+>^9D2g_}D<g`3?cv)P!E)xA9$d(?nqeEA}a^>@3Y?sCR}TeWL;d*gDbv0Gi)V(87yGBvBd<cwCH!;;Gt2_5BuoO^QJ?<l8N?D^7?>n&ZzR{9-cJ075N&!f*}*zye>^&Sq8kgnt57nJWl!(mR;iw#d)H&on!lHKFbW4|LEy?qXcZW@3=Ix5!;sN5MreuP9{7Qms8N7`r?%=}D4--YOC<?(K8_*?1d^omyBqtjh$ZSH4{YmXgwUAYq5lYa1&*Ab&5cKv0y3g(}IIl{sJ^R5M3yhwA_@@1wn!+anepS{e&718e+rZ_fSvzg4n%x-sc(aT?BmVg;h(?@v%qvl=C6ObG81m>R+I?5B+qMer<mTiW>7~jkp0={L$Y{xq01=ubh(~;g|q~uq&?}xmWZ8m^3oAv$kIupP*?>OgH?lS@ApIbV%Q``O8`e)3-h3up`<3;-CF)v`wR`A`Lm)w9^e?!j`xWzh-^9S4=`2&lb=PZKj+MBM-sF(8#q~$=YvkQFBmx)~-*Rl)FWfsh%o-+&FkXayyc?F>#JcLzVL1=jeVbJfF$}4!iUa!~dTR`!L==lQ}8x+5_AyyQBF$;=6MIJz%jHD9s0OB{?HztVRXX6(TzZLNWs*5urlmW&clVLmFNbqHV@u#5A2N$K}5ysCSnmB<_ktc!pJCy$=K>U)7QxJd6IWW<PvdS5!sCzN4qV-cl<%fpacQLj16`0@EH~!d+V)>n~7=B;hQ>hsKNRIC?{JC@E^SfVB4!ZbD7jxmK_qg%#cV&_-{=`E#f8uGJJ>?1FIdSV-KPP+Qp;1;QDL3ZYWU$jcJl(hDUb)xb$MJkr*;PBYXde`Q`S4?9r*<pu#4XJe&ni#cn&pc-@rOE2x{0eDR&mrGdjneEo%kylC+>lR%yQy(o;xMqnDO!P?N6WI{lJksUm)K1zYm?WZ+F5ZcH+j~i3jR<@d;mJ(2hU9|Lx2B6}=>|zBqCxvWW<K!6wY5Xz}~6U%&nzJRq4<')).decode())


_FITTED_INDEX_SEED_MAX = 1 << 62
_QSA_QPROBS7 = np.array([0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
_QSA_REGIME_PAGE_Q = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
_QSA_FITTED_ROWS = 4096
_QSA_GROUPS_PER_ROW = 512
_QSA_COMPRESS_RATIO = 4
_QSA_PAGE_GROUPS = 32
_QSA_UNIT_SIGMA = 0.42


def _fitted_index_rng_seed(step: int, tp_rank: int, index_seed: int) -> int:
    """Stable private seed for one fitted index draw.

    The law below is public, so the *only* thing keeping a scored index draw
    unreproducible is ``index_seed``, which lives in the withheld
    shape_valid.json.  A content hash is used instead of the harness RNG so the
    draw cannot be recovered from the public deterministic input seed formula.
    """
    tag = f"qwen38-core-qsa|fitted_step15|{step}|{tp_rank}|{int(index_seed)}".encode()
    return int.from_bytes(hashlib.blake2b(tag, digest_size=8).digest(), "big")


def _qsa_pmf_cdf(pmf):
    p = np.asarray(pmf, float)
    p = np.where(p > 0, p, 0.0)
    total = p.sum()
    if total <= 0:
        p = np.ones_like(p)
        total = p.sum()
    return np.cumsum(p / total)


def _qsa_pmf_draw(rng, cdf, size):
    return np.searchsorted(cdf, rng.random(size), side="left").astype(np.int64)


def _qsa_anchors(spec):
    """Rebuild candidate recency offsets, frequencies and transition rates."""
    deltas = np.fromiter((int(x) for x in spec["dd"].split(",")), np.int64)
    freq_pm = np.fromiter((int(x) for x in spec["f_pm"].split(",")), np.int64)
    ages = np.empty(deltas.size + 1, np.int64)
    ages[0] = int(spec["d0"])
    ages[1:] = int(spec["d0"]) + np.cumsum(deltas)
    freq = np.clip(freq_pm.astype(float) / 1000.0, 1e-6, 1.0)
    trans = np.asarray(spec["trans"], float)
    bins = np.fromiter((int(x) for x in spec["binof"].split(",")), np.int64)
    return ages, freq, trans[bins, 1], trans[bins, 2]


def _qsa_sink_groups(rng, spec):
    """Place the position-anchored sink groups at synthetic oldest-page offsets."""
    count = int(spec["n_sink_groups"])
    if count <= 0:
        return np.empty(0, np.int64)
    pages = max(1, int(spec.get("sink_pages", 1)))
    page0 = int(rng.integers(0, pages)) * max(1, int(spec.get("sink_span", 1)))
    return np.sort(page0 * _QSA_PAGE_GROUPS
                   + rng.choice(_QSA_PAGE_GROUPS, size=min(count, _QSA_PAGE_GROUPS), replace=False))


def _qsa_weighted_take(rng, cand, k, w):
    """Weighted sample of ``k`` entries without replacement (Efraimidis-Spirakis)."""
    if k <= 0 or cand.size == 0:
        return np.empty(0, np.int64)
    keys = rng.random(cand.size) ** (1.0 / np.maximum(np.asarray(w, float), 1e-9))
    if k >= cand.size:
        return cand
    return cand[np.argsort(keys)[-k:]]


def _qsa_order_units(selected, unit_cdf, rng):
    """Residue-mod-4 ascending units visited in a noisy (partially sorted) order."""
    n = len(selected)
    lens = np.maximum(1, _qsa_pmf_draw(rng, unit_cdf, 4 * n))
    units = []
    cursor = 0
    for residue in range(4):
        run = np.sort(selected[selected % 4 == residue])
        m = len(run)
        if m == 0:
            continue
        take = lens[cursor:cursor + m]
        cursor += m
        csum = np.cumsum(take)
        end = int(np.searchsorted(csum, m, side="left")) + 1
        csum = csum[:end]
        csum[-1] = m
        units.extend(np.split(run, csum[:-1]))
    if len(units) < 2:
        return np.sort(selected)
    keys = np.array([u[0] for u in units], float)
    rank_by_id = np.empty(len(units), np.int64)
    rank_by_id[np.argsort(keys)] = np.arange(len(units))
    noisy = rank_by_id + rng.normal(0.0, _QSA_UNIT_SIGMA * len(units), len(units))
    order = np.argsort(noisy)
    out = np.concatenate([units[i] for i in order])
    if len(out) != n:
        out = np.concatenate([out, np.setdiff1d(selected, out)])
    return out[:n]


def _qsa_regime_explore(rng, cap, need, regime, pool):
    """Page-granular clustered exploration used by the trailing regime rows.

    Production regime rows neither sample the visible history uniformly nor
    explore fresh pages: measured against the footprint of the preceding rows
    they add only a handful of new pages each, so the residual page budget is
    split between reusing the existing footprint and a fitted number of new
    pages.  Dropping the reuse term inflates the launch-level footprint well
    past production.
    """
    if need <= 0:
        return np.empty(0, np.int64)
    gpp_q = np.asarray(regime.get("groups_per_resid_page_q") or [3.14, 7.77, 8.85, 9.72, 13.83], float)
    pages_q = np.asarray(regime.get("resid_pages_q") or [23, 24, 29.8, 32, 33.5, 77.9, 101], float)
    new_q = regime.get("new_pages_q")
    reused_q = regime.get("reused_pages_q")
    region_frac = float(regime.get("region_frac") or 0.35)
    total_pages = max(1, cap // _QSA_PAGE_GROUPS)
    n_pages = int(round(float(np.interp(rng.random(), _QSA_QPROBS7, pages_q))))
    gpp = float(np.interp(rng.random(), _QSA_REGIME_PAGE_Q, gpp_q))
    n_pages = max(1, min(n_pages, int(round(need / max(1.0, gpp))), total_pages))

    n_new = n_pages
    if new_q and reused_q:
        a = float(np.interp(rng.random(), _QSA_QPROBS7, np.asarray(new_q, float)))
        b = float(np.interp(rng.random(), _QSA_QPROBS7, np.asarray(reused_q, float)))
        n_new = int(round(n_pages * a / max(1e-9, a + b)))
    n_new = max(0, min(n_new, n_pages))

    reused = np.empty(0, np.int64)
    if n_pages - n_new > 0 and pool is not None:
        cand = np.flatnonzero(pool[:total_pages])
        if cand.size:
            reused = cand[rng.choice(cand.size, min(n_pages - n_new, cand.size), replace=False)]
    n_fresh = max(0, n_pages - reused.size)

    fresh = np.empty(0, np.int64)
    if n_fresh > 0:
        n_regions = max(1, min(n_fresh, int(round(n_fresh * region_frac))))
        per = rng.multinomial(n_fresh, np.ones(n_regions) / n_regions)
        starts = rng.integers(0, total_pages, n_regions)
        fresh = np.unique(np.concatenate([
            np.arange(st, st + int(c)) % total_pages for st, c in zip(starts, per) if c > 0
        ]))
        if fresh.size > n_fresh:
            fresh = np.sort(fresh[rng.choice(fresh.size, n_fresh, replace=False)])
    pages = np.unique(np.concatenate((reused, fresh)))
    if pages.size == 0:
        return np.empty(0, np.int64)
    shares = rng.multinomial(need, np.ones(pages.size) / pages.size)
    parts = []
    for page, cnt in zip(pages, shares):
        cnt = int(min(int(cnt), _QSA_PAGE_GROUPS))
        if cnt <= 0:
            continue
        off = int(rng.integers(0, _QSA_PAGE_GROUPS - cnt + 1))
        parts.append(int(page) * _QSA_PAGE_GROUPS + off + np.arange(cnt, dtype=np.int64))
    if not parts:
        return np.empty(0, np.int64)
    return np.concatenate(parts)


def _qsa_fitted_groups(tp_rank: int, index_seed: int, rows: int) -> np.ndarray:
    """Synthesise the step-15 selected compression-group ids for one CP rank.

    The law is a *relative* description of the measured production selection:
    candidate recency offsets with per-offset selection frequencies, a
    frequency-binned two-state row-to-row persistence chain, a near-constant
    per-row active count, four position-anchored sink groups re-placed from the
    caller seed, a unit-run column order, and a trailing regime block.  No
    production page id, group id, hotspot identity or tensor value is stored or
    reproduced here.
    """
    spec = _FITTED_LAW[str(tp_rank)]
    rng = np.random.default_rng(_fitted_index_rng_seed(15, tp_rank, index_seed))
    base = 15 * 8192 + tp_rank * 4096
    completed = (base + np.arange(rows) + 1) // _QSA_COMPRESS_RATIO

    ages, freq, p_stay, p_turn = _qsa_anchors(spec)
    n_anc = len(ages)
    count_q = np.asarray(spec["anchor_count_q"], float)
    sink_groups = _qsa_sink_groups(rng, spec)
    regime = spec.get("regime") or {}
    n_regime = int(regime.get("rows") or 0)
    regime_rate = float(regime.get("anchor_rate") or 0.0)
    unit_cdf = _qsa_pmf_cdf(spec["unit_pmf"])

    state = rng.random(n_anc) < freq
    out = np.zeros((rows, _QSA_GROUPS_PER_ROW), np.int64)
    pool = np.zeros(max(1, int(completed[-1]) // _QSA_PAGE_GROUPS) + 1, bool)

    for r in range(rows):
        cap = int(completed[r])
        is_regime = bool(n_regime) and r >= rows - n_regime
        if is_regime:
            active = rng.random(n_anc) < regime_rate
        else:
            if r:
                u = rng.random(n_anc)
                state = np.where(state, u < p_stay, u < p_turn)
            target = int(round(float(np.interp(rng.random(), _QSA_QPROBS7, count_q))))
            target = min(max(target, 0), n_anc)
            n_act = int(state.sum())
            if n_act > target:
                cand = np.flatnonzero(state)
                drop = _qsa_weighted_take(rng, cand, n_act - target, 1.0 - freq[cand])
                state[drop] = False
            elif n_act < target:
                cand = np.flatnonzero(~state)
                add = _qsa_weighted_take(rng, cand, target - n_act, freq[cand])
                state[add] = True
            active = state
        usable = active & (ages < cap)
        selected = cap - 1 - ages[usable]
        if sink_groups.size:
            selected = np.concatenate((selected, sink_groups[sink_groups < cap]))
        if is_regime and selected.size < _QSA_GROUPS_PER_ROW:
            selected = np.concatenate(
                (selected, _qsa_regime_explore(rng, cap, _QSA_GROUPS_PER_ROW - selected.size, regime, pool))
            )
        selected = np.unique(selected)
        selected = selected[(selected >= 0) & (selected < cap)]
        if selected.size > _QSA_GROUPS_PER_ROW:
            keep = np.sort(rng.choice(selected.size, _QSA_GROUPS_PER_ROW, replace=False))
            selected = selected[keep]
        elif selected.size < _QSA_GROUPS_PER_ROW:
            need = _QSA_GROUPS_PER_ROW - selected.size
            lo = max(0, cap - 1 - int(0.02 * cap))
            rest = np.setdiff1d(np.arange(lo, cap), selected)
            if rest.size < need:
                rest = np.setdiff1d(np.arange(cap), selected)
            take = np.sort(rng.choice(rest.size, min(need, rest.size), replace=False))
            selected = np.sort(np.concatenate((selected, rest[take])))
        if selected.size != _QSA_GROUPS_PER_ROW:
            raise RuntimeError(f"fitted_step15 row {r}: built {selected.size} groups")
        pool[selected // _QSA_PAGE_GROUPS] = True
        out[r] = _qsa_order_units(selected, unit_cdf, rng)
    return out


def _qsa_fitted_tail(tp_rank: int, rows: int) -> np.ndarray:
    """Deterministic 0..3-token incomplete causal tail per row."""
    base = 15 * 8192 + tp_rank * 4096
    positions = base + np.arange(rows)
    visible = positions + 1
    completed = visible // _QSA_COMPRESS_RATIO
    offs = np.arange(_QSA_COMPRESS_RATIO - 1)[None, :]
    tail = completed[:, None] * _QSA_COMPRESS_RATIO + offs
    tail = np.where(offs < (visible % _QSA_COMPRESS_RATIO)[:, None], tail, -1)
    return tail.astype(np.int64)


def _fitted_step15_indices(
    positions: torch.Tensor,
    *,
    step: int,
    tp_rank: int,
    selected_tokens: int,
    index_seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand fitted step-15 group ids plus the causal tail into packed indices."""
    rows = positions.numel()
    if step != 15:
        raise ValueError("the fitted index law is measured on scheduler step 15 only")
    if tp_rank not in (0, 1):
        raise ValueError("the fitted index law covers the two captured CP ranks")
    if rows != _QSA_FITTED_ROWS:
        raise ValueError("the fitted index law is measured on 4096 CP-local rows")
    if not isinstance(index_seed, int) or isinstance(index_seed, bool):
        raise ValueError("index_seed must be an integer")
    if not 0 <= index_seed < _FITTED_INDEX_SEED_MAX:
        raise ValueError(f"index_seed must be within [0, {_FITTED_INDEX_SEED_MAX})")

    groups = _qsa_fitted_groups(tp_rank, index_seed, rows)
    tail = _qsa_fitted_tail(tp_rank, rows)
    offsets = torch.arange(_QSA_COMPRESS_RATIO, dtype=torch.int64)
    expanded = torch.from_numpy(groups)[:, :, None] * _QSA_COMPRESS_RATIO + offsets[None, None, :]
    expanded = expanded.reshape(rows, _QSA_GROUPS_PER_ROW * _QSA_COMPRESS_RATIO).to(torch.int32)
    logical = torch.cat((expanded, torch.from_numpy(tail).to(torch.int32)), dim=1)
    logical = logical.to(device=positions.device).contiguous()
    if logical.shape[1] != selected_tokens:
        raise ValueError("fitted expansion width does not match the declared selection width")
    if bool((logical > positions[:, None]).any()):
        raise RuntimeError("fitted indices violate the causal boundary")
    valid_count = (logical >= 0).sum(dim=1, dtype=torch.int32)
    columns = torch.arange(selected_tokens, device=positions.device)
    if not torch.equal(logical >= 0, columns[None, :] < valid_count[:, None]):
        raise RuntimeError("fitted indices are not a packed valid prefix")
    return logical, valid_count



def _make_inputs(
    step: int,
    tp_rank: int,
    sequence_length: int,
    row_start: int,
    q_shape: list[int],
    cache_shape: list[int],
    cache_stride: list[int],
    kv_offset: int,
    logical_indices_shape: list[int],
    block_table_shape: list[int],
    token_to_req_shape: list[int],
    chunk_tokens: int,
    softmax_scale: float,
    page_mapping: str,
    request_pattern: str,
    index_pattern: str,
    index_seed: int = 0,
    correctness_empty_row: bool = False,
) -> dict[str, object]:
    if not 0 <= step < 16 or tp_rank not in (0, 1):
        raise ValueError("the captured contract is 16 prefill steps on TP2/CP2")
    if len(q_shape) != 3:
        raise ValueError(f"q_shape must be rank 3, got {q_shape}")
    if len(cache_shape) != 4 or len(cache_stride) != 4:
        raise ValueError("cache_shape and cache_stride must both be rank 4")
    if len(logical_indices_shape) != 2 or len(block_table_shape) != 2:
        raise ValueError("logical_indices_shape and block_table_shape must be rank 2")
    if len(token_to_req_shape) != 1:
        raise ValueError("token_to_req_shape must be rank 1")

    local_rows, query_heads, head_dim = q_shape
    num_cache_blocks, page_size, kv_heads, cache_head_dim = cache_shape
    logical_rows, selected_tokens = logical_indices_shape
    block_table_rows, block_table_width = block_table_shape
    if cache_head_dim != head_dim:
        raise ValueError("query and cache head dimensions must match")
    expected_cache_stride = [cache_stride[0], kv_heads * head_dim, head_dim, 1]
    if cache_stride != expected_cache_stride:
        raise ValueError(
            f"cache_stride must preserve the captured K/V layout: {cache_stride}"
        )
    if logical_rows != local_rows or token_to_req_shape != [local_rows]:
        raise ValueError("all row-oriented inputs must use the CP-local row count")
    if block_table_rows != 1:
        raise ValueError("the fitted domain requires one block-table row")
    if page_mapping != "captured":
        raise ValueError(f"unsupported page_mapping: {page_mapping}")
    if request_pattern != "single":
        raise ValueError(f"unsupported request_pattern: {request_pattern}")
    if index_pattern != "fitted_step15":
        raise ValueError(f"unsupported index_pattern: {index_pattern}")
    if not isinstance(index_seed, int) or isinstance(index_seed, bool):
        raise ValueError("index_seed must be an integer")
    if not isinstance(correctness_empty_row, bool):
        raise ValueError("correctness_empty_row must be boolean")
    if sequence_length != (step + 1) * chunk_tokens:
        raise ValueError("sequence_length must match the captured scheduler step")
    if row_start != tp_rank * local_rows:
        raise ValueError("row_start must match the captured CP rank slice")
    if local_rows * 2 != chunk_tokens:
        raise ValueError("CP2 must split each 8192-token chunk into 4096 local rows")
    if selected_tokens != 2051 or page_size != 128:
        raise ValueError("selection width and page size are fixed by the live ABI")

    positions = step * chunk_tokens + row_start + torch.arange(
        local_rows, dtype=torch.int64, device="cuda"
    )

    block_table = torch.full(
        (block_table_rows, block_table_width), -1, dtype=torch.int32, device="cuda"
    )
    active_logical_pages = (sequence_length + page_size - 1) // page_size
    logical_pages = torch.arange(
        active_logical_pages, dtype=torch.int32, device="cuda"
    )
    page_rows = (1352 + logical_pages)[None, :]
    if int(page_rows.max()) >= num_cache_blocks:
        raise ValueError("captured physical cache pool is too small")
    block_table[:, :active_logical_pages] = page_rows
    physical_pages = page_rows.reshape(-1)

    # The production [2, blocks, page, kv_head, dim] cache interleaves K/V
    # inside a much larger physical page record. These views preserve the
    # observed post-unbind strides exactly without transforming the ABI.
    block_stride = cache_stride[0]
    storage_size = (
        (num_cache_blocks - 1) * block_stride
        + (page_size - 1) * (kv_heads * head_dim)
        + (kv_heads - 1) * head_dim
        + head_dim
        + kv_offset
    )
    cache_storage = torch.empty(
        storage_size, dtype=torch.bfloat16, device="cuda"
    )
    k_cache = cache_storage.as_strided(tuple(cache_shape), tuple(cache_stride), 0)
    v_cache = cache_storage.as_strided(
        tuple(cache_shape), tuple(cache_stride), kv_offset
    )
    active_shape = (physical_pages.numel(), page_size, kv_heads, head_dim)
    k_cache.index_copy_(
        0,
        physical_pages.long(),
        torch.randn(active_shape, dtype=torch.bfloat16, device="cuda") * 0.02,
    )
    v_cache.index_copy_(
        0,
        physical_pages.long(),
        torch.randn(active_shape, dtype=torch.bfloat16, device="cuda") * 0.02,
    )

    if correctness_empty_row:
        raise ValueError("fitted_step15 does not support an empty row")
    logical_indices, valid_count = _fitted_step15_indices(
        positions,
        step=step,
        tp_rank=tp_rank,
        selected_tokens=selected_tokens,
        index_seed=index_seed,
    )

    q = torch.randn(
        local_rows, query_heads, head_dim,
        dtype=torch.bfloat16, device="cuda",
    ) * 0.02
    packed_indices = torch.cat(
        (logical_indices, valid_count[:, None]), dim=1
    ).contiguous()
    output_gate = torch.randn_like(q) * 0.25
    token_to_req = torch.zeros(local_rows, dtype=torch.int32, device="cuda")
    return {
        "q": q,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "logical_indices": packed_indices,
        "block_table": block_table,
        "token_to_req": token_to_req,
        "use_prefill_config": True,
        "output_gate": output_gate,
        "out": torch.empty_like(q),
    }
