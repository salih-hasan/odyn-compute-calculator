/* Odyn Compute Calculator engine. Pure functions, no DOM.
   Models: Hugging Face config.json + safetensors (models.js). GPUs: vendor datasheets (gpus.js).
   Rates: SDS price_book_loader.py + public lists. Fit + cost cap: oracle_conformal.py. */
(function (root) {
  const isNode = typeof module !== "undefined" && module.exports;
  const MODELS = isNode ? require("./models.js").MODELS : root.MODELS;
  const GPUS = isNode ? require("./gpus.js").GPUS : root.GPUS;
  const MEAS = (isNode ? (process.env.ODYN_NO_MEASURED ? null : (() => { try { return require(process.env.ODYN_MEAS_FILE || "./measured.js"); } catch (e) { return null; } })()) : root.MEASURED) || { train: [], infer: [] };

  /* calibrated fine-tune fit on our 15 single-GPU runs: tok/s/GPU = exp(level) * N^-P */
  const P = 0.734, W = 0.70, B0 = 5.2932, B1 = 0.8783;
  const LVL = { a100: 9.0118, h100: 9.8507, h200: 9.9475 };
  const OVERHEAD = 1.10, DPO = 2.0;
  // Multi-GPU per-GPU efficiency. Shape: SDS scaling law from our benchmarks (4 GPUs: 8B 0.79, 70B 0.93; loss grows with
  // log n). Level: our 4 multi-GPU runs (70B-class, 2 and 4 GPUs) sit at 0.717, 0.840, 0.780, 0.750 of curve x law,
  // geometric mean 0.771. Above 4 GPUs is extrapolated.
  const MULTI_B = 0.771;
  const scaleLaw = (n, b) => { if (n <= 1) return 1; const base = b <= 8 ? 0.79 : b >= 70 ? 0.93 : 0.79 + (b - 8) / 62 * 0.14;
    return Math.max(0.3, 1 - (1 - base) * Math.log(n) / Math.log(4)); };
  const etaN = (n, b) => n > 1 ? MULTI_B * scaleLaw(n, b) : 1;
  const CAP = { 80: { f: 1.143, cov: "108/131" }, 90: { f: 1.247, cov: "117/131" } };   // conformal, from held-out errors of this engine (odyn-console/live_holdout.js); coverage calibrated without the GPU tested
  const KV_UTIL = 0.90;

  // measured content-token throughput per GPU at seq 2048; [tok/s, gpus in run]
  const ANCH = {
    "a100|qwen3-8b": [1980.82, 1], "a100|qwen3-14b": [1269.16, 1], "a100|gemma-3-12b": [1214.98, 1],
    "a100|gemma-3-27b": [653.27, 1], "a100|mistral-small-24b": [953.17, 1], "a100|llama-3.2-1b": [9457.03, 1],
    "a100|deepseek-r1-distill-qwen-1.5b": [5425.89, 1], "a100|deepseek-r1-distill-qwen-7b": [1952.62, 1],
    "a100|deepseek-r1-distill-qwen-14b": [984.27, 1], "a100|qwen2.5-72b": [236.80, 4],
    "h100|qwen3-8b": [3902.71, 1], "h100|qwen3-14b": [2598.44, 1], "h100|mistral-small-24b": [2046.67, 1],
    "h100|llama-3.3-70b": [655.83, 4],
    "h200|qwen3-8b": [4449.55, 1], "h200|qwen3-14b": [2873.14, 1], "h200|mistral-small-24b": [2170.36, 1],
    "h200|llama-3.3-70b": [696.08, 2], "h200|qwen2.5-72b": [655.03, 2]
  };

  const WQ = { bf16: 2, fp16: 2, fp8: 1, int8: 1, int4: 0.53 };   // INT4 AWQ/GPTQ group-128 is about 4.25 bits
  const KVQ = { fp16: 2, fp8: 1, int8: 1, int4: 0.5 };
    const CUDA_OVH = 1.6;                                         // GB framework/CUDA context per GPU

  const bn = x => x * 1e9;
  const UMA = 0.75;  // macOS lets the GPU wire about 75% of unified memory by default
  // vendors quote memory in binary gigabytes (an "80 GB" H100 holds 85.9e9 bytes); the model counts decimal GB
  const GIB = 1.073741824;
  const vramOf = g => (g.unified ? g.vram * UMA : g.vram) * GIB;
  // speed level: SDS fit for our fleet GPUs; for others, fitted on our RunPod LoRA runs at 2048 tokens per step; else from specs
  const LVLM = {};
  function levelMeasured(gk) {
    if (gk in LVLM) return LVLM[gk];
    const rs = ((MEAS && MEAS.train) || []).filter(r => r.gpu === gk && r.method === "lora" && r.n === 1 && r.batch * r.seq === 2048 &&
      r.ckpt === 1 && r.attn === "sdpa" && r.optim === "adamw" && r.rank === 16 && MODELS[r.model] && !MODELS[r.model].moe);
    return (LVLM[gk] = rs.length ? rs.reduce((a, r) => a + Math.log(r.tok / 1.3) + 0.734 * Math.log(MODELS[r.model].f), 0) / rs.length : null);
  }
  function level(gk) { const g = GPUS[gk]; if (gk in LVL) return LVL[gk]; const lm = levelMeasured(gk); return lm != null ? lm : B0 + B1 * (W * Math.log(g.tf) + (1 - W) * Math.log(g.bw)); }

  // KV cache bytes for one sequence of `ctx` tokens, honouring sliding windows and attention-only layers
  function kvBytes(m, ctx, kvq) {
    const per = (m.kve || 2 * m.kv * m.hd) * (kvq || 2), sw = m.sw || 0, L = m.attnL || m.L;
    const eff = (1 - sw) * ctx + sw * Math.min(ctx, m.win || 1024);
    return per * L * eff;
  }
  // weight-read bytes per decode step; MoE reads more experts as the batch grows
  function weightReadGB(m, wb, B) {
    if (!m.moe || !(m.k < m.E)) return wb * m.p;
    const kE = m.k / m.E, shared = Math.max(0, (m.a - kE * m.p) / (1 - kE)), touched = 1 - Math.pow(1 - kE, B);
    return wb * (shared + touched * (m.p - shared));
  }

  /* ---- inference ----
     Calibrated on NVIDIA NIM published results (Llama-3.1-8B/70B on H100, H200, L40S) and llama.cpp on RTX 4090:
     - decode step = max(bytes read / (bandwidth x eff), compute) + 1.4 ms fixed per step (launch, sampling, scheduling)
     - eff 0.745 for BF16/FP8/INT8, 0.56 for INT4 (dequantization overhead)
     - tensor parallel: 2 all-reduces per layer, ~36 us each over NVLink (fit on 70B TP4); PCIe or cross-node links slower
     - prefill (incl. attention FLOPs) at 58% of the precision peak (FP8 tensor cores double it) + 7.5 ms;
       in steady state the prompts feeding each decode step also run at 58%, decode tokens at 16% (fit: NIM 250 users, TRT-LLM max) */
  const DEC_EFF = { bf16: 0.745, fp16: 0.745, fp8: 0.745, int8: 0.745, int4: 0.56 };
  const T0 = 1.4e-3, TTFT0 = 7.5e-3, PRE_MFU = 0.58, DEC_MFU = 0.16;
  const AR = { nvlink: 36e-6, pcie: 90e-6, node: 120e-6 };            // per all-reduce latency (NVLink fit; others estimates)
  const LINKBW = { nvlink: 300e9, pcie: 25e9, node: 50e9 };           // effective all-reduce bandwidth for prefill activations
  const LINK = { cpu: 25, nvme: 7 }, PP_EFF = 0.9;
  // attention matmul FLOPs for one causal prompt of s tokens (QK^T and AV), honouring sliding windows
  function attnFlops(m, s) {
    const L = m.attnL || m.L, sw = m.sw || 0, w = m.win || 1024, per = 4 * (m.nh || 32) * (m.hd || 128) * L;
    const full = s * s / 2, win = s > w ? s * w - w * w / 2 : full;
    return per * ((1 - sw) * full + sw * win);
  }
  // GPU count to layout, as vLLM deploys it: tensor parallel inside one box, one pipeline stage per extra box.
  // ngpu 0 = auto: the fewest GPUs that hold every requested user, else the fewest that fit at all.
  function inf(gk, m, o) {
    if (o.ngpu === undefined) return inf1(gk, m, o);
    const g = o.gpu || GPUS[gk], box = o.gpn || g.maxN || 8, lay = N => ({ ...o, tp: Math.min(N, box), pp: Math.max(1, N / box), gpn: box, link: "auto" });
    if (o.ngpu > 0) return inf1(gk, m, lay(o.ngpu));
    const opts = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64].filter(N => N <= (o.maxGpus || 64) && (N <= box || N % box === 0));
    let first = null;
    for (const N of opts) { const r = inf1(gk, m, lay(N)); if (!r.fits) continue; if (r.cmax >= Math.max(1, o.C) * (o.batch || 1)) return r; first = first || r; }
    return first || inf1(gk, m, lay(opts[opts.length - 1] || 1));
  }
  function inf1(gk, m, o) {
    const g = o.gpu || GPUS[gk], wq = o.wq || "bf16", wb = WQ[wq] || 2, kvq = KVQ[o.kvq] || 2;
    const tp = o.tp || o.gpus || 1, pp = o.pp || 1, N = tp * pp, gpn = o.gpn || tp;
    const ctx = o.pin + o.pout, users = Math.max(1, o.C), batch = users * (o.batch || 1);
    const wGB = wb * m.p, seqGB = kvBytes(m, ctx, kvq) / 1e9, ovh = CUDA_OVH * N;
    const usable = vramOf(g) * N * KV_UTIL;
    let offGB = 0, budget = usable - wGB - ovh;
    if (budget < seqGB && o.offload) { offGB = Math.min(wGB, seqGB - budget + wGB * 0.02); budget = seqGB; }
    const cmax = Math.max(0, Math.floor(budget / seqGB));
    const r = { gk, N, tp, pp, wGB, seqGB, cmax, offGB, gpuName: g.name };
    const have = o.maxGpus || g.maxN || 8;   // GPUs one box can hold unless the caller sets up a cluster
    if (N > have) return { ...r, fits: false, over: true, why: "needs " + N + " GPUs, this setup has " + have };
    if (cmax < 1) return { ...r, fits: false };
    const B = Math.min(batch, cmax);
    const link = o.link && o.link !== "auto" ? o.link : tp > gpn ? "node" : hasNV(g, tp) ? "nvlink" : "pcie";
    const peak = g.tf * 1e12 * (wq === "fp8" && g.fp8 ? 2 : 1) * tp;       // FP8 weights use FP8 tensor cores
    const bw = g.bw * 1e12 * (DEC_EFF[wq] || 0.745) * tp;
    const Lyr = (m.attnL && m.attnL < m.L ? m.L : m.L), tpLat = tp > 1 ? 2 * Lyr * AR[link] : 0;
    const off = offGB > 0 ? offGB * 1e9 / ((LINK[o.offload] || 25) * 1e9) : 0, kvAvg = kvBytes(m, o.pin + o.pout / 2, kvq);
    // one decode step: every in-flight sequence advances one token; pp stages each hold 1/pp of the layers
    const stepT = c => {
      const s = c / Math.min(c, pp);
      const tm = ((weightReadGB(m, wb, s) - offGB) * 1e9 + s * kvAvg) / bw + off;
      // decode tokens run at the batched-decode efficiency; the prompts that feed them run at prefill efficiency
      const tc = s * (2 * bn(m.a) / (peak * DEC_MFU) + (o.pin / o.pout * 2 * bn(m.a) + attnFlops(m, o.pin) / o.pout) / (peak * PRE_MFU));
      const t = (Math.max(tm, tc) + T0 + tpLat) / (pp > 1 ? PP_EFF : 1);
      return { tm, tc, t };
    };
    const st = stepT(B);
    const preComm = tp > 1 ? 2 * Lyr * (o.pin * m.h * 2) / LINKBW[link] + 2 * Lyr * AR[link] : 0;
    let ttft = TTFT0 + (o.pin * 2 * bn(m.a) + attnFlops(m, o.pin)) / (peak * PRE_MFU) + preComm;
    // our vLLM measurements: exact match overrides; otherwise a per-GPU correction fitted on our runs
    const runs = [];
    if (!o.raw && !o.gpu) {
      const key = keyOf(m), q = wq === "fp8" ? "fp8" : wq === "bf16" || wq === "fp16" ? "none" : null;
      const exs = q ? inferRows(r => r.gpu === gk && r.model === key && r.quant === q && r.tp === tp && pp === 1 && r.inp === o.pin && r.out === o.pout && r.conc === users * (o.batch || 1) && !o.offload) : [];
      if (exs.length) {   // repeated runs averaged
        st.t = gmean(exs.map(r => r.tpot_ms)) / 1000; ttft = gmean(exs.map(r => r.ttft_ms)) / 1000; st.aggM = gmean(exs.map(r => r.out_tps)); runs.push(...exs.map(r => r.run_id)); }
      else {
        const cal = q ? inferCal(gk, q) : null;          // a correction only applies to the precision it was measured in
        if (cal) { st.t *= cal.dec; ttft *= cal.ttft1 * cal.load(users * (o.batch || 1)); runs.push(...cal.runs); }
      }
    }
    let bf = null;
    if (wq === "fp8" && !o.raw && !o.gpu && !st.aggM && (runs.length || inferCal(gk, "none"))) {   // FP8 weights read half the bytes of BF16: never slower (only where our vLLM corrections apply)
      const b = bf = inf(gk, m, { ...o, wq: "bf16" });
      if (b.fits && b.tpot / 1000 < st.t) st.t = b.tpot / 1000;
      if (b.fits && b.ttft < ttft) ttft = b.ttft;
    }
    const agg = st.aggM || B / st.t;
    const calD = (!o.raw && !o.gpu && runs.length && !st.aggM) ? (inferCal(gk, wq === "fp8" ? "fp8" : "none") || { dec: 1 }).dec : 1;
    let knee = null; for (let c = 1; c <= Math.min(cmax, 4096); c++) { const x = stepT(c); if (x.tc >= x.tm) { knee = c; break; } }
    const costH = g.rate == null ? null : g.rate * N * OVERHEAD;
    const act = B * ctx * m.h * 2 / 1e9 * 0.1;
    return {
      ...r, fits: true, B, users: Math.min(users, cmax), queued: batch > cmax, tmem: st.tm, tcomp: st.tc, t: st.t,
      agg, perUser: 1 / st.t, tpot: st.t * 1000, ttft, lat: ttft + o.pout * st.t, rph: agg * 3600 / o.pout, knee,
      kvUsed: B * seqGB, single: Math.max(1 / (stepT(1).t * calD), 1 / st.t, bf && bf.fits ? bf.single : 0),   // one user is never slower than many
      perM: costH == null ? null : costH / (agg * 3600) * 1e6,
      per1k: costH == null ? null : costH * (1000 * o.pout / agg) / 3600,
      wh1m: g.tdp * N / (agg * 3600) * 1e6,
      mem: { weights: wGB - offGB, kv: seqGB * B, act, ovh, off: offGB, total: wGB - offGB + seqGB * B + act + ovh, cap: vramOf(g) * N },
      runs, measured: runs.length > 0
    };
  }
  const hasNV = (g, n) => !!g.nvlink || (!!g.nvpair && n <= 2);
  const int0 = v => Math.round(v).toLocaleString("en-US");
  function attnType(m) { return m.mla ? "MLA" : m.kv === 1 ? "MQA" : m.kv >= m.nh ? "MHA" : "GQA"; }

  /* ---- fine-tune memory: full / LoRA / QLoRA, with apxml-style optimization toggles ----
     Baseline = our measured setup: LoRA, bf16, flash attention on, gradient checkpointing on,
     no packing, dynamic padding, AdamW with fp32 states. */
  /* ---- our RunPod measurements (odyn-prof, measured.js). Rows are real timings with run ids.
     Train rows are processed tok/s per GPU on full-length sequences (no padding), transformers + PEFT. */
  const seqAlign = (tok, from, to) => tok * Math.pow(to / from, -0.1051);        // same law as SEQ_B, fit on our runs
  const trainRows = (f) => (MEAS.train || []).filter(f);
  const isBase = r => r.n === 1 && r.batch === 1 && r.ckpt === 1 && r.attn === "sdpa" && r.optim === "adamw" && r.rank === 16;
  // measured speed of `method` relative to LoRA, same GPU and model, aligned to the same sequence length
  function methodFactor(gk, method) {
    const pairs = [];
    for (const r of trainRows(r => r.method === method && isBase(r))) {
      const lc = measuredCurve(r.gpu, r.model, "lora", 1, null);       // LoRA at the same tokens processed per step
      if (lc) { const T = r.batch * r.seq * (method === "dpo" ? 2 : 1), l = curveAt(lc, T, method === "dpo" ? null : r.seq);
        if (T >= lc[0].T * 0.99 && T <= lc[lc.length - 1].T * 1.01) pairs.push({ gpu: r.gpu, f: r.tok / l.tok, run: r.run_id }); }
    }
    const own = pairs.filter(p => p.gpu === gk), use = own.length ? own : pairs;
    if (!use.length) return null;
    return { f: Math.exp(use.reduce((a, p) => a + Math.log(p.f), 0) / use.length), own: !!own.length, runs: use.map(p => p.run) };
  }
  // measured runs for this GPU, model, method, GPU count and parallelism, as a curve over tokens per GPU per step
  const sameSettings = r => r.ckpt === 1 && r.attn === "sdpa" && r.optim === "adamw" && r.rank === 16;
  const distOf = z => z === 0 ? "ddp" : z >= 3 ? "fsdp" : null;
  function measuredCurve(gk, key, method, n, zero) {
    const d = n > 1 ? distOf(zero) : "none";
    if (!d) return null;
    const pts = trainRows(r => r.gpu === gk && r.model === key && r.method === method && r.n === n && r.dist === d && sameSettings(r))
      .map(r => ({ T: r.batch * r.seq, seq: r.seq, tok: r.tok, run: r.run_id }));
    // repeats of the same (batch, sequence) are averaged; the same tokens per step at a shorter sequence is a
    // different setup (less attention per token), so it stays a separate point
    const byK = {}; for (const p of pts) (byK[p.T + "|" + p.seq] = byK[p.T + "|" + p.seq] || []).push(p);
    const out = Object.values(byK).map(g => ({ T: g[0].T, seq: g[0].seq, tok: gmean(g.map(p => p.tok)), run: g.map(p => p.run).join("+"), reps: g.length }));
    out.sort((a, b) => a.T - b.T);
    return out.length ? out : null;
  }
  // processed tok/s at T tokens per step: log-linear between measured points, the nearest point outside them
  function curveAt(all, T, seq) {
    // exact setup first; else interpolate over tokens per step among runs at the same sequence length when they
    // bracket T; else among all runs (one point per step size, averaging different sequence lengths)
    if (seq) { const ex = all.find(p => p.seq === seq && Math.abs(p.T / T - 1) < 0.02); if (ex) return { tok: ex.tok, runs: [ex.run], exact: true }; }
    const same = seq ? all.filter(p => p.seq === seq) : [];
    let pts = same.length >= 2 && T >= same[0].T && T <= same[same.length - 1].T ? same : null;
    if (!pts) { const byT = {}; for (const p of all) (byT[p.T] = byT[p.T] || []).push(p);
      pts = Object.values(byT).map(g => ({ T: g[0].T, tok: gmean(g.map(p => p.tok)), run: g.map(p => p.run).join("+") })).sort((a, b) => a.T - b.T); }
    if (T <= pts[0].T) return { tok: pts[0].tok, runs: [pts[0].run], exact: Math.abs(T / pts[0].T - 1) < 0.02 };
    const last = pts[pts.length - 1];
    if (T >= last.T) return { tok: last.tok, runs: [last.run], exact: Math.abs(T / last.T - 1) < 0.02 };
    let i = 1; while (pts[i].T < T) i++;
    const a = pts[i - 1], b = pts[i], w = Math.log(T / a.T) / Math.log(b.T / a.T);
    return { tok: Math.exp(Math.log(a.tok) + w * (Math.log(b.tok) - Math.log(a.tok))), runs: [a.run, b.run], exact: false };
  }
  // measured board power as a share of TDP on this GPU (training runs)
  function powerFrac(gk) {
    const w = trainRows(r => r.gpu === gk && r.power_w).map(r => r.power_w);
    return w.length ? Math.min(1, w.reduce((a, b) => a + b, 0) / w.length / GPUS[gk].tdp) : null;
  }
  const gmean = a => Math.exp(a.reduce((x, y) => x + Math.log(y), 0) / a.length);
  const baseOf = r => trainRows(x => x.gpu === r.gpu && x.model === r.model && x.method === r.method && x.seq === r.seq &&
                                      x.batch === r.batch && x.n === 1 && x.dist === "none" && sameSettingsF(x))[0];
  const sameSettingsF = r => r.ckpt === 1 && r.attn === "sdpa" && r.optim === "adamw" && r.rank === 16;
  // speed of runs with one setting changed, relative to the same run with base settings; own GPU first, else all GPUs
  function settingFactor(gk, isVariant) {
    const pairs = [];
    for (const r of trainRows(r => r.n === 1 && isVariant(r))) { const b = baseOf(r); if (b) pairs.push({ gpu: r.gpu, f: r.tok / b.tok, run: r.run_id }); }
    const own = pairs.filter(p => p.gpu === gk), use = own.length ? own : pairs;
    return use.length ? { f: gmean(use.map(p => p.f)), own: !!own.length, runs: use.map(p => p.run) } : null;
  }
  // step-size law: speed vs tokens per GPU per step T, relative to T = 2048, from LoRA runs at several T per model.
  // Below 2048 speed falls as (T/2048)^beta, above it rises gently as (T/2048)^gamma; both exponents shrink with model size.
  const STEP = {};
  // per GPU when that GPU has runs at two model sizes (faster GPUs lose more at small steps), else pooled over all GPUs
  function stepLaw(gk) {
    if (gk in STEP) return STEP[gk];
    const own = stepLawFor(r => r.gpu === gk);
    return (STEP[gk] = own ? { ...own, own: true } : (STEP.__all !== undefined ? STEP.__all : (STEP.__all = stepLawFor(() => true))));
  }
  function stepLawFor(sel) {
    const lo = [], hi = [];
    const groups = {};
    for (const r of trainRows(r => r.method === "lora" && r.n === 1 && sameSettingsF(r) && MODELS[r.model] && sel(r)))
      (groups[r.gpu + "|" + r.model] = groups[r.gpu + "|" + r.model] || []).push(r);
    for (const g of Object.values(groups)) {
      const ref = g.find(r => r.batch * r.seq === 2048); if (!ref) continue;
      const N = MODELS[ref.model].f;
      for (const r of g) { const T = r.batch * r.seq; if (T === 2048) continue;
        const e = Math.log(r.tok / ref.tok) / Math.log(T / 2048); (T < 2048 ? lo : hi).push({ N, e, run: r.run_id }); }
    }
    // fit ln(e) = ln(e8) + k ln(8/N) by least squares (needs two model sizes)
    const fit = pts => { const q = pts.filter(p => p.e > 1e-3); if (new Set(q.map(p => p.N)).size < 2) return null;
      const xs = q.map(p => Math.log(8 / p.N)), ys = q.map(p => Math.log(p.e)), n = xs.length, mx = xs.reduce((a, b) => a + b) / n, my = ys.reduce((a, b) => a + b) / n;
      const k = xs.reduce((a, x, i) => a + (x - mx) * (ys[i] - my), 0) / xs.reduce((a, x) => a + (x - mx) ** 2, 0);
      return { e8: Math.exp(my - k * mx), k, runs: q.map(p => p.run) }; };
    const b = fit(lo), c = fit(hi);
    return b ? { lo: b, hi: c } : null;
  }
  function stepFactor(T, N, gk) {
    const s = stepLaw(gk); if (!s) return null;
    if (T < 2048) return Math.pow(T / 2048, Math.min(1, s.lo.e8 * Math.pow(8 / N, s.lo.k)));
    return s.hi ? Math.pow(T / 2048, Math.min(0.15, s.hi.e8 * Math.pow(8 / N, s.hi.k))) : 1;
  }
  // LoRA rank: f = (r/16)^e from runs at other ranks
  function rankFactor(gk, r) {
    const pts = []; for (const x of trainRows(x => x.n === 1 && x.rank !== 16 && x.ckpt === 1 && x.attn === "sdpa" && x.optim === "adamw")) {
      const b = baseOf(x); if (b) pts.push({ gpu: x.gpu, e: Math.log(x.tok / b.tok) / Math.log(x.rank / 16), run: x.run_id }); }
    const own = pts.filter(p => p.gpu === gk), use = own.length ? own : pts;
    return use.length ? { f: Math.pow(r / 16, use.reduce((a, p) => a + p.e, 0) / use.length), own: !!own.length, runs: use.map(p => p.run) } : null;
  }
  // MoE: measured speed vs the dense curve at active params, on MoE runs at T = 2048
  let MOE;
  function moeFactor() {
    if (MOE !== undefined) return MOE;
    const f = [], runs = [];
    for (const r of trainRows(r => r.method === "lora" && r.n === 1 && r.batch * r.seq === 2048 && sameSettingsF(r) && MODELS[r.model] && MODELS[r.model].moe &&
                                  !/routed experts/.test(r.lora_scope || ""))) {
      const curve = Math.exp(level(r.gpu) - P * Math.log(MODELS[r.model].f)) * PAD; f.push(r.tok / curve); runs.push(r.run_id); }
    return (MOE = f.length ? { f: gmean(f), runs } : null);
  }
  // activation memory without checkpointing, measured vs formula
  let ACTS;
  function actScaleNoCkpt() {
    if (ACTS !== undefined) return ACTS;
    const f = [];
    for (const r of trainRows(r => r.ckpt === 0 && r.n === 1 && r.peak_gb && MODELS[r.model] && r.method === "lora")) {
      const mm = ftMemRaw(MODELS[r.model], { method: "lora", rank: r.rank, batch: r.batch, seq: r.seq, maxseq: r.seq, packing: true, ckpt: false, flash: r.attn !== "eager" });
      const act = r.peak_gb - (mm.weights + mm.grad + mm.opt + mm.temp); if (act > 0 && mm.act > 0) f.push(act / mm.act); }
    return (ACTS = f.length ? gmean(f) : null);
  }
  // Multi-GPU per-GPU efficiency from our runs: speed at n GPUs over the same run on 1 GPU (same GPU, model, step size).
  // DDP (and ZeRO-1/2, which keep full weights on every GPU) is nearly flat. FSDP (ZeRO-3) costs a one-off step at
  // 2 GPUs that grows with model size, then stays nearly flat. Fitted per GPU when that GPU has runs, else pooled.
  const MULTI = {};
  function multiPoints(d) {
    const pts = [];
    for (const r of trainRows(r => r.n > 1 && r.dist === d && sameSettingsF(r) && r.method === "lora" && MODELS[r.model])) {
      const c = measuredCurve(r.gpu, r.model, "lora", 1, null); if (!c) continue;
      const T = r.batch * r.seq; if (T < c[0].T * 0.99 || T > c[c.length - 1].T * 1.01) continue;
      pts.push({ gpu: r.gpu, N: MODELS[r.model].f, n: r.n, e: r.tok / curveAt(c, T, r.seq).tok, run: r.run_id });
    }
    return pts;
  }
  function lsq(X, y) {   // tiny least squares via normal equations
    const k = X[0].length, A = Array.from({ length: k }, () => Array(k).fill(0)), b = Array(k).fill(0);
    X.forEach((row, i) => { for (let a = 0; a < k; a++) { b[a] += row[a] * y[i]; for (let c = 0; c < k; c++) A[a][c] += row[a] * row[c]; } });
    for (let i = 0; i < k; i++) { let p = i; for (let r = i + 1; r < k; r++) if (Math.abs(A[r][i]) > Math.abs(A[p][i])) p = r;
      [A[i], A[p]] = [A[p], A[i]]; [b[i], b[p]] = [b[p], b[i]]; if (Math.abs(A[i][i]) < 1e-12) return null;
      for (let r = 0; r < k; r++) if (r !== i) { const f = A[r][i] / A[i][i]; for (let c = i; c < k; c++) A[r][c] -= f * A[i][c]; b[r] -= f * b[i]; } }
    return b.map((v, i) => v / A[i][i]);
  }
  function multiLaw(gk, d) {
    const key = gk + "|" + d; if (key in MULTI) return MULTI[key];
    const all = multiPoints(d), own = all.filter(p => p.gpu === gk), use = own.length >= 2 ? own : all;
    if (!use.length) return (MULTI[key] = null);
    // ln e = a + s ln(N/8) + b log2(n/2); drop terms the data cannot identify
    const sizes = new Set(use.map(p => p.N)).size > 1, counts = new Set(use.map(p => p.n)).size > 1;
    const X = use.map(p => [1, ...(sizes ? [Math.log(p.N / 8)] : []), ...(counts ? [Math.log2(p.n / 2)] : [])]);
    const c = lsq(X, use.map(p => Math.log(p.e))); if (!c) return (MULTI[key] = null);
    const a = c[0], sN = sizes ? c[1] : 0, bn = counts ? c[sizes ? 2 : 1] : 0;
    return (MULTI[key] = { e: (N, n) => Math.min(1, Math.exp(a + sN * Math.log(N / 8) + bn * Math.log2(Math.max(2, n) / 2))),
      own: own.length >= 2, maxN: Math.max(...use.map(p => p.n)), runs: use.map(p => p.run) });
  }
  const inferRows = (f) => (MEAS.infer || []).filter(f);
  let KEYS = null;
  const keyOf = m => { if (!KEYS) { KEYS = new Map(); for (const k in MODELS) KEYS.set(MODELS[k], k); } return KEYS.get(m); };
  // per-GPU correction from our vLLM runs: decode step ratio, first-token ratio at 1 user, and first-token growth with load
  const CAL = {};
  function inferCal(gk, q) {
    const ck = gk + "|" + (q || "none");
    if (ck in CAL) return CAL[ck];
    const rows = inferRows(r => r.gpu === gk && r.quant === (q || "none") && r.tp === 1 && MODELS[r.model]);
    if (!rows.length) return (CAL[ck] = null);
    const pred = r => inf(gk, MODELS[r.model], { pin: r.inp, pout: r.out, C: r.conc, batch: 1, wq: q === "fp8" ? "fp8" : "bf16", kvq: "fp16", tp: 1, pp: 1, gpn: 8, link: "auto", raw: 1 });
    const gm = a => Math.exp(a.reduce((x, y) => x + Math.log(y), 0) / a.length);
    const one = rows.filter(r => r.conc === 1);
    const dec = gm(rows.map(r => r.tpot_ms / pred(r).tpot));
    const ttft1 = one.length ? gm(one.map(r => r.ttft_ms / (pred(r).ttft * 1000))) : 1;
    // first-token time under load relative to 1 user, as measured; log-linear in users between measured points
    const pts = [];
    for (const r of rows) { const b = one.find(x => x.model === r.model && x.inp === r.inp && x.out === r.out); if (b) pts.push([Math.log(r.conc), Math.log(r.ttft_ms / b.ttft_ms)]); }
    pts.sort((a, b) => a[0] - b[0]);
    const load = C => { if (pts.length < 2) return 1; const x = Math.log(Math.max(1, C)); let i = 1; while (i < pts.length - 1 && pts[i][0] < x) i++;
      const [x0, y0] = pts[i - 1], [x1, y1] = pts[i]; const y = x1 === x0 ? y0 : y0 + (y1 - y0) * (x - x0) / (x1 - x0); return Math.exp(Math.max(0, y)); };
    return (CAL[ck] = { dec: Math.min(1.5, Math.max(0.6, dec)), ttft1: Math.min(2, Math.max(0.5, ttft1)), load, runs: rows.map(r => r.run_id), n: rows.length });
  }

  const OSTATE = { adamw: 8, lion: 4, sgd: 4, adafactor: 0.1 };      // optimizer state bytes/trained param (fp32)
  const PAD = 1.3;       // median padding ratio (processed/content tokens) across our 35 measured runs
  const FT_UTIL = 0.90;   // keep 10% of VRAM free for allocator fragmentation (reproduces our measured GPU counts)
  const SEQ_B = 0.1051;  // speed vs processed length: (len/2048)^-0.1051, fit on our 9 runs at 512-4096 tokens (held-out error 4.3%)
  const CKPT_COST = 1.25; // checkpointing recompute cost, 20-30% per apxml course; estimate, not measured
  function ftMem(m, o) {
    const r = ftMemRaw(m, o);
    if (!r.flags.ckpt) { const sc = actScaleNoCkpt(); if (sc) { r.total += r.act * (sc - 1); r.act *= sc; r.actScaled = sc; } }
    return r;
  }
  function ftMemRaw(m, o) {
    const method = o.method === "dpo" ? "lora" : (o.method || "lora");
    const f = { flash: o.flash !== false, ckpt: o.ckpt !== false, opt8: !!o.opt8,
      fused: !!o.fused, packing: !!o.packing, dynpad: o.dynpad !== false, pure16: !!o.pure16 };
    const r = o.rank || 16, adapters = 2 * r * m.h * 7 * m.L / 1e9;
    const trainable = method === "full" ? m.p : adapters;
    const weights = (method === "qlora" ? 0.55 : 2) * m.p;              // NF4 ~0.55 B/param incl. scales
    const grad = 2 * trainable;
    let st = OSTATE[o.optim || "adamw"] ?? 8; if (f.opt8 && (o.optim === "adamw" || o.optim === "lion" || !o.optim)) st /= 4;
    else if (f.pure16 && method === "full") st /= 2;                     // optimizer states in bf16 alongside bf16 weights
    const master = method === "full" && !f.pure16 ? 4 : 0;               // fp32 master copy for mixed precision
    const opt = (st + master) * trainable;
    const maxseq = o.maxseq || 2048, avg = Math.min(o.seq || 1024, maxseq);
    const seqEff = f.packing ? maxseq : f.dynpad ? Math.min(maxseq, avg * PAD) : maxseq;
    const bs = (o.batch || 1) * (o.method === "dpo" ? 2 : 1);  // DPO: chosen + rejected
    // Activations per Korthikanti et al. 2022 (bf16, Flash Attention): a full layer stores ~34*h bytes per token.
    // With checkpointing only each layer's input (2*h bytes) is kept, plus one layer recomputed at a time.
    const tok = bs * seqEff, fullLayer = 34 * m.h;
    let act = f.ckpt ? tok * (2 * m.h * m.L + fullLayer) : tok * fullLayer * m.L;
    if (!f.flash) act += bs * m.nh * seqEff * seqEff * 2 * (f.ckpt ? 1 : m.L);    // stored attention scores
    const temp = tok * (m.vocab || 32000) * 4 * 2 * (f.fused ? 0.125 : 1);        // fp32 logits + their gradient; fused CE chunks them
    const g = v => v / 1e9;
    return { weights, grad, opt, act: g(act), temp: g(temp), ovh: CUDA_OVH, trainable, method, flags: f, seqEff,
      total: weights + grad + opt + g(act) + g(temp) + CUDA_OVH };
  }
  function ftTok(gk, m, key, n, zero) {
    const law = n > 1 ? multiLaw(gk, zero >= 3 ? "fsdp" : "ddp") : null;
    const eta = nn => nn <= 1 ? 1 : law ? law.e(m.f, nn) : etaN(nn, m.f);
    const within = law ? n <= law.maxN : n <= 4;                     // measured GPU counts, else extrapolated
    const a = ANCH[gk + "|" + key];
    if (a && a[1] === n) return [a[0], "M", law];
    if (a) return [a[0] * eta(n) / eta(a[1]), within ? "P" : "X", law];   // scale our measured run to this GPU count
    const tok = Math.exp(level(gk) - P * Math.log(m.f)) * eta(n);
    return [tok, ((gk in LVL || levelMeasured(gk) != null) && !m.moe && within) ? "P" : "X", law];
  }
  function ft(gk, m, key, o) {
    const g = GPUS[gk], avg = o.seq || 1024, tt = o.rows * avg * o.epochs;
    const mem = ftMem(m, o);
    // ZeRO / FSDP stage: 0 = plain DDP (every GPU holds everything), 1 shards optimizer state, 2 also gradients,
    // 3 (FSDP) also weights. Activations, logits and runtime overhead are always per GPU.
    const z = o.zero == null ? 3 : o.zero;
    const perGpu = n => (z >= 3 ? mem.weights / n : mem.weights) + (z >= 2 ? mem.grad / n : mem.grad) + (z >= 1 ? mem.opt / n : mem.opt) + mem.act + mem.temp + mem.ovh;
    let n = 1;
    if (o.ngpu) n = Math.min(o.ngpu, g.maxN);
    else while (n <= g.maxN && perGpu(n) > vramOf(g) * FT_UTIL) n *= 2;
    if (o.ngpu && perGpu(n) > vramOf(g) * FT_UTIL) return { gk, n, tt: o.rows * (o.seq || 1024) * o.epochs, mem, perGpuGB: perGpu(n), zero: z, fits: false, why: "does not fit on " + n + " GPU" + (n > 1 ? "s" : "") + " at ZeRO-" + z };
    const r = { gk, n, tt, mem, zero: z, perGpuGB: perGpu(Math.min(n, g.maxN)), perGpuAt: perGpu };
    if (n > g.maxN) return { ...r, fits: false, why: "more than " + g.maxN + " GPUs" };
    if (g.ftq === "no" && levelMeasured(gk) == null) return { ...r, fits: true, quoted: false, why: "needs a calibration run" };
    let [base, tag, mlaw] = ftTok(gk, m, key, n, o.zero == null ? 3 : o.zero);
    const adj = [];
    if (mlaw && n > 1) adj.push(`${n} GPUs x${mlaw.e(m.f, n).toFixed(2)} per GPU (${(o.zero == null ? 3 : o.zero) >= 3 ? "FSDP" : "DDP"}, measured${mlaw.own ? "" : " on other GPUs"})`);
    let runs = [];
    // an exact measured run replaces the curve; convert to the curve's convention (content tok/s at 2048, 1.3x padding)
    const meth0 = o.method || "lora";
    const Lp = Math.min(32768, Math.max(256, mem.seqEff)), Tstep = (o.batch || 1) * Lp * ((o.method || "lora") === "dpo" ? 2 : 1);
    // our RunPod runs are primary: known processed tokens, same basis as every other GPU and as the price cap.
    // (SDS fleet runs came out 2-16% faster than ours for identical setups, via an assumed 1.3x padding factor.)
    const curve = measuredCurve(gk, key, meth0, n, o.zero == null ? 3 : o.zero);
    let dr = null;
    if (curve) {   // measured processed tok/s at this step size; expressed in the curve's convention so the later factors line up
      const c = curveAt(curve, Tstep, Math.round(Lp)); dr = { run_id: c.runs.join(", "), exact: c.exact };
      base = c.tok / PAD / (stepFactor(Tstep, m.f, gk) ?? Math.pow(Lp / 2048, -SEQ_B)); tag = "M"; runs.push(...c.runs);
    }
    let k = 1;
    // tokens per GPU per step (batch x processed length) sets how busy the GPU is; the law is fitted on our runs
    const stf = stepFactor(Tstep, m.f, gk), sf = stf ?? Math.pow(Lp / 2048, -SEQ_B);
    if (Math.abs(sf - 1) > 0.005) { k *= sf; adj.push((stf != null ? "step size " + int0(Tstep) + " tokens" + (stepLaw(gk).own ? "" : " (law from other GPUs)") + " x" : "sequence length x") + sf.toFixed(2)); }
    if (!mem.flags.packing && !mem.flags.dynpad) {   // padding every sample to max length wastes compute
      const avg = Math.min(o.seq || 1024, o.maxseq || 2048), keep = Math.min(1, avg / (o.maxseq || 2048) * PAD);
      if (keep < 0.995) { k *= keep; adj.push("padding to max length x" + keep.toFixed(2)); }
    }
    if (mem.flags.packing) { k *= PAD; adj.push("packing x" + PAD); }
    const lab = (name, x) => `${name} x${x.f.toFixed(2)} (measured${x.own === false ? " on other GPUs" : ""})`;
    if (!mem.flags.ckpt) { const x = settingFactor(gk, r => r.ckpt === 0 && r.attn === "sdpa" && r.optim === "adamw" && r.rank === 16);
      if (x) { k *= x.f; runs.push(...x.runs); adj.push(lab("no checkpointing", x)); } else { k *= CKPT_COST; adj.push("no checkpointing x" + CKPT_COST + " (assumed)"); } }
    if (!mem.flags.flash) { const x = settingFactor(gk, r => r.attn === "eager" && r.ckpt === 1 && r.optim === "adamw" && r.rank === 16);
      if (x) { k *= x.f; runs.push(...x.runs); adj.push(lab("no Flash Attention", x)); } }
    if ((o.rank || 16) !== 16 && (o.method || "lora") !== "full") { const x = rankFactor(gk, o.rank);
      if (x && Math.abs(x.f - 1) > 0.005) { k *= x.f; runs.push(...x.runs); adj.push(lab("rank " + o.rank, x)); } }
    const optName = (o.optim || "adamw") === "adamw" && mem.flags.opt8 ? "adamw8bit" : (o.optim || "adamw") === "lion" && mem.flags.opt8 ? "lion" : (o.optim || "adamw");
    if (optName !== "adamw") { const x = settingFactor(gk, r => r.optim === optName && r.ckpt === 1 && r.attn === "sdpa" && r.rank === 16);
      if (x) { k *= x.f; runs.push(...x.runs); adj.push(lab(optName === "adamw8bit" ? "8-bit AdamW" : optName, x)); } }
    const meth = o.method || "lora";
    if (!dr && meth !== "lora") {                     // method factor: measured ratio if we have one, else the old assumption
      const mf = methodFactor(gk, meth);
      if (mf) { k *= mf.f; runs.push(...mf.runs); if (tag === "M") tag = "P"; adj.push(`${meth === "qlora" ? "QLoRA" : meth === "dpo" ? "DPO" : "full fine-tune"} x${mf.f.toFixed(2)} (measured${mf.own ? "" : " on other GPUs"})`); }
      else if (meth === "full") { const fr = mem.flags.ckpt ? 6 / 8 : 4 / 6; k *= fr; adj.push("full fine-tune x" + fr.toFixed(2) + " (FLOP ratio, not measured)"); }
      else if (meth === "qlora") adj.push("QLoRA speed taken as LoRA (not measured)");
      else if (meth === "dpo") { k /= DPO; adj.push("DPO x0.50 (assumed)"); }
    }
    if (dr) { const ids = String(dr.run_id).split(/[,+]\s*/).filter(Boolean);   // ids stay in the tooltip, not the text
      adj.push({ text: dr.exact ? "our measured run" : `interpolated between ${ids.length} measured run${ids.length > 1 ? "s" : ""}`, ids }); }
    if (m.moe && !dr) { const x = moeFactor();
      if (x) { k *= x.f; runs.push(...x.runs); adj.push(`MoE routing x${x.f.toFixed(2)} (measured on ${x.runs.length} MoE run${x.runs.length > 1 ? "s" : ""})`); }
      else adj.push("MoE speed from active params (not measured)"); }
    let eff = base * k;
    if (n > 1) {
      // per GPU per optimizer step: compute vs traffic. ZeRO-3 re-gathers the weights every micro-batch (forward + backward,
      // overlapped with compute by FSDP prefetch); the gradient all-reduce is not overlapped. On NVLink this stays hidden for
      // every measured multi-GPU run, so etaN already covers it; it bites for big MoE models (few active, many total params) and PCIe.
      const stepTok = (o.batch || 1) * (o.accum || 1) * Math.min(avg, o.maxseq || 2048), tc = stepTok / eff, fr = (n - 1) / n;
      const nvl = hasNV(g, n), bw = (nvl ? (g.nvbw || 240) : (g.pcie || 25)) * 1e9;
      const gather = z >= 3 ? (o.accum || 1) * 2 * mem.weights * 1e9 * fr / bw : 0;
      const ar = 2 * mem.grad * 1e9 * fr / bw;
      const t = Math.max(tc, gather) + ar, f = tc / t;
      if (f < 0.995) { eff *= f; adj.push((nvl ? "GPU-to-GPU traffic" : "PCIe traffic between GPUs") + " x" + f.toFixed(2) + " (estimate)"); }
    }
    const hours = tt / (eff * n) / 3600, gpuH = hours * n;
    const cost = g.rate == null ? null : gpuH * g.rate * OVERHEAD;
    const gbatch = (o.batch || 1) * (o.accum || 1) * n, samples = o.rows * o.epochs;
    const sps = eff * n / avg;
    return {
      ...r, fits: true, quoted: true, tok: base, eff, tag, adj, jobTok: eff * n, hours, gpuH, cost,
      cap: cost == null ? null : cost * CAP[o.cov].f, perM: cost == null ? null : cost / (tt / 1e6),
      mfu: 6 * bn(m.a) * eff * (o.method === "dpo" ? 2 : 1) / (g.tf * 1e12) * 100, kwh: g.tdp * (powerFrac(gk) || 1) * n * hours / 1000, powerMeasured: powerFrac(gk) != null, runs,
      gbatch, steps: Math.ceil(samples / gbatch), sps, stepsPerSec: sps / gbatch
    };
  }

  // what the engine derived from the measured runs, for reporting (the workbook's "Fitted laws" sheet)
  function diagnostics() {
    const gpus = [...new Set((MEAS.train || []).map(r => r.gpu))].filter(g => GPUS[g]);
    const out = { step: {}, multi: {}, method: {}, setting: {}, moe: moeFactor(), level: {}, actNoCkpt: actScaleNoCkpt(), cap: CAP };
    for (const g of gpus) {
      const st = stepLaw(g); if (st) out.step[g] = { below2048_e8: st.lo.e8, below2048_k: st.lo.k, above2048_e8: st.hi && st.hi.e8, above2048_k: st.hi && st.hi.k, own: !!st.own, runs: st.lo.runs.length };
      for (const d of ["ddp", "fsdp"]) { const L = multiLaw(g, d); if (L) out.multi[g + "|" + d] = { e_8B_2: L.e(8, 2), e_8B_8: L.e(8, 8), e_32B_8: L.e(32, 8), own: L.own, maxN: L.maxN, runs: L.runs.length }; }
      for (const m of ["qlora", "dpo", "full"]) { const f = methodFactor(g, m); if (f) out.method[g + "|" + m] = { f: f.f, own: f.own, runs: f.runs.length }; }
      const S = { "no checkpointing": r => r.ckpt === 0 && r.attn === "sdpa" && r.optim === "adamw" && r.rank === 16,
                  "eager attention": r => r.attn === "eager" && r.ckpt === 1 && r.optim === "adamw" && r.rank === 16 };
      for (const o of ["adamw8bit", "adafactor", "sgd", "lion"]) S[o] = r => r.optim === o && r.ckpt === 1 && r.attn === "sdpa" && r.rank === 16;
      for (const [k, f] of Object.entries(S)) { const x = settingFactor(g, f); if (x) out.setting[g + "|" + k] = { f: x.f, own: x.own, runs: x.runs.length }; }
      out.level[g] = { level: level(g), source: g in LVL ? "SDS fit" : levelMeasured(g) != null ? "our runs" : "datasheet" };
    }
    return out;
  }
  const api = { GPUS, MODELS, ANCH, CAP, WQ, KVQ, OSTATE, attnType, vramOf, ft, inf, ftMem, kvBytes, level, etaN, MEAS, diagnostics };
  if (isNode) module.exports = api; else Object.assign(root, api);
})(typeof window !== "undefined" ? window : globalThis);
