# Performance findings — RTX 5090, measured

A self-driven hardware measurement loop was built and used to profile LAAS on a
real RTX 5090. This is the data the "2× FPS" work is based on.

## The measurement rig (no human needed)
- **Windows headless Chrome** launched from WSL (`chrome.exe --headless=new
  --enable-unsafe-webgpu`) renders on the real RTX 5090 (`adapter: nvidia /
  blackwell`). Linux headless can't (SwiftShader only); Windows headless can.
- The dev server (gitignored `.cache/vite.https.config.mjs`) injects a telemetry
  beacon: each second the page POSTs `rafFps` (true presented frames counted via
  `requestAnimationFrame`), the engine's `frameMs`/`cpu.update`/`cpu.submit`
  counters, per-pass GPU timings, and draw/tri counts to `/__fps`, which the
  server appends to `.cache/fps.log`. WSL reads the log — no CDP needed (the CDP
  port is firewalled WSL→Windows; the outbound beacon is not).
- `/tmp/bench.sh "url-params" warmup collect label "WxH"` runs one measurement
  (kills only the bench Chrome via a PID filter, fresh-loads, warms up, reports
  median rafFps + CPU submit/update).

## Headline numbers (scene=world, shot=3, 1280×720, RTX 5090)
- **~74–82 fps** steady state. The rAF clock is **not** capped — the trivial
  `sanity` scene runs at **3511 fps**, so ~78 fps is a genuine workload limit.
- fps is **flat from 640p to 1440p** and only drops at 4K → the bottleneck is
  **resolution-independent** (fixed-cost passes / CPU), not pixel fill.

## The bottleneck: CPU encode, not the GPU
Frame ≈ 11.9 ms = **CPU update 3.2 ms + CPU submit 8.9 ms**, with GPU only
**5.3 ms** (overlapped). The frame is **CPU-submit-bound** — three.js spends ~9 ms
*encoding* the render every frame.

### Attribution (ablation, CPU submit / update at 720p)
| ablate | rafFps | CPU submit | CPU update | reads as |
|---|---|---|---|---|
| (full) | 74 | 10.9 | 5.1 | baseline |
| `shadows` | 91 | **4.9** | 5.8 | shadows ≈ **6 ms of encode** (veg crowns × 4 cascades) |
| `veg` | **403** | 2.0 | 0.1 | vegetation is the dominant cost overall |
| `grass` | 99 | 9.6 | **0.3** | grass `update()` ≈ 4.8 ms CPU; grass *draws* ≈ 3.4 ms encode |

Removing grass alone takes a frozen vista 72 → **118 fps**. Grass is ~10–20
indirect draws yet costs ~3.4 ms to *encode* (~200–340 µs/draw — pathological;
normal is single-digit µs). Same signature as the shadow-pass hash storm that
`ThreePatches` already patches, but on the **color pass** (grass/veg materials).

## The 2× levers (all CPU-encode, all resolution-independent)
1. **Grass/veg draw-encode cost** (~3.4 ms grass + veg share). Likely a per-frame
   node-material re-validation (cache-key memo miss) or the GI `setupLightMap`
   monkeypatch re-creating nodes. Highest ROI if it's a caching fix.
2. **Shadow caster encode** (~6 ms). Veg crowns cast into all 4 cascades; the far
   cascade (1–10 m texels) gets sub-texel crown shadows. Casting crowns into 2–3
   cascades instead of 4 cuts encode with no visible change (verify via a
   one-shot `chrome.exe --headless --screenshot` diff on the 5090).
3. **Grass `update()` 4.8 ms** when the camera moves (pose-gate already skips it
   when static — frozen update drops to 0.8 ms).

## What's already committed (safe, no-downside)
- Pose-gated GroundRing + Forests culls (skip dispatch on static frames). Real,
  but the culls are cheap on the 5090 (~0.1 ms) so the fps effect is small.
- Profiler gated to HUD-open (the per-frame timestamp readback only feeds F3).

## Honest status
The game already runs well on a 5090 (~78 fps at 720p, smooth). It is **CPU-encode
bound**, not GPU bound. A real 2× requires halving the per-frame draw-encode cost
of veg/grass/shadows — a three.js node-material caching investigation, done with
this measurement rig + screenshot verification so "no visual downside" is proven,
not assumed. The rig and the targets above are the path.
