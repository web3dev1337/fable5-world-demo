# Measuring FPS headlessly — the rAF soak technique

How to get a trustworthy frame-rate number out of a browser-rendered three.js
app without a human watching a HUD. This is the technique we used to profile the
meme-merge browser port overnight (UpsideEngine), adapted here for LAAS.

## The core idea

Drive the app into a representative state, then **count `requestAnimationFrame`
ticks over a fixed wall-clock window** and divide by elapsed seconds. That is the
*presented* frame rate — the thing a player actually perceives — and it is
independent of whatever the engine reports internally.

```js
// runs inside page.evaluate() — i.e. in the page's own JS context
const result = await page.evaluate(async (soakSec) => {
  // (optional) DOM-churn monitor — catches "rebuilds every frame" bugs
  let added = 0, removed = 0;
  const mo = new MutationObserver((muts) => {
    for (const m of muts) { added += m.addedNodes.length; removed += m.removedNodes.length; }
  });
  mo.observe(document.body, { childList: true, subtree: true });

  // the frame counter
  let frames = 0, raf = true;
  const count = () => { frames++; if (raf) requestAnimationFrame(count); };
  requestAnimationFrame(count);

  const t0 = performance.now();
  await new Promise((r) => setTimeout(r, soakSec * 1000));   // soak
  raf = false; mo.disconnect();
  const elapsed = (performance.now() - t0) / 1000;

  return {
    elapsedSec: +elapsed.toFixed(1),
    fps: +(frames / elapsed).toFixed(1),
    nodesAddedPerSec: +(added / elapsed).toFixed(1),
    heapMB: performance.memory ? +(performance.memory.usedJSHeapSize / 1048576).toFixed(1) : null,
  };
}, SOAK_SEC);
console.log(JSON.stringify(result, null, 2));
```

### Why count rAF instead of reading the engine's own fps

- **It is ground truth.** It measures frames the compositor presented, not an
  internal estimate, EMA, or a `setInterval`-based guess.
- **It needs no GPU timestamp-query.** LAAS's per-pass profiler calls
  `resolveTimestampsAsync` every frame; on a software adapter (SwiftShader) that
  path drops the WebGPU instance and the page dies. A bare rAF counter touches
  none of that, so it keeps working where the profiler can't.
- **It pairs with cheap monitors.** A `MutationObserver` (DOM churn) and
  `performance.memory` (heap) ride along for free, turning "what's my FPS" into
  "what's my FPS *and is something leaking/thrashing*" — that combination is how
  the overnight meme-merge lag was traced to a once-a-second full grid rebuild.

## The full headless harness shape

1. **Serve the build.** Start the dev server (or a static `vite preview`). For
   A/B, serve your branch and a clean `origin/master` worktree on two ports.
2. **Launch a real Chromium headless.** Playwright/Puppeteer. The browser must
   expose the renderer you're testing:
   - WebGL → `--use-gl=swiftshader --enable-webgl` (software, always available).
   - WebGPU → full Chromium new-headless (`channel: 'chromium'`), `--enable-unsafe-webgpu`.
     The default Puppeteer "headless shell" yields a **null** WebGPU adapter; you
     need the full build. (LAAS encodes this in `tools/launch.ts` and caches the
     winning flag set in `.cache/webgpu-flags.json`.)
3. **Drive into a representative state.** Load with the deterministic params your
   app supports, click through any intro, push it to a real scenario (for
   meme-merge: F2 dev cheats → unlock plots → hatch pets → leave a ready egg).
   For LAAS this is a URL param: `?shot=3` (a bookmark) or `?fly=1` (the 90 s
   flythrough), with `?freeze=0` so the world actually moves.
4. **Warm up, then soak.** Wait a few seconds for temporal effects (TAA,
   auto-exposure) and JIT to settle, then run the rAF counter for 15–30 s.
5. **Emit JSON** so runs are diffable and CI-friendly.

## Running it on LAAS

A ready-made adaptation lives at [`tools/fps-soak.ts`](../tools/fps-soak.ts),
wired to LAAS's `window.__laas` hooks and the WebGPU launcher:

```bash
npm run dev            # in one shell — the soak needs the dev server on :5173

# steady-state at a bookmark
npm run fps -- --scene world --shot 3 --soak 20

# under motion (the flythrough drives the camera through the soak)
npm run fps -- --scene world --fly --soak 30 --w 1920 --h 1080

# an exact pose
npm run fps -- --scene world --cam "10,80,40,1.2,-0.1" --soak 15
```

It prints both the **rAF-measured** rate and the engine's own attribution in one
object:

```json
{
  "elapsedSec": 20.0,
  "rafFps": 61.0,            // presented frame rate (ground truth)
  "frameMsAvg": 16.4,
  "frameMsP95": 18.9,        // tail latency — the stutters you feel
  "frameMsP99": 24.1,
  "longFrames": 0,           // frames slower than 50 ms during the soak
  "engineFps": 60.8,         // window.__laas.stats.fps (engine EMA, cross-check)
  "engineFrameMsP95": 18.7,
  "drawCalls": 412,
  "triangles": 9183744,
  "gpuPasses": { "scene": 6.1, "clouds": 2.3, "csm": 1.9, ... },  // per-pass ms
  "heapMB": 312.4,
  "adapter": "nvidia / ..." // which backend produced the number
}
```

Always read the `adapter` field. `google / swiftshader` means the number is
software rasterization (useful for relative A/B, meaningless as an absolute fps).
A real GPU vendor (`nvidia / …`) means it's representative.

## Getting a real number when the GPU is on the other side of WSL

This dev box renders headless Chromium through SwiftShader (no WebGPU ICD wired
to the NVIDIA GPU), so a headless soak here reports software fps. Two ways to get
the real figure:

1. **Run the soak where a real WebGPU adapter exists** — natively on the GPU
   machine, or once an NVIDIA/`dzn` Vulkan ICD is registered for headless Chrome.
2. **Measure the live browser via same-origin telemetry.** The page is already
   running in the real GPU's Chrome against the dev server, so let it phone its
   own FPS home. Add a Vite dev-server middleware that accepts `POST /__fps` and
   appends to a log, and a `transformIndexHtml` snippet that samples
   `window.__laas.stats` once a second and posts it:

   ```js
   // injected snippet (same origin as the page → no CORS / mixed-content)
   setInterval(() => {
     const s = window.__laas?.stats; if (!s) return;
     navigator.sendBeacon('/__fps', JSON.stringify({ t: Date.now(), fps: s.fps, frameMs: s.frameMs }));
   }, 1000);
   ```

   You then tail the server-side log while the user plays — real RTX-5090 frame
   rate, no headless GPU needed. (Inject it from the gitignored dev config so the
   tracked source stays clean.)

## Gotchas learned the hard way

- **The headless shell ≠ full Chromium for WebGPU.** Null adapter every time.
  Use `channel: 'chromium'`; cache the working launch recipe so reruns are fast.
- **WebGPU needs a secure context.** `navigator.gpu` is absent on a plain-HTTP
  LAN IP — probe over `http://localhost` (or HTTPS). Same reason the in-browser
  page must be opened via `https://` or `localhost`, never `http://<lan-ip>`.
- **Warm up before counting.** The first second is JIT + shader compile + TAA
  ramp; folding it into the average understates steady-state fps.
- **Soak, don't snapshot.** A single-frame timing is noise. 15–30 s of rAF plus
  p95/p99 frame-time captures the stutters that a mean hides.
- **Freeze kills the measurement.** If the world is frozen (`?freeze=1`) the
  per-frame cost collapses — measure with motion (`?freeze=0`, a bookmark or the
  flythrough) for a number that means something.
