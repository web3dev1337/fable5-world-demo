# LAAS Code Review — quality, DRY, performance, opportunities

Full read-only audit of `src/` (~19,180 LOC, 74 files) + `tools/`, run as 10 per-subsystem agents. Every finding cites `file:line` from code actually read. Severity: **HIGH** (structural / correctness risk), **MED** (quality / maintainability), **LOW** (polish).

## Verdict

This is a genuinely high-quality codebase, well above typical. Verified strengths:
- **Zero `any`** in `src/` (the strict-TS claim holds to the letter).
- **Fail-loud everywhere** — no silent black frames; the engine renders its own error overlay.
- **Comments explain *why*** — most carry bug history, probe findings, or measured rationale.
- **Allocation-aware hot paths** — module-scope scratch vectors, no per-frame JS churn in the camera/cull loops.
- **A real verification harness** — headless WebGPU A/B with frame-phase pinning and GPU-timing medians.
- **Exemplary data-driven design in places** — `Species.ts`, `ROCK_PRESETS`, `BARK_TABLE`, `ColorScript` keyframes.

The recurring weaknesses are three, and they repeat across every subsystem:
1. **God files / god functions** — a handful of files concentrate huge multi-responsibility functions.
2. **Inconsistent data-driven discipline** — some tuning is beautifully tabled; a lot is hardcoded magic spread across hundreds of lines, sometimes in coupled pairs that must stay in sync.
3. **`as unknown as` cast density** — `any` is avoided, but the type system is defeated at TSL/three boundaries per-call-site instead of via shared typed wrappers.

---

## ★ Top cross-cutting opportunities (do these first)

1. **Split the four god files.** `Forests.ts` (1016), `GroundRing.ts` (992), `Scatter.ts` (841 — one 477-LOC function), `PostStack.ts` (654 — 565-LOC constructor). Each holds 4–6 cleanly separable responsibilities. Highest maintainability ROI in the repo. **Effort: L each.**
2. **Single source of truth for the veg group/compact layout.** `Forests.ts` encodes the draw-group index math **twice** — once in TS (`groupOf`/`casterGroupOf`/`capOf`, 127-190) and again inline in the GPU cull kernels (757-877). The magic offsets `146/48/82/136/72` and caps must match across CPU and GPU or draws silently render the wrong pool/size. **This is the single highest-risk DRY violation. Effort: M.**
3. **Hoist a shared GPU-kernel toolkit.** `guard`/`cellXY`/`at`/`isBorder`/`OFFS`, bilinear sampling, and talus-relaxation are duplicated across `Erosion`, `FlowRivers`, and `BufferSample` (two divergent talus implementations). One tested `GridKernel` module. **Effort: M.**
4. **Extract a shared probe harness** (`tools/harness.ts`): `bootScene()` + `settle()` + `readDbg()` + `pixelDiff()` + one `parseArgs`. Collapses the boot-readiness wait duplicated in 10 files and the boot framing copy-pasted across 6 shadow probes; prune the ~5 resolved/abandoned probes. **Effort: M.**
5. **Convert per-call-site casts into typed wrappers.** Tighten `TSLTypes` aliases so `.sample()`/`.load()`/`.element()` are typed; add `elemV4(buf,i)`, a typed material-extension interface for `castShadowPositionNode`/`maskShadowNode`, and a typed `EngineDebug` for the `window.__laasDbg` handles. Removes most of the `as unknown as` density (Scatter 18+, Forests 42, GroundRing 50, PostStack 15, Gtao 27, VegMaterials 30). **Effort: M.**
6. **Audit the per-frame GPU dispatches that don't gate on change** — the biggest runtime wins are in §Performance below: GroundRing's ~10M-thread cull, Froxels' full grid, Caustics' bake, Clouds' shadow re-bake.

---

## Live visual artifacts (diagnosed from running the build)

Two artifacts you reported, both traced to the same root: **the renderer leans on temporal accumulation (TRAA) to clean up intentionally-noisy techniques, and where TRAA can't converge, the underlying noise shows through.**

### 1. "Things look like dots" — dithered LOD crossfade
Vegetation and grass cross-fade between LOD rings using **screen-door stochastic transparency**: `applyDitherFade` (`VegInstance.ts:127-151`) and `GroundRing.bandFade` (`GroundRing.ts:159-187`) compute `interleavedGradientNoise(screenCoordinate.xy)` and discard pixels below a per-distance threshold. That stipple is *meant* to be dissolved by TRAA into smooth alpha.

Root cause: the IGN is keyed on **`screenCoordinate.xy` only — it is spatially static, not animated per frame** (`VegInstance.ts:132`, `GroundRing.ts:179`). High-contrast on/off dither is exactly what TRAA's variance clipping refuses to blend, so on a still camera it survives as fixed dots, and while moving it shimmers — most visible as dotted rings at the LOD transition distances around the camera.

Fix (high confidence, low risk): **animate the dither** by offsetting the IGN coordinates with the existing frame counter, so successive frames present different patterns that TRAA averages to true alpha. The repo already has a `frameU` uniform doing exactly this for cloud/SSCS jitter (`PostStack.ts:80,103,162`); expose it (or three's `time` node) to `applyDitherFade`/`bandFade`:
```
interleavedGradientNoise(screenCoordinate.xy.add(vec2(frameU.mul(5.588), frameU.mul(5.588))))
```
Confirm visually (only your GPU can render it — this container has no GPU; see §verification note). `?ablate=taa` makes the dither far more visible, which confirms it's a TRAA-resolved stipple.

### 2. Cloud "noise flickers," worst at mountain tops
Clouds are a **half-res, per-frame-jittered raymarch** (`PostStack.ts:154-167` builds the half-res layer; `Clouds.march` `Clouds.ts:282-352`, 32 steps with a per-frame `jitter = hash12(... + frameU*0.618)`). The jitter + half-res are explicitly "absorbed by TRAA" (`PostStack.ts:150-151`).

Why it breaks at altitude: `CLOUD_BOTTOM=1250`, `CLOUD_TOP=1900` (`Clouds.ts:52-53`), and mountains peak ~2040 m. **At the top you are inside or right at the cloud slab**, where the slab-intersection math degenerates: `t0=(CLOUD_BOTTOM−camY)/dir.y` and the `inside` toggle (`Clouds.ts:286-291`) flip near the boundary, and grazing up-rays make `seg=(tExit−tEnter)/32` huge so 32 steps under-sample a long path. Frame-to-frame the jittered samples land on wildly different densities → heavy flicker the temporal filter can't stabilize. Camera rotation at the peak compounds it: TRAA history gets reprojection-rejected (clouds are at the far plane; the depth-aware half-res upsample `PostStack.ts:308-325` also breaks at the peak/sky silhouette).

Fixes (need your visual confirmation):
- **Cap `seg`** to a max world step (e.g. clamp to ~40–60 m) and/or scale `STEPS` with path length so long grazing/inside paths stay sampled — kills the dominant flicker source. `Clouds.ts:296-297`. **S/M.**
- **Damp the jitter** as `camY` approaches the slab (lerp `jitter`→0 within ~150 m of `CLOUD_BOTTOM/TOP`) so inside-slab marches are stable. **S.**
- Longer term: temporal reprojection of the cloud buffer rather than relying on full-frame TRAA at the far plane. **M.**

> Verification note: this WSL2 container has no GPU — headless Chromium falls back to SwiftShader, which can't sustain the full world. I can read and reason about the shaders but cannot render them here, so the two fixes above must be confirmed in your Chrome (the live HTTPS server auto-reloads on save).

---

## Findings by subsystem

### Core & GPU helpers (`src/core`, `src/gpu`)
**Quality**
- [MED] `Engine.ts:173` — per-frame `[...frameMsRing].sort()` allocates a 120-element copy + O(n log n) sort **every frame** for one p95 stat. `:171-172` `shift()` is O(n) too.
- [MED] `FlyCamera.ts:326-433` — `updateWalk` ~107-line god function mixing physics integration with cosmetic bob/dip/roll/FOV; the whole 434-LOC class bundles input + 2 movement models + effects + pointer-lock SM.
- [LOW] `FlyCamera.ts` — walk tuning is fully named (28-55, excellent) but fly mode hardcodes sensitivity `0.0022`, pitch clamp `1.55`, damp `9`, sprint `6`, wheel `1.15` (156-349). `Engine.ts:51-56,89` — `fov 55`/`near 0.3`/`far 30000`/dpr `1.5` unnamed.
- [LOW] A few `/** what */` docstrings violate the no-docstring rule (`Seed.ts:56,67`).
- **Good:** velocity-Verlet jump (frame-rate-independent apex), order-independent seed streams, `runiform` (CDP-measured win), fail-loud throughout, structural `*Shape` interfaces avoid `any` while patching three internals.

**DRY** — `GroundProbe` type declared twice (`FlyCamera.ts:23 ↔ Hooks.ts:60`); `parseCamString` return ≡ `CamPose` (`Params.ts:55 ↔ Hooks.ts:6-14`); the two `bilerp*` functions share a skeleton (`BufferSample.ts:12-25 ↔ 33-46`); fBm octave loop repeats in `NoiseTSL.ts:70-113`.

**⚠ Latent correctness** — `NoiseJS` (CPU) and `NoiseTSL` (GPU) use **different hash families**, so CPU-built mesh placement and GPU terrain/scatter noise fields do not agree. Confirm no consumer relies on them matching, or unify the hash. **Verify: S.**

### GPU compute passes (`src/gpu/passes`)
**Quality**
- [HIGH] `Scatter.ts:364-841` — `runScatter` god function (~477 LOC) builds 4 near-identical scatter pipelines inline; weight tables baked into the kernel body (`:398-428,514-542,…`) instead of an exported config like `BARK_TABLE`.
- [MED] `FlowRivers.ts:80-579` — ~500-LOC single function; `:503` shadows the `wB` storage buffer from `:92` (readability trap).
- [LOW] dead code: `BarkSynth.ts:206` (`void FloatType`), `BiomeSnow.ts:172` (`DIAG_COMPONENTS=false` dead branch), `Particles.ts:222-242` (always-built `partdbg` debug graph).

**DRY** — grid-kernel boilerplate (`guard`/`cellXY`/`at`/`isBorder`/`OFFS`) duplicated `Erosion.ts:90-113 ↔ FlowRivers.ts:101-126` (and a 3rd copy at `:203-226`); weighted-CDF if-ladder 4× in Scatter; two divergent talus implementations (`Erosion.ts:257 ↔ FlowRivers.ts:539`); Erosion reimplements bilinear sampling instead of reusing `BufferSample`.

**Performance**
- [HIGH-ish] `clumpField` recomputed 4× identically (same salt) across the largest dispatches — `Scatter.ts:397,513,614,727`. Bake it once into a coverage texture (like `buildCanopyMap` already does) and sample in all 4 layers.
- [MED] `Froxels` rebuilds the full 160×90×64 grid every frame with no skip even when `fogK==0` (`:234-240`) — add an early-out + consider half-res/temporal.
- [MED] no `dispose()` anywhere in `gpu/` — boot-only intermediates (fill pyramid, erosion temps) leak on any regenerate.
- [LOW] `Heightfield` does 2 sequential boot readbacks (`:190,192`) that could be `Promise.all`'d (Scatter already parallelizes its 4 counter readbacks — the right pattern).

### Render — post & terrain (`PostStack`, `TerrainMaterial`, `ColorScript`, `ThreePatches`, `HalfResMrt`)
**Quality**
- [HIGH] `PostStack.ts:64-629` — god constructor (~565 LOC); every sub-graph (cloud layer, bounce, aerial, AO upsample, SSCS, velocity reproject, exposure, grade) is a natural standalone builder. The `?cloudview`/`?skyveldbg` debug ladders (`:263-332`) roughly double `aerialNode`'s length.
- [MED] 15 `as unknown as` casts; worst is the fake `reprojectedVelocity = {load} as unknown as typeof depthTex` (`:506`) — root cause is too-narrow `TSLTypes` aliases.
- [MED] post tuning scattered inline (bloom `0.28/0.45/1.5` `:515`, exposure `:557-559`, grade edges/vignette/grain `:582-592`, bounce `:198,459`) — should be a `POST` const block mirroring `DISP`/`KEYFRAMES`.
- [MED] `ThreePatches.ts:53-99` rewrites the `getMaterialCacheKey` prototype globally on undocumented private fields — fragile; add a runtime existence assert so a three bump fails loudly instead of silently regressing the perf win.
- [LOW] `HalfResMrt.ts:35-36,70-73` — `red` (r8) attachment path is built but no caller uses it; the scalar AO layer lands in a full rgba16f attachment (the exact case `red` was for).
- **Good:** the analytic-velocity TRAA fix (`:490-503`) is correctly reasoned (three's `VelocityNode` is garbage for `positionNode`-displaced geometry); half-res for the cloud march + AO + bounce is the single biggest win; `ColorScript` is the data-driven ideal.

**DRY** — `isSkyDepth(d)` test copy-pasted 5× (`:159,193,252,398,607`); `getViewPosition(uv,d)` reconstructed 8× (also a redundant-recompute perf issue in the fused megashader); Rec.709 luma hardcoded twice while three's `luminance()` is also imported.

**Performance** — the fused aerial+AO-upsample+SSCS(12-tap)+bounce produces one very large full-res fragment; profile occupancy and consider half-res SSCS (`:394-440`). Auto-exposure does a 144-tap reduction in a single GPU invocation (`:526-560`) — a parallel/mip reduction is the textbook approach (tiny absolute cost).

### Render — shadows, AO, water (`ShadowSetup`, `CsmCached`, `Gtao`, `Caustics`, `WaterMaterial`)
**Quality**
- [MED] `waterMaterial()` (`WaterMaterial.ts:91-357`) — 266-LOC god function; split into `rippleNormal`/`refraction`/`reflection`/`foam`.
- [MED] GTAO X/Y horizon blocks (`Gtao.ts:182-221 ↔ 223-254`) are ~40 lines of near-identical copy (port fidelity is the stated reason; a `sampleHorizon(sign,comp)` halves it).
- [MED] shadow tuning hardcoded inline (`ShadowSetup.ts:124-134`: `mapSize 2048`, `bias -0.00012`, `normalBias 2.2`, `radius 1.15`, `maxFar*2.2`) — not in the named block with `SUN_TAN`/`MIN/MAX_PENUMBRA_M`. Water foam/SSR thresholds likewise all inline literals.
- **Good — the "no black shadows" law** is enforced and visible: `Gtao.ts:104-105,194-205,258-261` (sky `ao=1`, sub-texel rejection, NaN clamp), `WaterMaterial.ts:240-259` (occluded SSR falls back to the probe field), `ShadowSetup.ts:142-145` (single cloud-shadow NaN→1). `CsmCached` is the model caching design.

**DRY** — `CYC=0.45` re-declares the exported `FLOW_CYC` in the same file (`WaterMaterial.ts:133 ↔ 77`); two-phase flowmap advection implemented 3× (`WaterMaterial.ts:134-144,283-296 ↔ Caustics.ts:219-223`); the 4 caustic gate thresholds duplicated verbatim (`Caustics.ts:246-253 ↔ 262-265`); scene view-Z reconstruction repeated 4× in WaterMaterial.

**Performance** — Caustics re-bakes the full 512² tile **unconditionally every frame** (`Caustics.ts:171-173`) with no submersion/time/sun gate — the one uncached recompute in scope (mirror CsmCached's invalidation). WaterMaterial is the heavy per-pixel path (SSR `Loop(18)` + crowned-horizon march + ~6 redundant water-Y bilerps `:304-326`). AO is correctly half-res (not full-res).

### Render — vegetation & wind (`VegInstance`, `VegMaterials`, `VegPrepass`, `ImpostorRuntime`, `Wind`)
**Quality**
- [MED] `instanceVeg` (`VegInstance.ts:183-271`) ~90-line god function mixing transform with the delicate shadow-cutout/alpha logic; split into `applyTransform`/`applyWind`/`applyShadowCutout`/`applyDebug`.
- [MED] **Wind tuning is hardcoded magic** (`Wind.ts:131-198`: distance gates `380/100/160/140/40/80`, amplitude/profile coeffs throughout) — and wind is the most feedback-iterated subsystem. Move to a `WIND_TUNING` table. (`GUST_SPEED`/`LAG_M` *are* named — good.)
- [MED] structural escape hatches to set undeclared node fields (`castShadowPositionNode` `:229`, `maskShadowNode` `:263,267`, also `Forests.ts:457`) — one typed augmentation interface makes them checked.
- **Good:** shared LOD constants (`R0_FAR/BAND0…`) feed both the GPU cull kernel and CPU dither fade — single source of truth; `fetchInstance`/`applyDitherFade`/`applyInstanceTint` reused by ImpostorRuntime; per-frame work is just uniform writes (no material recompiles); depth prepass measured and scoped to cards only.

**DRY** — yaw rotation hand-inlined 4× (`VegInstance.ts:189-196,219-223 ↔ ImpostorRuntime.ts:69-73,115-119`) — add `rotateYaw(v,c,s)`; octahedral encode/decode CPU↔GPU pair (`Impostors.ts:46-53 ↔ ImpostorRuntime.ts:74-82`) needs a parity test; sqrt-decode inlined 5×; vdata.w AO law repeated 7×.

**Performance** — culling is fully GPU, no hot-path readback (good). The wind chain always fully evaluates ~5 vertex texture taps + 2 sins per veg vertex even when distance gates zero the amplitude (`Wind.ts:131-198`) — far R2 cards pay full cost for ~0 result; a cheap far-ring wind path *if* profiling shows vertex texture-fetch is a bottleneck.

### Sky & atmosphere (`Atmosphere`, `Clouds`, `SunSky`)
**Quality**
- [HIGH] **Clouds density model is hardcoded literals** (`Clouds.ts:235-257`), in sharp contrast to Atmosphere's named physical-constants block (`:49-77`). Worse, **three different extinction multipliers for the same medium** — shadow `-0.045` (`:193`), light `-0.04` (`:330`), view `-0.052` (`:345`) — collapse to one `CLOUD_EXTINCTION`.
- [MED] god methods `Atmosphere.init()` (`:165-332`, 3 inline kernels) and `Clouds.init()` (`:105-199`, 4 inline kernels) — split into `bake*()` methods.
- [MED] heavy `as unknown as` double-casts in Clouds (`windU.dir`/`uTime`/`uDriftBase` not typed as `NV2`/`NF`).
- **Good:** Atmosphere's physical-constants block + `SUN_E` coupling knob is exemplary; phase functions properly factored; dense raymarch loops carry intent comments.

**DRY** — ray-radius `sqrt(t²+r²+2tr·mu)` repeated 4× (`Atmosphere.ts:193,238,302,416`); extinction assembly repeated 4×; CPU `sunTransmittanceCpu` mirrors the GPU bake (inherent split — flag to keep in sync). **Cloud shadows correctly do NOT overlap the main CSM** — they're injected as a sun-term multiplier (good separation).

**Performance** — sky-view LUT re-bake correctly gated to sun-move only (good). **Cloud shadow map re-bakes every ~2.5 s unconditionally** (`Clouds.ts:211-221`, 768²×20-step) even with a static camera/ToD — gate on actual wind drift. Redundant per-pixel sun math: `sunDir.normalize()` per step (`Clouds.ts:283`, `Atmosphere.ts:291,357`) though `sunDir` is already normalized; `atan(sunDir.z,x)` recomputed per sky pixel (`:347`) — precompute as uniforms in `setSun`.

### World & terrain (`Heightfield`, `TerrainTiles`, `MacroMap`, …)
**Quality**
- [HIGH] `TerrainTiles` god constructor (`:75-382`, ~307 LOC) — split `buildTileMaterial`/`buildFarShell`/`applyDebugViews`.
- [MED] **dead constants violating data-driven intent** — `SNOWLINE_BASE/SUMMIT_MAX/VALLEY_FLOOR` (`WorldConst.ts:18-22`) never imported; the live snowline is re-encoded as magic lapse numbers in `BiomeSnow.ts:79-82`. Hardcoded `64` in `TerrainTiles.ts:393,435,438` should be `WORLD_SIZE/MIN_TILE`.
- [MED] CDLOD vertex morph uses bare `SPLIT_K` (`:134-135`) but the CPU splitter uses `SPLIT_K·errBoost` + altitude slack (`:468-475`) — morph completion no longer coincides with the real LOD handoff on rough/high tiles → potential popping (skirts hide cracks, so quality polish not a crack bug).
- **Good:** separate RNG streams per macro component; single-source height buffer; thorough *why* comments; constants imported where used.

**DRY** — [HIGH] whole grid-mesh builder duplicated byte-for-byte (`CanopyShell.ts:47-69 ↔ ShadowProxy.ts:23-45`, 3rd near-clone `WaterSurface.ts:30-60`); `GRID=512` declared twice. CPU bilinear sampler duplicated (`Heightfield.heightAtCpu ↔ waterYAtCpu`); neighbor-index block 4×; ridged-fBm loop 2× in MacroMap.

**Performance** — height synthesis runs twice (full-res + sim-res) — unavoidable but the biggest one-time cost. Camera-height readback is **one-time, not per-frame** (the "height readback for camera" the harness logs is at `Heightfield.ts:189-193`, feeding CPU mirrors — correct design). `WaterSurface.update` allocates 6 `new Vector4` per frame (`:111`) — mutate a scratch.

### Vegetation generation (`Forests`, `Skeleton`, `Species`, `VegLibrary`, …)
**Quality**
- [HIGH] **`Forests.ts` is the prime god file** — `init()` `:317-907` (~590 LOC) = layout math + draw/material wiring + indirect assembly + 6 TSL cull kernels + per-frame update/readback. The group-layout offsets (`:127-190`: `146+cls*4`, caps `8192/6144/…`) are an undocumented magic-number forest.
- [MED] `hasR2` class-set literal duplicated (`:345 ↔ 478`); 42 `as unknown as` casts; `object`-typed `kernels`/`csm` fields (`:266,275`) lose all safety.
- [MED] `growBranch` (`Skeleton.ts:70-282`, ~210 LOC) — extract the foliage-anchor placement block (`:219-279`).
- **Good:** `Species.ts` is a clean fully data-driven preset table (no per-species code branches in the grammar) — the standard done right; `MeshGrower` is a tidy single-responsibility accumulator.

**DRY** — [HIGH] group-index arithmetic encoded twice, CPU + GPU (`:127-159 ↔ 757-877`) — see ★#2; trees-cull ↔ extras-cull structural twins; `inFrustum ↔ inCascade` (`:686-695 ↔ 701-711`); `{geo,tris,make,castShadow}` part literal repeated 16× in VegLibrary.

**Performance** — [HIGH] **all CPU tree growth is synchronous at boot, no workers** — 72+ `buildTree` calls + hero mesh-leaf generation block the main thread (`VegLibrary.ts:182-227`); the dominant startup cost. [MED] impostor capture rebuilds variant-0's LOD1 tree identically (`:234 ↔ 196`) — 6 free builds saved by caching. Per-instance uniqueness cost is LOW by design (4 skeletons/species reused via GPU instancing — preserve this).

### Vegetation ground/understory/impostors (`GroundRing`, `GroundCover`, …)
**Quality**
- [HIGH] **`GroundRing.ts` god file** — `init()` `:340-775` (~435 LOC) = buffer alloc + 6 closures + 4 cull kernels + draw assembly; plus a 140-line `grassMaterial`. 50 `as unknown as`.
- [MED] **coupled salt literals 200 lines apart** — debris `salt^0x5dd5` at cull `:519` must match draw `:721`; far `^0x6f21` at `:611 ↔ 694`; grass density table `[0.18,0.7,…]` verbatim at `:462 ↔ 639`. A mismatch silently shifts every instance. Promote to named consts.
- [LOW] gallery-only legacy path (`GroundCover.grassPatch`/`scatterInstances`/`grassMaterial`) sits among the live builders.
- **Good:** `RockBuilder.ts` is exemplary (data-driven `ROCK_PRESETS`, no casts); stats readback throttled + non-blocking with `finally` reset.

**DRY** — [HIGH] cull-kernel preamble triplicated ~90 lines (`grassK/debrisK/farK` `:419-453/511-545/603-635`) — extract `sampleCell()`; [HIGH] `GroundRing.bandFade` is a near-verbatim copy of `VegInstance.applyDitherFade` (`:159-187 ↔ 127-151`) — one should call the other; `byBio` duplicates the private `Scatter.byBiome` (`:386-392 ↔ Scatter.ts:156-162`) — export and share.

**Performance** — [HIGH] **whole grass grid re-culled every frame unconditionally** — `update()` dispatches `grassK` (3072²≈9.44M threads) + `farK` (590k) + `debrisK` (262k) with no camera-moved gate (`:948-960,507`); a stationary camera re-derives ~10M identical results/frame. **Gating the cull on camera displacement is the single biggest runtime win in the repo.** Instance buffers are allocated once, no churn (good). Bake-time: `FoliageCards.dilate` allocates a full-atlas copy per pass (`:168`); `captureImpostor` serializes 192 GPU→CPU stalls.

### Debug scenes & harness (`src/debug`, `tools`)
**Quality**
- [HIGH] `GalleryScene.ts:94-625` — ~530-LOC god function; split into per-row modules.
- [MED] unguarded `barks.get(...) as BarkTextures` casts (`:225,316,…`) risk a crash if a layer is missing (parallel `atlases.get` *are* guarded — inconsistent); `as unknown as {…}` debug-handle casts (`TerrainScene.ts:42,60,62,213`) → typed `EngineDebug`.
- [LOW] `__laasDbg` assigned twice (`TerrainScene.ts:62,213`); gallery `ctx.progress` regresses `0.97→0.95`; `tools/probe-sun.ts:12` holds the only `any` in scope.
- **Good — the harness is a genuine asset:** `launch.ts:16-27` documents the WebGPU-headless gotcha and caches the recipe; `shoot.ts` `--framealign`/`--gpusample` is a serious A/B rig; determinism enforced end-to-end.

**DRY** — boot-readiness wait duplicated in 10 files; identical boot framing across 6 shadow probes; `csm/csm2/csm3` near-duplicates (csm3 supersets the others); pixel-diff implemented twice; `parseArgs` byte-identical `shoot.ts:19-39 ↔ compare.ts:19-39`; the esbuild `__name` trap worked around 4 different ways. → extract `tools/harness.ts` (★#4) and prune resolved probes.

**Performance** (offline, low stakes) — CPU-field probes (`find-water`, `probe-line`, `probe-wetmargin`) boot the entire pipeline (180-240 s) to read one CPU value; a `?scene=heights` fast boot would cut them to seconds.

---

## Effort-sorted backlog

**Small (mechanical, low risk)**
- Animate the LOD dither (the "dots" fix) — `VegInstance.ts:132`, `GroundRing.ts:179`.
- Name magic numbers into tables: wind (`Wind.ts`), shadow (`ShadowSetup.ts`), post (`PostStack.ts`), cloud (`Clouds.ts`), the coupled GroundRing salts, camera `fov/near/far`.
- Resolve dead `WorldConst` datums; replace hardcoded `64` with `WORLD_SIZE/MIN_TILE`.
- Remove dead code: `BarkSynth` `void FloatType`, `BiomeSnow` `DIAG_COMPONENTS`, duplicate `__laasDbg`, gallery progress regression.
- Unify `CYC/FLOW_CYC`; share types (`GroundProbe`, `CamPose`).
- `Promise.all` the two Heightfield readbacks; cache variant-0 LOD1 tree for impostor capture.

**Medium**
- Gate the GroundRing cull, Froxels dispatch, Caustics bake, and Clouds shadow re-bake on actual change. **(biggest runtime wins)**
- Single source of truth for the veg group/compact layout (★#2).
- Hoist `GridKernel` + `worldGrid()` + shared TSL helpers (`rotateYaw`, `sampleCell`, `isSkyDepth`, `viewPosAt`, two-phase flowmap, caustic gates, `neighbors`, `ridgedFbm`).
- Typed wrappers to kill `as unknown as` density; typed material-extension + `EngineDebug` interfaces.
- Extract `tools/harness.ts`; prune resolved probes.
- Cloud slab-boundary stabilization (the mountain-top flicker fix).
- Verify CPU↔GPU noise hash parity.

**Large (structural)**
- Split the four god files: `Forests.ts`, `GroundRing.ts`, `Scatter.ts`, `PostStack.ts` (and `GalleryScene.ts`, `TerrainTiles.ts`, `waterMaterial`, the two `init()` god methods).
- Move CPU tree growth off the main thread (worker pool / yield).
- Bake `clumpField` once into a coverage texture sampled by all 4 scatter layers.
