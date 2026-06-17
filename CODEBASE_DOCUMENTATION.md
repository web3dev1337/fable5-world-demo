# LAAS Codebase Documentation

Fully procedural 4×4 km open world in the browser on WebGPU. three.js `WebGPURenderer` + TSL node materials + raw WGSL compute. TypeScript strict, **zero `any`**, no WebGL fallback. Everything (mesh, texture, light) is generated at boot; the repo ships no image/model/audio assets. Reproducible from `?seed=N`.

~19,200 LOC across 75 `src/` files + a 19-file `tools/` verification harness.

## Quick Navigation

```
ENTRY:   index.html, src/main.ts
CORE:    src/core/         engine loop, camera, seeding, diagnostics, gate
GPU:     src/gpu/          TSL helpers + src/gpu/passes/ (compute: scatter, erosion, rivers, GI, froxels, particles, bakes)
WORLD:   src/world/        heightfield synth, CDLOD tiles, macro map, water/canopy/shadow proxies, constants
SKY:     src/sky/          Hillaire atmosphere LUTs, volumetric clouds, time-of-day driver
RENDER:  src/render/       post stack, materials (terrain/water/veg), shadows, GTAO, caustics, wind, instancing
VEG:     src/vegetation/   tree grammar + mesh builders, ground/understory, impostors, foliage cards, library
DEBUG:   src/debug/        scene registry + diagnostic scenes, HUD, bookmarks
TOOLS:   tools/            headless WebGPU screenshot/compare/diff harness + forensic probes
```

## Boot Flow

`index.html` → `src/main.ts:boot()`:
1. `browserGate()` (BrowserGate.ts) — mobile / non-Chromium / missing-WebGPU notices before any work.
2. `probeWebGPU()` (Diagnostics.ts) — adapter + required-limits, fail-loud on no adapter.
3. `Engine.create()` (Engine.ts) — `WebGPURenderer`, scene, camera, hooks.
4. `buildScene(?scene=…)` (debug/Scenes.ts) — default `world` → `buildTerrainScene` (debug/TerrainScene.ts), the integration hub that wires every subsystem.
5. `FlyCamera`, `HUD`, `installBookmarks`, then `engine.start()` (animation loop).

Everything tool-facing is mirrored to `window.__laas` (Hooks.ts): `ready`/`error`/`settle`/`stats`/`getPose`/`setPose`/`groundProbe`. Debug handles hang off `window.__laasDbg`.

## Core Systems

### Engine & frame loop — `src/core/`
- **Engine.ts** (210) — owns `WebGPURenderer`, `Scene`, `PerspectiveCamera`, time, per-frame stats. Subsystems attach via `onUpdate(fn)`. Private ctor + static async `create()` factory. `settle(frames)` for the test harness. Mirrors stats to `window.__laas.stats`. Per-pass GPU timing resolved every frame (documented pool-reset reason).
- **FlyCamera.ts** (434) — walk+fly rig: walk = gravity / velocity-Verlet jump / sprint / wade / stride-matched bob-dip-roll-FOV; fly = free flight + wheel speed. Pointer-lock cooldown state machine. `?cam=` pose I/O. Module-scope scratch vectors → zero per-frame alloc.
- **GpuProfiler.ts** (162) — labels GPU timestamp UIDs by monkey-patching the backend, aggregates per-pass durations into `stats.gpuPasses`. Uses structural `*Shape` interfaces to reach three internals without `any`.
- **Diagnostics.ts** (139) — `probeWebGPU`, `buildRequiredLimits`, `failLoud` overlay, global error hooks. Fail-loud everywhere (no silent black frames).
- **Seed.ts** (128) — FNV-1a string hash, murmur3 `fmix32`, `sfc32` PRNG (`Rng`), order-independent named-stream derivation (`WorldSeed`). Determinism backbone.
- **NoiseJS.ts** (92) — CPU value/fBm/ridged noise for one-time mesh generation. ⚠ Uses a *different* hash family than the GPU `NoiseTSL` — the two noise fields do not match (see review §latent).
- **Params.ts** (61) — URL-param parsing (`seed/scene/preset/cam/T/...`).
- **Hooks.ts** (93) — the `window.__laas` contract types + `initHooks`.
- **BootUI.ts** (35) — boot overlay progress bar/message.
- **BrowserGate.ts** (102) — pre-boot environment gate. ⚠ `navigator.gpu` only exists in a secure context (`https://` or `http://localhost`) — the gate fires on plain-HTTP LAN IPs even in a capable Chrome.
- **CameraSignature.ts** (25) — allocation-free camera matrix signature for skipping frustum/cull setup when projection and view matrices are unchanged.

### GPU helpers — `src/gpu/`
- **TSLTypes.ts** (30) — node-type aliases (`NF/NV2/NV3/NV4/NB/NI`) + the single sanctioned vector-transcendental cast (`vexp3`). The intended containment point for `as unknown as` TSL-typing-gap casts.
- **RenderUniform.ts** (24) — `runiform()`: a `uniform()` retagged into three's `renderGroup` update group (per-shader revalidation vs per-object). CDP-measured ~3.7 ms/frame win.
- **BufferSample.ts** (46) — `bilerpFloatBuffer`/`bilerpVec2Buffer`/`uvToGrid` — bilinear sampling of non-filterable storage buffers in TSL.
- **noise/NoiseTSL.ts** (120) — GPU TSL noise builders (Hoskins sinless hashes, value/fBm/ridged/warp). Pure expression builders (no `.toVar()`) so they work in both compute `Fn` and material graphs.

### GPU compute passes — `src/gpu/passes/`
The procedural-generation engine. All dispatch via `renderer.compute(N)` (default [64,1,1] workgroup).
- **Scatter.ts** (841) — clustered-Poisson scatter for 4 layers (trees/understory/extras/stones) via atomic-append into per-layer `vec4` buffers; builds the canopy coverage texture. The single largest function in the repo (`runScatter` ~477 LOC). [god function]
- **FlowRivers.ts** (579) — depression fill (multigrid pyramid) → 3M-particle flow accumulation (260-step descent) → river carve → talus relax → moisture. Heaviest boot kernel.
- **Erosion.ts** (308) — pipe-model hydraulic + thermal erosion, 5 ping-ponged per-cell kernels.
- **ProbeGI.ts** (386) — irradiance probe field via heightfield ray-march (16 dirs × 16 steps), SH-L1 EMA into 3 `Storage3DTexture`s, time-sliced (3072 probes/frame). Sampled in materials via `irradiance()`.
- **Froxels.ts** (257) — froxel volumetrics, 160×90×64 grid **rebuilt every frame** (per-froxel density + sun horizon march + cloud shadow), front-to-back integrated.
- **Particles.ts** (272) — 131,072 GPU particles (snow/pollen/leaf), toroidal wind-advected, drawn as one instanced billboard mesh.
- **BarkSynth.ts** (250) — per-species 2048² tileable bark maps from an exported `BARK_TABLE` recipe (the data-driven pattern done right).
- **BiomeSnow.ts** (184) — biome/snow/veg-density/rock classification into an rgba8 texture.
- **NoiseBake.ts** (115) — bakes tileable noise textures + pre-derived gradients (replaces ~52 ms/frame of live fBm with fetches).
- **HeightSynthesis.ts** (50) — bakes `macroTerrain` into height+hardness buffers (run at full-res and sim-res).

### World & terrain — `src/world/`
- **WorldConst.ts** (64) — central const table: `WORLD_SIZE/HALF`, `HEIGHT_RES/SIM_RES`, vertical datums, `Biome` enum, `QualityConfig`/`qualityConfig()`. ⚠ Some datums (`SNOWLINE_BASE/SUMMIT_MAX/VALLEY_FLOOR`) are declared but never imported — the live snowline is re-encoded as magic lapse-rate numbers in BiomeSnow.
- **Heightfield.ts** (567) — owns all terrain GPU state; `static generate()` orchestrates synth→erosion→hydrology→classification; builds waterY field + CPU mirrors; exposes TSL samplers (`sampleHeight*`/`sampleWaterY*`) and `heightAtCpu`. One-time camera-height readback (~64 MB) feeds CPU mirrors so nothing reads back per frame.
- **TerrainTiles.ts** (493) — CDLOD quadtree (`update`/`recurse`), instanced patch mesh + crack-hiding skirts + vertex morph, far vista shell, range-mip pyramid. God constructor (~307 LOC). Camera-move-gated rebuild (≥20 m).
- **MacroMap.ts** (407) — art-directed macro graph: seed-jittered spline valleys, alpine ridges, karst towers (`macroTerrain`, `valleyFields`, `zoneMasks`).
- **CanopyShell.ts** (133) — far-forest aggregate as a static 512² lit surface, dithers in past impostors.
- **WaterSurface.ts** (114) — 6-level camera-following water clipmap on one shared 128² grid.
- **ShadowProxy.ts** (67) — coarse 512² height-lifted grid that casts CSM shadows so the real CDLOD mesh doesn't.

### Sky & atmosphere — `src/sky/`
- **Atmosphere.ts** (429) — Hillaire LUT atmosphere (km units): transmittance 256×64, multiple-scatter 32×32, sky-view 192×108 (re-baked on sun move only). `backgroundNode`/`skyColor`/`aerial`/`sampleTransmittance` + CPU `sunTransmittanceCpu` for light color. Exemplary named physical-constants block.
- **Clouds.ts** (353) — 2-layer raymarched Worley–Perlin volumetric clouds. `CLOUD_BOTTOM=1250`, `CLOUD_TOP=1900`. `march()` runs **half-res in the post chain** with per-frame jitter, relies on TRAA to resolve. Top-down shadow map re-baked every ~2.5 s. ⚠ density model is hardcoded literals (not data-driven); three different extinction coefficients for the same medium.
- **SunSky.ts** (131) — time-of-day driver: hours → sun world dir, transmittance-tinted `DirectionalLight` + `HemisphereLight`, IBL/PMREM re-bake on ToD change.

### Render & post — `src/render/`
- **PostStack.ts** (654) — the whole HDR post pipeline built in one constructor: scene MRT → merged half-res MRT (clouds march + GTAO + SS bounce) → aerial perspective + froxel fog → joint-bilateral AO upsample → SSCS contact shadows → bounce composite → TRAA (analytic camera-reprojection velocity) → bloom → GPU auto-exposure → filmic grade + AgX. God constructor (~565 LOC) with inline `?cloudview`/`?skyveldbg` debug ladders.
- **TerrainMaterial.ts** (378) — terrain TSL shading shared by near tiles + far shell: continuous splat classes, macro/meso/micro detail, analytic normal perturbation. Data-driven `DISP` block.
- **ColorScript.ts** (88) — per-ToD grade keyframe table + lerp. Cleanest file in render.
- **ThreePatches.ts** (100) — two monkeypatches on three 0.184 internals to kill a shadow-pass material-hash storm. Fragile by nature; mitigated by pinned three + idempotency guard + `THREE-NOTES.md`.
- **StaticRefresh.ts** (147) — veg/grass per-frame refresh skip (Phase 8). three's `NodeMaterialObserver` refreshes every node-material draw every frame (`hasNode`); for ~700 veg/grass draws that's the dominant CPU encode cost. Patches `needsRefresh` so only the FIRST draw per SHARED bind group refreshes (flushing the per-frame jitter/time/camera); the rest read the flushed buffer and skip — pixel-identical (render order guarantees flush-before-read), motion-safe, ~62% veg encode cut on a frozen vista. Veg draws share the renderGroup buffer because the per-draw compacted-list `base` offset was moved out of renderGroup → object group (see VegInstance/groundring). `markVegRefresh` tags draws; `tickVegRefresh` advances the flush clock each frame; `?vrf=off` disables.
- **HalfResMrt.ts** (138) — custom `TempNode` driving one half-res multi-attachment pass (merges 3 former passes). Has an unused `red` (r8) attachment path the AO layer could use.
- **ShadowSetup.ts** (199) — 4-cascade CSM + PCSS contact-hardening filter (blocker search → world-metric penumbra → Vogel PCF). Enforces the "no black shadows" law.
- **CsmCached.ts** (159) — `CachedCsmShadowNode`: per-cascade re-render cadence (cascades render at 1/2, 1/3, 1/6 rate) with forced refresh on sun/FOV/center change. ~13–19 ms/frame saved. The model caching pattern in the repo.
- **Gtao.ts** (349) — ground-truth AO as a half-res fragment expression (faithful three 0.184 port + 3 documented deviations).
- **Caustics.ts** (283) — analytic water caustics, 512² tile re-baked **every frame**.
- **WaterMaterial.ts** (357) — stream/lake material: refraction (Beer–Lambert), SSR march + probe fallback, two-phase flowmap ripples/foam, wet margins. God function (~266 LOC).
- **VegInstance.ts** (271) — rewires a veg material for GPU-driven compacted-indirect instancing: yaw+lean transform, **dithered LOD ring crossfade** (`applyDitherFade`), per-instance tint, wind hook, shadow-pass cutout. `instanceVeg` is a ~90-line god function.
- **VegMaterials.ts** (338) — TSL builders for bark/rock/deadwood/flower/mushroom/foliage/card materials + shared `translucency`/`hueShift`.
- **VegPrepass.ts** (105) — depth-only prepass twin + WGSL `@invariant` clip-pos injection (depth-match correctness).
- **ImpostorRuntime.ts** (134) — octahedral impostor draw material; reuses VegInstance's fetch/fade/tint.
- **Wind.ts** (198) — one global wind field: advected-fbm gusts, lean ∝ strength², per-instance sway, lagged branch motion, flutter. The most feedback-iterated subsystem; tuning is hardcoded (not data-driven).

### Vegetation generation — `src/vegetation/`
- **Forests.ts** (1016) — GPU-driven cull/LOD/shadow-caster orchestration + draw/indirect setup + TSL cull kernels + stats readback. The repo's prime **god file** (`init()` ~590 LOC, 5+ responsibilities). Per-frame culling is fully GPU (6 kernels), readback throttled to every 90 frames.
- **GroundRing.ts** (992) — camera-following near-field grass (3 LOD bands + super-tuft) + debris (5 classes) as toroidal clipmaps, re-culled on GPU into indirect draws every frame. **god file** (`init()` ~435 LOC, 3 near-duplicate cull kernels). The grass grid dispatches ~9.4M threads/frame.
- **VegLibrary.ts** (557) — boot-time pool baking: 6 species × 4 variants × 3 LODs + impostors/understory/deadfall/rocks/stones/branches → `VegLib`. All CPU tree growth is synchronous at boot.
- **Understory.ts** (382) — shrub species (hazel/pink/juniper), fern rosette, 4 flower builders. Data-driven species.
- **TubeMesh.ts** (351) — `MeshGrower` accumulator + skeleton→generalized-cylinder meshing via parallel transport.
- **Skeleton.ts** (345) — the recursive branching grammar (`growSkeleton`/`growBranch`). `growBranch` ~210 LOC.
- **GroundCover.ts** (332) — grass blade/clump + debris geometry & materials. Also holds a gallery-only `InstancedMesh` preview path (legacy, not the live GroundRing path).
- **Impostors.ts** (321) — 8×8 hemi-octahedral impostor bake (albedo/normal/depth) + BFS RGB dilation.
- **FoliageCards.ts** (318) — per-species 2×2 cluster-card atlas bake (render → readback → flip → dilate).
- **Species.ts** (314) — six declarative species presets (`SPRUCE…SNAG`) + `TREE_SPECIES`. 100% data, zero logic — the data-driven standard done right.
- **RockBuilder.ts** (240) — welded icosphere displaced by macro/strata/ridge/fracture fields; data-driven `ROCK_PRESETS`. The cleanest file in the vegetation set.
- **LeafMesh.ts** (210) — leaf/needle geometry into a `MeshGrower`.
- **Dressing.ts** (189) — hanging vines + mushrooms.
- **VegTypes.ts** (174) — shared grammar schema (every field carries a *why* comment).
- **TreeBuilder.ts** (141) — `buildTree`: species+seed+LOD → bark/foliage geometries.
- **Deadfall.ts** (103) — fallen logs (3 decay states) + stumps; decay drives a moss channel.

### Debug & scenes — `src/debug/`
- **Scenes.ts** (36) — `?scene=` registry/dispatcher (`world`/`sanity`/`terrain`/`gallery`/`shadowtest`).
- **TerrainScene.ts** (319) — the real world; `?scene=world`. Orchestrates heightfield → sky → scatter → GI → tiles/water → forests/grass → clouds → CSM → particles → froxels → post. The integration hub.
- **GalleryScene.ts** (625) — specimen gallery (every species × seeds on pedestals + rocks/ground/dead/hero/impostor rows). Art-review surface. God function (~530 LOC).
- **SanityScene.ts** (179) — Phase-0 end-to-end GPU-stack proof (boots fast; used for verification).
- **ShadowTestScene.ts** (138) — minimal shadow repro with control casters.
- **Bookmarks.ts** (137) — keys 1–9 composed viewpoints + F 92 s Catmull-Rom flythrough.
- **ScatterDebug.ts** (118) — `?view=scatter` instanced-marker view of GPU scatter buffers.
- **HUD.ts** (105) — diagnostics HUD; F3 toggles per-pass GPU timings. Clean provider pattern.

### Verification harness — `tools/`
A genuine asset — headless WebGPU A/B testing rig.
- **launch.ts** — WebGPU-capable Chromium launcher (probes recipes; full `channel:'chromium'` headless required — the default headless shell yields `adapter=null`) + `laasUrl` builder; caches the winning recipe to `.cache/webgpu-flags.json`.
- **shoot.ts** — the workhorse: boot → ready → settle → screenshot → stats JSON; `--framealign` pins frame-phase for bit-comparable A/B, `--gpusample` medians GPU timings.
- **compare.ts** / **diff.ts** — side-by-side compositor + pixel sampler / amplified pixel-delta.
- **herotris.ts** / **vegtris.ts** — CPU-only tri-count reporters.
- **probe-*.ts** (11 files) — one-off forensic bug-hunt probes (mostly from a resolved 2026-06-11 shadow investigation; abandoned but kept). `probe-pointerlock.ts` is a genuine headed PASS/FAIL regression test.
- **find-water.ts** / **probe-line.ts** — CPU-field scanners.

## Conventions

- **Strict TS, zero `any`.** Verified: 0 `any` tokens in `src/`. The escape hatch is `as unknown as N*` at TSL/three type-gap boundaries, conceptually centralized in `TSLTypes.ts` (but in practice spread across call sites — the biggest density is in `Scatter.ts`, `Forests.ts`, `GroundRing.ts`, `PostStack.ts`).
- **Comments explain *why*, not *what*** — strongly adhered to; most carry bug history / probe findings / measured rationale.
- **Data-driven constants** — done excellently in some places (`Species.ts`, `ROCK_PRESETS`, `BARK_TABLE`, `ColorScript` keyframes, `WorldConst`, `DISP`) and poorly in others (cloud density, wind tuning, scatter weight tables, shadow/water magic numbers).
- **Fail-loud** — pervasive; no silent black frames.
- **Determinism end-to-end** — seed-driven, `?freeze=1`, frame alignment for the test harness.

## Commands

```
npm run dev        # vite dev server (:5173). For WebGPU, open over https:// or http://localhost
npm run build      # tsc --noEmit && vite build
npm run typecheck  # tsc --noEmit
npm run shoot      # tsx tools/shoot.ts — headless WebGPU screenshot + stats
npm run compare    # tsx tools/compare.ts
```

⚠ **WebGPU needs a secure context.** Open `https://…` or `http://localhost`. A plain-HTTP LAN IP (`http://172.x.x.x:5173`) hides `navigator.gpu` and trips the gate even in a capable Chrome.

## Useful URL params

`?seed=N` · `?T=hours` (0–24) · `?shot=1..9` · `?cam=x,y,z,yaw,pitch[,fov]` · `?preset=low|high|ultra` · `?freeze=1` · `?hud=1` · `?scene=world|sanity|terrain|gallery|shadowtest` · `?ablate=taa|clouds|ao|bounce|bloom|pcss` (subsystem bisect) · `?view=scatter` · `?nogate=1`.

## Known issues / open items

See **STATUS.md** (the model's working memory) for the live issue list and diagnosis logs, and **docs/CODE_REVIEW.md** for the full quality/DRY/performance/opportunities audit and the live visual-artifact diagnosis (LOD-dither "dots" + cloud flicker at altitude).
