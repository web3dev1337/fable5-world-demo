/**
 * Forests — GPU-driven rendering of the scattered world (spec §3.6 core).
 *
 * Per frame, compute passes run before render:
 *   clear counters → cull each scatter layer → write indirect args.
 * The cull kernel does: per-class distance bound → frustum sphere test (6
 * planes) → terrain-occlusion march (heightfield ray test camera→crown-top,
 * the "Hi-Z" of a heightfield world) → LOD ring classification with overlap
 * bands → atomic append of the instance slot into per-(pool,ring) compact
 * regions. Draw instance counts go straight into an indirect buffer
 * (geometry.setIndirect) — instance data and counts never touch the CPU.
 * Cull granularity = instance (tree/shrub/rock), not 64-tri meshlets — the
 * deviation and rationale are documented in DEVIATIONS D-5.
 *
 * SHADOW CASTERS ARE CULLED PER CASCADE, not by the view frustum: the same
 * cull kernel also tests every instance against each CSM cascade's ortho
 * frustum (24 extra plane uniforms, refreshed from the cascade cameras each
 * frame — one frame stale, hidden inside the CSM lightMargin slack) and
 * appends into per-(pool,ring,cascade) caster regions. Each casting draw
 * gets 4 shadow-only sibling meshes on layers 2+c that ONLY cascade c's
 * shadow camera renders (ShadowNode keeps a custom camera layer mask), while
 * the main meshes stop casting. This fixes shadows of off-screen casters
 * (sun behind you, golden-hour edges) — the view-frustum compact lists used
 * to silently drop them from every cascade map — and skips the
 * camera-occlusion march for casters (a ridge-hidden tree still casts into
 * the visible slope).
 *
 * LOD rings (dithered crossfades in the materials):
 *   trees:  R0 hero ≤26 m (full bark + cards + real mesh leaves, ≥100k tris)
 *           → R1 full cards ≤150 m → R2 branch-cards ≤460 m → octahedral
 *           impostors beyond (4-tile view blend, relit — D-4 runtime)
 *   understory: single ring with per-class max distance
 *   extras: boulders/slabs swap to low-detail rock at 120 m, live to 700 m
 *
 * This file is a thin orchestrator: the compact-region index math and named
 * layout constants live in forests/layout.ts (shared with the cull kernel so
 * the two never drift), the crown shadow proxy in forests/crownProxy.ts, the
 * draw/material wiring in forests/drawSetup.ts, and the TSL cull/LOD kernels
 * in forests/cullKernels.ts.
 */

import {
  Color,
  Group,
  Mesh,
  Vector3,
  Vector4,
} from 'three';
import type { PerspectiveCamera } from 'three';
import { Frustum, Matrix4 } from 'three';
import {
  IndirectStorageBufferAttribute,
  MeshStandardNodeMaterial,
  StorageBufferAttribute,
  type Renderer,
  type StorageBufferNode,
  type StorageTexture,
} from 'three/webgpu';
import { IrradianceNode } from 'three/webgpu';
import {
  instancedArray,
  normalWorld,
  positionWorld,
  storage,
  uniform,
  uniformArray,
  varying,
  vec3,
  vec4,
} from 'three/tsl';
import type { Heightfield } from '../world/Heightfield';
import type { ProbeGI } from '../gpu/passes/ProbeGI';
import { canopyAt, type ScatterLayer, type ScatterResult } from '../gpu/passes/Scatter';
import { impostorQuad, impostorRuntimeMaterial } from '../render/ImpostorRuntime';
import { instanceVeg, updateVegViewPos } from '../render/VegInstance';
import type { NF, NV3 } from '../gpu/TSLTypes';
import type { VegLib } from './VegLibrary';
import {
  CASCADES,
  EXTRAS_BASE,
  GROUPS,
  IMPOSTOR_BASE,
  MAIN_GROUPS,
  TREE_MAIN_BASE,
  UNDER_BASE,
  capOf,
  casterGroupOf,
  groupOf,
} from './forests/layout';
import {
  CROWN_SHADOW_DENSITY,
  crownProxyGeometry,
  type CrownDims,
} from './forests/crownProxy';
import {
  fadeFor,
  geoView,
  makeAddDraw,
  proxyCasterMat,
  type DrawSpec,
} from './forests/drawSetup';
import { buildCullKernels } from './forests/cullKernels';

export class Forests {
  readonly group = new Group();
  /** depth twins render before color draws (renderOrder; kept in one child
   *  group so a future bundle path can rely on traversal order too) */
  private prepassGroup = new Group();

  private compact!: StorageBufferNode<'uint'>;
  private counters!: ReturnType<StorageBufferNode<'uint'>['toAtomic']>;
  private kernels: object[] = [];
  private camU = uniform(new Vector3());
  private planesU = uniformArray(
    Array.from({ length: 6 }, () => new Vector4()),
  );
  /** 6 planes × 4 cascade ortho frusta; w=-1e9 ⇒ reject-all until CSM exists */
  private planesCsmU = uniformArray(
    Array.from({ length: 6 * CASCADES }, () => new Vector4(0, 0, 0, -1e9)),
  );
  private csm: object | null = null;
  private frustum = new Frustum();
  private projView = new Matrix4();
  private cascM = new Matrix4();
  private cascFrustum = new Frustum();
  private indirectAttr!: IndirectStorageBufferAttribute;
  private groupTris = new Float32Array(GROUPS);
  private groupCaps = new Uint32Array(GROUPS);
  private reading = false;
  private frame = 0;
  private hud: Record<string, number> = {};
  // pose-gate (see GroundRing): the cull is a pure function of the camera +
  // cascade frusta over a static world, so its indirect draw args are
  // bit-identical when every cull input is unchanged — skip the dispatch and
  // reuse last frame's args. Bit-exact compare ⇒ zero visual difference.
  private lastCamX = NaN;
  private lastCamY = NaN;
  private lastCamZ = NaN;
  private lastPlanes = new Float32Array(6 * 4);
  private lastCascPlanes = new Float32Array(6 * CASCADES * 4);
  // motion cadence (see GroundRing): re-cull at most every other frame while
  // moving — a 1-frame-stale visible set only lags instances at the cull ring
  // boundary by one frame, imperceptible during motion.
  private framesSinceCull = 99;

  constructor(
    private hf: Heightfield,
    private scatter: ScatterResult,
    private lib: VegLib,
    private gi: ProbeGI | null,
    private canopyTex: StorageTexture | null = null,
  ) {}

  private patchGI(mat: MeshStandardNodeMaterial): void {
    const gi = this.gi;
    if (!gi) return;
    let irr = gi.irradiance(positionWorld as unknown as NV3, normalWorld as unknown as NV3);
    if (this.canopyTex) {
      // probe field is canopy-aware (crown-slab extinction in the gather) —
      // this is only the 4 m-texel residual the 16 m probe grid can't carry
      irr = irr.mul(
        canopyAt(this.canopyTex, (positionWorld as unknown as NV3).xz)
          .mul(0.12)
          .oneMinus(),
      ) as typeof irr;
    }
    // vertex-stage probe GI (Phase 7 perf): the probe grid is 16 m and the
    // canopy residual 4 m — across ≤2 m cards or cm-tessellated hero meshes
    // vertex eval + interpolation is identical, and drops 4 texture taps
    // from every overdrawn foliage fragment
    const irrV = varying(irr as unknown as Parameters<typeof varying>[0]);
    (mat as unknown as { setupLightMap: () => unknown }).setupLightMap = () =>
      new IrradianceNode(irrV as unknown as ConstructorParameters<typeof IrradianceNode>[0]);
  }

  init(renderer: Renderer): void {
    void renderer;
    this.group.add(this.prepassGroup);
    const lib = this.lib;

    // ---- compact regions / group tables ------------------------------------
    const offsets = new Uint32Array(GROUPS);
    let off = 0;
    for (let g = 0; g < GROUPS; g++) {
      offsets[g] = off;
      this.groupCaps[g] = capOf(g);
      off += capOf(g);
    }
    this.compact = instancedArray(off, 'uint');
    this.counters = instancedArray(GROUPS, 'uint').toAtomic();
    const offBuf = storage(new StorageBufferAttribute(offsets, 1), 'uint', GROUPS);
    const capBuf = storage(
      new StorageBufferAttribute(this.groupCaps.slice(), 1),
      'uint',
      GROUPS,
    );

    // per-class cull info: (height, radius, maxDist, hasR2)
    const clsInfo = new Float32Array(24 * 4);
    for (let c = 0; c < 24; c++) {
      clsInfo[c * 4 + 0] = this.lib.clsHeight[c] ?? 1;
      clsInfo[c * 4 + 1] = this.lib.clsRadius[c] ?? 1;
      clsInfo[c * 4 + 2] = this.lib.clsMaxDist[c] ?? 150;
      const hasR2 = c < 6 || c === 18 || c === 19 || c === 20 || c === 21 || c === 23;
      clsInfo[c * 4 + 3] = hasR2 ? 1 : 0;
    }
    const clsBuf = storage(new StorageBufferAttribute(clsInfo, 4), 'vec4', 24);

    // ---- draws ---------------------------------------------------------------
    const draws: DrawSpec[] = [];
    const meshes: Mesh[] = [];

    const prepassQ = new URLSearchParams(window.location.search).get('prepass');
    const noPrepass = prepassQ === '0' || prepassQ === 'grass';
    const addDraw = makeAddDraw({
      draws,
      meshes,
      groupTris: this.groupTris,
      group: this.group,
      prepassGroup: this.prepassGroup,
      noPrepass,
    });

    const layerOf = (cls: number): ScatterLayer =>
      cls < 6
        ? this.scatter.trees
        : cls < 15
          ? this.scatter.understory
          : cls < 20
            ? this.scatter.extras
            : this.scatter.stones;

    for (const pool of lib.pools) {
      const layer = layerOf(pool.cls);
      const rings: { ring: 0 | 1 | 2; parts: typeof pool.r1 }[] = [];
      if (pool.r0) rings.push({ ring: 0, parts: pool.r0 });
      if (pool.r1) rings.push({ ring: 1, parts: pool.r1 });
      if (pool.r2) rings.push({ ring: 2, parts: pool.r2 });
      // shadow budget: tree rings 0–2 cast (per-cascade caster lists);
      // understory is grounded by contact shadows + AO instead.
      // ?ablate=casters drops ALL veg caster draws (perf attribution).
      const ablateCasters = (
        new URLSearchParams(window.location.search).get('ablate') ?? ''
      )
        .split(',')
        .includes('casters');
      const ringCasts =
        !ablateCasters && (pool.cls < 6 ? true : pool.cls < 15 ? false : true);
      const crownDensity = pool.cls < 6 ? CROWN_SHADOW_DENSITY[pool.cls] ?? 0 : 0;
      // fit the shadow proxy to THIS pool's real extents (R1 union bbox)
      let poolDims: CrownDims | null = null;
      if (crownDensity > 0) {
        const fitParts = pool.r1 ?? pool.r2 ?? null;
        if (fitParts) {
          let top = 2;
          let rxz = 0.8;
          for (const part of fitParts) {
            part.geo.computeBoundingBox();
            const bb = part.geo.boundingBox;
            if (!bb) continue;
            top = Math.max(top, bb.max.y);
            rxz = Math.max(
              rxz,
              Math.abs(bb.min.x),
              bb.max.x,
              Math.abs(bb.min.z),
              bb.max.z,
            );
          }
          // cards overhang the foliage mass — pull the core in
          const bot = top * 0.32;
          poolDims = {
            cy: (top + bot) / 2,
            ry: ((top - bot) / 2) * 0.95,
            rxz: rxz * 0.74,
          };
        }
      }
      // wind opt-in: living vegetation sways (trees + understory); deadfall,
      // stumps and stones stay rigid (their vdata.y means moss/decay, and
      // a swaying log reads broken instantly). Trees rock slowly around the
      // trunk-bend knee; understory is light + springy (faster natural
      // frequency, knee near the ground); bare snags are stiff dead wood.
      const windBind =
        pool.cls < 6
          ? pool.cls === 5
            ? { k: 0.45, freq: 0.8, h0: 6 }
            : { k: 1, freq: 1, h0: 6 }
          : pool.cls < 15
            ? { k: 1, freq: 1.8, h0: 0.9 }
            : undefined;
      for (const { ring, parts } of rings) {
        if (!parts) continue;
        const g = groupOf(pool.cls, pool.variant, ring);
        for (const part of parts) {
          const mat = part.make();
          instanceVeg(mat, {
            bufA: layer.bufA,
            bufB: layer.bufB,
            compact: this.compact,
            groupBase: offsets[g] ?? 0,
            fade: fadeFor(pool.cls, ring, this.lib.clsMaxDist),
            wind: windBind,
          });
          this.patchGI(mat);
          // ?clsdbg=1 — flat-color every draw by VegClass (artifact triage:
          // "which pool is that?"); keeps alpha cutouts so silhouettes read
          if (new URLSearchParams(window.location.search).get('clsdbg') === '1') {
            const hue = (pool.cls * 47) % 360;
            const cdbg = new Color().setHSL(hue / 360, 0.95, 0.55);
            const op = mat.opacityNode as unknown as NF | null;
            mat.colorNode = vec4(vec3(cdbg.r, cdbg.g, cdbg.b), 1);
            if (op) mat.opacityNode = op;
          }
          addDraw(part.geo, mat, g, part.tris);
          // per-cascade caster siblings. Tree R2 skips its card/bark parts —
          // the crown proxy below carries the whole far shadow (a cascade
          // texel ≥0.5 m out there; 1.8k-tri cards bought nothing but raster)
          const proxyOwnsRing = pool.cls < 6 && ring === 2 && crownDensity > 0;
          // far cascades have 0.5–21 m texels — card-level caster geometry
          // buys nothing there; the crown proxies (added below for rings
          // 1+2 in EVERY cascade) own the far shadow. Ring-1 real casters
          // only feed the two near cascades (~15 ms → ~7 ms caster raster).
          const cascadeMax =
            pool.cls < 6 && ring === 1 && crownDensity > 0 ? 2 : CASCADES;
          if (part.castShadow && ringCasts && !proxyOwnsRing) {
            for (let c = 0; c < cascadeMax; c++) {
              const cg = casterGroupOf(c, pool.cls, pool.variant, ring);
              const cmat = part.make();
              instanceVeg(cmat, {
                bufA: layer.bufA,
                bufB: layer.bufB,
                compact: this.compact,
                groupBase: offsets[cg] ?? 0,
                fade: null,
                wind: windBind,
              });
              addDraw(geoView(part.geo), cmat, cg, 0, 2 + c);
            }
          }
        }
        // crown shadow proxy (rings 1+2): bulk occlusion at CROWN density —
        // cards alone leak 40%+ and PCSS flattens the speckle into a half-lit
        // wash; the core brings noon interiors down to real 5–15% so dapple
        // pools only survive at TRUE crown gaps
        if (ring > 0 && ringCasts && crownDensity > 0 && poolDims) {
          const proxyGeo = crownProxyGeometry(poolDims);
          for (let c = 0; c < CASCADES; c++) {
            const cg = casterGroupOf(c, pool.cls, pool.variant, ring);
            const pmat = proxyCasterMat(
              {
                bufA: layer.bufA,
                bufB: layer.bufB,
                compact: this.compact,
                groupBase: offsets[cg] ?? 0,
                fade: null,
              },
              crownDensity,
              poolDims,
              false,
            );
            addDraw(geoView(proxyGeo), pmat, cg, 0, 2 + c);
          }
          // impostor band (variant 0 carries it: one caster group per cls)
          if (ring === 2 && pool.variant === 0) {
            for (let c = 0; c < CASCADES; c++) {
              const cg = casterGroupOf(c, pool.cls, 0, 3);
              const pmat = proxyCasterMat(
                {
                  bufA: layer.bufA,
                  bufB: layer.bufB,
                  compact: this.compact,
                  groupBase: offsets[cg] ?? 0,
                  fade: null,
                },
                crownDensity,
                poolDims,
                true,
              );
              addDraw(geoView(crownProxyGeometry(poolDims)), pmat, cg, 0, 2 + c);
            }
          }
        }
      }
    }

    // tree impostors: one billboard draw per species
    for (const [cls, atlas] of lib.impostors) {
      const g = groupOf(cls, 0, 3);
      const mat = impostorRuntimeMaterial(atlas, {
        bufA: this.scatter.trees.bufA,
        bufB: this.scatter.trees.bufB,
        compact: this.compact,
        groupBase: offsets[g] ?? 0,
        fade: fadeFor(cls, 3, this.lib.clsMaxDist),
      });
      this.patchGI(mat);
      addDraw(impostorQuad(), mat, g, 2);
    }

    // ---- indirect buffer -------------------------------------------------------
    const D = draws.length;
    const indirectData = new Uint32Array(D * 5);
    const drawGroups = new Uint32Array(D);
    for (let d = 0; d < D; d++) {
      const spec = draws[d] as DrawSpec;
      indirectData[d * 5] = spec.indexCount;
      drawGroups[d] = spec.group;
    }
    this.indirectAttr = new IndirectStorageBufferAttribute(indirectData, 5);
    for (let d = 0; d < D; d++) {
      (meshes[d] as Mesh).geometry.setIndirect(this.indirectAttr, d * 20);
    }
    const indirectStore = storage(this.indirectAttr, 'uint', D * 5);
    const drawGroupBuf = storage(new StorageBufferAttribute(drawGroups, 1), 'uint', D);

    // ---- kernels ---------------------------------------------------------------
    this.kernels = buildCullKernels({
      counters: this.counters,
      compact: this.compact,
      capBuf,
      offBuf,
      clsBuf,
      drawGroupBuf,
      indirectStore,
      drawCount: D,
      camU: this.camU,
      planesU: this.planesU,
      planesCsmU: this.planesCsmU,
      hf: this.hf,
      scatter: this.scatter,
    });
  }

  /** wire the CSM rig (cascade cameras feed the caster cull) */
  setCSM(csm: object | null): void {
    this.csm = csm;
  }

  /** per-frame: update frustum/camera uniforms, run cull+indirect computes */
  update(renderer: Renderer, camera: PerspectiveCamera): void {
    this.camU.value.copy(camera.position);
    updateVegViewPos(camera);
    this.projView.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    this.frustum.setFromProjectionMatrix(this.projView);
    const arr = this.planesU.array as Vector4[];
    for (let p = 0; p < 6; p++) {
      const pl = this.frustum.planes[p];
      if (!pl) continue;
      (arr[p] as Vector4).set(pl.normal.x, pl.normal.y, pl.normal.z, pl.constant);
    }
    // cascade ortho frusta → caster-cull planes (read one frame stale —
    // CSMShadowNode positions its lwLights during the upcoming render; the
    // lightMargin slack swallows the lag). Also pins each cascade camera to
    // its caster layer once the lazy CSM init has built the lights.
    interface CascCam {
      projectionMatrix: Matrix4;
      matrixWorldInverse: Matrix4;
      layers: { enable(ch: number): void };
      left?: number;
    }
    const lights = (
      this.csm as { lights?: { shadow?: { camera?: CascCam } }[] } | null
    )?.lights;
    if (lights) {
      const carr = this.planesCsmU.array as Vector4[];
      for (let c = 0; c < CASCADES; c++) {
        const cam = lights[c]?.shadow?.camera;
        if (!cam || !Number.isFinite(cam.left ?? NaN)) continue;
        cam.layers.enable(2 + c);
        this.cascM.multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse);
        this.cascFrustum.setFromProjectionMatrix(this.cascM);
        for (let p = 0; p < 6; p++) {
          const pl = this.cascFrustum.planes[p];
          if (!pl) continue;
          (carr[c * 6 + p] as Vector4).set(
            pl.normal.x,
            pl.normal.y,
            pl.normal.z,
            pl.constant,
          );
        }
      }
    }
    let changed =
      camera.position.x !== this.lastCamX ||
      camera.position.y !== this.lastCamY ||
      camera.position.z !== this.lastCamZ;
    this.lastCamX = camera.position.x;
    this.lastCamY = camera.position.y;
    this.lastCamZ = camera.position.z;
    const mainArr = this.planesU.array as Vector4[];
    for (let p = 0; p < 6; p++) {
      const v = mainArr[p] as Vector4;
      const b = p * 4;
      if (
        v.x !== this.lastPlanes[b] ||
        v.y !== this.lastPlanes[b + 1] ||
        v.z !== this.lastPlanes[b + 2] ||
        v.w !== this.lastPlanes[b + 3]
      ) {
        changed = true;
      }
      this.lastPlanes[b] = v.x;
      this.lastPlanes[b + 1] = v.y;
      this.lastPlanes[b + 2] = v.z;
      this.lastPlanes[b + 3] = v.w;
    }
    const cascArr = this.planesCsmU.array as Vector4[];
    for (let p = 0; p < 6 * CASCADES; p++) {
      const v = cascArr[p] as Vector4;
      const b = p * 4;
      if (
        v.x !== this.lastCascPlanes[b] ||
        v.y !== this.lastCascPlanes[b + 1] ||
        v.z !== this.lastCascPlanes[b + 2] ||
        v.w !== this.lastCascPlanes[b + 3]
      ) {
        changed = true;
      }
      this.lastCascPlanes[b] = v.x;
      this.lastCascPlanes[b + 1] = v.y;
      this.lastCascPlanes[b + 2] = v.z;
      this.lastCascPlanes[b + 3] = v.w;
    }
    this.framesSinceCull++;
    if (changed && this.framesSinceCull >= 2) {
      for (const k of this.kernels) {
        renderer.compute(k as Parameters<Renderer['compute']>[0]);
      }
      this.framesSinceCull = 0;
    }
    this.frame++;
    if (this.frame % 90 === 0 && !this.reading) {
      this.reading = true;
      void this.readStats(renderer);
    }
  }

  /** HUD stats (throttled async readback of the group counters) */
  counterSnapshot(): Record<string, number> {
    return this.hud;
  }

  private async readStats(renderer: Renderer): Promise<void> {
    try {
      const attr = (this.counters as unknown as { value: unknown }).value;
      const ab = await renderer.getArrayBufferAsync(
        attr as Parameters<Renderer['getArrayBufferAsync']>[0],
      );
      const counts = new Uint32Array(ab);
      let hero = 0;
      let r1 = 0;
      let r2 = 0;
      let imp = 0;
      let under = 0;
      let extras = 0;
      let cast = 0;
      let tris = 0;
      for (let g = 0; g < GROUPS; g++) {
        const n = Math.min(counts[g] ?? 0, this.groupCaps[g] ?? 0);
        tris += n * (this.groupTris[g] ?? 0);
        if (g >= MAIN_GROUPS) {
          cast += n;
        } else if (g < IMPOSTOR_BASE) {
          if (g % 2 === 0) r1 += n;
          else r2 += n;
        } else if (g < UNDER_BASE) imp += n;
        else if (g < EXTRAS_BASE) under += n;
        else if (g < TREE_MAIN_BASE) extras += n;
        else hero += n;
      }
      this.hud = {
        'veg.hero': hero,
        'veg.r1': r1,
        'veg.r2': r2,
        'veg.imp': imp,
        'veg.underDrawn': under,
        'veg.extraDrawn': extras,
        'veg.cast': cast,
        'veg.tris': Math.round(tris),
      };
    } finally {
      this.reading = false;
    }
  }
}
