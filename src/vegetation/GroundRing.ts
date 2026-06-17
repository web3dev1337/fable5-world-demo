/**
 * GroundRing — camera-following near-field carpets: GRASS (≥800k blades,
 * spec floor) and DEBRIS (≥80k: cobbles/pebbles/twigs/bark chips/litter).
 *
 * Streaming without uploads: each instance slot maps to the unique world
 * cell congruent to it (mod GRID) nearest the camera — the classic clipmap
 * wrap. All per-instance parameters re-derive from pcg(worldCell), so a
 * slot's content changes only when its world cell does. A per-frame cull
 * compute samples biome/water/canopy fields, thins density toward the ring
 * edge, frustum-tests, picks the LOD band, and appends (cell, groundY) into
 * compact lists → indirect draws.
 *
 * Grass LODs: 4-seg blade ≤26 m → 2-seg ≤60 m → wide tuft cross beyond
 * (dither-crossfaded). Debris types: cobbles/pebbles bias toward stream beds
 * (flowStrength — "water-rounded near streams"), litter/twigs/chips under
 * canopy, pebbles on rocky ground.
 *
 * The cull kernels, draw materials and geometry builders live in
 * ./groundring/{cull,materials,geo}.ts; tuning constants + the cull↔draw
 * couplings (cell sizes, band widths, thinning) in ./groundring/constants.ts.
 * This shell owns buffer allocation, the draw/prepass assembly, the indirect
 * count kernel, per-frame dispatch and the GPU stats readback.
 */

import {
  BufferGeometry,
  DoubleSide,
  Frustum,
  Group,
  Matrix4,
  Mesh,
  Vector3,
  Vector4,
} from 'three';
import type { DataTexture, PerspectiveCamera } from 'three';
import {
  IndirectStorageBufferAttribute,
  IrradianceNode,
  MeshStandardNodeMaterial,
  StorageBufferAttribute,
  type Renderer,
  type StorageBufferNode,
  type StorageTexture,
} from 'three/webgpu';
import {
  Fn,
  If,
  Return,
  atomicLoad,
  instanceIndex,
  instancedArray,
  positionWorld,
  storage,
  uniform,
  uniformArray,
  varying,
  vec3,
} from 'three/tsl';
import { canopyAt } from '../gpu/passes/Scatter';
import { cameraSignatureChanged, createCameraSignature } from '../core/CameraSignature';
import { rockMaterial } from '../render/VegMaterials';
import { markVegRefresh } from '../render/StaticRefresh';
import { depthPrepassTwin } from '../render/VegPrepass';
import type { NU, NV3 } from '../gpu/TSLTypes';
import type { Heightfield } from '../world/Heightfield';
import type { ProbeGI } from '../gpu/passes/ProbeGI';
import {
  barkChipGeometry,
  debrisMaterial,
  litterMaterial,
  twigGeometry,
} from './GroundCover';
import { buildRock } from './RockBuilder';
import type { WorldSeed } from '../core/Seed';
import { bladeClump, litterQuad, tuftGeometry } from './groundring/geo';
import { debrisTransform, grassMaterial, type RingBind } from './groundring/materials';
import { buildCullKernels } from './groundring/cull';
import {
  DEB_CAPS,
  DEB_CELL,
  FAR_CAP,
  FAR_CELL,
  GRASS_CAPS,
  GRASS_CELL,
  G_MID,
  G_NEAR,
} from './groundring/constants';

export class GroundRing {
  readonly group = new Group();
  /** depth twins render before color draws (renderOrder; grouped) */
  private prepassGroup = new Group();
  private kernels: object[] = [];
  private camU = uniform(new Vector3());
  private planesU = uniformArray(Array.from({ length: 6 }, () => new Vector4()));
  private frustum = new Frustum();
  private projView = new Matrix4();
  private hud: Record<string, number> = {};
  private reading = false;
  private frame = 0;
  private counters!: ReturnType<StorageBufferNode<'uint'>['toAtomic']>;
  private caps: number[] = [...GRASS_CAPS, ...DEB_CAPS, FAR_CAP];
  // pose-gate: the cull is a pure function of camera position + frustum over a
  // static world (no time/wind term), so its compact draw lists are identical
  // across frames where the pose is unchanged — skip the ~10M-thread dispatch
  // and reuse last frame's lists. Bit-exact compare ⇒ zero visual difference.
  private lastCamX = NaN;
  private lastCamY = NaN;
  private lastCamZ = NaN;
  private lastPlanes = new Float32Array(24);
  private cameraSignature = createCameraSignature();
  // motion cadence: re-cull at most every other frame while moving. The ring is
  // a camera-following clipmap; a 1-frame-stale visible set lags only the far
  // ring edge by one frame (≈6 cm at walk speed) — same imperceptible-latency
  // principle CsmCached already uses for far shadow cascades. Resumes instantly
  // from a static hold (framesSinceCull is large by then).
  private framesSinceCull = 99;

  constructor(
    private hf: Heightfield,
    private canopyTex: StorageTexture,
    private seed: WorldSeed,
    private gi: ProbeGI | null = null,
  ) {}

  /**
   * Probe ambient for the carpets (same field as terrain/veg — without it
   * the grass keeps the dimmed hemisphere and reads as a pale glowing mat
   * inside canopy-shadowed interiors). Up-normal: a carpet integrates the
   * down-welling irradiance.
   */
  private patchGI(mat: MeshStandardNodeMaterial): void {
    const gi = this.gi;
    if (!gi) return;
    let irr = gi.irradiance(
      positionWorld as unknown as NV3,
      vec3(0, 1, 0) as unknown as NV3,
    );
    irr = irr.mul(
      canopyAt(this.canopyTex, (positionWorld as unknown as NV3).xz)
        .mul(0.12)
        .oneMinus(),
    ) as typeof irr;
    // probe field varies at ≥1.5 m — vertex-stage eval on ≤0.3 m carpet
    // geometry is identical, and skips 4 texture fetches per overdrawn px
    const irrV = varying(irr as unknown as Parameters<typeof varying>[0]);
    (mat as unknown as { setupLightMap: () => unknown }).setupLightMap = () =>
      new IrradianceNode(irrV as unknown as ConstructorParameters<typeof IrradianceNode>[0]);
  }

  init(beechAtlas: DataTexture | null): void {
    this.group.add(this.prepassGroup);
    const hf = this.hf;
    const salt = this.seed.sub('groundring') & 0x7fffffff;
    const canopyTex = this.canopyTex;

    const offsets: number[] = [];
    let off = 0;
    for (const cap of this.caps) {
      offsets.push(off);
      off += cap;
    }
    const cells = instancedArray(off, 'uint');
    const heights = instancedArray(off, 'float');
    this.counters = instancedArray(this.caps.length, 'uint').toAtomic();
    const counters = this.counters;
    const capBuf = storage(
      new StorageBufferAttribute(new Uint32Array(this.caps), 1),
      'uint',
      this.caps.length,
    );
    const offBuf = storage(
      new StorageBufferAttribute(new Uint32Array(offsets), 1),
      'uint',
      this.caps.length,
    );

    const { clearK, grassK, debrisK, farK } = buildCullKernels({
      hf,
      canopyTex,
      camU: this.camU,
      planesU: this.planesU,
      salt,
      cells,
      heights,
      counters,
      capBuf,
      offBuf,
      caps: this.caps,
    });

    // ---------------- draws -------------------------------------------------------
    const draws: { geo: BufferGeometry; mat: MeshStandardNodeMaterial; g: number }[] = [];

    const grassGeos = [bladeClump(5, 4), bladeClump(3, 2), tuftGeometry()];
    const grassFades: [number | null, number | null][] = [
      [null, G_NEAR],
      [G_NEAR, G_MID],
      [G_MID, null],
    ];
    for (let l = 0; l < 3; l++) {
      const bindL: RingBind = {
        cells,
        heights,
        base: offsets[l] ?? 0,
        cell: GRASS_CELL,
        salt,
      };
      const mat = grassMaterial(hf, canopyTex, bindL, grassFades[l] ?? [null, null], l === 2);
      this.patchGI(mat);
      draws.push({ geo: grassGeos[l] as BufferGeometry, mat, g: l });
    }

    // far super-tufts: one draw on the coarse list, wide cards, full
    // terrain-normal shading (mode 'far' in grassMaterial)
    {
      const bindF: RingBind = {
        cells,
        heights,
        base: offsets[8] ?? 0,
        cell: FAR_CELL,
        salt: salt ^ 0x6f21,
      };
      const matF = grassMaterial(hf, canopyTex, bindF, [null, null], true, true);
      this.patchGI(matF);
      draws.push({ geo: tuftGeometry(0.21), mat: matF, g: 8 });
    }

    const rng = this.seed.rng('groundring/geo');
    const debrisGeos: BufferGeometry[] = [
      buildRock('cobble', rng.fork('cobble'), 2).geometry,
      buildRock('cobble', rng.fork('pebble'), 1).geometry,
      twigGeometry(rng.fork('twig')),
      barkChipGeometry(rng.fork('chip')),
      litterQuad(),
    ];
    const debrisScale = [0.16, 0.05, 1, 1, 1];
    for (let t = 0; t < 5; t++) {
      let mat: MeshStandardNodeMaterial;
      if (t === 4 && beechAtlas) mat = litterMaterial(beechAtlas);
      else if (t === 2) mat = debrisMaterial('twig');
      else if (t === 3) mat = debrisMaterial('chip');
      else mat = rockMaterial({ moss: t === 0 ? 0.18 : 0.05 });
      const bindD: RingBind = {
        cells,
        heights,
        base: offsets[3 + t] ?? 0,
        cell: DEB_CELL,
        salt: salt ^ 0x5dd5,
      };
      debrisTransform(mat, bindD, debrisScale[t] ?? 1);
      this.patchGI(mat);
      draws.push({ geo: debrisGeos[t] as BufferGeometry, mat, g: 3 + t });
    }

    const D = draws.length;
    const indirectData = new Uint32Array(D * 5);
    const drawGroups = new Uint32Array(D);
    const indirectAttr = new IndirectStorageBufferAttribute(indirectData, 5);
    for (let d = 0; d < D; d++) {
      const spec = draws[d];
      if (!spec) continue;
      const geo = spec.geo;
      indirectData[d * 5] = geo.index ? geo.index.count : geo.attributes.position?.count ?? 0;
      drawGroups[d] = spec.g;
      geo.setIndirect(indirectAttr, d * 20);
      const mesh = new Mesh(geo, spec.mat);
      mesh.frustumCulled = false;
      mesh.castShadow = false;
      mesh.receiveShadow = true;
      markVegRefresh(mesh);
      this.group.add(mesh);
      // grass layers shade 2-8x per pixel without a prepass (random draw
      // order defeats early-Z); twin shares geometry = same indirect slot
      const noPrepass =
        new URLSearchParams(window.location.search).get('prepass') === '0';
      if (!noPrepass && (spec.g <= 2 || spec.g === 8)) {
        const matS = spec.mat as unknown as { positionNode: unknown; maskNode: unknown };
        const twin = depthPrepassTwin(mesh, {
          positionNode: matS.positionNode,
          maskNode: matS.maskNode ?? undefined,
          side: DoubleSide,
        });
        markVegRefresh(twin);
        this.prepassGroup.add(twin);
      }
    }
    const indirectStore = storage(indirectAttr, 'uint', D * 5);
    const drawGroupBuf = storage(new StorageBufferAttribute(drawGroups, 1), 'uint', D);

    const indirectK = Fn(() => {
      const i = instanceIndex;
      If(i.greaterThanEqual(D), () => {
        Return();
      });
      const g = drawGroupBuf.element(i) as unknown as NU;
      const raw = atomicLoad(counters.element(g)) as unknown as NU;
      const cap = capBuf.element(g) as unknown as NU;
      indirectStore.element(i.mul(5).add(1)).assign(raw.greaterThan(cap).select(cap, raw));
    })().compute(D);
    indirectK.setName('ringIndirect');

    this.kernels = [clearK, grassK, debrisK, farK, indirectK];
  }

  update(renderer: Renderer, camera: PerspectiveCamera, diagnosticsVisible = false): void {
    const cameraChanged = cameraSignatureChanged(camera, this.cameraSignature);
    if (!cameraChanged) {
      this.frame++;
      if (diagnosticsVisible && this.frame % 90 === 30 && !this.reading) {
        this.reading = true;
        void this.readStats(renderer);
      }
      return;
    }
    this.camU.value.copy(camera.position);
    this.projView.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    this.frustum.setFromProjectionMatrix(this.projView);
    const arr = this.planesU.array as Vector4[];
    let changed =
      cameraChanged ||
      camera.position.x !== this.lastCamX ||
      camera.position.y !== this.lastCamY ||
      camera.position.z !== this.lastCamZ;
    for (let p = 0; p < 6; p++) {
      const pl = this.frustum.planes[p];
      if (!pl) continue;
      const v = arr[p] as Vector4;
      v.set(pl.normal.x, pl.normal.y, pl.normal.z, pl.constant);
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
    this.lastCamX = camera.position.x;
    this.lastCamY = camera.position.y;
    this.lastCamZ = camera.position.z;
    this.framesSinceCull++;
    if (changed && this.framesSinceCull >= 2) {
      for (const k of this.kernels) {
        renderer.compute(k as Parameters<Renderer['compute']>[0]);
      }
      this.framesSinceCull = 0;
    }
    this.frame++;
    if (diagnosticsVisible && this.frame % 90 === 30 && !this.reading) {
      this.reading = true;
      void this.readStats(renderer);
    }
  }

  counterSnapshot(): Record<string, number> {
    return this.hud;
  }

  private async readStats(renderer: Renderer): Promise<void> {
    try {
      const attr = (this.counters as unknown as { value: unknown }).value;
      const ab = await renderer.getArrayBufferAsync(
        attr as Parameters<Renderer['getArrayBufferAsync']>[0],
      );
      const c = new Uint32Array(ab);
      const n = (g: number): number => Math.min(c[g] ?? 0, this.caps[g] ?? 0);
      this.hud = {
        'veg.grass': n(0) + n(1) + n(2) + n(8),
        'veg.g0': n(0),
        'veg.g1': n(1),
        'veg.g2': n(2),
        'veg.g3': n(8),
        'veg.debris': n(3) + n(4) + n(5) + n(6) + n(7),
      };
    } finally {
      this.reading = false;
    }
  }
}
