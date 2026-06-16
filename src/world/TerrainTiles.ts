/**
 * Terrain rendering: CDLOD quadtree of instanced grid patches + far vista shell.
 *
 * - One InstancedMesh draws every active tile; per-tile data (origin, size,
 *   lod) lives in a CPU-writable instanced storage buffer, updated only when
 *   the quadtree changes (camera moved) — never per-frame per-instance.
 * - CDLOD vertex morphing: odd vertices slide toward their even-grid
 *   positions across the outer 35% of each LOD ring → no cracks, no pops.
 * - Far shell: radial ring 1.95–14 km, analytic macro height (far branch),
 *   blended to the baked field across the world edge.
 *
 * The per-concern node graphs (near material, debug surfaces, far shell) live
 * in ./terrain/* builders; this file owns the quadtree + per-tile buffer.
 */

import { InstancedMesh, PlaneGeometry, Mesh, type PerspectiveCamera } from 'three';
import type { StorageBufferNode } from 'three/webgpu';
import { instancedArray } from 'three/tsl';
import type { Heightfield } from './Heightfield';
import { WORLD_SIZE } from './WorldConst';
import {
  MAX_TILES,
  MIN_TILE,
  MIN_TILE_ROUGH,
  PATCH_SEGS,
  RANGE_BASE,
  SPLIT_K,
} from './terrain/TileConstants';
import { buildTileMaterial, type TerrainTilesOptions } from './terrain/TileMaterial';
import { applyDebugViews } from './terrain/DebugViews';
import { buildFarShell } from './terrain/FarShell';

export type { TerrainTilesOptions };

export class TerrainTiles {
  readonly mesh: InstancedMesh;
  readonly farShell: Mesh;
  private tileData: Float32Array;
  private tileBuf: StorageBufferNode<'vec4'>;
  private hf: Heightfield;
  private lastCamX = Infinity;
  private lastCamZ = Infinity;
  activeTiles = 0;
  /** per-level height ranges: level 0 = 64×64 grid of 64 m cells, then halves */
  private rangePyr: Float32Array[] = [];

  constructor(
    hf: Heightfield,
    debugView: string | null = null,
    opts: TerrainTilesOptions = {},
  ) {
    this.hf = hf;
    this.buildRangePyramid();
    // ?ablate=mat → neutral clay (perf attribution for the splat material)
    const ablate = new Set(
      (new URLSearchParams(window.location.search).get('ablate') ?? '').split(','),
    );
    if (ablate.has('mat')) opts = { ...opts, neutral: true };
    // --- per-tile buffer -------------------------------------------------------
    this.tileData = new Float32Array(MAX_TILES * 4);
    this.tileBuf = instancedArray(this.tileData, 'vec4');
    const heightBuf = opts.heightBuf ?? hf.height;

    // --- patch geometry ----------------------------------------------------------
    // one extra quad ring beyond ±0.5 = skirt vertices: the shader clamps
    // them onto the edge then drops them down — hides cracks from the
    // error-biased (non-uniform) quadtree splits
    const s = 1 / PATCH_SEGS;
    const patch = new PlaneGeometry(1 + 2 * s, 1 + 2 * s, PATCH_SEGS + 2, PATCH_SEGS + 2);
    patch.rotateX(-Math.PI / 2); // local xz in [-0.5-s, 0.5+s], +y up

    // --- near-tile material + debug surfaces -------------------------------------
    const tileMat = buildTileMaterial(hf, this.tileBuf, heightBuf, opts, ablate);
    applyDebugViews(tileMat, hf, heightBuf, debugView, opts);

    this.mesh = new InstancedMesh(patch, tileMat.mat, MAX_TILES);
    this.mesh.frustumCulled = false;
    this.mesh.receiveShadow = true;
    // mountain shadows come from the coarse ShadowProxy grid — casting the
    // full CDLOD mesh re-rasterized ~11M tris across the four cascades
    this.mesh.castShadow = false;

    // --- far shell -----------------------------------------------------------------
    this.farShell = buildFarShell(hf, opts, ablate);
  }

  /**
   * Height-range mip pyramid from the CPU height mirror — drives error-biased
   * splits (steep/rough tiles refine deeper, flat meadows stay coarse).
   */
  private buildRangePyramid(): void {
    const heights = this.hf.cpuHeights;
    if (!heights) return;
    const res = Math.sqrt(heights.length) | 0;
    const base = RANGE_BASE; // cells per side; one cell = MIN_TILE meters
    const cellPx = res / base;
    const l0 = new Float32Array(base * base);
    for (let cy = 0; cy < base; cy++) {
      for (let cx = 0; cx < base; cx++) {
        let mn = Infinity;
        let mx = -Infinity;
        const x0 = cx * cellPx;
        const y0 = cy * cellPx;
        // 4-px stride: range estimate, not exact min/max (16× cheaper)
        for (let y = y0; y < y0 + cellPx; y += 4) {
          const row = y * res;
          for (let x = x0; x < x0 + cellPx; x += 4) {
            const v = heights[row + x] as number;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
          }
        }
        l0[cy * base + cx] = mx - mn;
      }
    }
    this.rangePyr = [l0];
    for (let side = base >> 1; side >= 1; side >>= 1) {
      const prev = this.rangePyr[this.rangePyr.length - 1] as Float32Array;
      const pSide = side * 2;
      const lvl = new Float32Array(side * side);
      for (let cy = 0; cy < side; cy++) {
        for (let cx = 0; cx < side; cx++) {
          lvl[cy * side + cx] = Math.max(
            prev[cy * 2 * pSide + cx * 2] as number,
            prev[cy * 2 * pSide + cx * 2 + 1] as number,
            prev[(cy * 2 + 1) * pSide + cx * 2] as number,
            prev[(cy * 2 + 1) * pSide + cx * 2 + 1] as number,
          );
        }
      }
      this.rangePyr.push(lvl);
    }
  }

  /** height range (m) within a tile (≥ MIN_TILE sizes use the exact level) */
  private heightRange(ox: number, oz: number, size: number): number {
    if (this.rangePyr.length === 0) return 0;
    const lvl = Math.max(0, Math.min(Math.round(Math.log2(Math.max(size, MIN_TILE) / MIN_TILE)), this.rangePyr.length - 1));
    const side = RANGE_BASE >> lvl;
    const cell = WORLD_SIZE / side;
    const cx = Math.max(0, Math.min(Math.floor((ox + WORLD_SIZE / 2) / cell), side - 1));
    const cy = Math.max(0, Math.min(Math.floor((oz + WORLD_SIZE / 2) / cell), side - 1));
    return (this.rangePyr[lvl] as Float32Array)[cy * side + cx] as number;
  }

  /** rebuild the quadtree when the camera has moved enough */
  update(camera: PerspectiveCamera): void {
    const cx = camera.position.x;
    const cz = camera.position.z;
    if (Math.hypot(cx - this.lastCamX, cz - this.lastCamZ) < 20 && this.activeTiles > 0) return;
    this.lastCamX = cx;
    this.lastCamZ = cz;

    let n = 0;
    const data = this.tileData;
    const emit = (ox: number, oz: number, size: number, lod: number): void => {
      if (n >= MAX_TILES) return;
      data[n * 4] = ox;
      data[n * 4 + 1] = oz;
      data[n * 4 + 2] = size;
      data[n * 4 + 3] = lod;
      n++;
    };
    const cy = camera.position.y;
    const recurse = (ox: number, oz: number, size: number, lod: number): void => {
      const dx = Math.max(Math.abs(cx - ox) - size / 2, 0);
      const dz = Math.max(Math.abs(cz - oz) - size / 2, 0);
      // 3D distance: from high altitude the ground straight below does not
      // need MIN_TILE resolution (slack absorbs in-tile height spread)
      const groundY = this.hf.heightAtCpu(ox, oz);
      const dy = Math.max(Math.abs(cy - groundY) - 250, 0) * 0.8;
      const dist = Math.hypot(dx, dz, dy);
      // error bias: tiles with big internal relief split earlier AND deeper
      // (cliff close-ups got 1 m quads stretched over ~10 m vertical)
      const range = this.heightRange(ox, oz, size);
      const errBoost = Math.min(1 + (range / size) * 0.8, 1.8);
      const minTile = range > size * 0.85 ? MIN_TILE_ROUGH : MIN_TILE;
      if (size > minTile && dist < size * SPLIT_K * errBoost) {
        const q = size / 4;
        const h = size / 2;
        recurse(ox - q, oz - q, h, lod + 1);
        recurse(ox + q, oz - q, h, lod + 1);
        recurse(ox - q, oz + q, h, lod + 1);
        recurse(ox + q, oz + q, h, lod + 1);
      } else {
        emit(ox, oz, size, lod);
      }
    };
    recurse(0, 0, WORLD_SIZE, 0);

    this.activeTiles = n;
    this.mesh.count = n;
    const attr = this.tileBuf.value;
    attr.needsUpdate = true;
  }
}
