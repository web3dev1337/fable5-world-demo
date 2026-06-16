/**
 * Diagnostic terrain surfaces: the `?view=` (probes / lod / caust / caust2 /
 * snow / bioR / bioB), neutral-clay, and split-screen branches. Each overwrites
 * the relevant material nodes — kept in their original order so the last write
 * wins exactly as it did inline in the constructor.
 *
 * Pure code motion from the TerrainTiles constructor; verbatim node graphs.
 */

import {
  float,
  positionWorld,
  positionLocal,
  screenUV,
  texture,
  transformNormalToView,
  vec2,
  vec3,
} from 'three/tsl';
import { CAUSTIC_TILE, causticTintParts } from '../../render/Caustics';
import type { NV4 } from '../../gpu/TSLTypes';
import type { Heightfield } from '../Heightfield';
import { WORLD_SIZE } from '../WorldConst';
import type { TerrainTilesOptions, TileMaterialResult } from './TileMaterial';

export function applyDebugViews(
  ctx: TileMaterialResult,
  hf: Heightfield,
  heightBuf: Heightfield['height'],
  debugView: string | null,
  opts: TerrainTilesOptions,
): void {
  const { mat, shading, cctx, tile } = ctx;
  if (debugView === 'probes' && opts.gi) {
    // ambient-only view: probe irradiance × albedo, no sun/shadows
    mat.colorNode = vec3(0.0);
    mat.emissiveNode = opts.gi
      .irradiance(positionWorld, shading.worldNormalNode)
      .mul(shading.colorNode);
  }
  if (debugView === 'lod') {
    // distinct color per LOD level + faint grid along tile edges
    const lod = tile.w;
    const edge = positionLocal.xz.abs().x.max(positionLocal.xz.abs().y);
    const grid = edge.greaterThan(0.492).select(float(0.25), float(1));
    mat.colorNode = vec3(0.02);
    mat.emissiveNode = vec3(
      lod.mul(0.9173).add(0.13).fract(),
      lod.mul(0.3719).add(0.41).fract(),
      lod.mul(0.7177).add(0.79).fract(),
    ).mul(grid);
  }
  if (debugView === 'caust' && cctx) {
    // raw caustic tile painted on the terrain (bake verification);
    // ?caustmip=N forces a mip bias — verifies the auto-generated chain
    // that depth-defocus sampling depends on (black ⇒ mips never built)
    const mip = Number(
      new URLSearchParams(window.location.search).get('caustmip') ?? '0',
    );
    mat.colorNode = vec3(0.0);
    mat.emissiveNode = vec3(
      (
        texture(cctx.bake.tex, positionWorld.xz.div(CAUSTIC_TILE)).bias(
          float(mip),
        ) as unknown as NV4
      ).x,
    );
  }
  if (debugView === 'caust2' && cctx) {
    // tint triage: r = gated tint, g = gate product, b = ungated pattern
    mat.colorNode = vec3(0.0);
    mat.emissiveNode = causticTintParts(positionWorld);
  }
  if ((debugView === 'snow' || debugView === 'bioR' || debugView === 'bioB') && hf.biomeTex) {
    // single-channel classification view: white = channel value
    const b = texture(hf.biomeTex, positionWorld.xz.div(WORLD_SIZE).add(0.5));
    mat.colorNode = vec3(0.02);
    const ch = debugView === 'bioR' ? b.r : debugView === 'bioB' ? b.b : b.g;
    mat.emissiveNode = vec3(ch);
  }
  if (opts.neutral) {
    // neutral clay shading for the erosion split view: fragment-space
    // finite-difference normals from the bound height buffer
    const eH = 1.6;
    const pxz = positionWorld.xz;
    const hC = hf.sampleHeightFrom(heightBuf, pxz);
    const hX = hf.sampleHeightFrom(heightBuf, pxz.add(vec2(eH, 0)));
    const hZ = hf.sampleHeightFrom(heightBuf, pxz.add(vec2(0, eH)));
    const nFD = vec3(hC.sub(hX), float(eH), hC.sub(hZ)).normalize();
    mat.colorNode = vec3(0.55, 0.53, 0.5);
    mat.normalNode = transformNormalToView(nFD);
    mat.roughnessNode = float(0.92);
  }
  if (opts.screenHalf) {
    // split-screen via alpha test: keep only one half of the screen
    const keepLeft = opts.screenHalf === 'left';
    const keep = keepLeft
      ? screenUV.x.lessThanEqual(0.5)
      : screenUV.x.greaterThan(0.5);
    mat.opacityNode = keep.select(float(1), float(0));
    mat.alphaTest = 0.5;
  }
}
