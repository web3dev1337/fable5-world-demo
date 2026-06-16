/**
 * Per-candidate site sampling: the shared kernel preamble (grid cell → jitter →
 * world pos) and `sampleSite`, which gathers the density inputs (height, slope,
 * ecotone-warped biome id, snow/veg/rock fields, moisture, water) at a world xz.
 */

import {
  If,
  Return,
  float,
  instanceIndex,
  texture,
  vec2,
  vec3,
} from 'three/tsl';
import type { Heightfield } from '../../../world/Heightfield';
import { WORLD_SIZE } from '../../../world/WorldConst';
import { fbm3 } from '../../noise/NoiseTSL';
import type { NF, NI, NV2, NV4 } from '../../TSLTypes';
import { cellHash2 } from './helpers';

export interface SiteSamples {
  h: NF;
  slope: NF;
  bioId: NI; // ecotone-warped biome id
  snow: NF;
  vegDens: NF;
  rockExp: NF;
  moisture: NF;
  riverDepth: NF;
  standing: NF; // W − h (standing-water depth)
  nrmXZ: NV2;
}

export function sampleSite(hf: Heightfield, wpos: NV2): SiteSamples {
  const uv = wpos.div(WORLD_SIZE).add(0.5);
  const h = hf.sampleHeight(wpos);
  const ns = texture(hf.normalTex, uv, 0) as unknown as NV4;
  // ecotone warp: read the biome classification through a ±26 m wobble
  const warp = vec2(
    fbm3(vec3(wpos.x.mul(0.011), 3.7, wpos.y.mul(0.011)), 2),
    fbm3(vec3(wpos.x.mul(0.011), 91.2, wpos.y.mul(0.011)), 2),
  ).mul(26);
  const uvW = wpos.add(warp).div(WORLD_SIZE).add(0.5);
  const bio = texture(
    hf.biomeTex as NonNullable<typeof hf.biomeTex>,
    uvW,
    0,
  ) as unknown as NV4;
  const bioExact = texture(
    hf.biomeTex as NonNullable<typeof hf.biomeTex>,
    uv,
    0,
  ) as unknown as NV4;
  const fields = texture(
    hf.fieldsTex as NonNullable<typeof hf.fieldsTex>,
    uv,
    0,
  ) as unknown as NV4;
  return {
    h,
    slope: ns.w,
    bioId: bio.x.mul(8).add(0.5).floor().toInt(),
    snow: bioExact.y, // snow/veg-density/rock read unwarped (physical fields)
    vegDens: bioExact.z,
    rockExp: bioExact.w,
    moisture: fields.x,
    riverDepth: fields.z,
    standing: fields.w.sub(h),
    nrmXZ: vec2(ns.x, ns.z),
  };
}

export interface CellSite {
  cell: NV2;
  wpos: NV2;
  s: SiteSamples;
}

/**
 * Shared kernel preamble: out-of-range threads return early, then the candidate
 * cell is jittered to a world position and sampled. Identical across all four
 * scatter layers (only grid size `g` and `salt` differ).
 */
export function gridSite(hf: Heightfield, g: number, salt: number): CellSite {
  const i = instanceIndex;
  If(i.greaterThanEqual(g * g), () => {
    Return();
  });
  const cell = vec2(float(i.mod(g)), float(i.div(g)));
  const jit = cellHash2(cell, salt);
  const wpos = cell.add(jit).div(g).sub(0.5).mul(WORLD_SIZE);
  const s = sampleSite(hf, wpos);
  return { cell, wpos, s };
}
