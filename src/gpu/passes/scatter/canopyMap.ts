/**
 * Canopy occlusion map: tree crowns splatted into a world-space coverage
 * texture (4 m/texel). Lighting uses it to pull probe ambient DOWN under
 * canopy (probes ray-march only the heightfield, so without this the forest
 * interior glows with full open-sky irradiance and every sun shadow washes
 * out to an AO-like smudge). Doubles as the spec's canopy-shadow density
 * field for later passes.
 */

import type { Renderer } from 'three/webgpu';
import { StorageTexture } from 'three/webgpu';
import {
  Fn,
  If,
  Return,
  atomicAdd,
  atomicLoad,
  float,
  instanceIndex,
  instancedArray,
  texture,
  textureStore,
  uint,
  uvec2,
  vec2,
  vec4,
} from 'three/tsl';
import { WORLD_SIZE } from '../../../world/WorldConst';
import type { NF, NU, NV2, NV4 } from '../../TSLTypes';
import { byBiome } from './helpers';
import type { ScatterLayer } from './types';

export const CANOPY_RES = 1024;

export async function buildCanopyMap(
  renderer: Renderer,
  trees: ScatterLayer,
): Promise<StorageTexture> {
  const accum = instancedArray(CANOPY_RES * CANOPY_RES, 'uint').toAtomic();
  const texel = WORLD_SIZE / CANOPY_RES; // 4 m

  // crown radius (m at scale 1) and skylight opacity per tree class
  const crownR = [2.9, 2.7, 3.8, 2.7, 3.2, 0.9];
  const opacity = [0.85, 0.7, 0.9, 0.65, 0.8, 0.12];

  const splatK = Fn(() => {
    const i = instanceIndex;
    If(i.greaterThanEqual(uint(Math.max(trees.count, 1))), () => {
      Return();
    });
    const A = trees.bufA.element(i) as unknown as NV4;
    const B = trees.bufB.element(i) as unknown as NV4;
    const cls = B.w.div(8).floor().toInt();
    const r = byBiome(cls, crownR).mul(A.w).clamp(1, 11);
    const op = byBiome(cls, opacity);
    const gx = A.x.div(WORLD_SIZE).add(0.5).mul(CANOPY_RES);
    const gy = A.z.div(WORLD_SIZE).add(0.5).mul(CANOPY_RES);
    for (let dy = -3; dy <= 3; dy++) {
      for (let dx = -3; dx <= 3; dx++) {
        const tx = gx.add(dx).floor();
        const ty = gy.add(dy).floor();
        const inB = tx.greaterThanEqual(0)
          .and(tx.lessThan(CANOPY_RES))
          .and(ty.greaterThanEqual(0))
          .and(ty.lessThan(CANOPY_RES));
        const d = vec2(tx.add(0.5).sub(gx), ty.add(0.5).sub(gy)).length().mul(texel);
        const w = float(1).sub(d.div(r)).max(0).pow(1.5).mul(op).mul(255);
        If(inB.and(w.greaterThan(1)), () => {
          atomicAdd(
            accum.element(ty.toInt().mul(CANOPY_RES).add(tx.toInt())),
            w.toUint(),
          );
        });
      }
    }
  })().compute(Math.max(trees.count, 1));
  splatK.setName('canopySplat');
  await renderer.computeAsync(splatK);

  const tex = new StorageTexture(CANOPY_RES, CANOPY_RES);
  tex.generateMipmaps = false;
  const packK = Fn(() => {
    const i = instanceIndex;
    If(i.greaterThanEqual(CANOPY_RES * CANOPY_RES), () => {
      Return();
    });
    const x = i.mod(CANOPY_RES);
    const y = i.div(CANOPY_RES);
    // 3×3 box blur of the fixed-point accumulation → soft canopy field
    const sum = float(0).toVar();
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        const xx = float(x).add(dx).clamp(0, CANOPY_RES - 1).toInt();
        const yy = float(y).add(dy).clamp(0, CANOPY_RES - 1).toInt();
        sum.addAssign(
          float(
            atomicLoad(
              accum.element(yy.mul(CANOPY_RES).add(xx)),
            ) as unknown as NU,
          ),
        );
      }
    }
    const cov = sum.div(9 * 255).div(1.6).clamp(0, 1).pow(0.75);
    textureStore(tex, uvec2(x.toUint(), y.toUint()), vec4(cov, cov, cov, 1)).toWriteOnly();
  })().compute(CANOPY_RES * CANOPY_RES);
  packK.setName('canopyPack');
  await renderer.computeAsync(packK);
  return tex;
}

/** sample the canopy coverage field at a world xz (filtered) */
export function canopyAt(tex: StorageTexture, wxz: NV2): NF {
  const uv = wxz.div(WORLD_SIZE).add(0.5);
  return (texture(tex, uv) as unknown as NV4).x;
}
