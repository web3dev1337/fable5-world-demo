/**
 * GroundRing draw-side materials. Each instance fetches its packed world cell
 * + ground height (fetchRing), re-derives its transform/shading from
 * pcg(worldCell), and the grass material additionally drives the continuous
 * LOD width compensation, wind bend and band crossfade. The cull-side jitter
 * salts (bind.salt) MUST match the ones the cull kernels pack with, or a
 * slot's draw position diverges from where it was culled.
 */

import { DoubleSide } from 'three';
import { MeshStandardNodeMaterial } from 'three/webgpu';
import type { StorageBufferNode, StorageTexture } from 'three/webgpu';
import {
  Fn,
  cameraPosition,
  float,
  instanceIndex,
  interleavedGradientNoise,
  mix,
  normalLocal,
  positionLocal,
  screenCoordinate,
  smoothstep,
  texture,
  time,
  transformNormalToView,
  uint,
  uv,
  varying,
  vec2,
  vec3,
} from 'three/tsl';
import { canopyAt, cellHash2 } from '../../gpu/passes/Scatter';
import { grassTranslucency } from '../../render/VegMaterials';
import { gustAt, windContext, windExposure, windU } from '../../render/Wind';
import { WORLD_SIZE } from '../../world/WorldConst';
import { runiform } from '../../gpu/RenderUniform';
import { CELL_BIAS, DEB_R, G_BAND, grassThin } from './constants';
import type { NB, NF, NU, NV2, NV3, NV4 } from '../../gpu/TSLTypes';
import type { Heightfield } from '../../world/Heightfield';

export interface RingBind {
  cells: StorageBufferNode<'uint'>;
  heights: StorageBufferNode<'float'>;
  base: number;
  cell: number;
  salt: number;
}

/** vertex-stage fetch: packed world cell + ground height for this instance */
function fetchRing(bind: RingBind): { wc: NV2; y: NF; wpos: NV2 } {
  const at = instanceIndex.add(runiform(uint(bind.base)) as unknown as NU);
  const packed = bind.cells.element(at) as unknown as NU;
  const wc = vec2(
    float(packed.shiftRight(uint(16))).sub(CELL_BIAS),
    float(packed.bitAnd(uint(0xffff))).sub(CELL_BIAS),
  );
  const y = bind.heights.element(at) as unknown as NF;
  const jit = cellHash2(wc, bind.salt);
  return { wc, y, wpos: wc.add(jit).mul(bind.cell) };
}

/**
 * Dithered band crossfade by camera distance. COMPLEMENTARY partition (same
 * scheme as VegInstance.applyDitherFade): the outgoing layer draws where
 * IGN < fadeOut, the incoming one where IGN >= 1 − fadeIn — with the shared
 * band width the two layers split the pixel set exactly, so blade density
 * stays constant through the band. Same-comparison dithering halved the
 * drawn pixels at every grass-layer boundary (visible thin rings).
 */
function bandFade(
  mat: MeshStandardNodeMaterial,
  dist: NF,
  fadeIn: number | null,
  fadeOut: number | null,
  band: number,
): void {
  const inV =
    fadeIn !== null
      ? varying(smoothstep(fadeIn - band, fadeIn + band, dist))
      : null;
  const outV =
    fadeOut !== null
      ? varying(float(1).sub(smoothstep(fadeOut - band, fadeOut + band, dist)))
      : null;
  if (!inV && !outV) return;
  // maskNode (not colorNode Discards) so the depth-prepass twin can share
  // the EXACT same draw condition — a depth-vs-color discard mismatch
  // punches holes at the fade bands. Main pass only (carpets cast no
  // shadows, so maskShadowNode never consults this).
  // animate the dither each frame so TRAA dissolves the LOD-crossfade stipple
  // into smooth alpha — a screen-static IGN leaves fixed screen-door dots
  const ign = interleavedGradientNoise(
    screenCoordinate.xy.add(vec2(time.mul(53).mod(128), time.mul(97).mod(128))),
  );
  let cond: NB | null = null;
  if (inV) cond = ign.greaterThanEqual(float(1).sub(inV)) as unknown as NB;
  if (outV) {
    const c2 = ign.lessThan(outV) as unknown as NB;
    cond = cond ? ((cond as unknown as { and(o: NB): NB }).and(c2)) : c2;
  }
  mat.maskNode = cond as unknown as typeof mat.maskNode;
}

/** blade/tuft material — color matched to the terrain grass palette */
export function grassMaterial(
  hf: Heightfield,
  canopyTex: StorageTexture,
  bind: RingBind,
  fades: [number | null, number | null],
  tuft: boolean,
  far = false,
): MeshStandardNodeMaterial {
  const mat = new MeshStandardNodeMaterial();
  const { wc, y, wpos } = fetchRing(bind);
  const h2 = cellHash2(wc, bind.salt ^ 0x9191);
  // patch-level (≈1.6 m) dryness/hue so meadows read as drifts, not noise
  const patch = cellHash2(wc.mul(0.125).floor(), bind.salt ^ 0x3333);
  const tilt = cellHash2(wc, bind.salt ^ 0x4545).sub(0.5).mul(0.5);
  const dist = wpos.sub(vec2(cameraPosition.x, cameraPosition.z)).length();
  // width compensation for the continuous thinning — coverage conserved.
  // far mode: coarse-grid super-tufts have their own fixed footprint
  const widen = far
    ? h2.y.mul(0.8).add(1.6)
    : float(1).div(grassThin(dist).sqrt()).clamp(1, 4);
  const bladeH = h2.x
    .pow(1.3)
    .mul(far ? 0.42 : 0.3)
    .add(far ? 0.34 : 0.2)
    .mul(tuft && !far ? 2.0 : 1)
    .mul(widen.sub(1).mul(0.3).add(1));
  const yawA = h2.y.mul(6.2831853);
  const c = yawA.cos();
  const s = yawA.sin();
  const ls = positionLocal.mul(
    vec3(widen.mul(tuft ? 1.5 : 1.15), bladeH, 1),
  );
  const rx = ls.x.mul(c).add(ls.z.mul(s));
  const rz = ls.z.mul(c).sub(ls.x.mul(s));
  // wind: cantilever bend (tip²) riding the traveling gust field + a fine
  // per-blade shimmer; tips dip as they deflect. Same field as the trees
  // (Wind.ts) so meadow waves and canopy surges line up.
  let dx: NF = float(0);
  let dy: NF = float(0);
  let dz: NF = float(0);
  if (windContext()) {
    const wd = vec2(windU.dir as unknown as NV2);
    const tN = positionLocal.y; // 0..1 along the blade
    const st = windU.strength as unknown as NF;
    const amp = st
      .mul(gustAt(wpos).mul(0.9).add(0.3))
      .mul(windExposure(wpos));
    // lean² rule (matches the tree rework): strong wind flattens the
    // sward — deflection grows superlinearly, the tempo doesn't change
    const bend = amp
      .mul(st.mul(0.55).add(0.6))
      .mul(tN.mul(tN))
      .mul(bladeH.mul(0.42));
    const flut = far
      ? (float(0) as NF)
      : time
          .mul(5.2)
          .add(h2.x.mul(6.2832))
          .add(wpos.x.add(wpos.y).mul(0.9))
          .sin()
          .mul(tN)
          .mul(amp)
          .mul(0.05);
    dx = wd.x.mul(bend).sub(wd.y.mul(flut));
    dz = wd.y.mul(bend).add(wd.x.mul(flut));
    dy = bend.mul(tN).mul(-0.4);
  }
  // random lean (shear) — vertical uniform blades read as planted corn
  mat.positionNode = vec3(
    rx.add(tilt.x.mul(ls.y)).add(dx).add(wpos.x),
    ls.y.add(y).add(dy),
    rz.add(tilt.y.mul(ls.y)).add(dz).add(wpos.y),
  );
  // Blade shading normal (feedback 2.7+2.9): yaw-rotate the baked rounded
  // normal, then pull it toward the TERRAIN normal — a sward lights like
  // the hillside it grows on (the GoT move; per-blade card normals made
  // meadows sparkle gray). Harder pull with distance: near keeps blade
  // curvature, far converges on the splat so the g2 band dissolves clean.
  const nR = vec3(
    normalLocal.x.mul(c).add(normalLocal.z.mul(s)),
    normalLocal.y,
    normalLocal.z.mul(c).sub(normalLocal.x.mul(s)),
  );
  const tNrm = (
    texture(
      hf.normalTex,
      wpos.div(WORLD_SIZE).add(0.5),
      0,
    ) as unknown as NV4
  ).xyz.normalize();
  const upK = far
    ? (float(1) as NF)
    : smoothstep(8, 70, dist).mul(0.35).add(0.5);
  // VERTEX-stage shading hoist (Phase 7 perf): every term below varies at
  // ≥ blade scale (per-cell hashes, 1.5 m+ probe/canopy/heightfield
  // fields) — evaluating them per fragment re-ran the ring storage reads,
  // 4 hashes and 2 texture fetches for every overdrawn pixel of a 1–4 px
  // blade. varying() moves them to the vertex stage; interpolation across
  // a few-cm triangle is sub-quantization (verified by pixel diff).
  const nBlendV = varying(
    mix(nR.normalize(), tNrm, upK) as unknown as Parameters<typeof varying>[0],
  ) as unknown as NV3;
  mat.normalNode = transformNormalToView(
    nBlendV.normalize() as unknown as NV3,
  ) as unknown as typeof mat.normalNode;

  const t = uv().y as unknown as NF;
  const fresh = mix(
    vec3(0.02, 0.062, 0.011),
    vec3(0.065, 0.148, 0.028),
    t.mul(t),
  ) as unknown as NV3;
  const dry = mix(
    vec3(0.085, 0.07, 0.024),
    vec3(0.21, 0.17, 0.075),
    t,
  ) as unknown as NV3;
  // shade-grown grass: under crowns the sward stays deep cool green (dry
  // straw patches are a full-sun phenomenon) — without this the carpet
  // reads as a pale glowing mat inside forest interiors
  const cov = canopyAt(canopyTex, wpos);
  const dryK = smoothstep(0.7, 0.95, patch.x).mul(
    float(1).sub(cov.mul(0.85)),
  );
  let albedo = mix(fresh, dry, dryK) as unknown as NV3;
  albedo = albedo.mul(patch.y.sub(0.5).mul(0.3).add(1)) as unknown as NV3;
  albedo = mix(albedo, vec3(0.018, 0.052, 0.014), cov.mul(0.55)) as unknown as NV3;
  mat.colorNode = varying(
    albedo as unknown as Parameters<typeof varying>[0],
  ) as unknown as typeof mat.colorNode;
  mat.emissiveNode = varying(
    grassTranslucency(albedo, t) as unknown as Parameters<typeof varying>[0],
  ) as unknown as typeof mat.emissiveNode;
  mat.aoNode = varying(
    smoothstep(0.0, 0.55, t).mul(0.55).add(0.45) as unknown as Parameters<typeof varying>[0],
  ) as unknown as typeof mat.aoNode;
  mat.roughness = 0.88;
  mat.metalness = 0;
  mat.side = DoubleSide;
  bandFade(mat, dist, fades[0], fades[1], G_BAND);
  return mat;
}

/** cobbles/pebbles/twigs/chips/litter placement (yaw + scale + sink) */
export function debrisTransform(
  mat: MeshStandardNodeMaterial,
  bind: RingBind,
  scaleK: number,
): void {
  const { wc, y, wpos } = fetchRing(bind);
  const h2 = cellHash2(wc, bind.salt ^ 0x7777);
  const scl = h2.x.mul(0.9).add(0.55).mul(scaleK);
  const yawA = h2.y.mul(6.2831853);
  const c = yawA.cos();
  const s = yawA.sin();
  const ls = positionLocal.mul(scl);
  const rx = ls.x.mul(c).add(ls.z.mul(s));
  const rz = ls.z.mul(c).sub(ls.x.mul(s));
  const sink = scl.mul(0.22);
  mat.positionNode = Fn(() => {
    const n = vec3(
      normalLocal.x.mul(c).add(normalLocal.z.mul(s)),
      normalLocal.y,
      normalLocal.z.mul(c).sub(normalLocal.x.mul(s)),
    ).toVar();
    normalLocal.assign(n);
    return vec3(rx.add(wpos.x), ls.y.add(y).sub(sink), rz.add(wpos.y));
  })();
  const dist = wpos.sub(vec2(cameraPosition.x, cameraPosition.z)).length();
  bandFade(mat, dist, null, DEB_R - 6, 5);
}
