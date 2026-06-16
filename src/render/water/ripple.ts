/**
 * Ripple surface normal: two fbm-gradient layers advected along the hydrology
 * flow field with the classic two-phase flowmap blend (no sliding-texture
 * artifact). |flowDir| encodes speed and is ZERO in lakes — they fall back to
 * a faint breeze ripple. Normals come from NoiseBake's pre-derived d(fbm)/dxz,
 * so a ripple layer costs one texture fetch.
 *
 * The two-phase advection offsets (offA/offB) and crossfade weight (w2) are
 * returned alongside the normal because the foam builder reuses the exact same
 * advection so its pattern tracks the ripples (see foam.ts).
 */

import { abs, float, fract, mix, positionWorld, texture, time, vec2, vec3 } from 'three/tsl';
import type { StorageTexture } from 'three/webgpu';
import { PERIOD_FBM } from '../../gpu/passes/NoiseBake';
import type { NF, NV2, NV3, NV4 } from '../../gpu/TSLTypes';
import { FLOW_CYC } from './constants';

export interface RippleResult {
  /** ripple surface normal (object space, +Y up) */
  n: NV3;
  /** two-phase crossfade weight — shared with foam */
  w2: NF;
  /** phase-A advection offset — shared with foam */
  offA: NV2;
  /** phase-B advection offset — shared with foam */
  offB: NV2;
}

export function rippleNormal(noiseA: StorageTexture, fdir: NV2, spd: NF): RippleResult {
  const ph1 = fract(time.mul(FLOW_CYC));
  const ph2 = fract(time.mul(FLOW_CYC).add(0.5));
  const w2 = abs(ph1.sub(0.5)).mul(2);
  // advection velocity (m/s): rivers stream, lakes get a faint breeze drift
  const vel = fdir.mul(spd.mul(1.9)).add(vec2(0.045, 0.03));
  const gradAt = (s: number, off: NV2): NV2 =>
    (texture(noiseA, positionWorld.xz.sub(off).div(s * PERIOD_FBM)) as unknown as NV4).zw.div(s);
  const offA = vel.mul(ph1.div(FLOW_CYC));
  const offB = vel.mul(ph2.div(FLOW_CYC)).add(vec2(3.71, 1.13));
  const layer = (off: NV2): NV2 => gradAt(0.9, off).add(gradAt(3.4, off.mul(0.62)).mul(0.5));
  const grad = mix(layer(offA), layer(offB), w2);
  // baked fbm gradients are ±(3..10)/m at these scales — the old amp
  // (0.018+0.085·spd) tilted normals 8–30° everywhere, saturating fresnel
  // to ~1 and turning every stream into a sky mirror ("white sheet")
  const rippleAmp = float(0.007).add(spd.mul(0.028));
  const slope = grad.mul(rippleAmp);
  const n = vec3(slope.x.negate(), 1, slope.y.negate()).normalize();
  return { n, w2, offA, offB };
}
