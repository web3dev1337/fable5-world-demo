/**
 * Foam: shore/rapid foam + wet margin.
 *
 * Two-phase advection like the ripple normals — a linearly time-advected
 * pattern slides coherently and its thresholded fbm level sets read as sharp
 * white stripes (user-reported). Two decorrelated scales multiply into clumpy
 * patches instead of bands. Shares the ripple builder's advection offsets
 * (offA/offB) and crossfade weight (w2) so the foam tracks the ripples.
 */

import { clamp, mix, positionWorld, smoothstep, texture } from 'three/tsl';
import type { StorageTexture } from 'three/webgpu';
import { PERIOD_FBM } from '../../gpu/passes/NoiseBake';
import type { NF, NV2, NV4 } from '../../gpu/TSLTypes';

export function foam(
  noiseA: StorageTexture,
  offA: NV2,
  offB: NV2,
  w2: NF,
  vDepth: NF,
  sampleY: (q: NV2) => NF,
  fdir: NV2,
  spd: NF,
): NF {
  const foamUv = (off: NV2, s: number): NV2 => positionWorld.xz.sub(off).div(s * PERIOD_FBM);
  // EVERY octave must live inside the two-phase blend: phase A's offset
  // snaps to zero at its cycle wrap, and the blend only hides that for
  // terms weighted by w2 — a detail octave pinned to offA alone snapped
  // visibly once per cycle (user-reported "sharp stop in the loop").
  const fA = (texture(noiseA, foamUv(offA, 0.55)) as unknown as NV4).y;
  const fB = (texture(noiseA, foamUv(offB.mul(1.13), 0.55)) as unknown as NV4).y;
  const dA = (texture(noiseA, foamUv(offA.mul(0.6), 0.21)) as unknown as NV4).y;
  const dB = (texture(noiseA, foamUv(offB.mul(0.71), 0.21)) as unknown as NV4).y;
  // renormalize the crossfade variance — averaging two uncorrelated fields
  // flattens the pattern at blend midpoints and thresholded coverage pulses
  const varNorm = w2.mul(w2).add(w2.oneMinus().mul(w2.oneMinus())).sqrt();
  const fblend = mix(fA, fB, w2).sub(0.5).div(varNorm).add(0.5);
  const fDetail = mix(dA, dB, w2).sub(0.5).div(varNorm).add(0.5);
  const foamPat = smoothstep(0.42, 0.85, fblend.mul(0.62).add(fDetail.mul(0.38)));
  const shoreFoam = smoothstep(0.16, 0.03, vDepth).mul(0.42);
  // rapids key on the DROP of the water surface along flow (a large calm
  // river has high strength but no whitewater — slope is what froths).
  // Window starts at ~3% grade: a 1.5% start blanketed every gorge reach
  // in white (real streams run clear on smooth grades and froth at STEPS,
  // which survive in the smoothed field as locally steeper sub-reaches).
  const drop = sampleY(positionWorld.xz)
    .sub(sampleY(positionWorld.xz.add(fdir.mul(3))))
    .div(3);
  const rapidFoam = smoothstep(0.09, 0.24, drop).mul(smoothstep(0.18, 0.55, spd)).mul(0.8);
  return clamp(shoreFoam.add(rapidFoam), 0, 1).mul(foamPat).clamp(0, 0.68) as NF;
}
