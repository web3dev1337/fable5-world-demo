/**
 * Joint-bilateral GTAO upsample (the AO itself renders half-res in the merged
 * pass; faithful GTAONode port lives in Gtao.ts). Full-res depth guides a
 * 4-tap bilateral; a gated fallback to the plain average prevents the
 * bilateral collapse that fabricated ao=0 black bands on grazing slopes. AO is
 * a near-field cue — faded with distance and shed on sun-lit (high-luminance)
 * pixels (indirect-only approximation; see DEVIATIONS.md).
 */

import type { TextureNode } from 'three/webgpu';
import {
  Fn,
  exp2,
  float,
  getViewPosition,
  luminance,
  mix,
  screenSize,
  screenUV,
  smoothstep,
  vec2,
} from 'three/tsl';
import type { NF, NM4, NV4 } from '../../gpu/TSLTypes';

export function buildAoUpsample(args: {
  aoSrc: NV4 | null;
  depthTex: TextureNode;
  uProjInv: NM4;
  beauty: TextureNode;
}): NF | null {
  const { aoSrc, depthTex, uProjInv, beauty } = args;
  if (!aoSrc) return null;
  return Fn((): NF => {
    const viewC = getViewPosition(screenUV, depthTex.x, uProjInv);
    const dist = viewC.length();
    const k = smoothstep(700, 1800, dist);
    // indirect-only approximation: sun-lit pixels (high HDR luminance)
    // shed most of the post-AO — occlusion belongs to ambient light.
    // (True aoNode-into-lighting wiring lands with the Phase-4 material
    // restructure; see DEVIATIONS.md.)
    const directK = smoothstep(1.2, 4.0, luminance(beauty.rgb)).mul(0.75);
    const halfTexel = vec2(1).div(screenSize.mul(0.5));
    const zC = viewC.z;
    const acc = float(0).toVar();
    const avg = float(0).toVar();
    const wsum = float(1e-4).toVar();
    for (const [ox, oy] of [
      [-0.5, -0.5],
      [0.5, -0.5],
      [-0.5, 0.5],
      [0.5, 0.5],
    ] as const) {
      const uvi = screenUV.add(halfTexel.mul(vec2(ox, oy)));
      const ai = ((aoSrc as unknown as { sample(uv: unknown): unknown }).sample(uvi) as NV4).x;
      const zi = getViewPosition(uvi, (depthTex.sample(uvi) as unknown as NV4).x, uProjInv).z;
      const w = exp2(zi.sub(zC).abs().mul(-3.5));
      acc.addAssign(ai.mul(w));
      avg.addAssign(ai);
      wsum.addAssign(w);
    }
    // GATED fallback for bilateral collapse: on grazing slopes near the
    // horizon a half-res texel spans tens of meters of view depth, every
    // tap rejects, and acc/1e-4 → 0 — the upsampler FABRICATED ao=0 and
    // painted the far field black (horizon-black band; same collapse on
    // grazing water = bm2 far-rim stripe). Support-free pixels fall back
    // to the plain 4-tap average; wsum > 0.02 (any tap within ~2 m)
    // keeps the bilateral result EXACT — zero deviation on healthy
    // pixels (a global +0.01 weight floor printed a ~1% AO wash on the
    // bm7 hero trunk and was rejected).
    const aoRaw = mix(avg.mul(0.25), acc.div(wsum), smoothstep(0.002, 0.02, wsum));
    return mix(mix(aoRaw, float(1), directK), float(1), k);
  })();
}
