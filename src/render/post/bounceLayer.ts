/**
 * Half-res screen-space bounce / colour-bleed layer (DEVIATIONS D-2).
 * Depth-gated gather of nearby on-screen radiance; composited after AO with
 * the receiver's chroma. Subtle by design — probes carry large-scale bounce;
 * this adds local green-on-trunk / warm-on-rock bleed.
 */

import type { TextureNode } from 'three/webgpu';
import {
  Fn,
  If,
  clamp,
  float,
  getViewPosition,
  screenUV,
  smoothstep,
  texture,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import type { NM4, NV4 } from '../../gpu/TSLTypes';
import { isSkyDepth } from './skyDepth';

export function buildBounceLayer(args: {
  depthTex: TextureNode;
  beauty: TextureNode;
  uProjInv: NM4;
}): NV4 {
  const { depthTex, beauty, uProjInv } = args;
  return Fn((): NV4 => {
    const res = vec4(0).toVar();
    const d = depthTex.x;
    const isSky = isSkyDepth(d);
    If(isSky.not(), () => {
      const viewPos = getViewPosition(screenUV, d, uProjInv);
      const dist = viewPos.length();
      // ≈0.6 m world-space gather radius projected to screen
      const rPx = clamp(float(0.55).div(dist), 0.004, 0.07);
      const sum = vec3(0).toVar();
      const wsum = float(0).toVar();
      for (let i = 0; i < 8; i++) {
        const ga = i * 2.399963 + 0.7;
        const rr = Math.sqrt((i + 0.5) / 8);
        const offX = Math.cos(ga) * rr;
        const offY = Math.sin(ga) * rr;
        const uvS = screenUV.add(vec2(offX, offY).mul(rPx));
        const dS = texture(depthTex.value, uvS).x;
        const pS = getViewPosition(uvS, dS, uProjInv);
        const w = smoothstep(1.8, 0.25, pS.sub(viewPos).length());
        sum.addAssign(texture(beauty.value, uvS).rgb.mul(w));
        wsum.addAssign(w);
      }
      res.assign(vec4(sum.div(wsum.max(1e-3)), wsum.mul(0.125)));
    });
    return res;
  })();
}
