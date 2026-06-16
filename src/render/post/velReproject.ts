/**
 * Analytic per-pixel camera velocity for TRAA history reprojection. Exact for
 * a static world INCLUDING translation parallax; the far-plane limit covers
 * sky (clouds at quasi-infinity) with no branch. Emitted in VelocityNode's
 * convention (ndcCur−ndcPrev, y-up NDC). uv space is TOP-LEFT origin so the
 * forward projection flips v back (getViewPosition flips it internally), or the
 * reprojection comes out vertically mirrored.
 */

import type { TextureNode } from 'three/webgpu';
import { getViewPosition, screenSize, vec2, vec4 } from 'three/tsl';
import type { NM4, NV2, NV4 } from '../../gpu/TSLTypes';

export function buildVelReproject(args: {
  depthTex: TextureNode;
  uProjInv: NM4;
  uCamWorld: NM4;
  uPrevView: NM4;
  uPrevProj: NM4;
}): (texel: NV2) => NV2 {
  const { depthTex, uProjInv, uCamWorld, uPrevView, uPrevProj } = args;
  return (texel: NV2): NV2 => {
    // texel = uv*size, already carrying the +0.5 center. screenSize == the
    // full-res MRT/resolve dims in every pass that calls this
    // (velocityTex.size() on the MRT attachment returned 0 — NaN uvs).
    const uvv = texel.div(screenSize);
    const d = (depthTex.load(texel as unknown as Parameters<typeof depthTex.load>[0]) as unknown as NV4).x;
    const posV = getViewPosition(uvv, d, uProjInv);
    const posW = uCamWorld.mul(vec4(posV, 1)).xyz;
    const posVPrev = uPrevView.mul(vec4(posW, 1)).xyz;
    const clipPrev = uPrevProj.mul(vec4(posVPrev, 1));
    const uvPrevRaw = clipPrev.xy.div(clipPrev.w).mul(0.5).add(0.5);
    const uvPrev = vec2(uvPrevRaw.x, uvPrevRaw.y.oneMinus());
    return uvv.sub(uvPrev).mul(vec2(2, -2));
  };
}
