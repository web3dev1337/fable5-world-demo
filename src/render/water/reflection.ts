/**
 * Reflection: screen-space march with sky/probe fallback.
 *
 * Streams at grazing angles must reflect the far bank / trees (dark), not
 * bright horizon haze — sky-only reflection read as a white sheet. March the
 * opaque depth buffer along the reflected ray; misses fall back to the
 * sky-view LUT, gated by a crowned-horizon occlusion test so the "sky" is only
 * trusted where the reflected ray actually clears terrain AND tree crowns.
 */

import {
  Break,
  Fn,
  If,
  Loop,
  cameraFar,
  cameraNear,
  cameraProjectionMatrix,
  cameraViewMatrix,
  clamp,
  float,
  getScreenPosition,
  interleavedGradientNoise,
  mix,
  perspectiveDepthToViewZ,
  positionView,
  positionWorld,
  reflect,
  screenCoordinate,
  smoothstep,
  vec2,
  vec3,
  vec4,
  viewportDepthTexture,
  viewportSharedTexture,
} from 'three/tsl';
import type { StorageTexture } from 'three/webgpu';
import { canopyAt } from '../../gpu/passes/Scatter';
import type { ProbeGI } from '../../gpu/passes/ProbeGI';
import type { Atmosphere } from '../../sky/Atmosphere';
import type { Heightfield } from '../../world/Heightfield';
import type { NF, NI, NV2, NV3, NV4 } from '../../gpu/TSLTypes';

export function reflection(
  n: NV3,
  viewDir: NV3,
  dist: NF,
  atm: Atmosphere,
  hf: Heightfield,
  canopyTex: StorageTexture | null,
  gi: ProbeGI | null,
): NV3 {
  const rdir = reflect(viewDir.negate(), vec3(n.x.mul(0.55), n.y, n.z.mul(0.55)).normalize());
  const reflectionNode = Fn((): NV3 => {
    const dirV = cameraViewMatrix.mul(vec4(rdir, 0)).xyz;
    // far cap 28 m: a grazing lake reflects its far tree line — with a
    // 12 m cap the march died ~200 m short and the whole far band fell to
    // the FLAT probe fallback (read as a dark slab hovering on the lake)
    const stepLen = clamp(dist.mul(0.09), 0.25, 28);
    const jitter = interleavedGradientNoise(screenCoordinate.xy);
    const hit = float(0).toVar();
    const hitUv = vec2(0, 0).toVar();
    Loop(18, ({ i }: { readonly i: NI }) => {
      const t = float(i).add(jitter).mul(stepLen);
      const pV = positionView.add(dirV.mul(t));
      const uvS = getScreenPosition(pV, cameraProjectionMatrix) as unknown as NV2;
      If(
        uvS.x.lessThan(0).or(uvS.x.greaterThan(1)).or(uvS.y.lessThan(0)).or(uvS.y.greaterThan(1)),
        () => {
          Break();
        },
      );
      const zS = perspectiveDepthToViewZ(
        (viewportDepthTexture(uvS) as unknown as NV4).x,
        cameraNear,
        cameraFar,
      );
      // hit: scene surface just in front of the ray point (viewZ is negative)
      If(
        zS.greaterThan(pV.z.add(0.06)).and(zS.lessThan(pV.z.add(stepLen.mul(2.6).add(0.7)))),
        () => {
          hit.assign(1);
          hitUv.assign(uvS);
          Break();
        },
      );
    });
    // sky fallback (horizon-clamped so the LUT never samples below ground)
    const rdirUp = vec3(rdir.x, rdir.y.max(0.035), rdir.z).normalize();
    const sky = atm.skyColor(rdirUp);
    // CROWNED-HORIZON occlusion: when the SSR march misses, "sky" is only
    // correct if the reflected ray clears terrain AND tree crowns. March
    // the height field at log-spaced ranges with the canopy map raising
    // the tested horizon by crown height — one test covers both regimes:
    // steep gorge-stream rays get caught by overhead crowns (dark wall/
    // canopy mirror, scene1), grazing lake rays clear the far tree line
    // into open sky (a blanket canopy multiply here used to crush the
    // far-lake band to black). Occluded rays fall back to the probe field
    // toward the ray — it already encodes wall/canopy brightness.
    const horizonVis = float(1).toVar();
    for (const dRay of [9, 24, 65, 180]) {
      const q = positionWorld.xz.add(rdir.xz.mul(dRay));
      const rayY = positionWorld.y.add(rdir.y.mul(dRay));
      let hQ = hf.sampleHeightNearest(q) as NF;
      if (canopyTex) {
        hQ = hQ.add(canopyAt(canopyTex, q).mul(16)) as NF;
      }
      // wide knee — a hard threshold printed razor-edged reflection bands
      horizonVis.mulAssign(smoothstep(-16, 7, rayY.sub(hQ)));
    }
    const wallAmb = gi
      ? (gi.irradiance(positionWorld, rdir).mul(0.65) as unknown as NV3)
      : (sky.mul(0.18) as unknown as NV3);
    // ripple-jittered blend breaks the residual banding at the transition
    const vJit = n.x.add(n.z).mul(0.18);
    const fallback = mix(wallAmb, sky as unknown as NV3, horizonVis.add(vJit).clamp(0, 1));
    // fade SSR toward the screen border so hits don't pop at the edge
    const e = hitUv.sub(0.5).abs().mul(2);
    const edgeFade = smoothstep(1.0, 0.82, e.x.max(e.y));
    const scene = (viewportSharedTexture(hitUv) as unknown as NV4).rgb;
    return mix(fallback, scene, hit.mul(edgeFade));
  })();
  return reflectionNode as unknown as NV3;
}
