/**
 * Screen-space contact shadows (spec §2 floor). Short depth-buffer march
 * toward the sun picks up the ~0.1–2 m contact occlusion the 2048² cascades
 * can't resolve. Near field only; floored so it stays a contact CUE (never
 * pitch black — no-black-shadows law).
 */

import type { TextureNode } from 'three/webgpu';
import {
  Fn,
  If,
  float,
  getScreenPosition,
  getViewPosition,
  screenUV,
  smoothstep,
  texture,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import { hash12 } from '../../gpu/noise/NoiseTSL';
import type { NF, NM4 } from '../../gpu/TSLTypes';
import type { Atmosphere } from '../../sky/Atmosphere';
import { isSkyDepth } from './skyDepth';

const SSCS_STEPS = 12;

export function buildContactShadow(args: {
  depthTex: TextureNode;
  uProjInv: NM4;
  atmosphere: Atmosphere;
  uView: NM4;
  uProj: NM4;
  frameU: NF;
}): NF {
  const { depthTex, uProjInv, atmosphere, uView, uProj, frameU } = args;
  return Fn((): NF => {
    const result = float(1).toVar();
    const d = depthTex.x;
    const isSky = isSkyDepth(d);
    const viewPos = getViewPosition(screenUV, d, uProjInv);
    const dist = viewPos.length();
    If(isSky.not().and(dist.lessThan(240)), () => {
      const sunW = vec3(atmosphere.sunDir).normalize();
      const sunV = uView.mul(vec4(sunW, 0)).xyz;
      const jit = hash12(screenUV.mul(vec2(517.7, 893.3)).add(float(frameU).mul(0.7548)))
        .mul(0.8)
        .add(0.4);
      const range = float(1.7);
      // first-hit-wins early exit: the contribution 1−f·0.5 strictly
      // DECREASES with step index, so once any step hits, later steps can
      // never raise the max — identical output, and whole wavefronts skip
      // the remaining taps (contact hits are spatially coherent). hitF
      // sentinel 2 = no hit yet.
      const hitF = float(2).toVar();
      for (let s = 1; s <= SSCS_STEPS; s++) {
        // quadratic step distribution: dense near the surface
        const f = (s / SSCS_STEPS) ** 1.6;
        If(hitF.greaterThan(1.5), () => {
          const sampleV = viewPos.add(sunV.mul(range).mul(jit).mul(f));
          const uvS = getScreenPosition(sampleV, uProj);
          const inFrame = uvS.x
            .greaterThan(0.001)
            .and(uvS.x.lessThan(0.999))
            .and(uvS.y.greaterThan(0.001))
            .and(uvS.y.lessThan(0.999));
          const dS = texture(depthTex.value, uvS).x;
          const bufV = getViewPosition(uvS, dS, uProjInv);
          const dz = bufV.z.sub(sampleV.z); // >0: buffer closer to camera
          const hit = dz.greaterThan(0.05).and(dz.lessThan(1.4)).and(inFrame);
          If(hit, () => {
            hitF.assign(f);
          });
        });
      }
      const occl = hitF.lessThan(1.5).select(float(1).sub(hitF.mul(0.5)), float(0));
      // distance fade + floor
      const fade = smoothstep(240, 140, dist);
      result.assign(float(1).sub(occl.mul(0.6).mul(fade)));
    });
    return result;
  })();
}
