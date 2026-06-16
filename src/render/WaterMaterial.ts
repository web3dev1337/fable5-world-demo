/**
 * Stream/lake water shading (Phase 6). One material per clipmap level of the
 * WaterSurface mesh (levels differ only in uniforms — pipeline is shared).
 *
 * Composition (all scene-linear, post stack tonemaps later):
 *   emissive = (1−foam) · mix(refraction, skyReflection, fresnel)
 *     refraction: viewportSharedTexture sampled at a ripple-refracted uv,
 *       depth-validated (samples landing on geometry IN FRONT of the water
 *       fall back to the straight uv), Beer–Lambert absorbed by the water
 *       column thickness from viewportDepthTexture, plus turbidity
 *       in-scatter tied to the sky so it tracks time-of-day.  (refraction.ts)
 *     reflection: sky-view LUT along the reflected ray (streams reflect sky;
 *       lakes upgrade to a planar pass later in Phase 6).            (reflection.ts)
 *   diffuse (colorNode) = foam albedo — lit by sun/CSM/GI like any surface,
 *     so foam in cliff shade goes properly dim.                      (foam.ts)
 *   PBR spec from the scene sun (roughness ~0.05 + ripple normals) supplies
 *   glints with cast shadows; the emissive reflection is sky-dome only, so
 *   nothing double-counts.
 *
 * Ripples: two fbm-gradient layers advected along the hydrology flow field
 * with the classic two-phase flowmap blend (no sliding-texture artifact).
 * |flowDir| encodes speed and is ZERO in lakes — they fall back to a faint
 * breeze ripple. Normals come from NoiseBake's pre-derived d(fbm)/dxz, so a
 * ripple layer costs one texture fetch.                              (ripple.ts)
 *
 * Each concern lives in its own builder under ./water/; this function wires
 * the flow field + view basis and composes the builder outputs.
 */

import { MeshStandardNodeMaterial } from 'three/webgpu';
import {
  cameraPosition,
  clamp,
  float,
  mix,
  positionLocal,
  positionView,
  positionWorld,
  smoothstep,
  transformNormalToView,
  vec2,
  vec3,
} from 'three/tsl';
import type { StorageTexture } from 'three/webgpu';
import { bilerpVec2Buffer } from '../gpu/BufferSample';
import type { ProbeGI } from '../gpu/passes/ProbeGI';
import type { NF, NV2, NV3, NV4 } from '../gpu/TSLTypes';
import type { Atmosphere } from '../sky/Atmosphere';
import type { Heightfield } from '../world/Heightfield';
import { WORLD_HALF } from '../world/WorldConst';
import { foam } from './water/foam';
import { reflection } from './water/reflection';
import { refraction } from './water/refraction';
import { rippleNormal } from './water/ripple';

export { FLOW_CYC } from './water/constants';

export interface WaterLevelHandles {
  /** snapped world origin of this clipmap level (uniform, updated per frame) */
  origin: NV2;
  /** world rect (minX, minZ, maxX, maxZ) of the next-finer level — discarded here */
  innerRect: NV4;
  /** cell size in meters (compile-time constant per level) */
  cell: number;
  /** coarse level → sample the min-reduced far field (narrow channels
   *  vanish at distance instead of stretching across whole cells) */
  far: boolean;
}

export function waterMaterial(
  hf: Heightfield,
  atm: Atmosphere,
  canopyTex: StorageTexture | null,
  gi: ProbeGI | null,
  lvl: WaterLevelHandles,
): MeshStandardNodeMaterial {
  const flow = hf.flow;
  const noiseA = hf.noiseA;
  if (!flow || !noiseA) throw new Error('waterMaterial needs hydrology + baked noise');

  const mat = new MeshStandardNodeMaterial();
  mat.transparent = true;
  mat.depthWrite = true;
  mat.metalness = 0;

  // ---- vertex: clipmap grid (cell units) → world water surface ----------------
  const sampleY = (q: NV2): NF => (lvl.far ? hf.sampleWaterYFar(q) : hf.sampleWaterY(q));
  const wxz = lvl.origin.add(positionLocal.xz.mul(lvl.cell));
  mat.positionNode = vec3(wxz.x, sampleY(wxz), wxz.y);

  // ---- inner-level cutout + hard world bounds ----------------------------------
  // Outside ±WORLD_HALF the field samples clamp to the border texel — a wet
  // border cell would extend an infinite water band into the far shell.
  const p = positionWorld.xz;
  const r = lvl.innerRect;
  const insideInner = p.x
    .greaterThan(r.x)
    .and(p.y.greaterThan(r.y))
    .and(p.x.lessThan(r.z))
    .and(p.y.lessThan(r.w));
  const inWorld = p.x.abs().lessThan(WORLD_HALF - 4).and(p.y.abs().lessThan(WORLD_HALF - 4));
  mat.maskNode = insideInner.not().and(inWorld);

  // ---- flow field --------------------------------------------------------------
  const simRes = hf.simRes;
  const g = clamp(positionWorld.xz.div(4096).add(0.5), 0, 1).mul(simRes).sub(0.5);
  const flowV = bilerpVec2Buffer(flow.flowDir, simRes, g);
  const spd = flowV.length();
  const fdir = flowV.div(spd.max(1e-4));

  // ---- ripple normal: two-phase flowmap over fbm gradients ---------------------
  const { n, w2, offA, offB } = rippleNormal(noiseA, fdir, spd);
  mat.normalNode = transformNormalToView(n);

  // ---- view / depth ------------------------------------------------------------
  const toCam = cameraPosition.sub(positionWorld);
  const dist = toCam.length();
  const viewDir = toCam.div(dist.max(1e-4));
  const fragZ = positionView.z; // negative

  // ---- refraction: Beer–Lambert depth-validated transmission -------------------
  const { refr, thick } = refraction(n, dist, fragZ, atm);
  // vertical water column under this fragment (foam/shore feather)
  const vDepth = thick.mul(viewDir.y.abs().max(0.06));

  // ---- reflection: screen-space march with sky/probe fallback ------------------
  const skyRefl = reflection(n, viewDir, dist, atm, hf, canopyTex, gi);
  // fresnel on a FLATTENED normal (standard water practice): per-pixel
  // ripple tilt makes (1−cosθ)^5 explode at any view angle — reflectance
  // weight should follow the mean surface, the ripples only shape WHAT is
  // reflected (rdir inside reflection keeps the full normal)
  const nFres = vec3(n.x.mul(0.3), n.y, n.z.mul(0.3)).normalize();
  const cosT = clamp(viewDir.dot(nFres), 0.0, 1.0);
  const fres = float(0.02).add(float(0.98).mul(cosT.oneMinus().pow(5)));

  // ---- foam --------------------------------------------------------------------
  const foamV = foam(noiseA, offA, offB, w2, vDepth, sampleY, fdir, spd);

  // ---- compose --------------------------------------------------------------------
  mat.colorNode = vec3(0.74, 0.76, 0.74).mul(foamV);
  mat.emissiveNode = mix(refr, skyRefl, fres).mul(foamV.oneMinus());
  mat.roughnessNode = mix(float(0.05), float(0.55), foamV);
  // shoreline feather: mm-deep water fades out over the bed. ALSO fade
  // steep surface RAMPS: the field dives ~2 m to the dry sentinel past
  // every shoreline — across a FLAT far beach seen edge-on that dive
  // renders as a thick dark band hugging the shore (twin-lake artifact).
  // Hydrology gates real water to gentle slopes (rdGate), so any render
  // slope ≥ ~30° is a dive, never water.
  const eS = 2.0;
  const gWx = sampleY(positionWorld.xz.add(vec2(eS, 0)))
    .sub(sampleY(positionWorld.xz.sub(vec2(eS, 0))))
    .div(2 * eS);
  const gWz = sampleY(positionWorld.xz.add(vec2(0, eS)))
    .sub(sampleY(positionWorld.xz.sub(vec2(0, eS))))
    .div(2 * eS);
  // near levels only: far levels carry the min-reduction's shore dip by
  // design (see Heightfield.reduceWaterY) — fading it would expose the
  // dark silt bed as a rim band instead
  const rampK = lvl.far
    ? float(1)
    : smoothstep(0.55, 0.3, vec2(gWx, gWz).length());
  mat.opacityNode = smoothstep(0.004, 0.05, vDepth).mul(rampK).mul(0.985);

  // ?waterdbg=N — component probe ladder (1 foam, 2 fresnel, 3 refraction,
  // 4 reflection, 5 column thickness, 6 SSR hit/horizon mix)
  const dbg = Number(new URLSearchParams(window.location.search).get('waterdbg') ?? '0');
  if (dbg > 0) {
    const paint =
      dbg === 1
        ? vec3(foamV)
        : dbg === 2
          ? vec3(fres)
          : dbg === 3
            ? refr
            : dbg === 4
              ? skyRefl
              : dbg === 5
                ? vec3(thick.mul(0.25), vDepth.mul(0.25), 0)
                : (skyRefl as NV3);
    mat.colorNode = vec3(0);
    mat.emissiveNode = paint;
    mat.opacityNode = float(1);
  }

  return mat;
}
