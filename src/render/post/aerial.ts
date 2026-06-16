/**
 * Aerial perspective from depth (Hillaire in-scatter — Pillar D haze) plus the
 * froxel volumetrics under-blend and the half-res cloud composite. Froxel
 * shafts/valley fog integrate first (≤ ~480 m), the km-scale Hillaire haze on
 * top, then clouds are composited with a depth-aware upsample gate.
 *
 * The ?cloudview bisect ladder lives in postProbes.ts; here it is a single
 * build-time branch (`probe !== null`) so the production path stays legible.
 */

import type { TextureNode } from 'three/webgpu';
import { Fn, float, getViewPosition, screenUV, vec4 } from 'three/tsl';
import type { NM4, NV3, NV4 } from '../../gpu/TSLTypes';
import type { Atmosphere } from '../../sky/Atmosphere';
import { CLOUD_BOTTOM, CLOUD_TOP, type Clouds } from '../../sky/Clouds';
import type { Froxels } from '../../gpu/passes/Froxels';
import { isSkyDepth } from './skyDepth';
import { cloudviewCloudProbe, cloudviewContextProbe } from './postProbes';

export function buildAerial(args: {
  depthTex: TextureNode;
  beauty: TextureNode;
  uProjInv: NM4;
  uCamWorld: NM4;
  camPosW: NV3;
  atmosphere: Atmosphere;
  froxels: Froxels | null;
  clouds: Clouds | null;
  cloudTex: NV4 | null;
  cloudview: string | null;
  ablate: Set<string>;
}): NV3 {
  const { depthTex, beauty, uProjInv, uCamWorld, camPosW, atmosphere, froxels, clouds, cloudTex, cloudview, ablate } =
    args;
  return Fn((): NV3 => {
    const d = depthTex.x.toVar();
    const col = beauty.rgb.toVar();
    // ray direction from a FIXED finite depth (the far-plane depth value
    // degenerates through the inverse projection)
    const viewDirV = getViewPosition(screenUV, float(0.5), uProjInv).normalize();
    const dirW = uCamWorld.mul(vec4(viewDirV, 0)).xyz.normalize().toVar();
    const viewPos = getViewPosition(screenUV, d, uProjInv);
    const dist = viewPos.length();
    const distKm = dist.div(1000);
    const camAltKm = camPosW.y.div(1000).max(0.005);
    // sky = cleared depth; tolerate either depth convention (0 or 1 at far)
    const isSky = isSkyDepth(d);
    // froxel volumetrics first (local shafts/valley fog ≤ ~480 m), the
    // km-scale Hillaire haze integrates on top of the fogged radiance
    if (froxels) {
      const fogDist = isSky.select(float(1e5), dist);
      col.assign(froxels.apply(col, fogDist, screenUV));
    }
    const hazed = atmosphere.aerial(col, dirW, camAltKm, distKm);
    // reversed-z: far plane clears to 0 → sky already carries the atmosphere
    const scenePart = isSky.select(col, hazed).toVar();

    if (clouds && !ablate.has('clouds')) {
      const maxD = isSky.select(float(1e9), dist);
      const probe = cloudviewCloudProbe(cloudview, {
        scenePart,
        dirW,
        camPosW,
        dist,
        maxD,
        isSky,
        cloudTex,
      });
      if (probe !== null) {
        scenePart.assign(probe);
      } else if (cloudTex) {
        // depth-aware upsample gate: the cloud RTT is half-res, and
        // bilinear upsampling smears sky texels (visible through leaf
        // gaps) onto near geometry — clouds painted over close trees in
        // a woven pattern (user screenshot). A solid surface nearer
        // than the cloud-slab entry can have no cloud in front of it:
        // zero the contribution there. 300 m floor covers downward /
        // near-horizontal rays where the slab math degenerates.
        const t0 = float(CLOUD_BOTTOM).sub(camPosW.y).div(dirW.y);
        const t1 = float(CLOUD_TOP).sub(camPosW.y).div(dirW.y);
        const ins = camPosW.y
          .greaterThan(CLOUD_BOTTOM)
          .and(camPosW.y.lessThan(CLOUD_TOP));
        const tEnter = ins.select(float(0), t0.min(t1).max(0));
        const nearSolid = isSky.not().and(dist.lessThan(tEnter.max(300)));
        const k = nearSolid.select(float(0), float(1));
        const cl4 = cloudTex;
        scenePart.assign(
          scenePart.mul(float(1).sub(cl4.a.mul(k))).add(cl4.rgb.mul(k)),
        );
      }
    }
    const ctxProbe = cloudviewContextProbe(cloudview, d);
    if (ctxProbe !== null) {
      scenePart.assign(ctxProbe);
    }
    return scenePart;
  })();
}
