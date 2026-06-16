/**
 * Debug-probe ladders kept OUT of the production post graph so the real
 * stages stay legible:
 *   - ?cloudview=N   — per-step bisect probes injected into the aerial /
 *     cloud-composite node (build-time branch; exactly one is selected).
 *   - ?skyveldbg=err|raw|ana — TRAA velocity diagnostics over far geometry.
 * These paint raw values, so the renderer runs with NoToneMapping when active.
 */

import type { TextureNode } from 'three/webgpu';
import {
  Fn,
  clamp,
  float,
  getViewPosition,
  mix,
  screenSize,
  screenUV,
  vec3,
} from 'three/tsl';
import type { NB, NF, NM4, NV2, NV3, NV4 } from '../../gpu/TSLTypes';
import { CLOUD_BOTTOM, CLOUD_TOP } from '../../sky/Clouds';
import { isSkyDepth } from './skyDepth';

/**
 * ?cloudview override for the in-clouds composite block. Returns the value to
 * assign to `scenePart`, or null to run the production depth-aware upsample.
 */
export function cloudviewCloudProbe(
  cloudview: string | null,
  ctx: {
    scenePart: NV3;
    dirW: NV3;
    camPosW: NV3;
    dist: NF;
    maxD: NF;
    isSky: NB;
    cloudTex: NV4 | null;
  },
): NV3 | null {
  const { scenePart, dirW, camPosW, dist, maxD, isSky, cloudTex } = ctx;
  if (cloudview === '2') {
    // constant output; march not built at all (graph-pollution bisect)
    return vec3(1, 0, 0);
  }
  if (cloudview === '7') {
    // ray-direction probe: R = dir.y, G = -dir.y, B = horizontalness
    return vec3(
      clamp(dirW.y, 0, 1),
      clamp(dirW.y.negate(), 0, 1),
      dirW.y.abs().lessThan(1e-3).select(float(1), float(0)),
    );
  }
  if (cloudview === '6') {
    // slab-intersection probe: R = valid, G = tEnter/10km, B = tExit/10km
    const t0 = float(CLOUD_BOTTOM).sub(camPosW.y).div(dirW.y);
    const t1 = float(CLOUD_TOP).sub(camPosW.y).div(dirW.y);
    const tEnterRaw = t0.min(t1);
    const tExitRaw = t0.max(t1);
    const ins = camPosW.y
      .greaterThan(CLOUD_BOTTOM)
      .and(camPosW.y.lessThan(CLOUD_TOP));
    const tEnter = ins.select(float(0), tEnterRaw.max(0));
    const tExit = tExitRaw.min(maxD).min(26000);
    const valid = tExit.greaterThan(tEnter).and(dirW.y.abs().greaterThan(1e-4));
    return vec3(
      valid.select(clamp(tExit.div(10000), 0, 1), float(0)),
      clamp(tEnter.div(10000), 0, 1),
      clamp(dist.div(10000), 0, 1),
    );
  }
  if (cloudview === '5') {
    // camera uniform probe: gray = camera height / 3000
    return vec3(camPosW.y.div(3000));
  }
  if (cloudview === '3') {
    // isSky probe: white = far-plane depth
    return isSky.select(vec3(1), vec3(0));
  }
  if (cloudview === '1' && cloudTex) {
    // march alpha as magenta overlay
    return mix(scenePart, vec3(1, 0, 1), clamp(cloudTex.a, 0, 1));
  }
  return null;
}

/** ?cloudview=4 context probe applied after the clouds block. */
export function cloudviewContextProbe(cloudview: string | null, d: NF): NV3 | null {
  if (cloudview === '4') {
    // context probe: R/G = screenUV gradients, B = raw depth ×100
    return vec3(screenUV.x, screenUV.y, clamp(d.mul(100), 0, 1));
  }
  return null;
}

/**
 * ?skyveldbg=err|raw|ana — paints TRAA velocity over far geometry (>1.5 km).
 * `ana` = analytic camera reprojection (what TRAA consumes), `raw` = velocity
 * MRT, `err` = their difference. R = x ×20, G = y ×20, B = mask.
 */
export function buildSkyVelDbg(args: {
  velocityTex: TextureNode;
  depthTex: TextureNode;
  uProjInv: NM4;
  velReproject: (texel: NV2) => NV2;
  mode: string | null;
}): NV3 {
  const { velocityTex, depthTex, uProjInv, velReproject, mode } = args;
  return Fn((): NV3 => {
    const texel = screenUV.mul(screenSize);
    const raw = (velocityTex.load(texel as unknown as Parameters<typeof velocityTex.load>[0]) as unknown as NV4).xy;
    const d = (depthTex.load(texel as unknown as Parameters<typeof depthTex.load>[0]) as unknown as NV4).x;
    const isSky = isSkyDepth(d);
    const dist = getViewPosition(screenUV, d, uProjInv).length();
    const farGeo = isSky.not().and(dist.greaterThan(1500));
    const ana = velReproject(texel);
    const v = mode === 'raw' ? raw : mode === 'ana' ? ana : ana.sub(raw);
    const err = v.abs().mul(20);
    const mask = farGeo.select(float(1), float(0));
    return vec3(err.x.mul(mask), err.y.mul(mask), mask);
  })();
}
