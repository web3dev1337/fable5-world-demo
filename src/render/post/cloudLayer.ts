/**
 * Half-res volumetric cloud-march layer for the merged HalfResMrtNode pass.
 * Marches the cloud slab from each pixel's view ray, gated by scene depth.
 */

import type { TextureNode } from 'three/webgpu';
import { Fn, float, getViewPosition, screenUV, vec2, vec4 } from 'three/tsl';
import { hash12 } from '../../gpu/noise/NoiseTSL';
import type { NF, NM4, NV3, NV4 } from '../../gpu/TSLTypes';
import type { Clouds } from '../../sky/Clouds';
import { isSkyDepth } from './skyDepth';

export function buildCloudLayer(args: {
  clouds: Clouds;
  depthTex: TextureNode;
  uProjInv: NM4;
  uCamWorld: NM4;
  camPosW: NV3;
  frameU: NF;
}): NV4 {
  const { clouds, depthTex, uProjInv, uCamWorld, camPosW, frameU } = args;
  return Fn((): NV4 => {
    const d = depthTex.x;
    const viewDirV = getViewPosition(screenUV, float(0.5), uProjInv).normalize();
    const dirW = uCamWorld.mul(vec4(viewDirV, 0)).xyz.normalize().toVar();
    const dist = getViewPosition(screenUV, d, uProjInv).length();
    const isSky = isSkyDepth(d);
    const maxD = isSky.select(float(1e9), dist);
    const jitter = hash12(
      screenUV.mul(vec2(911.3, 423.7)).add(float(frameU).mul(0.61803)),
    );
    const cl = clouds.march(camPosW, dirW, maxD, jitter);
    return vec4(cl.color, cl.alpha);
  })();
}
