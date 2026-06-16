/**
 * Filmic grade node (per-ToD colour script: white balance, teal–orange split
 * toning, saturation, gentle contrast around mid-gray, restrained vignette and
 * static grain) feeding AgX via renderer.toneMapping. Owns its grade uniforms
 * and returns a `refresh` closure the ToD setter calls to re-push them.
 */

import type { StorageBufferNode } from 'three/webgpu';
import { Fn, dot, float, mix, screenUV, smoothstep, vec2, vec3 } from 'three/tsl';
import { hash12 } from '../../gpu/noise/NoiseTSL';
import type { NV3 } from '../../gpu/TSLTypes';
import { runiform } from '../../gpu/RenderUniform';
import type { GradeUniforms } from '../ColorScript';

export function buildGrade(
  grade: GradeUniforms,
  withBloom: NV3,
  exposureBuf: StorageBufferNode<'float'>,
): { node: NV3; refresh: () => void } {
  const uWB = runiform(grade.whiteBalance);
  const uShadowTint = runiform(grade.shadowTint);
  const uHighlightTint = runiform(grade.highlightTint);
  const uShadowAmt = runiform(0.3);
  const uHighlightAmt = runiform(0.2);
  const uSat = runiform(1.0);
  const uContrast = runiform(1.03);
  const refresh = (): void => {
    uShadowAmt.value = grade.shadowAmt;
    uHighlightAmt.value = grade.highlightAmt;
    uSat.value = grade.saturation;
    uContrast.value = grade.contrast;
  };

  const node = Fn((): NV3 => {
    let c: NV3 = withBloom.mul(exposureBuf.element(0));
    c = c.mul(vec3(uWB));
    const lum = dot(c, vec3(0.2126, 0.7152, 0.0722));
    const shadowMask = smoothstep(0.45, 0.08, lum).mul(float(uShadowAmt));
    c = mix(c, c.mul(vec3(uShadowTint)), shadowMask);
    const hiMask = smoothstep(0.35, 0.95, lum).mul(float(uHighlightAmt));
    c = mix(c, c.mul(vec3(uHighlightTint)), hiMask);
    // saturation + gentle contrast around mid-gray
    c = mix(vec3(dot(c, vec3(0.2126, 0.7152, 0.0722))), c, float(uSat));
    c = c.div(0.18).pow(vec3(float(uContrast))).mul(0.18);
    // restrained vignette + static grain (freeze-deterministic)
    const v = screenUV.sub(0.5);
    const vig = float(1).sub(dot(v, v).mul(0.42));
    const grain = hash12(screenUV.mul(vec2(1923.7, 1671.3))).sub(0.5).mul(0.012);
    return c.mul(vig).add(grain);
  })();

  return { node, refresh };
}
