/**
 * GPU-only auto-exposure feedback (no readback). A single compute invocation
 * samples a 12×12 center-weighted grid of the beauty buffer, computes the
 * log-average luminance, and smooths the exposure scalar into a 2-float
 * storage buffer the grade node reads. `buildExposureInit` seeds the buffer
 * to 1 so frame 0 is never black.
 */

import type { Renderer, StorageBufferNode, TextureNode } from 'three/webgpu';
import {
  Fn,
  If,
  Return,
  clamp,
  exp2,
  float,
  instanceIndex,
  log2,
  luminance,
  mix,
  texture,
  vec2,
} from 'three/tsl';

export function buildExposureInit(
  exposureBuf: StorageBufferNode<'float'>,
): Parameters<Renderer['compute']>[0] {
  return Fn(() => {
    exposureBuf.element(0).assign(1);
    exposureBuf.element(1).assign(1);
  })().compute(1);
}

export function buildExposureKernel(
  exposureBuf: StorageBufferNode<'float'>,
  beautyForMeter: TextureNode,
): Parameters<Renderer['compute']>[0] {
  const kernel: Parameters<Renderer['compute']>[0] = Fn(() => {
    If(instanceIndex.greaterThanEqual(1), () => {
      Return();
    });
    const logSum = float(0).toVar();
    const N = 12;
    for (let gy = 0; gy < N; gy++) {
      for (let gx = 0; gx < N; gx++) {
        const u = (gx + 0.5) / N;
        const v = (gy + 0.5) / N;
        // center-weighted metering
        const w = 1 - 0.55 * Math.hypot(u - 0.5, (v - 0.5) * 0.9);
        const c = texture(beautyForMeter.value, vec2(u, v)).rgb;
        const lum = luminance(c).max(1e-4);
        logSum.addAssign(log2(lum).mul(w));
      }
    }
    let wTot = 0;
    for (let gy = 0; gy < N; gy++) {
      for (let gx = 0; gx < N; gx++) {
        wTot += 1 - 0.55 * Math.hypot((gx + 0.5) / N - 0.5, ((gy + 0.5) / N - 0.5) * 0.9);
      }
    }
    const avgLum = exp2(logSum.div(wTot));
    // key: auto-exposure normalizes the frame to mid-gray — the key sets
    // WHICH gray. 0.125 floated forest scenes into a washy high-key; 0.1
    // keeps deep canopy darks so the sun reads (user: "washed out").
    // Gain cap 4 (was 7): a fully canopy-shadowed interior must STAY a
    // dark frame (scene1 value structure: dark frame → lit mid → bright
    // bg) — at ×7 the meter dragged it to pastel mid-gray and noon
    // interiors read overcast.
    const target = clamp(float(0.1).div(avgLum), 0.18, 4.0);
    const prev = exposureBuf.element(0);
    exposureBuf.element(0).assign(mix(prev, target, 0.07));
  })().compute(1);
  kernel.setName('autoExposure');
  return kernel;
}
