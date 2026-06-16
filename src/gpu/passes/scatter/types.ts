/**
 * Scatter result shapes: per-layer storage buffers (two vec4 streams) plus the
 * accepted instance count read back once at boot.
 *
 * Instance layout (two vec4 buffers):
 *   A = (x, y, z, scale)
 *   B = (yaw, leanX, leanZ, idF)   idF = class·8 + variant  (exact in f32)
 */

import type { StorageBufferNode } from 'three/webgpu';

export interface ScatterLayer {
  bufA: StorageBufferNode<'vec4'>;
  bufB: StorageBufferNode<'vec4'>;
  cap: number;
  /** accepted instances (clamped to cap) — read back once at boot */
  count: number;
}

export interface ScatterResult {
  trees: ScatterLayer;
  understory: ScatterLayer;
  extras: ScatterLayer;
  /** stones (3 size classes) + fallen branches — ground-solid coverage */
  stones: ScatterLayer;
}
