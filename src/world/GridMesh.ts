/**
 * Shared flat XZ grid geometry (y = 0) spanning the whole world as GRID×GRID
 * quads. The far canopy shell and the terrain shadow proxy both build this
 * exact mesh and then lift its vertices to terrain height in their own vertex
 * (positionNode) stage. Kept in one place so the resolution, world mapping,
 * and triangle winding can't drift between the two consumers.
 */

import { BufferAttribute, BufferGeometry } from 'three';
import { WORLD_SIZE } from './WorldConst';

export function worldGridGeometry(grid: number): BufferGeometry {
  const n = grid + 1;
  const pos = new Float32Array(n * n * 3);
  for (let z = 0; z < n; z++) {
    for (let x = 0; x < n; x++) {
      const i = (z * n + x) * 3;
      pos[i] = (x / grid - 0.5) * WORLD_SIZE;
      pos[i + 1] = 0;
      pos[i + 2] = (z / grid - 0.5) * WORLD_SIZE;
    }
  }
  const idx = new Uint32Array(grid * grid * 6);
  let w = 0;
  for (let z = 0; z < grid; z++) {
    for (let x = 0; x < grid; x++) {
      const a = z * n + x;
      idx[w++] = a;
      idx[w++] = a + n;
      idx[w++] = a + 1;
      idx[w++] = a + 1;
      idx[w++] = a + n;
      idx[w++] = a + n + 1;
    }
  }
  const geo = new BufferGeometry();
  geo.setAttribute('position', new BufferAttribute(pos, 3));
  geo.setIndex(new BufferAttribute(idx, 1));
  return geo;
}
