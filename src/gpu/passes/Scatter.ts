/**
 * Scatter — GPU vegetation/rock placement (spec §3.5), boot-time.
 *
 * Clustered Poisson, fully parallel: a jittered child grid (one thread per
 * candidate cell) is gated by per-class density functions (biome, slope,
 * altitude/treeline, moisture, snow, rock exposure, water) × a parent clump
 * field (hashed parent points per coarse cell → light-competition clumping;
 * the SAME parent field feeds the understory pass as a canopy proxy: ferns
 * gather under tree clumps, flowers in gaps, pink shrubs at clump edges).
 * Ecotones: the biome id is read through a low-frequency warp so boundaries
 * interdigitate instead of tracing classification isolines.
 *
 * Accepted instances are atomically appended into storage buffers — instance
 * data never touches the CPU (only the final counts are read back once for
 * HUD/draw bookkeeping). Deterministic: all randomness is pcg2d(cell, salt),
 * an integer hash — sin-based hashes band at 4-digit cell coordinates.
 *
 * Instance layout (two vec4 buffers):
 *   A = (x, y, z, scale)
 *   B = (yaw, leanX, leanZ, idF)   idF = class·8 + variant  (exact in f32)
 *
 * This file orchestrates the four layer kernels and re-exports the public
 * surface; the kernels and primitives live in ./scatter/*.
 */

import type { Renderer } from 'three/webgpu';
import type { WorldSeed } from '../../core/Seed';
import type { Heightfield } from '../../world/Heightfield';
import { readCount } from './scatter/helpers';
import {
  scatterExtras,
  scatterStones,
  scatterTrees,
  scatterUnderstory,
} from './scatter/layers';
import type { ScatterResult } from './scatter/types';

// public surface — kept importable from this module so consumers don't churn
export { cellHash, cellHash2 } from './scatter/helpers';
export { CANOPY_RES, buildCanopyMap, canopyAt } from './scatter/canopyMap';
export { TREE_VARIANTS, VegClass } from './scatter/classes';
export type { ScatterLayer, ScatterResult } from './scatter/types';

export async function runScatter(
  renderer: Renderer,
  hf: Heightfield,
  seed: WorldSeed,
): Promise<ScatterResult> {
  const sT = seed.sub('scatter/trees') & 0x7fffffff;
  const sU = seed.sub('scatter/understory') & 0x7fffffff;
  const sE = seed.sub('scatter/extras') & 0x7fffffff;
  const sS = seed.sub('scatter/stones') & 0x7fffffff;

  // sequential dispatch (one compute pass each); understory/extras/stones reuse
  // the TREE clump salt (sT) so their canopy proxy lines up with the trees
  const trees = await scatterTrees(renderer, hf, sT);
  const understory = await scatterUnderstory(renderer, hf, sU, sT);
  const extras = await scatterExtras(renderer, hf, sE, sT);
  const stones = await scatterStones(renderer, hf, sS, sT);

  // ---- counts (single boot-time readback; instance data stays on GPU) ----
  const [tc, uc, ec, sc] = await Promise.all([
    readCount(renderer, trees.counter, trees.cap),
    readCount(renderer, understory.counter, understory.cap),
    readCount(renderer, extras.counter, extras.cap),
    readCount(renderer, stones.counter, stones.cap),
  ]);

  return {
    trees: { bufA: trees.bufA, bufB: trees.bufB, cap: trees.cap, count: tc },
    understory: {
      bufA: understory.bufA,
      bufB: understory.bufB,
      cap: understory.cap,
      count: uc,
    },
    extras: { bufA: extras.bufA, bufB: extras.bufB, cap: extras.cap, count: ec },
    stones: { bufA: stones.bufA, bufB: stones.bufB, cap: stones.cap, count: sc },
  };
}
