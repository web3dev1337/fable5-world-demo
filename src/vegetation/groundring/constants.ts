/**
 * GroundRing tuning constants and shared helpers. Grid/cell/range/cap tables
 * for the grass, debris and far-tuft clipmap layers, the 16|16 cell-pack bias,
 * and the continuous grass-thinning curve. These are referenced across the
 * cull kernels, the draw materials and the class shell, so they live in one
 * place to keep the cull↔draw couplings (cell sizes, band widths, thinning)
 * numerically identical.
 */

import { float } from 'three/tsl';
import type { NF } from '../../gpu/TSLTypes';

export const GRASS_GRID = 3072;
export const GRASS_CELL = 0.105; // m → ±161 m ring, ~90 slots/m²
export const GRASS_R = 155;
export const G_NEAR = 30;
export const G_MID = 70;
/** crossfade half-width — cull overlap and material fade MUST share it */
export const G_BAND = 12;
export const GRASS_CAPS = [524288, 1048576, 1835008]; // near/mid/far compact regions

/**
 * Continuous distance thinning, conserved by blade widening (1/√thin in the
 * vertex stage). thin(0..~40 m) = 1; ~0.45 at 100 m; ~0.16 at 155 m.
 * Beyond ~120 m an extra collapse folds coverage into ever-wider
 * super-tufts so the band reaches GRASS_R without a vertex explosion
 * (feedback 2.8: grass should render much farther, cheaply).
 */
export function grassThin(dist: NF): NF {
  const base = float(58).div(dist.max(1).add(42)).min(1).pow(1.15);
  const far = float(120).div(dist.max(120)).pow(1.6);
  return base.mul(far);
}

export const DEB_GRID = 512;
export const DEB_CELL = 0.3; // ±77 m ring
export const DEB_R = 74;
// cobble / pebble / twig / chip / litter
export const DEB_CAPS = [24576, 49152, 49152, 32768, 65536];

// far super-tuft layer (g3, feedback 2.8): its own COARSE toroidal grid —
// the fine grid physically ends at ±161 m. Wide merged tufts carry the
// meadow silhouette 150→265 m; beyond that the terrain splat owns it.
export const FAR_GRID = 768;
export const FAR_CELL = 0.7; // ±269 m ring, ~2 slots/m²
export const FAR_R0 = 150;
export const FAR_R = 265;
export const FAR_CAP = 196608;

/** world cells span ±~10k — bias before the 16-bit pack */
export const CELL_BIAS = 20000;
