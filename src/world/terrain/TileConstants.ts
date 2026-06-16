/**
 * Shared tuning constants for the CDLOD terrain tile system. Split out so the
 * tile-material / far-shell / debug-view builders and the quadtree in
 * TerrainTiles share one source of truth (no value duplication / drift).
 */

import { WORLD_SIZE } from '../WorldConst';

export const MAX_TILES = 2048;
export const PATCH_SEGS = 64;
/** split while camDist < size·SPLIT_K */
export const SPLIT_K = 2.1;
export const MIN_TILE = 64;
/** rough/steep tiles may refine below MIN_TILE (cliff close-ups) */
export const MIN_TILE_ROUGH = 32;
/** range-pyramid base resolution: one cell = MIN_TILE meters across the world */
export const RANGE_BASE = WORLD_SIZE / MIN_TILE;
