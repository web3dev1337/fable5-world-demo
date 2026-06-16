import type { NF } from '../../gpu/TSLTypes';

/** far-plane sky test — tolerates either depth convention (0 or 1 at far) */
export function isSkyDepth(d: NF) {
  return d.lessThanEqual(1e-7).or(d.greaterThanEqual(0.9999999));
}
