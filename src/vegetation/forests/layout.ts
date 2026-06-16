/**
 * Forests compact-region layout — the single source of truth for the
 * draw-group index math.
 *
 * Every scattered instance is appended into one per-(pool,ring[,cascade])
 * compact region; a region's index is computed by `groupOf` (main view) and
 * `casterGroupOf` (per-cascade shadow casters). The SAME arithmetic is encoded
 * a second time inside the GPU cull kernel (cullKernels.ts) — the two must
 * produce byte-identical indices or draws silently render the wrong pool. To
 * keep them in lock-step, the additive group-base offsets live here as named
 * constants and BOTH the TS functions below and the TSL kernel reference them.
 *
 * The LOD-ring distance/band constants (R0_FAR…EX_BAND) are likewise the
 * single source feeding both the cull kernel (ring classification) and the CPU
 * `fadeFor` (dithered crossfade boundaries).
 */

// ---- LOD ring distances (m) + dither bands (user feedback: transitions read
// too close — full-card trees hold to 150 m, impostors start at 460 m).
// Hero ring 0 (≤26 m): full bark + cards + REAL mesh leaves — the nanite-
// equivalence near field (spec floor: hero tree ≥100k tris).
export const R0_FAR = 26;
export const BAND0 = 5;
export const R1_FAR = 150;
export const BAND1 = 14;
export const R2_FAR = 460;
export const BAND2 = 36;
export const EX_R1_FAR = 120;
export const EX_BAND = 15;

// ---- group block sizing ----------------------------------------------------
export const MAIN_GROUPS = 170;
/**
 * Per-cascade caster groups: trees r1/r2 (48) + hero r0 (24) + extras/stones
 * (64) + impostor-band crown proxies per species (6). The impostor band casts
 * so tree shadows don't end in a hard circle at the R2 boundary — they fade
 * out by IMP_CAST_FAR instead.
 */
export const CASC_LOCALS = 142;
export const CASCADES = 4;
export const GROUPS = MAIN_GROUPS + CASCADES * CASC_LOCALS;

/** crown-proxy shadows fade out across this band (m from camera) */
export const IMP_CAST_FADE0 = 620;
export const IMP_CAST_FAR = 1100;

// ---- main-view group block base offsets (0 .. MAIN_GROUPS-1) ----------------
/** tree rings r1/r2: pool*2 + (ring-1) — base 0 is implicit (no literal) */
export const TREE_R12_BASE = 0;
/** tree impostor billboard, one per species: + cls */
export const IMPOSTOR_BASE = 48;
/** understory: (cls-8)*4 + variant */
export const UNDER_BASE = 54;
/** extras/stones: pe*2 + (ring-1) */
export const EXTRAS_BASE = 82;
/** hero r0 per pool: + pool */
export const TREE_MAIN_BASE = 146;
/** boundary group: extras (cls 16–19) below, size-stratified stones above */
export const EXTRAS_STONE_SPLIT = 114;

// ---- per-cascade caster block locals (0 .. CASC_LOCALS-1) -------------------
/** caster tree r1/r2: pool*2 + (ring-1) — base 0 is implicit (no literal) */
export const CAST_TREE_R12_LOCAL = 0;
/** caster hero r0: + pool */
export const CAST_HERO_LOCAL = 48;
/** caster extras/stones: pe*2 + (ring-1) */
export const CAST_EXTRAS_LOCAL = 72;
/** caster impostor-band crown proxies: + cls */
export const CAST_IMP_LOCAL = 136;

// ---- compact-region capacities ---------------------------------------------
// main view
export const CAP_HERO = 48;
export const CAP_TREE_R1 = 6144;
export const CAP_TREE_R2 = 8192;
export const CAP_IMPOSTOR = 49152;
export const CAP_UNDER = 4096;
export const CAP_EX_R1 = 1024;
export const CAP_EX_R2 = 2048;
// main-view size-stratified stones/branches (cls 20–23)
export const CAP_STONE_L_R1 = 4096; // StoneL → 900 m
export const CAP_STONE_L_R2 = 24576;
export const CAP_STONE_M_R1 = 8192; // StoneM → 280 m
export const CAP_STONE_M_R2 = 16384;
export const CAP_STONE_S_R1 = 24576; // StoneS — single ring
export const CAP_STONE_S_R2 = 64;
export const CAP_BRANCH = 8192;
// per-cascade caster regions (a cascade box covers a slice of the frustum, so
// the worst case is well under the main-view caps)
export const CAP_CAST_IMP = 8192; // impostor-band crown proxies (per cls)
export const CAP_CAST_TREE_R1 = 3072;
export const CAP_CAST_TREE_R2 = 6144;
export const CAP_CAST_EX_R1 = 512;
export const CAP_CAST_EX_R2 = 1024;
export const CAP_CAST_STONE_L_R1 = 2048;
export const CAP_CAST_STONE_L_R2 = 12288;
export const CAP_CAST_STONE_M_R1 = 4096;
export const CAP_CAST_STONE_M_R2 = 8192;
export const CAP_CAST_STONE_S_R1 = 12288;
export const CAP_CAST_STONE_S_R2 = 64;
export const CAP_CAST_BRANCH = 4096;

export function groupOf(cls: number, variant: number, ring: 0 | 1 | 2 | 3): number {
  if (cls < 6) {
    if (ring === 0) return TREE_MAIN_BASE + cls * 4 + variant;
    if (ring === 3) return IMPOSTOR_BASE + cls;
    return (cls * 4 + variant) * 2 + (ring - 1);
  }
  if (cls < 15) return UNDER_BASE + (cls - 8) * 4 + variant;
  const pe = (cls - 16) * 4 + variant;
  return EXTRAS_BASE + pe * 2 + (ring - 1);
}

/**
 * Caster-group index for cascade c. Local layout:
 *   0..47   tree pools × rings r1/r2  (pool*2 + ring-1)
 *   48..71  hero r0 per pool
 *   72..135 extras/stones pe × rings  (72 + pe*2 + ring-1)
 */
export function casterGroupOf(
  c: number,
  cls: number,
  variant: number,
  ring: 0 | 1 | 2 | 3,
): number {
  const base = MAIN_GROUPS + c * CASC_LOCALS;
  if (cls < 6) {
    if (ring === 3) return base + CAST_IMP_LOCAL + cls;
    const pool = cls * 4 + variant;
    if (ring === 0) return base + CAST_HERO_LOCAL + pool;
    return base + pool * 2 + (ring - 1);
  }
  const pe = (cls - 16) * 4 + variant;
  return base + CAST_EXTRAS_LOCAL + pe * 2 + (ring - 1);
}

export function capOf(g: number): number {
  if (g >= MAIN_GROUPS) {
    // caster regions: a cascade box covers a slice of the frustum, so the
    // worst case is well under the main-view caps
    const local = (g - MAIN_GROUPS) % CASC_LOCALS;
    if (local >= CAST_IMP_LOCAL) return CAP_CAST_IMP; // impostor-band crown proxies (per cls)
    if (local < CAST_HERO_LOCAL) return local % 2 === 0 ? CAP_CAST_TREE_R1 : CAP_CAST_TREE_R2; // tree r1/r2
    if (local < CAST_EXTRAS_LOCAL) return CAP_HERO;
    const pe = (local - CAST_EXTRAS_LOCAL) >> 1;
    const cls = 16 + (pe >> 2);
    const isR1 = (local - CAST_EXTRAS_LOCAL) % 2 === 0;
    if (cls < 20) return isR1 ? CAP_CAST_EX_R1 : CAP_CAST_EX_R2; // extras
    if (cls === 20) return isR1 ? CAP_CAST_STONE_L_R1 : CAP_CAST_STONE_L_R2; // StoneL → 900 m
    if (cls === 21) return isR1 ? CAP_CAST_STONE_M_R1 : CAP_CAST_STONE_M_R2; // StoneM
    if (cls === 22) return isR1 ? CAP_CAST_STONE_S_R1 : CAP_CAST_STONE_S_R2; // StoneS — single ring
    return CAP_CAST_BRANCH; // Branch
  }
  if (g < IMPOSTOR_BASE) return g % 2 === 0 ? CAP_TREE_R1 : CAP_TREE_R2;
  if (g < UNDER_BASE) return CAP_IMPOSTOR;
  if (g < EXTRAS_BASE) return CAP_UNDER;
  if (g >= TREE_MAIN_BASE) return CAP_HERO;
  if (g < EXTRAS_STONE_SPLIT) return (g - EXTRAS_BASE) % 2 === 0 ? CAP_EX_R1 : CAP_EX_R2;
  // size-stratified stones/branches (cls 20–23)
  const cls = 16 + ((g - EXTRAS_BASE) >> 3);
  const isR1 = (g - EXTRAS_BASE) % 2 === 0;
  if (cls === 20) return isR1 ? CAP_STONE_L_R1 : CAP_STONE_L_R2; // StoneL → 900 m
  if (cls === 21) return isR1 ? CAP_STONE_M_R1 : CAP_STONE_M_R2; // StoneM → 280 m
  if (cls === 22) return isR1 ? CAP_STONE_S_R1 : CAP_STONE_S_R2; // StoneS — single ring
  return CAP_BRANCH; // Branch
}
