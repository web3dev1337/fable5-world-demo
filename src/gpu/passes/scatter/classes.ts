/**
 * Vegetation/rock class ids — the `idF` channel packs `class·8 + variant`
 * (exact in f32). Shared by the scatter kernels (placement) and the draw-side
 * vegetation library (geometry-pool lookup).
 */

/** geometry-pool class ids (variant index lives in the low 3 bits of idF) */
export const enum VegClass {
  // trees — order matches TREE_SPECIES
  Spruce = 0,
  Pine = 1,
  Beech = 2,
  Birch = 3,
  KarstGnarl = 4,
  Snag = 5,
  // understory
  BushHazel = 8,
  BushPink = 9,
  Juniper = 10,
  Fern = 11,
  FlowerUmbel = 12,
  FlowerBell = 13,
  FlowerDaisy = 14,
  // ground extras
  Log = 16,
  Stump = 17,
  Boulder = 18,
  Slab = 19,
  // size-stratified ground solids (the "no bare ground" layer): each class
  // draws to the range where it still covers >~2 px — constant screen-space
  // granularity, the aggregate equivalent of nanite cluster selection
  StoneL = 20, // 0.6–2.2 m → 900 m
  StoneM = 21, // 0.2–0.6 m → 280 m
  StoneS = 22, // 6–20 cm → 90 m
  Branch = 23, // fallen branches on forest floors → 230 m
}

/** structural variants baked per tree species (geometry reuse, D5) */
export const TREE_VARIANTS = 4;
