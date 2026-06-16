/**
 * The four clustered-Poisson scatter layers (trees, understory, ground extras,
 * stones+branches). Each builds a jittered child grid (one thread per candidate
 * cell), gates it with per-class density × the parent clump field, picks a
 * class, and atomically appends accepted instances.
 *
 * Kept as four named functions rather than one parametrized kernel: the bodies
 * share the preamble (`gridSite`), append, and the weighted-CDF picker, but
 * diverge materially past that — different exclusion rules (stones skip the
 * river test and use a higher standing-water cutoff), the extras log-on-slope
 * rejection, the stones uphill talus march + threshold size-pick, and per-layer
 * scale/sink/lean/variant logic. Collapsing them would mean `if (layer===…)`
 * special-casing throughout, so they stay explicit.
 */

import type { Renderer, StorageBufferNode } from 'three/webgpu';
import {
  Fn,
  If,
  Return,
  float,
  instancedArray,
  int,
  smoothstep,
  vec4,
} from 'three/tsl';
import type { Heightfield } from '../../../world/Heightfield';
import { LAKE_LEVEL, TREELINE, WORLD_SIZE } from '../../../world/WorldConst';
import type { NV4 } from '../../TSLTypes';
import { VegClass, TREE_VARIANTS } from './classes';
import {
  append,
  byBiome,
  cellHash,
  cellHash2,
  clumpField,
  pickWeighted,
  type AtomicCounter,
} from './helpers';
import { gridSite } from './site';

// child-grid cell sizes (m) — jitter spans the full cell, so no grid reads
const TREE_CELL = 3.4;
const UNDER_CELL = 2.4;
const EXTRA_CELL = 5.5;
const STONE_CELL = 2.1;
const TREE_CAP = 600_000;
const UNDER_CAP = 700_000;
const EXTRA_CAP = 180_000;
const STONE_CAP = 1_500_000;

const TAU = 6.2831853;

/** the raw outputs of a layer dispatch — counts are read back later, in parallel */
export interface LayerBuffers {
  bufA: StorageBufferNode<'vec4'>;
  bufB: StorageBufferNode<'vec4'>;
  counter: AtomicCounter;
  cap: number;
}

// ---------------------------------------------------------------- trees --
export async function scatterTrees(
  renderer: Renderer,
  hf: Heightfield,
  sT: number,
): Promise<LayerBuffers> {
  const treeG = Math.round(WORLD_SIZE / TREE_CELL);
  const treeA = instancedArray(TREE_CAP, 'vec4');
  const treeB = instancedArray(TREE_CAP, 'vec4');
  const treeCount = instancedArray(1, 'uint').toAtomic();

  const treeK = Fn(() => {
    const { cell, wpos, s } = gridSite(hf, treeG, sT);

    // hard exclusions: open/standing water, river channels, lake shelf
    If(s.h.lessThan(LAKE_LEVEL + 0.4), () => {
      Return();
    });
    If(s.riverDepth.greaterThan(0.22).or(s.standing.greaterThan(0.3)), () => {
      Return();
    });

    const clump = clumpField(wpos, sT ^ 0x51f3);
    const dens = byBiome(s.bioId, [0, 0.22, 0.8, 0.85, 0.06, 0.26]);
    const clumpFloor = byBiome(s.bioId, [0, 0.15, 0.3, 0.35, 0.04, 0.12]);
    const slopeFade = float(1).sub(smoothstep(0.5, 0.95, s.slope));
    const treelineFade = float(1).sub(
      smoothstep(TREELINE - 110, TREELINE + 50, s.h),
    );
    const snowFade = float(1).sub(s.snow.mul(0.85));
    const accept = dens
      .mul(clumpFloor.add(float(1).sub(clumpFloor).mul(clump)))
      .mul(slopeFade)
      .mul(treelineFade)
      .mul(snowFade)
      .mul(s.vegDens.mul(0.85).add(0.15))
      .mul(float(1).sub(s.rockExp.mul(0.65)));
    If(cellHash(cell, sT ^ 0x1234f).greaterThanEqual(accept), () => {
      Return();
    });

    // species weights: per-biome table × moisture response
    const m = s.moisture;
    const w0 = byBiome(s.bioId, [0, 0.6, 0.58, 0.07, 0.05, 0.12]) // spruce
      .mul(m.mul(0.5).add(0.75));
    const w1 = byBiome(s.bioId, [0, 0.22, 0.27, 0.02, 0.15, 0]) // pine
      .mul(float(1.45).sub(m.mul(0.9)));
    const w2 = byBiome(s.bioId, [0, 0, 0.02, 0.5, 0.42, 0.05]) // beech
      .mul(m.mul(0.9).add(0.55));
    const w3 = byBiome(s.bioId, [0, 0.03, 0.08, 0.16, 0.3, 0.55]) // birch
      .mul(m.mul(0.6).add(0.7));
    const w4 = byBiome(s.bioId, [0, 0, 0, 0.2, 0, 0]) // karst gnarl
      .mul(s.rockExp.mul(1.6).add(0.4));
    const w5 = byBiome(s.bioId, [0, 0.15, 0.05, 0.05, 0.08, 0.28]); // snag

    const r = cellHash(cell, sT ^ 0x77e1).mul(
      w0.add(w1).add(w2).add(w3).add(w4).add(w5),
    );
    const sp = pickWeighted(
      r,
      [w0, w1, w2, w3, w4, w5],
      [
        VegClass.Spruce,
        VegClass.Pine,
        VegClass.Beech,
        VegClass.Birch,
        VegClass.KarstGnarl,
        VegClass.Snag,
      ],
    );

    // size: power-biased jitter; krummholz shrink toward the treeline;
    // subalpine biome additionally stunted
    const h2 = cellHash2(cell, sT ^ 0x3b8d);
    const krumm = smoothstep(TREELINE - 170, TREELINE + 10, s.h);
    const stunt = s.bioId.equal(int(1)).select(float(0.72), float(1));
    const scale = h2.x
      .pow(1.6)
      .mul(0.85)
      .add(0.62)
      .mul(float(1).sub(krumm.mul(0.55)))
      .mul(stunt);

    const yaw = h2.y.mul(TAU);
    const leanR = cellHash2(cell, sT ^ 0x6c2f).sub(0.5).mul(0.12);
    const lean = s.nrmXZ.mul(0.18).add(leanR);
    const variant = cellHash(cell, sT ^ 0x49a1)
      .mul(TREE_VARIANTS)
      .floor()
      .min(TREE_VARIANTS - 1);
    const idF = float(sp).mul(8).add(variant);
    const y = s.h.sub(scale.mul(0.12)); // sink — root flare covers the seam

    append(
      treeCount,
      TREE_CAP,
      treeA,
      treeB,
      vec4(wpos.x, y, wpos.y, scale) as unknown as NV4,
      vec4(yaw, lean.x, lean.y, idF) as unknown as NV4,
    );
  })().compute(treeG * treeG);
  treeK.setName('scatterTrees');
  await renderer.computeAsync(treeK);

  return { bufA: treeA, bufB: treeB, counter: treeCount, cap: TREE_CAP };
}

// ----------------------------------------------------------- understory --
export async function scatterUnderstory(
  renderer: Renderer,
  hf: Heightfield,
  sU: number,
  sT: number,
): Promise<LayerBuffers> {
  const underG = Math.round(WORLD_SIZE / UNDER_CELL);
  const underA = instancedArray(UNDER_CAP, 'vec4');
  const underB = instancedArray(UNDER_CAP, 'vec4');
  const underCount = instancedArray(1, 'uint').toAtomic();

  const underK = Fn(() => {
    const { cell, wpos, s } = gridSite(hf, underG, sU);

    If(s.h.lessThan(LAKE_LEVEL + 0.35), () => {
      Return();
    });
    If(s.riverDepth.greaterThan(0.2).or(s.standing.greaterThan(0.3)), () => {
      Return();
    });

    // canopy proxy = the TREE clump field (same salt → same parents)
    const canopy = clumpField(wpos, sT ^ 0x51f3);
    const dens = byBiome(s.bioId, [0, 0.25, 0.55, 0.6, 0.55, 0.45]);
    const slopeFade = float(1).sub(smoothstep(0.55, 0.9, s.slope));
    const treelineFade = float(1).sub(
      smoothstep(TREELINE - 40, TREELINE + 140, s.h),
    );
    const accept = dens
      .mul(slopeFade)
      .mul(treelineFade)
      .mul(float(1).sub(s.snow.mul(0.9)))
      .mul(s.vegDens.mul(0.9).add(0.1))
      .mul(float(1).sub(s.rockExp.mul(0.85)));
    If(cellHash(cell, sU ^ 0x2477).greaterThanEqual(accept), () => {
      Return();
    });

    const m = s.moisture;
    const edge = canopy.mul(float(1).sub(canopy)).mul(4); // 1 at clump rims
    const w0 = byBiome(s.bioId, [0, 0.05, 0.15, 0.3, 0.04, 0.1]); // hazel
    const w1 = byBiome(s.bioId, [0, 0, 0.02, 0.12, 0.1, 0.02]) // pink shrub
      .mul(edge.mul(1.3).add(0.2));
    const w2 = byBiome(s.bioId, [0, 0.55, 0.3, 0.02, 0.03, 0]) // juniper
      .mul(float(1.3).sub(m.mul(0.8)));
    const w3 = byBiome(s.bioId, [0, 0.1, 0.4, 0.38, 0.03, 0.5]) // fern
      .mul(m.mul(1.1).add(0.3))
      .mul(canopy.mul(1.1).add(0.35));
    const gapK = float(1.25).sub(canopy.mul(0.9));
    const w4 = byBiome(s.bioId, [0, 0.1, 0.05, 0.06, 0.3, 0.2]).mul(gapK); // umbel
    const w5 = byBiome(s.bioId, [0, 0.08, 0.04, 0.06, 0.22, 0.1]).mul(gapK); // bell
    const w6 = byBiome(s.bioId, [0, 0.12, 0.04, 0.06, 0.28, 0.08]).mul(gapK); // daisy

    const r = cellHash(cell, sU ^ 0x59d3).mul(
      w0.add(w1).add(w2).add(w3).add(w4).add(w5).add(w6),
    );
    const cls = pickWeighted(
      r,
      [w0, w1, w2, w3, w4, w5, w6],
      [
        VegClass.BushHazel,
        VegClass.BushPink,
        VegClass.Juniper,
        VegClass.Fern,
        VegClass.FlowerUmbel,
        VegClass.FlowerBell,
        VegClass.FlowerDaisy,
      ],
    );

    const h2 = cellHash2(cell, sU ^ 0x71c9);
    const scale = h2.x.pow(1.4).mul(0.7).add(0.6);
    const yaw = h2.y.mul(TAU);
    const variant = cellHash(cell, sU ^ 0x1ee7).mul(4).floor().min(3);
    const idF = float(cls).mul(8).add(variant);

    append(
      underCount,
      UNDER_CAP,
      underA,
      underB,
      vec4(wpos.x, s.h.sub(0.03), wpos.y, scale) as unknown as NV4,
      vec4(yaw, 0, 0, idF) as unknown as NV4,
    );
  })().compute(underG * underG);
  underK.setName('scatterUnderstory');
  await renderer.computeAsync(underK);

  return { bufA: underA, bufB: underB, counter: underCount, cap: UNDER_CAP };
}

// --------------------------------------------------------------- extras --
export async function scatterExtras(
  renderer: Renderer,
  hf: Heightfield,
  sE: number,
  sT: number,
): Promise<LayerBuffers> {
  const extraG = Math.round(WORLD_SIZE / EXTRA_CELL);
  const extraA = instancedArray(EXTRA_CAP, 'vec4');
  const extraB = instancedArray(EXTRA_CAP, 'vec4');
  const extraCount = instancedArray(1, 'uint').toAtomic();

  const extraK = Fn(() => {
    const { cell, wpos, s } = gridSite(hf, extraG, sE);

    If(s.h.lessThan(LAKE_LEVEL + 0.3), () => {
      Return();
    });
    If(s.riverDepth.greaterThan(0.3).or(s.standing.greaterThan(0.35)), () => {
      Return();
    });

    const canopy = clumpField(wpos, sT ^ 0x51f3);
    const forestK = byBiome(s.bioId, [0, 0.3, 1, 1, 0.25, 0.6]).mul(
      canopy.mul(0.7).add(0.3),
    );
    const m = s.moisture;
    const w0 = forestK.mul(0.3).mul(m.mul(0.6).add(0.4)); // log
    const w1 = forestK.mul(0.12); // stump
    const w2 = s.rockExp.mul(1.1).add(0.12).mul(0.42); // boulder
    const w3 = s.rockExp.mul(0.9).mul(0.2); // slab

    const dens = byBiome(s.bioId, [0.08, 0.25, 0.62, 0.65, 0.22, 0.5]);
    const slopeFade = float(1).sub(smoothstep(0.55, 1.1, s.slope));
    const wSum = w0.add(w1).add(w2).add(w3);
    const accept = dens.mul(slopeFade).mul(wSum.min(1));
    If(cellHash(cell, sE ^ 0x3f21).greaterThanEqual(accept), () => {
      Return();
    });

    const r = cellHash(cell, sE ^ 0x6d05).mul(wSum);
    const cls = pickWeighted(
      r,
      [w0, w1, w2, w3],
      [VegClass.Log, VegClass.Stump, VegClass.Boulder, VegClass.Slab],
    );

    // logs slide off steep ground; decay class follows moisture
    If(cls.equal(int(VegClass.Log)).and(s.slope.greaterThan(0.5)), () => {
      Return();
    });
    const h2 = cellHash2(cell, sE ^ 0x15bd);
    const mJit = m.add(h2.x.mul(0.3).sub(0.15));
    const decay = mJit
      .greaterThan(0.62)
      .select(float(2), mJit.greaterThan(0.35).select(float(1), float(0)));
    const isRock = cls.greaterThanEqual(int(VegClass.Boulder));
    // boulder/slab variants are context-keyed like StoneL: 0/1 pale bedrock
    // blocks on exposed rock, scree slopes, or dry pale soil (everywhere
    // the splat is pale — they must match the ground), 2/3 dark mossy
    // forest rocks
    const paleCtx = s.rockExp
      .greaterThan(0.35)
      .or(s.slope.greaterThan(0.42))
      .or(s.moisture.lessThan(0.32));
    const rockV = cellHash(cell, sE ^ 0x44d7)
      .mul(2)
      .floor()
      .min(1)
      .add(paleCtx.select(float(0), float(2)));
    const variant = cls
      .equal(int(VegClass.Log))
      .select(
        decay,
        isRock.select(rockV, cellHash(cell, sE ^ 0x44d7).mul(4).floor().min(3)),
      );

    const scale = isRock.select(
      h2.y.pow(2).mul(1.9).add(0.5),
      h2.y.mul(0.6).add(0.7),
    );
    // rocks bed deeper on slopes — a perched block on an incline floats
    const bed = s.slope.mul(0.9).add(1);
    const sink = isRock.select(scale.mul(0.28).mul(bed), float(0.08));
    const yaw = cellHash(cell, sE ^ 0x2a6b).mul(TAU);
    const idF = float(cls).mul(8).add(variant);

    append(
      extraCount,
      EXTRA_CAP,
      extraA,
      extraB,
      vec4(wpos.x, s.h.sub(sink), wpos.y, scale) as unknown as NV4,
      vec4(yaw, s.nrmXZ.x.mul(0.3), s.nrmXZ.y.mul(0.3), idF) as unknown as NV4,
    );
  })().compute(extraG * extraG);
  extraK.setName('scatterExtras');
  await renderer.computeAsync(extraK);

  return { bufA: extraA, bufB: extraB, counter: extraCount, cap: EXTRA_CAP };
}

// ------------------------------------------------- stones + branches --
// size-stratified ground solids: stones everywhere geology says so
// (scree slopes, rock exposure, streambeds, talus under cliffs) plus a
// light scatter on all soil; fallen branches on forest floors. This is
// the "no bare ground" layer — references show ground GEOMETRY at every
// distance, never naked splat.
export async function scatterStones(
  renderer: Renderer,
  hf: Heightfield,
  sS: number,
  sT: number,
): Promise<LayerBuffers> {
  const stoneG = Math.round(WORLD_SIZE / STONE_CELL);
  const stoneA = instancedArray(STONE_CAP, 'vec4');
  const stoneB = instancedArray(STONE_CAP, 'vec4');
  const stoneCount = instancedArray(1, 'uint').toAtomic();

  const stoneK = Fn(() => {
    const { cell, wpos, s } = gridSite(hf, stoneG, sS);
    If(s.h.lessThan(LAKE_LEVEL + 0.25), () => {
      Return();
    });
    If(s.standing.greaterThan(0.5), () => {
      Return();
    });

    const canopy = clumpField(wpos, sT ^ 0x51f3);
    const streamK = smoothstep(0.05, 0.3, s.riverDepth);
    // angle of repose: loose rock can't rest above ~42° — anything clinging
    // to steeper faces reads as stuck-on blobs (user feedback: "random
    // protruding circles along cliffs")
    const repose = float(1).sub(smoothstep(0.72, 0.98, s.slope));
    // talus: march uphill — steep ground above sheds rock onto this site,
    // so stones concentrate in fans BELOW cliffs rather than on them
    const upLen = s.nrmXZ.length().max(0.02);
    const up = s.nrmXZ.div(upLen).negate();
    const h8 = hf.sampleHeight(wpos.add(up.mul(8)));
    const h18 = hf.sampleHeight(wpos.add(up.mul(18)));
    const riseNear = h8.sub(s.h).div(8);
    const riseFar = h18.sub(h8).div(10);
    const cliffAbove = smoothstep(0.7, 1.3, riseNear.max(riseFar));
    // shared rockiness clumps: one field gates ALL size classes, so big
    // blocks sit inside aprons of smaller fragments with bare gaps between
    // (real scree is patchy and size-mixed, never uniform speckle)
    const patch = clumpField(wpos, sS ^ 0x77aa).mul(0.78).add(0.22);
    const scree = smoothstep(0.42, 0.8, s.slope);
    const stoneBase = byBiome(s.bioId, [0.55, 0.4, 0.26, 0.32, 0.14, 0.18])
      .mul(
        s.rockExp
          .mul(0.85)
          .add(scree.mul(0.85))
          .add(streamK.mul(1.5))
          .add(cliffAbove.mul(1.15))
          .add(0.16),
      )
      .mul(patch)
      .mul(repose)
      .mul(float(1).sub(s.snow.mul(0.85)));
    // branches need ground that holds them — steep bare slopes grew
    // floating white sticks (user-visible artifact)
    const branchFlat = float(1).sub(smoothstep(0.45, 0.75, s.slope));
    const branchW = canopy.mul(0.6).mul(
      byBiome(s.bioId, [0, 0.2, 1, 1, 0.3, 0.7]),
    ).mul(branchFlat);
    const accept = stoneBase.add(branchW).min(1);
    If(cellHash(cell, sS ^ 0x71f1).greaterThanEqual(accept), () => {
      Return();
    });

    // class pick: branch vs stone, stones split L/M/S by size budget.
    // Stones embed deeper on slopes (a perched sphere on an incline reads
    // as a stuck-on blob; a bedded one reads as an outcrop).
    const bed = s.slope.mul(0.9).add(1);
    const r = cellHash(cell, sS ^ 0x2e2e).mul(stoneBase.add(branchW));
    const h2 = cellHash2(cell, sS ^ 0x6b6b);
    const cls = int(VegClass.Branch).toVar();
    const scale = float(1).toVar();
    const sink = float(0.05).toVar();
    const variant = cellHash(cell, sS ^ 0x5c5c).mul(4).floor().min(3).toVar();
    If(r.lessThan(stoneBase), () => {
      // streambeds skew LARGE: scene1 beds are built from rounded boulders
      const sr = h2.x.sub(streamK.mul(0.16));
      If(sr.lessThan(0.13), () => {
        cls.assign(int(VegClass.StoneL));
        scale.assign(h2.y.pow(1.7).mul(1.6).add(0.6)); // 0.6–2.2 m
        sink.assign(scale.mul(0.3).mul(bed));
        // variant by context: 0/1 pale faceted talus on scree/exposed rock/
        // dry pale soil (matches the pale splat), 2/3 dark rounded stones
        // in streambeds and on moist mossy forest floor
        const paleCtx = s.rockExp
          .greaterThan(0.35)
          .or(s.slope.greaterThan(0.42))
          .or(s.moisture.lessThan(0.32))
          .and(streamK.lessThan(0.35));
        const vr = cellHash(cell, sS ^ 0x1d2d).mul(2).floor().min(1);
        variant.assign(vr.add(paleCtx.select(float(0), float(2))));
      }).Else(() => {
        If(sr.lessThan(0.45), () => {
          cls.assign(int(VegClass.StoneM));
          scale.assign(h2.y.mul(0.4).add(0.2)); // 0.2–0.6 m
          sink.assign(scale.mul(0.26).mul(bed));
        }).Else(() => {
          cls.assign(int(VegClass.StoneS));
          scale.assign(h2.y.mul(0.14).add(0.06)); // 6–20 cm
          sink.assign(scale.mul(0.22).mul(bed));
        });
      });
    }).Else(() => {
      scale.assign(h2.y.mul(0.8).add(0.6));
      sink.assign(0.04);
    });

    const yaw = cellHash(cell, sS ^ 0x3d3d).mul(TAU);
    const idF = float(cls).mul(8).add(variant);
    append(
      stoneCount,
      STONE_CAP,
      stoneA,
      stoneB,
      vec4(wpos.x, s.h.sub(sink), wpos.y, scale) as unknown as NV4,
      vec4(yaw, s.nrmXZ.x.mul(0.4), s.nrmXZ.y.mul(0.4), idF) as unknown as NV4,
    );
  })().compute(stoneG * stoneG);
  stoneK.setName('scatterStones');
  await renderer.computeAsync(stoneK);

  return { bufA: stoneA, bufB: stoneB, counter: stoneCount, cap: STONE_CAP };
}
