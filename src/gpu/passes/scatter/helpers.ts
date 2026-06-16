/**
 * Scatter primitives: deterministic integer hashing (pcg2d over cell+salt),
 * per-biome value tables, the parent clump field, the atomic append/readback
 * pair, and the weighted-CDF class picker shared by the layer kernels.
 */

import type { Renderer } from 'three/webgpu';
import type { StorageBufferNode } from 'three/webgpu';
import {
  If,
  atomicAdd,
  float,
  int,
  smoothstep,
  uint,
  vec2,
} from 'three/tsl';
import type { NF, NI, NU, NV2, NV4 } from '../../TSLTypes';

// parent clump field (shared by trees + understory — canopy correlation)
const PARENT_CELL = 26;
const PARENT_PROB = 0.62;

// ---------------------------------------------------------------------------
// integer hash: pcg2d over (cell + salt) — stable at any cell magnitude
// ---------------------------------------------------------------------------

function pcg2d(p: NV2, salt: number): NV2 {
  // PURE expression chain — no toVar/assign, so it works in material node
  // graphs too (assign needs a Fn() stack). +40000 keeps negative ring cell
  // coords positive before the uint cast (world cells span ±~10k).
  const M = uint(1664525);
  const C = uint(1013904223);
  const a0 = p.x.add(40000 + (salt & 0x3fff)).toUint();
  const b0 = p.y.add(40000 + ((salt >> 14) & 0x3fff)).toUint();
  const a1 = a0.mul(M).add(C);
  const b1 = b0.mul(M).add(C);
  const a2 = a1.add(b1.mul(M));
  const b2 = b1.add(a2.mul(M));
  const a3 = a2.bitXor(a2.shiftRight(uint(16)));
  const b3 = b2.bitXor(b2.shiftRight(uint(16)));
  const a4 = a3.add(b3.mul(M));
  const b4 = b3.add(a4.mul(M));
  const a5 = a4.bitXor(a4.shiftRight(uint(16)));
  const b5 = b4.bitXor(b4.shiftRight(uint(16)));
  const inv = 1 / 16777216;
  return vec2(
    float(a5.bitAnd(uint(0xffffff))).mul(inv),
    float(b5.bitAnd(uint(0xffffff))).mul(inv),
  );
}

export function cellHash2(cell: NV2, salt: number): NV2 {
  return pcg2d(cell, salt);
}

export function cellHash(cell: NV2, salt: number): NF {
  return pcg2d(cell, salt).x;
}

// ---------------------------------------------------------------------------

/** per-biome value tables → TSL select chain (biome ids 0..5) */
export function byBiome(bioId: NI, vals: readonly number[]): NF {
  let e: NF = float(vals[5] ?? 0);
  for (let b = 4; b >= 0; b--) {
    e = bioId.equal(int(b)).select(float(vals[b] ?? 0), e) as NF;
  }
  return e;
}

/**
 * Parent clump field: hashed parent points on a coarse grid; weight = max
 * kernel over the 3×3 neighborhood. ~1 at clump hearts, 0 in gaps.
 */
export function clumpField(wpos: NV2, salt: number): NF {
  const base = wpos.div(PARENT_CELL).floor();
  const w = float(0).toVar();
  for (let dy = -1; dy <= 1; dy++) {
    for (let dx = -1; dx <= 1; dx++) {
      const c = base.add(vec2(dx, dy)).add(8192); // parents span negatives
      const h2 = cellHash2(c, salt);
      const exists = cellHash(c, salt ^ 0x9e3779).lessThan(PARENT_PROB);
      const ppos = c.sub(8192).add(0.15).add(h2.mul(0.7)).mul(PARENT_CELL);
      const r = float(PARENT_CELL).mul(h2.x.mul(0.55).add(0.5));
      const d = wpos.sub(ppos).length();
      const k = float(1)
        .sub(smoothstep(r.mul(0.22), r, d))
        .mul(exists.select(float(1), float(0)));
      w.assign(w.max(k));
    }
  }
  return w;
}

export type AtomicCounter = ReturnType<StorageBufferNode<'uint'>['toAtomic']>;

/** append helper: idx = old counter value; write when under cap */
export function append(
  counter: AtomicCounter,
  cap: number,
  bufA: StorageBufferNode<'vec4'>,
  bufB: StorageBufferNode<'vec4'>,
  a: NV4,
  b: NV4,
): void {
  const idx = atomicAdd(counter.element(0), uint(1)) as unknown as NU;
  If(idx.lessThan(uint(cap)), () => {
    bufA.element(idx).assign(a);
    bufB.element(idx).assign(b);
  });
}

export async function readCount(
  renderer: Renderer,
  counter: AtomicCounter,
  cap: number,
): Promise<number> {
  const attr = (counter as unknown as { value: unknown }).value;
  const ab = await renderer.getArrayBufferAsync(
    attr as Parameters<Renderer['getArrayBufferAsync']>[0],
  );
  const n = new Uint32Array(ab)[0] ?? 0;
  return Math.min(n, cap);
}

/**
 * Weighted-CDF class pick: walks `weights` as an inclusive prefix sum and
 * returns the `classIds` entry for the bucket `r` lands in. Emits the exact
 * nested-If ladder the layer kernels used inline (accumulate, branch; the
 * innermost branch assigns without a trailing add). `r` must already be scaled
 * by Σweights by the caller.
 */
export function pickWeighted(
  r: NF,
  weights: readonly NF[],
  classIds: readonly number[],
): NI {
  const n = weights.length;
  const w0 = weights[0];
  const c0 = classIds[0];
  if (w0 === undefined || c0 === undefined) {
    throw new Error('pickWeighted: empty weight/class tables');
  }
  const out = int(c0).toVar();
  const acc = w0.toVar();
  const build = (k: number): void => {
    const wk = weights[k];
    const ck = classIds[k];
    if (wk === undefined || ck === undefined) return;
    out.assign(int(ck));
    if (k < n - 1) {
      acc.addAssign(wk);
      If(r.greaterThan(acc), () => build(k + 1));
    }
  };
  If(r.greaterThan(acc), () => build(1));
  return out as unknown as NI;
}
