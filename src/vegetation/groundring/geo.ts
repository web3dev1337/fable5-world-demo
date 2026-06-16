/**
 * GroundRing geometry builders — self-contained BufferGeometry factories for
 * the near-field carpets (flat litter quad, multi-blade grass clump, far tuft
 * cross). Pure CPU mesh construction; no GPU/TSL state.
 */

import { BufferAttribute, BufferGeometry } from 'three';
import { grassBladeGeometry } from '../GroundCover';

/** simple flat litter quad (5×7 cm), uv 0..1, normal up */
export function litterQuad(): BufferGeometry {
  const g = new BufferGeometry();
  const w = 0.038;
  const l = 0.05;
  g.setAttribute(
    'position',
    new BufferAttribute(new Float32Array([-w, 0, -l, w, 0, -l, w, 0, l, -w, 0, l]), 3),
  );
  g.setAttribute(
    'normal',
    new BufferAttribute(new Float32Array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]), 3),
  );
  g.setAttribute('uv', new BufferAttribute(new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]), 2));
  g.setIndex([0, 2, 1, 0, 3, 2]);
  return g;
}

/**
 * N-blade clump in one instance — the SOTA near-grass move: per-pixel blade
 * overlap is what reads as "lush", and single thin blades can't do it at
 * walking distance no matter the density. Deterministic mini-rng; per-cell
 * variety still comes from the instance transform/hash.
 */
export function bladeClump(blades: number, segs: number): BufferGeometry {
  let s = 1234567 + blades * 77 + segs * 13;
  const rnd = (): number => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
  const pos: number[] = [];
  const nrm: number[] = [];
  const uvA: number[] = [];
  const idx: number[] = [];
  for (let b = 0; b < blades; b++) {
    const base = grassBladeGeometry(segs);
    const yaw = rnd() * Math.PI * 2;
    const c = Math.cos(yaw);
    const sn = Math.sin(yaw);
    const ox = (rnd() - 0.5) * 0.16;
    const oz = (rnd() - 0.5) * 0.16;
    const hk = 0.62 + rnd() * 0.65;
    const lean = (rnd() - 0.5) * 0.42;
    const p = base.attributes.position as BufferAttribute;
    const nA = base.attributes.normal as BufferAttribute;
    const uA = base.attributes.uv as BufferAttribute;
    const v0 = pos.length / 3;
    for (let i = 0; i < p.count; i++) {
      const x = p.getX(i) * 1.25;
      const y = p.getY(i) * hk;
      const z = p.getZ(i);
      pos.push(x * c + z * sn + ox + lean * y * c, y, z * c - x * sn + oz + lean * y * sn);
      nrm.push(nA.getX(i) * c + nA.getZ(i) * sn, nA.getY(i), nA.getZ(i) * c - nA.getX(i) * sn);
      uvA.push(uA.getX(i), uA.getY(i));
    }
    const ix = base.index as BufferAttribute;
    for (let i = 0; i < ix.count; i++) idx.push(v0 + ix.getX(i));
  }
  const g = new BufferGeometry();
  g.setAttribute('position', new BufferAttribute(new Float32Array(pos), 3));
  g.setAttribute('normal', new BufferAttribute(new Float32Array(nrm), 3));
  g.setAttribute('uv', new BufferAttribute(new Float32Array(uvA), 2));
  g.setIndex(idx);
  return g;
}

/** three crossed wide blades — far-band tuft (≈ a small clump in one card) */
export function tuftGeometry(W = 0.04): BufferGeometry {
  const pos: number[] = [];
  const nrm: number[] = [];
  const uvA: number[] = [];
  const idx: number[] = [];
  for (let k = 0; k < 3; k++) {
    const a = k * 1.92 + 0.4;
    const c = Math.cos(a);
    const s = Math.sin(a);
    const base = pos.length / 3;
    for (const [u, v] of [
      [-W, 0],
      [W, 0],
      [W * 0.55, 1],
      [-W * 0.55, 1],
    ] as const) {
      pos.push(u * c, v, u * s);
      // rounded cross-section (see grassBladeGeometry): edges tilt ±38°
      // toward the width axis (c,0,s)
      const sgn = u < 0 ? -1 : 1;
      nrm.push(
        -s * 0.97 * 0.788 + sgn * 0.616 * c,
        0.25,
        c * 0.97 * 0.788 + sgn * 0.616 * s,
      );
      uvA.push(u < 0 ? 0 : 1, v);
    }
    idx.push(base, base + 2, base + 1, base, base + 3, base + 2);
  }
  const g = new BufferGeometry();
  g.setAttribute('position', new BufferAttribute(new Float32Array(pos), 3));
  g.setAttribute('normal', new BufferAttribute(new Float32Array(nrm), 3));
  g.setAttribute('uv', new BufferAttribute(new Float32Array(uvA), 2));
  g.setIndex(idx);
  return g;
}
