/**
 * Crown shadow proxy — the bulk-occlusion caster geometry for trees.
 *
 * Real closed canopy transmits 2–5% at noon; hollow card-shell crowns leak
 * 40%+ through their alpha gradients and PCSS averages the speckle into a flat
 * half-lit wash (no dapple, no dark interior). The shadow proxy core (dithered
 * to a per-class density) restores bulk occlusion; cards keep the edges ragged
 * in the near ring.
 */

import { BufferAttribute, BufferGeometry, IcosahedronGeometry } from 'three';

/**
 * Crown shadow density per tree class (spruce/pine/beech/birch/karst/snag).
 * Snag crowns are bare — no core.
 */
export const CROWN_SHADOW_DENSITY = [0.9, 0.84, 0.92, 0.74, 0.85, 0] as const;

/** crown proxy dims, FITTED to a pool's actual ring geometry (meters, scale 1) */
export interface CrownDims {
  cy: number;
  ry: number;
  rxz: number;
}

/**
 * Shadow-proxy tree: 80-tri ellipsoid crown + 12-tri trunk prism, fitted to
 * the pool's own geometry bounds (class-max dims made small variants throw
 * giant blob shadows — user-reported). This is the ONLY tree caster beyond
 * R1 (a cascade texel out there is ≥0.5 m — card raggedness is invisible)
 * and the bulk-density core inside R1's card edges.
 */
export function crownProxyGeometry(d: CrownDims): BufferGeometry {
  // PolyhedronGeometry is non-indexed: 80 faces × 3 verts at detail 1
  const core = new IcosahedronGeometry(1, 1);
  const cpos = core.attributes.position as BufferAttribute;
  const cy = d.cy;
  const nCore = cpos.count;
  const tr = 0.035 * d.rxz + 0.03;
  const merged = new Float32Array(nCore * 3 + 6 * 3);
  for (let i = 0; i < nCore; i++) {
    merged[i * 3] = cpos.getX(i) * d.rxz;
    merged[i * 3 + 1] = cpos.getY(i) * d.ry + cy;
    merged[i * 3 + 2] = cpos.getZ(i) * d.rxz;
  }
  // trunk prism: 3 quads, base→crown center
  const idx: number[] = [];
  for (let i = 0; i < nCore; i++) idx.push(i);
  for (let k = 0; k < 3; k++) {
    const a = (k / 3) * Math.PI * 2;
    const o = (nCore + k * 2) * 3;
    merged[o] = Math.cos(a) * tr;
    merged[o + 1] = 0;
    merged[o + 2] = Math.sin(a) * tr;
    merged[o + 3] = Math.cos(a) * tr * 0.6;
    merged[o + 4] = cy;
    merged[o + 5] = Math.sin(a) * tr * 0.6;
  }
  for (let k = 0; k < 3; k++) {
    const n = (k + 1) % 3;
    idx.push(
      nCore + k * 2, nCore + n * 2, nCore + k * 2 + 1,
      nCore + n * 2, nCore + n * 2 + 1, nCore + k * 2 + 1,
    );
  }
  const nrm = new Float32Array(merged.length);
  for (let i = 0; i < nrm.length; i += 3) nrm[i + 1] = 1;
  const g = new BufferGeometry();
  g.setAttribute('position', new BufferAttribute(merged, 3));
  g.setAttribute('normal', new BufferAttribute(nrm, 3));
  g.setIndex(idx);
  return g;
}
