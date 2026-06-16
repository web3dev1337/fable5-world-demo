/**
 * GroundRing cull kernels (spec §9). One compute thread per toroidal slot maps
 * to its nearest congruent world cell, samples biome/water/canopy fields,
 * thins density toward the ring edge, frustum-tests, and atomically appends
 * (packed cell, groundY) into the per-layer compact lists that feed the
 * indirect draws. clearK zeroes the counters first.
 *
 * The three placement kernels share a ~30-line preamble (slot → world cell →
 * jitter → distance/range gate → biome/fields/normal fetch → height/water
 * sample): sampleCell() factors that byte-identical prologue. Each kernel's
 * post-preamble density/type logic is otherwise verbatim, and the per-layer
 * jitter salts (grass = salt, debris = salt ^ 0x5dd5, far = salt ^ 0x6f21)
 * MUST match the draw-side fetchRing salts.
 */

import type { Vector3 } from 'three';
import type {
  StorageBufferNode,
  StorageTexture,
  UniformArrayNode,
  UniformNode,
} from 'three/webgpu';
import {
  Fn,
  If,
  Return,
  atomicAdd,
  atomicStore,
  float,
  instanceIndex,
  int,
  smoothstep,
  texture,
  uint,
  vec2,
  vec3,
} from 'three/tsl';
import { canopyAt, cellHash, cellHash2 } from '../../gpu/passes/Scatter';
import { WORLD_SIZE } from '../../world/WorldConst';
import {
  CELL_BIAS,
  DEB_CELL,
  DEB_GRID,
  DEB_R,
  FAR_CELL,
  FAR_GRID,
  FAR_R,
  FAR_R0,
  GRASS_CELL,
  GRASS_GRID,
  GRASS_R,
  G_BAND,
  G_MID,
  G_NEAR,
  grassThin,
} from './constants';
import type { NB, NF, NI, NU, NV2, NV3, NV4 } from '../../gpu/TSLTypes';
import type { Heightfield } from '../../world/Heightfield';

export interface CullCtx {
  hf: Heightfield;
  canopyTex: StorageTexture;
  camU: UniformNode<'vec3', Vector3>;
  planesU: UniformArrayNode<unknown>;
  salt: number;
  cells: StorageBufferNode<'uint'>;
  heights: StorageBufferNode<'float'>;
  counters: ReturnType<StorageBufferNode<'uint'>['toAtomic']>;
  capBuf: StorageBufferNode<'uint'>;
  offBuf: StorageBufferNode<'uint'>;
  caps: number[];
}

interface CellSample {
  wc: NV2;
  wpos: NV2;
  dist: NF;
  bio: NV4;
  fl: NV4;
  ns: NV4;
  bioId: NI;
  h: NF;
  water: NF;
}

export function buildCullKernels(ctx: CullCtx): {
  clearK: object;
  grassK: object;
  debrisK: object;
  farK: object;
} {
  const { hf, canopyTex, camU, planesU, salt, cells, heights, counters, capBuf, offBuf, caps } =
    ctx;

  const clearK = Fn(() => {
    const i = instanceIndex;
    If(i.greaterThanEqual(caps.length), () => {
      Return();
    });
    atomicStore(counters.element(i), uint(0));
  })().compute(caps.length);
  clearK.setName('ringClear');

  /** toroidal slot → nearest congruent world cell */
  const worldCell = (sx: NF, sy: NF, grid: number, cell: number): NV2 => {
    const camC = vec2(camU.x, camU.z).div(cell);
    const wx = camC.x.sub(sx).div(grid).round().mul(grid).add(sx);
    const wy = camC.y.sub(sy).div(grid).round().mul(grid).add(sy);
    return vec2(wx, wy);
  };

  const byBio = (b: NI, vals: number[]): NF => {
    let e: NF = float(vals[5] ?? 0);
    for (let i = 4; i >= 0; i--) {
      e = b.equal(float(i).toInt()).select(float(vals[i] ?? 0), e) as NF;
    }
    return e;
  };

  const inFrustum = (center: NV3, slack: number): NF => {
    let inside: NF = float(1);
    for (let p = 0; p < 6; p++) {
      const pl = planesU.element(float(p).toInt()) as unknown as NV4;
      inside = inside.mul(
        pl.xyz.dot(center).add(pl.w).greaterThan(-slack).select(float(1), float(0)),
      );
    }
    return inside;
  };

  const appendRing = (g: NI, wc: NV2, y: NF): void => {
    const idx = atomicAdd(counters.element(g), uint(1)) as unknown as NU;
    If(idx.lessThan(capBuf.element(g) as unknown as NU), () => {
      const at = (offBuf.element(g) as unknown as NU).add(idx);
      // pack biased cell coords 16|16 (cells span ±~10k)
      cells.element(at).assign(
        wc.x.add(CELL_BIAS).toUint().shiftLeft(uint(16)).bitOr(wc.y.add(CELL_BIAS).toUint()),
      );
      heights.element(at).assign(y);
    });
  };

  /**
   * Shared placement prologue: slot index → world cell → jitter → distance,
   * the per-kernel range early-out, then the biome/fields/normal texture
   * fetch, biome id and height/water sample. The post-preamble water gate
   * differs per layer (grass/far gate on height above water, debris on
   * submersion depth), so it stays in each kernel.
   */
  const sampleCell = (
    grid: number,
    cell: number,
    jitSalt: number,
    rangeOut: (d: NF) => NB,
  ): CellSample => {
    const i = instanceIndex;
    If(i.greaterThanEqual(grid * grid), () => {
      Return();
    });
    const sx = float(i.mod(grid));
    const sy = float(i.div(grid));
    const wc = worldCell(sx, sy, grid, cell);
    const jit = cellHash2(wc, jitSalt);
    const wpos = wc.add(jit).mul(cell);
    const dist = wpos.sub(vec2(camU.x, camU.z)).length();
    If(rangeOut(dist), () => {
      Return();
    });
    const uvW = wpos.div(WORLD_SIZE).add(0.5);
    const bio = texture(
      hf.biomeTex as NonNullable<typeof hf.biomeTex>,
      uvW,
      0,
    ) as unknown as NV4;
    const fl = texture(
      hf.fieldsTex as NonNullable<typeof hf.fieldsTex>,
      uvW,
      0,
    ) as unknown as NV4;
    const ns = texture(hf.normalTex, uvW, 0) as unknown as NV4;
    const bioId = bio.x.mul(8).add(0.5).floor().toInt();
    const h = hf.sampleHeight(wpos);
    const water = hf.sampleWaterYNearest(wpos);
    return { wc, wpos, dist, bio, fl, ns, bioId, h, water };
  };

  // ---------------- grass cull -----------------------------------------------
  const grassK = Fn(() => {
    const { wc, wpos, dist, bio, fl, ns, bioId, h, water } = sampleCell(
      GRASS_GRID,
      GRASS_CELL,
      salt,
      (d) => d.greaterThan(GRASS_R),
    );
    // gate on the ACTUAL water surface, not the carve apron: riverDepth
    // is widen-blurred and flags whole gorge floors as "river" — grass
    // vanished from every dry bank (scene1 banks are green to the line)
    const above = h.sub(water);
    If(above.lessThan(0.04), () => {
      Return();
    });
    const canopy = canopyAt(canopyTex, wpos);
    // soft bank margin: full grass from ~0.5 m above the waterline. The
    // channel scar (deep riverDepth) thins hard — the debris ring's
    // cobbles take over there (scene1: cobbled floor with grassy banks,
    // not a meadow blanket to the waterline) — but never zeroes, so
    // tufts still break the gravel.
    const bank = smoothstep(0.06, 0.5, above).mul(
      float(1).sub(smoothstep(0.2, 1.1, fl.z).mul(0.78)),
    );
    let dens = byBio(bioId, [0.18, 0.7, 0.62, 0.7, 1.5, 1.1])
      .mul(bank)
      .mul(bio.z.mul(0.85).add(0.15))
      .mul(float(1).sub(bio.w.mul(0.55)))
      .mul(float(1).sub(canopy.mul(0.45)))
      .mul(fl.x.mul(0.35).add(0.75));
    // near-field scruff floor: NOTHING within ~12 m may be totally bald
    // (Pillar A) — thin dry blades survive even on poor soil. Hard gates
    // (water, snow, steep rock) still apply below.
    dens = dens.max(
      float(0.3).mul(float(1).sub(smoothstep(8, 14, dist))).mul(bank),
    );
    dens = dens
      .mul(float(1).sub(bio.y.mul(0.95)))
      .mul(float(1).sub(smoothstep(0.55, 0.95, ns.w)));
    // coverage-conserving continuous LOD ("cheap nanite for aggregates"):
    // accept thins SMOOTHLY with distance — survivors widen by 1/sqrt(thin)
    // in the vertex stage, so screen coverage stays constant and there are
    // no density bands; the ring then dissolves into the field-matched
    // terrain splat instead of ending at an edge.
    const thin = grassThin(dist);
    const edge = float(1).sub(smoothstep(GRASS_R * 0.9, GRASS_R, dist));
    If(cellHash(wc, salt ^ 0x77a1).greaterThanEqual(dens.mul(edge).mul(thin)), () => {
      Return();
    });
    If(inFrustum(vec3(wpos.x, h.add(0.5), wpos.y), 1.4).lessThan(0.5), () => {
      Return();
    });
    // Boundary-band cells append to BOTH adjacent layers — the
    // complementary dither in grassMaterial then draws each pixel from
    // exactly one layer, holding blade density constant through the band.
    // Single-list assignment + dither halved density at every boundary
    // (the visible "transparent rings" around the camera).
    If(dist.lessThan(G_NEAR + G_BAND), () => {
      appendRing(int(0), wc, h);
    });
    If(
      dist.greaterThanEqual(G_NEAR - G_BAND).and(dist.lessThan(G_MID + G_BAND)),
      () => {
        appendRing(int(1), wc, h);
      },
    );
    If(dist.greaterThanEqual(G_MID - G_BAND), () => {
      appendRing(int(2), wc, h);
    });
  })().compute(GRASS_GRID * GRASS_GRID);
  grassK.setName('grassRingCull');

  // ---------------- debris cull ------------------------------------------------
  const debrisK = Fn(() => {
    const { wc, wpos, dist, bio, fl, ns, bioId, h, water } = sampleCell(
      DEB_GRID,
      DEB_CELL,
      salt ^ 0x5dd5,
      (d) => d.greaterThan(DEB_R),
    );
    // cobbles stay visible THROUGH shallow water (scene1: the trickle
    // runs over them) — only drop debris under genuinely deep water
    const submergedBy = water.sub(h);
    If(submergedBy.greaterThan(0.55), () => {
      Return();
    });
    const canopy = canopyAt(canopyTex, wpos);
    const streamK = smoothstep(0.32, 0.7, fl.y).max(smoothstep(0.02, 0.2, fl.z));
    // bank margin: too shallow for the bed override, too wet for grass —
    // gravel it or it reads as a bare strip along every wash
    const marginK = smoothstep(0.005, 0.06, fl.z).mul(float(1).sub(streamK));
    // organic debris floats off — submerged cells keep only stone classes
    const dry = smoothstep(0.05, -0.02, submergedBy);
    // channel core (deep scar or submerged) leans hard into cobbles —
    // scene1's bed is packed rounded stone, not occasional rocks
    const coreK = smoothstep(0.25, 1.0, fl.z).max(smoothstep(-0.05, 0.15, submergedBy));
    const wCobble = streamK
      .mul(2.2)
      .add(marginK.mul(1.4))
      .add(bio.w.mul(0.3))
      .add(coreK.mul(2.6))
      .mul(0.5);
    const wPebble = bio.w.mul(0.9).add(streamK).add(marginK.mul(1.4)).add(0.15).mul(0.6);
    const wTwig = canopy.mul(1.8).add(0.12).mul(float(1).sub(streamK)).mul(dry);
    const wChip = canopy.mul(0.8).mul(float(1).sub(streamK)).mul(dry);
    const wLitter = canopy.mul(3.0).add(0.08).mul(float(1).sub(streamK.mul(0.8))).mul(dry);
    const wSum = wCobble.add(wPebble).add(wTwig).add(wChip).add(wLitter);
    // streambeds are FULLY cobbled geometry (spec §9) — override biome density
    const dens = byBio(bioId, [0.4, 0.6, 1.0, 1.0, 0.6, 0.75])
      .mul(float(1).sub(bio.y.mul(0.9)))
      .mul(wSum.mul(0.5).min(1))
      .max(streamK.mul(0.95))
      .max(marginK.mul(0.85))
      .mul(float(1).sub(smoothstep(0.7, 1.05, ns.w)));
    const edge = float(1).sub(smoothstep(DEB_R * 0.72, DEB_R, dist));
    If(cellHash(wc, salt ^ 0x132f).greaterThanEqual(dens.mul(edge)), () => {
      Return();
    });
    If(inFrustum(vec3(wpos.x, h.add(0.3), wpos.y), 0.8).lessThan(0.5), () => {
      Return();
    });
    const r = cellHash(wc, salt ^ 0x4c11).mul(wSum);
    const ty = float(0).toVar();
    const acc = wCobble.toVar();
    If(r.greaterThan(acc), () => {
      ty.assign(1);
      acc.addAssign(wPebble);
      If(r.greaterThan(acc), () => {
        ty.assign(2);
        acc.addAssign(wTwig);
        If(r.greaterThan(acc), () => {
          ty.assign(3);
          acc.addAssign(wChip);
          If(r.greaterThan(acc), () => {
            ty.assign(4);
          });
        });
      });
    });
    appendRing(ty.add(3).toInt(), wc, h);
  })().compute(DEB_GRID * DEB_GRID);
  debrisK.setName('debrisRingCull');

  // ---------------- far super-tuft cull (g3) -----------------------------------
  const farK = Fn(() => {
    const { wc, wpos, dist, bio, fl, ns, bioId, h, water } = sampleCell(
      FAR_GRID,
      FAR_CELL,
      salt ^ 0x6f21,
      (d) => d.lessThan(FAR_R0 - 16).or(d.greaterThan(FAR_R)),
    );
    const above = h.sub(water);
    If(above.lessThan(0.06), () => {
      Return();
    });
    const canopy = canopyAt(canopyTex, wpos);
    const bank = smoothstep(0.06, 0.5, above).mul(
      float(1).sub(smoothstep(0.2, 1.1, fl.z).mul(0.78)),
    );
    const dens = byBio(bioId, [0.18, 0.7, 0.62, 0.7, 1.5, 1.1])
      .mul(bank)
      .mul(bio.z.mul(0.85).add(0.15))
      .mul(float(1).sub(bio.w.mul(0.55)))
      .mul(float(1).sub(canopy.mul(0.45)))
      .mul(float(1).sub(bio.y.mul(0.95)))
      .mul(float(1).sub(smoothstep(0.55, 0.95, ns.w)));
    // ramp IN over the fine band's dissolve, OUT at the splat handoff
    const fadeIn = smoothstep(FAR_R0 - 16, FAR_R0 + 14, dist);
    const edge = float(1).sub(smoothstep(FAR_R * 0.93, FAR_R, dist));
    If(
      cellHash(wc, salt ^ 0x55aa).greaterThanEqual(
        dens.mul(fadeIn).mul(edge).mul(0.55),
      ),
      () => {
        Return();
      },
    );
    If(inFrustum(vec3(wpos.x, h.add(0.6), wpos.y), 1.6).lessThan(0.5), () => {
      Return();
    });
    appendRing(int(8), wc, h);
  })().compute(FAR_GRID * FAR_GRID);
  farK.setName('farTuftCull');

  return { clearK, grassK, debrisK, farK };
}
