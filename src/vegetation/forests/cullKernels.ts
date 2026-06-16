/**
 * Forests GPU cull/LOD compute kernels (TSL).
 *
 * Per frame: clear counters → cull each scatter layer → write indirect args.
 * The cull kernel does, per instance: per-class distance bound → frustum sphere
 * test (6 planes) → terrain-occlusion march (heightfield ray test camera→crown
 * top) → LOD ring classification with overlap bands → atomic append of the
 * instance slot into per-(pool,ring) compact regions. Shadow casters are culled
 * per cascade against each CSM cascade's ortho frustum (24 extra plane uniforms)
 * and appended into per-(pool,ring,cascade) caster regions; casters skip the
 * camera-occlusion march (a ridge-hidden tree still casts into the visible
 * slope).
 *
 * The group-index arithmetic here is the GPU mirror of layout.ts's groupOf /
 * casterGroupOf — both reference the SAME named base offsets so the two never
 * drift (a mismatch silently renders the wrong pool).
 */

import type { Vector3 } from 'three';
import type { StorageBufferNode, UniformArrayNode, UniformNode } from 'three/webgpu';
import {
  Fn,
  If,
  Return,
  atomicAdd,
  atomicLoad,
  atomicStore,
  float,
  instanceIndex,
  int,
  uint,
  vec2,
  vec3,
} from 'three/tsl';
import type { Heightfield } from '../../world/Heightfield';
import type { ScatterLayer, ScatterResult } from '../../gpu/passes/Scatter';
import type { NF, NI, NU, NV3, NV4 } from '../../gpu/TSLTypes';
import {
  BAND0,
  BAND1,
  BAND2,
  CASC_LOCALS,
  CASCADES,
  CAST_EXTRAS_LOCAL,
  CAST_HERO_LOCAL,
  CAST_IMP_LOCAL,
  EX_BAND,
  EX_R1_FAR,
  EXTRAS_BASE,
  GROUPS,
  IMP_CAST_FAR,
  IMPOSTOR_BASE,
  MAIN_GROUPS,
  R0_FAR,
  R1_FAR,
  R2_FAR,
  TREE_MAIN_BASE,
  UNDER_BASE,
} from './layout';

export interface CullKernelCtx {
  /** atomic per-group counters */
  counters: StorageBufferNode<'uint'>;
  /** flat compact slot list (all regions concatenated) */
  compact: StorageBufferNode<'uint'>;
  /** per-group region capacity */
  capBuf: StorageBufferNode<'uint'>;
  /** per-group region start offset into `compact` */
  offBuf: StorageBufferNode<'uint'>;
  /** per-class cull info (height, radius, maxDist, hasR2) */
  clsBuf: StorageBufferNode<'vec4'>;
  /** per-draw → group lookup (indirect pass) */
  drawGroupBuf: StorageBufferNode<'uint'>;
  /** indirect draw args (5 uint per draw) */
  indirectStore: StorageBufferNode<'uint'>;
  /** number of draws */
  drawCount: number;
  /** main-camera position */
  camU: UniformNode<'vec3', Vector3>;
  /** 6 main-view frustum planes */
  planesU: UniformArrayNode<unknown>;
  /** 6 planes × CASCADES cascade ortho frusta */
  planesCsmU: UniformArrayNode<unknown>;
  /** terrain heightfield (occlusion march) */
  hf: Heightfield;
  /** scatter layers fed to the cull kernels */
  scatter: ScatterResult;
}

/**
 * Builds the per-frame compute kernel list:
 *   [ clearK, cull(trees), cull(under), cull(extras), cull(stones), indirectK ]
 */
export function buildCullKernels(ctx: CullKernelCtx): object[] {
  const counters = ctx.counters;
  const compact = ctx.compact;
  const capBuf = ctx.capBuf;
  const offBuf = ctx.offBuf;
  const clsBuf = ctx.clsBuf;
  const drawGroupBuf = ctx.drawGroupBuf;
  const indirectStore = ctx.indirectStore;
  const camU = ctx.camU;
  const planesU = ctx.planesU;
  const planesCsmU = ctx.planesCsmU;
  const hf = ctx.hf;
  const D = ctx.drawCount;

  const clearK = Fn(() => {
    const i = instanceIndex;
    If(i.greaterThanEqual(GROUPS), () => {
      Return();
    });
    atomicStore(counters.element(i), uint(0));
  })().compute(GROUPS);
  clearK.setName('vegClear');

  const inFrustum = (center: NV3, rad: NF): NF => {
    // product of per-plane step(−r ≤ dist) — 1 inside, 0 outside
    let inside: NF = float(1);
    for (let p = 0; p < 6; p++) {
      const pl = planesU.element(int(p)) as unknown as NV4;
      const d = pl.xyz.dot(center).add(pl.w);
      inside = inside.mul(d.greaterThan(rad.negate()).select(float(1), float(0)));
    }
    return inside;
  };

  // +30 m slack: the planes are one frame stale (CSM fits its boxes during
  // the upcoming render) — without it, casters at box edges pop while the
  // camera moves
  const inCascade = (c: number, center: NV3, rad: NF): NF => {
    let inside: NF = float(1);
    for (let p = 0; p < 6; p++) {
      const pl = planesCsmU.element(int(c * 6 + p)) as unknown as NV4;
      const d = pl.xyz.dot(center).add(pl.w);
      inside = inside.mul(
        d.greaterThan(rad.add(30).negate()).select(float(1), float(0)),
      );
    }
    return inside;
  };

  const appendTo = (g: NI | NU, slot: NU): void => {
    const idx = atomicAdd(counters.element(g), uint(1)) as unknown as NU;
    If(idx.lessThan(capBuf.element(g) as unknown as NU), () => {
      compact
        .element((offBuf.element(g) as unknown as NU).add(idx))
        .assign(slot);
    });
  };

  const makeCull = (
    layer: ScatterLayer,
    kind: 'trees' | 'under' | 'extras',
  ): object => {
    const N = layer.count;
    const k = Fn(() => {
      const i = instanceIndex;
      If(i.greaterThanEqual(uint(Math.max(N, 1))), () => {
        Return();
      });
      const A = layer.bufA.element(i) as unknown as NV4;
      const B = layer.bufB.element(i) as unknown as NV4;
      const idF = B.w;
      const cls = idF.div(8).floor();
      const variant = idF.sub(cls.mul(8));
      const info = clsBuf.element(cls.toInt()) as unknown as NV4;
      const scl = A.w;
      const hgt = info.x.mul(scl);
      const rad = info.y.mul(scl);
      const center = A.xyz.add(vec3(0, 1, 0).mul(hgt.mul(0.5)));
      const dist = A.xyz.sub(camU).length();

      if (kind !== 'trees') {
        // hard reach bound — applies to main view AND casters (beyond it
        // no ring geometry exists at all)
        If(dist.greaterThanEqual(info.z), () => {
          Return();
        });
      }

      if (kind === 'under') {
        // understory never casts — keep the cheap early-out path
        If(inFrustum(center, rad).lessThan(0.5), () => {
          Return();
        });
        const g = cls.sub(8).mul(4).add(variant).add(UNDER_BASE).toInt();
        appendTo(g as unknown as NI, i as unknown as NU);
        return;
      }

      // main-view visibility: frustum + terrain-occlusion march (camera
      // sight line) — casters intentionally skip BOTH (an off-screen or
      // ridge-hidden tree still casts into the visible scene)
      const visMain = inFrustum(center, rad).toVar();
      If(visMain.greaterThan(0.5).and(dist.greaterThan(140)), () => {
        const top = vec3(A.x, A.y.add(hgt), A.z);
        const occ = float(0).toVar();
        for (let st = 1; st <= 7; st++) {
          const t = st / 8;
          const sp = camU.mul(1 - t).add(top.mul(t)) as unknown as NV3;
          const th = hf.sampleHeightNearest(vec2(sp.x, sp.z));
          occ.assign(occ.max(th.sub(sp.y)));
        }
        If(occ.greaterThan(4), () => {
          visMain.assign(0);
        });
      });

      if (kind === 'trees') {
        const pool = cls.mul(4).add(variant).toInt();
        If(visMain.greaterThan(0.5), () => {
          If(dist.lessThan(R0_FAR + BAND0), () => {
            appendTo(pool.add(TREE_MAIN_BASE) as unknown as NI, i as unknown as NU);
          });
          If(
            dist.greaterThanEqual(R0_FAR - BAND0).and(dist.lessThan(R1_FAR + BAND1)),
            () => {
              appendTo(pool.mul(2) as unknown as NI, i as unknown as NU);
            },
          );
          If(
            dist.greaterThanEqual(R1_FAR - BAND1).and(dist.lessThan(R2_FAR + BAND2)),
            () => {
              appendTo(pool.mul(2).add(1) as unknown as NI, i as unknown as NU);
            },
          );
          If(dist.greaterThanEqual(R2_FAR - BAND2), () => {
            appendTo(cls.add(IMPOSTOR_BASE).toInt() as unknown as NI, i as unknown as NU);
          });
        });
        // casters per cascade — same ring choice as the main view so the
        // shadow silhouette matches the rendered crown
        for (let c = 0; c < CASCADES; c++) {
          const base = MAIN_GROUPS + c * CASC_LOCALS;
          If(inCascade(c, center, rad).greaterThan(0.5), () => {
            If(dist.lessThan(R0_FAR + BAND0), () => {
              appendTo(
                pool.add(base + CAST_HERO_LOCAL) as unknown as NI,
                i as unknown as NU,
              );
            });
            If(
              dist.greaterThanEqual(R0_FAR - BAND0).and(dist.lessThan(R1_FAR + BAND1)),
              () => {
                appendTo(pool.mul(2).add(base) as unknown as NI, i as unknown as NU);
              },
            );
            If(
              dist.greaterThanEqual(R1_FAR - BAND1).and(dist.lessThan(R2_FAR + BAND2)),
              () => {
                appendTo(
                  pool.mul(2).add(base + 1) as unknown as NI,
                  i as unknown as NU,
                );
              },
            );
            // impostor band: crown proxies keep casting past R2 so the
            // shadow field fades out instead of ending in a camera circle
            If(
              dist.greaterThanEqual(R2_FAR - BAND2).and(dist.lessThan(IMP_CAST_FAR)),
              () => {
                appendTo(
                  cls.add(base + CAST_IMP_LOCAL).toInt() as unknown as NI,
                  i as unknown as NU,
                );
              },
            );
          });
        }
      } else {
        const pe = cls.sub(16).mul(4).add(variant);
        const hasR2 = info.w.greaterThan(0.5);
        If(visMain.greaterThan(0.5), () => {
          If(hasR2, () => {
            If(dist.lessThan(EX_R1_FAR + EX_BAND), () => {
              appendTo(pe.mul(2).add(EXTRAS_BASE).toInt() as unknown as NI, i as unknown as NU);
            });
            If(dist.greaterThanEqual(EX_R1_FAR - EX_BAND), () => {
              appendTo(pe.mul(2).add(EXTRAS_BASE + 1).toInt() as unknown as NI, i as unknown as NU);
            });
          }).Else(() => {
            appendTo(pe.mul(2).add(EXTRAS_BASE).toInt() as unknown as NI, i as unknown as NU);
          });
        });
        for (let c = 0; c < CASCADES; c++) {
          const base = MAIN_GROUPS + c * CASC_LOCALS + CAST_EXTRAS_LOCAL;
          If(inCascade(c, center, rad).greaterThan(0.5), () => {
            If(hasR2, () => {
              If(dist.lessThan(EX_R1_FAR + EX_BAND), () => {
                appendTo(
                  pe.mul(2).add(base).toInt() as unknown as NI,
                  i as unknown as NU,
                );
              });
              If(dist.greaterThanEqual(EX_R1_FAR - EX_BAND), () => {
                appendTo(
                  pe.mul(2).add(base + 1).toInt() as unknown as NI,
                  i as unknown as NU,
                );
              });
            }).Else(() => {
              appendTo(
                pe.mul(2).add(base).toInt() as unknown as NI,
                i as unknown as NU,
              );
            });
          });
        }
      }
    })().compute(Math.max(N, 1));
    k.setName(`vegCull_${kind}`);
    return k;
  };

  const indirectK = Fn(() => {
    const i = instanceIndex;
    If(i.greaterThanEqual(D), () => {
      Return();
    });
    const g = drawGroupBuf.element(i) as unknown as NU;
    const raw = atomicLoad(counters.element(g)) as unknown as NU;
    const cap = capBuf.element(g) as unknown as NU;
    const n = raw.greaterThan(cap).select(cap, raw);
    indirectStore.element(i.mul(5).add(1)).assign(n);
  })().compute(D);
  indirectK.setName('vegIndirect');

  return [
    clearK,
    makeCull(ctx.scatter.trees, 'trees'),
    makeCull(ctx.scatter.understory, 'under'),
    makeCull(ctx.scatter.extras, 'extras'),
    makeCull(ctx.scatter.stones, 'extras'),
    indirectK,
  ];
}
