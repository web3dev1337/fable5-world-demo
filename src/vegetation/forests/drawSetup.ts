/**
 * Forests draw/material wiring helpers — the pieces the per-pool draw loop in
 * Forests.init() composes:
 *   - geoView:        a geometry view sharing attributes/index but with its own
 *                     indirect slot
 *   - fadeFor:        the LOD-ring dithered-crossfade boundaries (CPU side; the
 *                     ring distances are shared with the GPU cull kernel via
 *                     layout.ts so the two never drift)
 *   - proxyCasterMat: the crown shadow-proxy caster material
 *   - makeAddDraw:    the per-draw registration (indirect spec + mesh + optional
 *                     depth-prepass twin / per-cascade caster layer)
 */

import { BufferGeometry, Mesh } from 'three';
import type { BufferAttribute, Group } from 'three';
import { MeshStandardNodeMaterial } from 'three/webgpu';
import { float, positionLocal, positionWorld, smoothstep, vec3 } from 'three/tsl';
import { hash12 } from '../../gpu/noise/NoiseTSL';
import { instanceVeg, type RingFade } from '../../render/VegInstance';
import { markVegRefresh } from '../../render/StaticRefresh';
import { depthPrepassTwin } from '../../render/VegPrepass';
import type { NF } from '../../gpu/TSLTypes';
import type { CrownDims } from './crownProxy';
import {
  BAND0,
  BAND1,
  BAND2,
  EX_BAND,
  EX_R1_FAR,
  IMP_CAST_FADE0,
  IMP_CAST_FAR,
  R0_FAR,
  R1_FAR,
  R2_FAR,
} from './layout';

export interface DrawSpec {
  group: number;
  indexCount: number;
}

/** geometry view sharing attributes/index but with its own indirect slot */
export function geoView(src: BufferGeometry): BufferGeometry {
  const g = new BufferGeometry();
  for (const [name, attr] of Object.entries(src.attributes)) {
    g.setAttribute(name, attr as BufferAttribute);
  }
  if (src.index) g.setIndex(src.index);
  return g;
}

/**
 * Shadow-proxy caster material: world-anchored hash dither (screen-space IGN
 * swims when CSM refits its boxes — user-visible shadow flicker) at species
 * density, with a crown-edge falloff so the rim breaks up into a ragged crown
 * instead of a solid oval. Impostor-band proxies fade out toward IMP_CAST_FAR
 * (fade distance uses vegViewPos via instanceVeg).
 */
export function proxyCasterMat(
  bind: Parameters<typeof instanceVeg>[1],
  density: number,
  dims: CrownDims,
  impostorBand: boolean,
): MeshStandardNodeMaterial {
  const pmat = new MeshStandardNodeMaterial();
  const handles = instanceVeg(pmat, bind);
  const e = positionLocal
    .sub(vec3(0, dims.cy, 0))
    .div(vec3(dims.rxz, dims.ry, dims.rxz))
    .length();
  let dens: NF = float(density).mul(
    float(1).sub(e.pow(3).mul(0.55)),
  );
  if (impostorBand) {
    dens = dens.mul(
      float(1).sub(smoothstep(IMP_CAST_FADE0, IMP_CAST_FAR - 50, handles.dist)),
    );
  }
  (pmat as unknown as { maskShadowNode: unknown }).maskShadowNode = hash12(
    positionWorld.xz.mul(13.73).add(positionWorld.yy.mul(5.19)),
  ).lessThan(dens);
  return pmat;
}

export function fadeFor(cls: number, ring: 0 | 1 | 2 | 3, clsMaxDist: number[]): RingFade {
  if (cls < 6) {
    if (ring === 0) return { fadeOutAt: R0_FAR, band: BAND0 };
    if (ring === 1)
      return { fadeInAt: R0_FAR, inBand: BAND0, fadeOutAt: R1_FAR, band: BAND1 };
    // bands MUST match across each boundary (in-band here = out-band of the
    // nearer ring) or the complementary dither doesn't partition pixels and
    // holes reappear — hence inBand: ring2's out edge pairs with the impostor's
    // BAND2 while its in edge pairs with BAND1.
    if (ring === 2)
      return { fadeInAt: R1_FAR, inBand: BAND1, fadeOutAt: R2_FAR, band: BAND2 };
    return { fadeInAt: R2_FAR, band: BAND2 };
  }
  const maxD = clsMaxDist[cls] ?? 150;
  if (cls < 15) return { fadeOutAt: maxD - 15, band: 15 };
  const hasR2 = cls === 18 || cls === 19 || cls === 20 || cls === 21 || cls === 23;
  if (ring === 1)
    return hasR2
      ? { fadeOutAt: EX_R1_FAR, band: EX_BAND }
      : { fadeOutAt: maxD - 20, band: 20 };
  return { fadeInAt: EX_R1_FAR, fadeOutAt: maxD - 20, band: EX_BAND };
}

export interface AddDrawCtx {
  draws: DrawSpec[];
  meshes: Mesh[];
  groupTris: Float32Array;
  group: Group;
  prepassGroup: Group;
  noPrepass: boolean;
}

/**
 * Registers a draw: pushes its indirect spec, builds the instanced mesh, and
 * either (visible draw) adds a card-only depth-prepass twin or (caster) pins
 * the mesh to a per-cascade shadow layer. Casting for visible draws is owned
 * by the per-cascade sibling meshes, so visible meshes never cast.
 */
export function makeAddDraw(
  ctx: AddDrawCtx,
): (
  geo: BufferGeometry,
  mat: MeshStandardNodeMaterial,
  g: number,
  tris: number,
  shadowLayer?: number | null,
) => void {
  return (geo, mat, g, tris, shadowLayer = null): void => {
    const indexCount = geo.index ? geo.index.count : geo.attributes.position?.count ?? 0;
    ctx.draws.push({ group: g, indexCount });
    const mesh = new Mesh(geo, mat);
    mesh.frustumCulled = false;
    markVegRefresh(mesh);
    if (shadowLayer === null) {
      // visible draw — casting is owned by the per-cascade sibling meshes
      ctx.groupTris[g] += tris;
      mesh.castShadow = false;
      mesh.receiveShadow = true;
      // depth prepass for CARD parts only: crowns shade 3-8x per covered pixel
      // (alpha-tested cutouts defeat early-Z); opaque bark/rock overdraw is
      // shallow and each twin costs a CPU draw (~29 us) + doubled vertex wind,
      // so twinning everything LOST wall time. (A DoubleSide quad rasterizes
      // each triangle ONCE — no duplicate face depth-ties; EQUAL is safe.
      // Verified vs no-prepass at a frame-aligned capture: differences at the
      // deterministic floor.)
      if (!ctx.noPrepass && mat.alphaTest > 0) {
        const matS = mat as unknown as {
          positionNode: unknown;
          maskNode: unknown;
          opacityNode: unknown;
        };
        const twin = depthPrepassTwin(mesh, {
          positionNode: matS.positionNode,
          maskNode: matS.maskNode ?? undefined,
          ...(mat.alphaTest > 0
            ? { opacityNode: matS.opacityNode, alphaTest: mat.alphaTest }
            : {}),
          side: mat.side,
        });
        markVegRefresh(twin);
        ctx.prepassGroup.add(twin);
      }
    } else {
      // shadow-only caster: lives on the cascade's layer, so ONLY that
      // cascade's shadow camera ever renders it
      mesh.castShadow = true;
      mesh.receiveShadow = false;
      mesh.layers.set(shadowLayer);
    }
    ctx.meshes.push(mesh);
    ctx.group.add(mesh);
  };
}
