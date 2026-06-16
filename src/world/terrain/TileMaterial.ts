/**
 * Near-tile CDLOD material: world-space vertex morph + skirt drop, micro-
 * displacement (Pillar A geometric relief), terrain shading, the Phase 6
 * water/caustic response, and probe-GI light-map injection.
 *
 * Pure code motion from the TerrainTiles constructor — every node, uniform and
 * numeric value is verbatim. Returns the material plus the handful of nodes the
 * debug-view builder needs to keep painting onto the same material.
 */

import {
  IrradianceNode,
  MeshPhysicalNodeMaterial,
  type StorageBufferNode,
} from 'three/webgpu';
import { canopyAt } from '../../gpu/passes/Scatter';
import {
  cameraPosition,
  clamp,
  float,
  fract,
  smoothstep,
  instanceIndex,
  mix,
  positionLocal,
  positionWorld,
  texture,
  varying,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import type { ProbeGI } from '../../gpu/passes/ProbeGI';
import {
  causticContext,
  causticDepth,
  causticTint,
  causticTintParts,
  type CausticCtx,
} from '../../render/Caustics';
import {
  DISP,
  buildTerrainShading,
  type TerrainShading,
} from '../../render/TerrainMaterial';
import { PERIOD_FBM, PERIOD_RID, PERIOD_VAL } from '../../gpu/passes/NoiseBake';
import type { Heightfield } from '../Heightfield';
import { WORLD_SIZE } from '../WorldConst';
import { PATCH_SEGS, SPLIT_K } from './TileConstants';
import type { NV4 } from '../../gpu/TSLTypes';
import type { StorageTexture } from 'three/webgpu';

export interface TerrainTilesOptions {
  heightBuf?: Heightfield['height'];
  neutral?: boolean;
  screenHalf?: 'left' | 'right';
  gi?: ProbeGI;
  /** canopy coverage map — attenuates probe ambient under tree crowns */
  canopyTex?: StorageTexture;
}

/** material + the nodes the debug-view pass paints onto */
export interface TileMaterialResult {
  mat: MeshPhysicalNodeMaterial;
  shading: TerrainShading;
  cctx: CausticCtx | null;
  tile: NV4;
}

export function buildTileMaterial(
  hf: Heightfield,
  tileBuf: StorageBufferNode<'vec4'>,
  heightBuf: Heightfield['height'],
  opts: TerrainTilesOptions,
  ablate: Set<string>,
): TileMaterialResult {
  // --- material ---------------------------------------------------------------
  // physical for specularIntensity: the dielectric F0 0.04 sheen at
  // glancing sun desaturates whole hillsides to silver (user feedback —
  // 'terrain gets too silvery'); rock keeps a modest glint
  const mat = new MeshPhysicalNodeMaterial();
  mat.specularIntensity = 0.35;
  const tile = tileBuf.element(instanceIndex);
  const tileOrigin = tile.xy; // world xz of tile center
  const tileSize = tile.z;

  // CDLOD morph: world-space vertex, odd-vertex snap toward even grid.
  // Skirt verts (|local| > 0.5) clamp onto the edge, then drop down.
  const rawLocal = positionLocal.xz;
  const clampedLocal = clamp(rawLocal, -0.5, 0.5);
  const isSkirt = rawLocal
    .abs()
    .x.max(rawLocal.abs().y)
    .greaterThan(0.5001)
    .select(float(1), float(0));
  const local = clampedLocal.mul(tileSize);
  const wpos0 = local.add(tileOrigin).toVar();
  const quad = tileSize.div(PATCH_SEGS); // quad size in meters
  const gridUV = clampedLocal.add(0.5).mul(PATCH_SEGS); // 0..SEGS
  const odd = fract(gridUV.mul(0.5)).mul(2); // 1 where odd, 0 where even
  const snapped = wpos0.sub(odd.mul(quad)); // snap odd verts down-grid
  const camD = wpos0.sub(cameraPosition.xz).length();
  // morph across the outer band of this LOD's range
  const rangeEnd = tileSize.mul(SPLIT_K).mul(2); // parent split distance
  const morphK = clamp(camD.sub(rangeEnd.mul(0.7)).div(rangeEnd.mul(0.24)), 0, 1);
  const wpos = mix(wpos0, snapped, morphK);

  // instance + object matrices are identity → positionNode is world space
  const skirtDrop = isSkirt.mul(tileSize.mul(0.045).add(2.5));
  const hSample = hf.sampleHeightFrom(heightBuf, wpos).sub(skirtDrop);

  // --- micro-displacement (5×-detail / Pillar A): geometric relief ≤85 m.
  // The splat's bump normals imply 10–35 cm of relief the silhouette never
  // had — grazing close-ups read blob-smooth ("bare smooth ground" ban).
  // Crack-free: skirt verts sample the same world-space field at their
  // clamped edge position, and CDLOD morph makes shared-edge verts
  // coincide across LODs. Veg sits on the UNDISPLACED field — amplitude
  // stays ≤9 cm where grass grows (blade sink hides it), full on bare
  // rock/scree; snow smooths it back out.
  const uvV = wpos.div(WORLD_SIZE).add(0.5);
  const nsV = texture(hf.normalTex, uvV, 0);
  const bioV = hf.biomeTex ? texture(hf.biomeTex, uvV, 0) : vec4(0, 0, 0, 0);
  const fldV = hf.fieldsTex ? texture(hf.fieldsTex, uvV, 0) : vec4(0, 0, 0, 0);
  const rockK = smoothstep(DISP.slopeKnee0, DISP.slopeKnee1, nsV.w).max(
    bioV.a.mul(0.85),
  );
  const gravelK = smoothstep(0.32, 0.7, fldV.y)
    .max(smoothstep(0.02, 0.2, fldV.z))
    .mul(float(DISP.gravel));
  const dispAmp = mix(float(DISP.base), float(DISP.rock), rockK)
    .max(gravelK)
    .mul(bioV.g.mul(0.75).oneMinus())
    .mul(clamp(float(DISP.fade1).sub(camD).div(DISP.fade1 - DISP.fade0), 0, 1));
  const noiseA = hf.noiseA as NonNullable<typeof hf.noiseA>;
  const noiseB = hf.noiseB as NonNullable<typeof hf.noiseB>;
  const f1 = texture(noiseA, wpos.div(DISP.sF1 * PERIOD_FBM), 0)
    .y.mul(2)
    .sub(1);
  const f2 = texture(noiseA, wpos.div(DISP.sF2 * PERIOD_VAL).add(vec2(0.31, 0.77)), 0)
    .x.mul(2)
    .sub(1);
  // ridged creases (1−|n| sharp valleys) carry the "rock" read — weighted
  // toward rock faces, soft elsewhere
  const r1 = texture(noiseB, wpos.div(DISP.sRid * PERIOD_RID), 0)
    .z.mul(2)
    .sub(1);
  const disp = f1
    .mul(DISP.wF1)
    .add(f2.mul(DISP.wF2))
    .add(r1.mul(rockK.mul(1 - DISP.ridBase).add(DISP.ridBase)).mul(DISP.wRid))
    .mul(dispAmp);
  mat.positionNode = vec3(wpos.x, hSample.add(disp), wpos.y);
  // shadow casting: skip the morph + bilinear (4 reads → 1); cascade texels
  // are meters wide, normalBias absorbs the nearest-fetch steps
  mat.castShadowPositionNode = vec3(
    wpos0.x,
    hf.sampleHeightNearest(wpos0).sub(skirtDrop),
    wpos0.y,
  );

  const shading = buildTerrainShading({
    normalTex: hf.normalTex,
    biomeTex: hf.biomeTex as NonNullable<typeof hf.biomeTex>,
    fieldsTex: hf.fieldsTex as NonNullable<typeof hf.fieldsTex>,
    noiseA: hf.noiseA as NonNullable<typeof hf.noiseA>,
    noiseB: hf.noiseB as NonNullable<typeof hf.noiseB>,
    mp: hf.mp,
    far: false,
  });
  mat.colorNode = shading.colorNode;
  mat.normalNode = shading.normalNode;
  mat.roughnessNode = shading.roughnessNode;
  mat.metalnessNode = float(0);
  // Phase 6 water response (near tiles only): capillary-wet band hugging
  // the true waterline (the splat's moisture wetness is sim-res blurry)
  // + animated caustics on submerged beds. d = water column above the
  // fragment; the band covers d ∈ (−0.45, 0) and saturates under water.
  const cctx = causticContext();
  if (cctx && !opts.neutral) {
    const d = causticDepth(positionWorld);
    const fringe = smoothstep(-0.45, -0.04, d);
    const caust = causticTint(positionWorld, d);
    // permanently submerged beds grow biofilm/algae: darker and olive —
    // without this the sunlit gravel splat shines straight through the
    // water and the whole stream reads as a pale sheet (vs scene1's dark
    // glassy trickle)
    const biofilm = smoothstep(0.04, 0.5, d);
    let wetCol = shading.colorNode
      .mul(fringe.mul(0.38).oneMinus())
      .mul(biofilm.mul(0.42).oneMinus());
    wetCol = mix(wetCol, wetCol.mul(vec3(0.72, 0.86, 0.55)), biofilm.mul(0.65));
    mat.colorNode = wetCol.mul(caust.mul(1.7).add(1));
    mat.roughnessNode = shading.roughnessNode.sub(fringe.mul(0.42)).clamp(0.18, 1);
    // ?caustlit=1 — paint the lit graph's own caustic chain (triage):
    // r = gated tint×4, g = gate product, b = ungated pattern
    if (new URLSearchParams(window.location.search).get('caustlit') === '1') {
      const parts = causticTintParts(positionWorld, d);
      mat.emissiveNode = vec3(parts.x.mul(4), parts.y, parts.z);
    }
  }
  // ?dispdbg=1 — paint micro-displacement (green=+, red=−, dark=none);
  // must land AFTER the shading assignment or it gets overwritten
  if (new URLSearchParams(window.location.search).get('dispdbg') === '1') {
    const dv = varying(disp);
    mat.colorNode = vec3(0.02);
    mat.emissiveNode = vec3(dv.negate().max(0).mul(2), dv.max(0).mul(2), 0.02);
  }
  if (opts.gi && !ablate.has('gi')) {
    // probe-GI irradiance replaces the hemisphere ambient (Phase 3) —
    // injected through the lighting context like a light map. The probe
    // field is canopy-aware (crown-slab extinction in the gather); this
    // receiver factor only adds the 4 m-texel spatial detail the 16 m
    // probe grid can't resolve.
    let irr = opts.gi.irradiance(positionWorld, shading.worldNormalNode);
    if (opts.canopyTex && !ablate.has('canopy')) {
      irr = irr.mul(
        canopyAt(opts.canopyTex, positionWorld.xz).mul(0.18).oneMinus(),
      ) as typeof irr;
    }
    (mat as unknown as { setupLightMap: () => unknown }).setupLightMap = () =>
      new IrradianceNode(irr as unknown as ConstructorParameters<typeof IrradianceNode>[0]);
  }

  return { mat, shading, cctx, tile };
}
