/**
 * Far vista shell: a single radial ring (world edge → FAR_RADIUS) carrying the
 * analytic far-mode macro height, finite-difference normal, and far-detail
 * terrain shading. Blends to the baked field across the world edge so the
 * distant mountains continue the near terrain seamlessly.
 *
 * Pure code motion from the TerrainTiles constructor; verbatim node graphs.
 */

import { Mesh, RingGeometry } from 'three';
import { IrradianceNode, MeshPhysicalNodeMaterial } from 'three/webgpu';
import {
  clamp,
  float,
  mix,
  positionLocal,
  positionWorld,
  varying,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import { buildTerrainShading } from '../../render/TerrainMaterial';
import type { Heightfield } from '../Heightfield';
import { macroTerrain } from '../MacroMap';
import { FAR_RADIUS, WORLD_HALF } from '../WorldConst';
import type { TerrainTilesOptions } from './TileMaterial';

export function buildFarShell(
  hf: Heightfield,
  opts: TerrainTilesOptions,
  ablate: Set<string>,
): Mesh {
  const ring = new RingGeometry(WORLD_HALF * 0.952, FAR_RADIUS, 160, 42);
  ring.rotateX(-Math.PI / 2);
  const farMat = new MeshPhysicalNodeMaterial();
  farMat.specularIntensity = 0.35;
  const fxz = positionLocal.xz;
  const farMacro = macroTerrain(fxz, hf.mp, 'far');
  const baked = hf.sampleHeight(fxz);
  const edgeBlend = clamp(
    fxz.abs().x.max(fxz.abs().y).sub(WORLD_HALF * 0.95).div(WORLD_HALF * 0.05),
    0,
    1,
  );
  // sit well below the tile mesh inside the world (coarse far tiles deviate
  // several meters — the shell poked through and showed far-mode shading)
  const farH = mix(baked, farMacro.height, edgeBlend).sub(
    mix(float(9), float(2.5), edgeBlend),
  );
  farMat.positionNode = vec3(fxz.x, farH, fxz.y);
  // analytic per-vertex normal (no baked maps beyond the world edge):
  // finite-difference the far macro height, interpolated via varying
  const eN = 60;
  const hX = macroTerrain(fxz.add(vec2(eN, 0)), hf.mp, 'far').height;
  const hZ = macroTerrain(fxz.add(vec2(0, eN)), hf.mp, 'far').height;
  const farNormal = vec3(farMacro.height.sub(hX), float(eN), farMacro.height.sub(hZ))
    .normalize();
  const farSlope = vec2(farMacro.height.sub(hX), farMacro.height.sub(hZ))
    .length()
    .div(eN);
  const farNS = varying(vec4(farNormal, farSlope));
  const farShading = buildTerrainShading({
    normalTex: hf.normalTex,
    biomeTex: hf.biomeTex as NonNullable<typeof hf.biomeTex>,
    fieldsTex: hf.fieldsTex as NonNullable<typeof hf.fieldsTex>,
    noiseA: hf.noiseA as NonNullable<typeof hf.noiseA>,
    noiseB: hf.noiseB as NonNullable<typeof hf.noiseB>,
    mp: hf.mp,
    far: true,
    baseNormalSlope: farNS,
  });
  farMat.colorNode = farShading.colorNode;
  farMat.normalNode = farShading.normalNode;
  farMat.roughnessNode = farShading.roughnessNode;
  farMat.metalnessNode = float(0);
  if (opts.gi && !ablate.has('gi')) {
    const farIrr = opts.gi.irradiance(positionWorld, farShading.worldNormalNode);
    (farMat as unknown as { setupLightMap: () => unknown }).setupLightMap = () =>
      new IrradianceNode(farIrr as unknown as ConstructorParameters<typeof IrradianceNode>[0]);
  }
  const farShell = new Mesh(ring, farMat);
  farShell.frustumCulled = false;
  farShell.receiveShadow = true;
  return farShell;
}
