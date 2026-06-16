/**
 * ShadowProxy — coarse terrain stand-in for the shadow cascades.
 *
 * The CDLOD tiles run 2.8M+ triangles; letting them cast re-rasterizes that
 * into all four CSM cascades (~11M tri-passes — the "terrain 20M tris" debt).
 * Mountain/ridge shadows only need macro shape, so a static 512² grid (8 m
 * quads, heights from the height buffer in the vertex stage) casts instead:
 * colorWrite/depthWrite off make its main-pass cost vertex-only, while the
 * shadow pass swaps in its depth material as usual. Near-field terrain
 * self-shadow detail below 8 m is covered by the screen-space contact
 * shadows. The real terrain keeps castShadow = false.
 */

import { Mesh } from 'three';
import { MeshStandardNodeMaterial } from 'three/webgpu';
import { positionLocal, vec2, vec3 } from 'three/tsl';
import type { Heightfield } from './Heightfield';
import { worldGridGeometry } from './GridMesh';

const GRID = 512;

export function buildTerrainShadowProxy(hf: Heightfield): Mesh {
  const geo = worldGridGeometry(GRID);

  const mat = new MeshStandardNodeMaterial();
  const lifted = vec3(
    positionLocal.x,
    hf.sampleHeight(vec2(positionLocal.x, positionLocal.z)),
    positionLocal.z,
  );
  mat.positionNode = lifted;
  (mat as unknown as { castShadowPositionNode: unknown }).castShadowPositionNode = lifted;
  mat.colorWrite = false;
  mat.depthWrite = false;
  mat.depthTest = false;

  const mesh = new Mesh(geo, mat);
  mesh.frustumCulled = false;
  mesh.castShadow = true;
  mesh.receiveShadow = false;
  return mesh;
}
