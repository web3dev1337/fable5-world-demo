/** Gallery tree row: 6 species × 3 seeds on labeled pedestals. */

import { Mesh } from 'three';
import { barkTexturedMaterial, foliageCardMaterial } from '../../render/VegMaterials';
import type { BarkTextures } from '../../gpu/passes/BarkSynth';
import { TREE_SPECIES } from '../../vegetation/Species';
import { buildTree } from '../../vegetation/TreeBuilder';
import { ROW_Z, type GalleryContext } from './shared';

export async function buildTreesRow(g: GalleryContext): Promise<void> {
  const { engine, seed, atlases, barks, exhibit, progress } = g;
  let totalTris = 0;
  const spacing = 13;
  const groupGap = 6;
  const nSpecies = TREE_SPECIES.length;
  const rowWidth = nSpecies * 3 * spacing + (nSpecies - 1) * groupGap;
  let x = -rowWidth / 2;
  for (let si = 0; si < nSpecies; si++) {
    const sp = TREE_SPECIES[si];
    if (!sp) continue;
    for (let vi = 0; vi < 3; vi++) {
      progress(
        0.1 + (0.8 * (si * 3 + vi)) / (nSpecies * 3),
        `gallery: growing ${sp.id} #${vi}`,
      );
      // yield so boot UI can paint between heavy builds
      await new Promise((r) => setTimeout(r, 0));
      const rng = seed.rng(`tree/${sp.id}/${vi}`);
      const built = buildTree(sp, rng);
      totalTris += built.stats.tris;
      const at = exhibit(
        x,
        ROW_Z.trees,
        sp.label,
        `seed ${vi} · ${(built.stats.tris / 1000).toFixed(0)}k tris · ${built.stats.height.toFixed(1)} m`,
      );
      const barkTex = barks.get(sp.barkLayer) as BarkTextures;
      const barkMesh = new Mesh(built.bark, barkTexturedMaterial(barkTex));
      barkMesh.position.set(at.x, 0.42, at.z);
      barkMesh.castShadow = true;
      barkMesh.receiveShadow = true;
      engine.scene.add(barkMesh);
      const atlas = atlases.get(sp.id);
      if (built.foliage && atlas) {
        const folMesh = new Mesh(
          built.foliage,
          foliageCardMaterial(atlas, { color: sp.foliageColor }),
        );
        folMesh.position.copy(barkMesh.position);
        folMesh.castShadow = true;
        folMesh.receiveShadow = true;
        engine.scene.add(folMesh);
      }
      x += spacing;
    }
    x += groupGap;
  }
  engine.stats.counters['veg.tris'] = totalTris;
}
