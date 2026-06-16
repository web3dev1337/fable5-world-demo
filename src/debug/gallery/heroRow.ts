/** Gallery hero row: mesh-foliage hero trees (>=100k tris) with base dressing. */

import { InstancedMesh, Mesh, PlaneGeometry } from 'three';
import {
  barkTexturedMaterial,
  foliageCardMaterial,
  foliageMaterial,
  mushroomMaterial,
} from '../../render/VegMaterials';
import type { BarkTextures } from '../../gpu/passes/BarkSynth';
import { litterMaterial, scatterInstances } from '../../vegetation/GroundCover';
import { buildMushroom } from '../../vegetation/Dressing';
import { TREE_SPECIES } from '../../vegetation/Species';
import { buildTree } from '../../vegetation/TreeBuilder';
import { ROW_Z, type GalleryContext } from './shared';

export async function buildHeroRow(g: GalleryContext): Promise<void> {
  const { engine, seed, atlases, barks, exhibit, progress } = g;
  progress(0.96, 'gallery: hero trees');
  await new Promise((r) => setTimeout(r, 0));
  {
    const HZ = ROW_Z.hero;
    const heroSpecs = [TREE_SPECIES[0], TREE_SPECIES[2]];
    let hx = -14;
    for (const sp of heroSpecs) {
      if (!sp) continue;
      const built = buildTree(sp, seed.rng(`hero/${sp.id}`), { foliageMode: 'hybrid' });
      const at = exhibit(hx, HZ, `HERO ${sp.label}`, `${(built.stats.tris / 1000).toFixed(0)}k tris (mesh foliage)`);
      const bm = new Mesh(built.bark, barkTexturedMaterial(barks.get(sp.barkLayer) as BarkTextures));
      bm.position.set(at.x, 0.42, at.z);
      bm.castShadow = true;
      bm.receiveShadow = true;
      engine.scene.add(bm);
      const heroAtlas = atlases.get(sp.id);
      if (built.foliage && heroAtlas) {
        const fm = new Mesh(built.foliage, foliageCardMaterial(heroAtlas, { color: sp.foliageColor }));
        fm.position.copy(bm.position);
        fm.castShadow = true;
        fm.receiveShadow = true;
        engine.scene.add(fm);
      }
      if (built.foliageMesh) {
        const fm2 = new Mesh(built.foliageMesh, foliageMaterial({ color: sp.foliageColor }));
        fm2.position.copy(bm.position);
        fm2.castShadow = true;
        fm2.receiveShadow = true;
        engine.scene.add(fm2);
      }
      engine.stats.counters[`hero.${sp.id}`] = built.stats.tris;
      hx += 28;
    }
    // tree-base dressing: mushroom cluster + litter ring at the beech hero
    const mrng = seed.rng('fungi');
    for (let i = 0; i < 6; i++) {
      const mush = new Mesh(buildMushroom(mrng.fork(String(i)), 'cap'), mushroomMaterial());
      const a2 = mrng.float() * 6.28;
      const rr = 0.5 + mrng.float() * 1.3;
      mush.position.set(14 + Math.cos(a2) * rr, 0.42, HZ + Math.sin(a2) * rr);
      mush.castShadow = true;
      engine.scene.add(mush);
    }
    const beechAtlas2 = atlases.get('beech');
    if (beechAtlas2) {
      const lg = new PlaneGeometry(0.16, 0.16);
      lg.rotateX(-Math.PI / 2);
      const li = new InstancedMesh(lg, litterMaterial(beechAtlas2), 160);
      scatterInstances(li, seed.rng('hero/litter'), 5, 0.04, [0.8, 2.0], true);
      li.position.set(14, 0.44, HZ);
      engine.scene.add(li);
    }
  }
}
