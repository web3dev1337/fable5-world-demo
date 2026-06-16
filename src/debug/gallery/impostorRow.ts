/** Gallery impostor row: 8×8 octahedral capture demo (albedo+normal+depth). */

import { Mesh, PlaneGeometry } from 'three';
import type { BarkTextures } from '../../gpu/passes/BarkSynth';
import { TREE_SPECIES } from '../../vegetation/Species';
import { buildTree } from '../../vegetation/TreeBuilder';
import {
  captureImpostor,
  impostorPreviewMaterial,
  type ImpostorPart,
} from '../../vegetation/Impostors';
import { ROW_Z, type GalleryContext } from './shared';

export async function buildImpostorRow(g: GalleryContext): Promise<void> {
  const { engine, seed, atlases, barks, exhibit, progress } = g;
  progress(0.97, 'gallery: capturing impostors');
  await new Promise((r) => setTimeout(r, 0));
  {
    const sp = TREE_SPECIES[0];
    const atlas0 = sp ? atlases.get(sp.id) : undefined;
    if (sp && atlas0) {
      const built = buildTree(sp, seed.rng(`tree/${sp.id}/0`));
      const parts: ImpostorPart[] = [
        { geometry: built.bark, kind: 'bark', barkTex: barks.get(sp.barkLayer) as BarkTextures },
      ];
      if (built.foliage) parts.push({ geometry: built.foliage, kind: 'cards', atlas: atlas0 });
      const imp = await captureImpostor(engine.renderer, parts, {
        centerY: built.stats.height * 0.5,
        radius: built.stats.height * 0.62,
      });
      // preview cards: three captured views beside the real tree
      // side-on tiles (grid center is the zenith view in hemi-oct mapping)
      const views = [
        { gx: 7, gy: 4 },
        { gx: 4, gy: 7 },
        { gx: 6, gy: 6 },
      ];
      for (let i = 0; i < views.length; i++) {
        const card = new Mesh(
          new PlaneGeometry(imp.radius * 2, imp.radius * 2),
          impostorPreviewMaterial(imp, views[i] as { gx: number; gy: number }),
        );
        card.position.set(-150 - i * 0.01, imp.centerY + 0.42, ROW_Z.trees + i * 0.01);
        engine.scene.add(card);
      }
      exhibit(-150, ROW_Z.trees, 'Impostor preview', '8×8 oct capture', { pedestal: false });
    }
  }
}
