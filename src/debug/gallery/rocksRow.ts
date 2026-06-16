/** Gallery rock row: hero tor, boulders, scree, slab, cobbles, wall, dressed cliff. */

import { Mesh } from 'three';
import {
  barkTexturedMaterial,
  foliageCardMaterial,
  rockMaterial,
} from '../../render/VegMaterials';
import type { BarkTextures } from '../../gpu/passes/BarkSynth';
import { buildVines } from '../../vegetation/Dressing';
import { buildRock, type RockPreset } from '../../vegetation/RockBuilder';
import { buildFern, FERN_CAPTURE } from '../../vegetation/Understory';
import { ROW_Z, type GalleryContext } from './shared';

export async function buildRocksRow(g: GalleryContext): Promise<void> {
  const { engine, seed, atlases, barks, exhibit, progress } = g;
  progress(0.9, 'gallery: carving rocks');
  await new Promise((r) => setTimeout(r, 0));
  let rockTris = 0;
  const addRock = (
    preset: RockPreset,
    detail: number,
    seedTag: string,
    x: number,
    z: number,
    moss: number,
    label?: string,
  ): void => {
    const rock = buildRock(preset, seed.rng(`rock/${preset}/${seedTag}`), detail);
    rockTris += rock.stats.tris;
    const m = new Mesh(rock.geometry, rockMaterial({ moss }));
    // settle into the ground a bit
    const bs = rock.geometry.boundingSphere;
    m.position.set(x, bs ? bs.radius * 0.52 : 1, z);
    m.rotation.y = seed.rng(`rockrot/${preset}/${seedTag}`).float() * Math.PI * 2;
    m.castShadow = true;
    m.receiveShadow = true;
    engine.scene.add(m);
    if (label) {
      exhibit(x, z, label, `${(rock.stats.tris / 1000).toFixed(0)}k tris`);
    }
  };
  const RZ = ROW_Z.rocks;
  addRock('hero', 7, '0', -60, RZ, 0.35, 'Hero rock (tor)');
  for (let i = 0; i < 3; i++) addRock('boulder', 5, String(i), -38 + i * 9, RZ, 0.45, i === 0 ? 'Boulders ×3' : undefined);
  for (let i = 0; i < 3; i++) addRock('angular', 5, String(i), -8 + i * 8, RZ, 0.1, i === 0 ? 'Angular scree ×3' : undefined);
  addRock('slab', 6, '0', 18, RZ, 0.2, 'Slab');
  // cobble cluster
  for (let i = 0; i < 14; i++) {
    const r = seed.rng(`cob/${i}`);
    addRock('cobble', 3, String(i), 28 + r.float() * 5 - 2.5, RZ + r.float() * 4 - 2, 0.12);
  }
  exhibit(28, RZ, 'Cobbles', 'water-rounded');
  // rock wall: stacked slabs
  for (let row = 0; row < 3; row++) {
    for (let col = 0; col < 4; col++) {
      const r = seed.rng(`wall/${row}/${col}`);
      const rock = buildRock('slab', r.fork('s'), 5);
      rockTris += rock.stats.tris;
      const m = new Mesh(rock.geometry, rockMaterial({ moss: 0.3 }));
      m.position.set(44 + col * 2.6 + (row % 2) * 1.2, 0.8 + row * 1.35, RZ + (r.float() - 0.5));
      m.rotation.set((r.float() - 0.5) * 0.3, r.float() * 6.28, (r.float() - 0.5) * 0.3);
      m.scale.setScalar(0.85 + r.float() * 0.4);
      m.castShadow = true;
      m.receiveShadow = true;
      engine.scene.add(m);
    }
  }
  exhibit(48, RZ, 'Rock wall', 'stacked slabs');

  // dressed cliff: leaning slab + dirt streaks + hanging vines + ledge ferns
  {
    const cliffRock = buildRock('cliffFace', seed.rng('cliff/0'), 6);
    rockTris += cliffRock.stats.tris;
    const m = new Mesh(cliffRock.geometry, rockMaterial({ moss: 0.4 }));
    m.scale.set(1.5, 1.7, 1.2);
    m.position.set(72, 4.2, RZ);
    m.rotation.set(0.05, 0.2, -0.02);
    m.castShadow = true;
    m.receiveShadow = true;
    engine.scene.add(m);
    const vineRng = seed.rng('cliff/vines');
    const vines = buildVines(vineRng, 5.2, 4.2, 9);
    const stemMat = barkTexturedMaterial(barks.get(4) as BarkTextures);
    const vs = new Mesh(vines.stems, stemMat);
    vs.position.set(71.8, 7.6, RZ + 1.7);
    vs.castShadow = true;
    engine.scene.add(vs);
    const hazelAtlas = atlases.get('bushHazel');
    if (hazelAtlas) {
      const vl = new Mesh(
        vines.leaves,
        foliageCardMaterial(hazelAtlas, { color: { r: 0.05, g: 0.12, b: 0.035, hueVar: 0.25 } }),
      );
      vl.position.copy(vs.position);
      vl.castShadow = true;
      engine.scene.add(vl);
    }
    const fernAtlas2 = atlases.get('fern');
    if (fernAtlas2) {
      for (let i = 0; i < 2; i++) {
        const lf = new Mesh(
          buildFern(seed.rng(`cliff/fern${i}`)),
          foliageCardMaterial(fernAtlas2, { color: FERN_CAPTURE.foliageColor }),
        );
        lf.position.set(70.4 + i * 2.2, 1.5 + i * 1.3, RZ + 1.75 - i * 0.35);
        lf.castShadow = true;
        engine.scene.add(lf);
      }
    }
    exhibit(72, RZ + 5, 'Dressed cliff', 'streaks+vines+ledge ferns', { pedestal: false });
  }
  engine.stats.counters['rock.tris'] = rockTris;
}
