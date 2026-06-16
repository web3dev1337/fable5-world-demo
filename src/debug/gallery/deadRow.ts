/** Gallery dead row: logs in three decay states, shelf fungi, stumps. */

import { Mesh } from 'three';
import { deadwoodMaterial, mushroomMaterial } from '../../render/VegMaterials';
import type { BarkTextures } from '../../gpu/passes/BarkSynth';
import { buildLog, buildStump, type DecayState } from '../../vegetation/Deadfall';
import { buildMushroom } from '../../vegetation/Dressing';
import { ROW_Z, type GalleryContext } from './shared';

export function buildDeadRow(g: GalleryContext): void {
  const { engine, seed, barks, exhibit, progress } = g;
  progress(0.95, 'gallery: deadfall');
  const DZ = ROW_Z.dead;
  const spruceBark = barks.get(0) as BarkTextures;
  const decays: DecayState[] = ['fresh', 'mossy', 'rotten'];
  for (let i = 0; i < decays.length; i++) {
    const log = buildLog(seed.rng(`log/${i}`), decays[i] as DecayState);
    const m = new Mesh(log.geometry, deadwoodMaterial(spruceBark));
    m.position.set(-22 + i * 9, 0, DZ);
    // keep logs near-perpendicular to the row so they present their length
    m.rotation.y = (seed.rng(`logr/${i}`).float() - 0.5) * 0.8;
    m.castShadow = true;
    m.receiveShadow = true;
    engine.scene.add(m);
    exhibit(-22 + i * 9, DZ + 2.5, `Log (${decays[i]})`, `${log.length.toFixed(1)} m`, { pedestal: false });
  }
  {
    const shelfRng = seed.rng('shelf');
    for (let i = 0; i < 4; i++) {
      const sh = new Mesh(buildMushroom(shelfRng.fork(String(i)), 'shelf'), mushroomMaterial());
      sh.position.set(-13.6 + i * 0.5, 0.32 + (i % 2) * 0.12, DZ + 0.28);
      sh.rotation.z = Math.PI / 2 - 0.3;
      sh.rotation.y = -Math.PI / 2;
      sh.castShadow = true;
      engine.scene.add(sh);
    }
  }
  for (let i = 0; i < 2; i++) {
    const st = buildStump(seed.rng(`stump/${i}`));
    const m = new Mesh(st.geometry, deadwoodMaterial(spruceBark));
    m.position.set(8 + i * 6, 0, DZ);
    m.castShadow = true;
    m.receiveShadow = true;
    engine.scene.add(m);
  }
  exhibit(11, DZ + 2.5, 'Stumps ×2', 'root flare, jagged top', { pedestal: false });
}
