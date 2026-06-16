/** Gallery ground row: grass patches, debris square, ferns, flowers, shrubs. */

import { InstancedMesh, Mesh, PlaneGeometry } from 'three';
import {
  barkTexturedMaterial,
  flowerMaterial,
  foliageCardMaterial,
  rockMaterial,
} from '../../render/VegMaterials';
import type { BarkTextures } from '../../gpu/passes/BarkSynth';
import { buildRock } from '../../vegetation/RockBuilder';
import {
  barkChipGeometry,
  debrisMaterial,
  grassPatch,
  litterMaterial,
  scatterInstances,
  twigGeometry,
} from '../../vegetation/GroundCover';
import {
  buildFern,
  buildFlower,
  buildShrub,
  FERN_CAPTURE,
  type FlowerKind,
  UNDERSTORY_SPECIES,
} from '../../vegetation/Understory';
import { ROW_Z, type GalleryContext } from './shared';

export async function buildGroundRow(g: GalleryContext): Promise<void> {
  const { engine, seed, atlases, barks, exhibit, progress } = g;
  progress(0.93, 'gallery: ground cover');
  await new Promise((r) => setTimeout(r, 0));
  const GZ = ROW_Z.ground;
  {
    // meadow squares
    const small = grassPatch(seed.rng('grass/small'), 60_000, 6);
    small.position.set(-62, 0, GZ);
    engine.scene.add(small);
    exhibit(-62, GZ + 4, 'Grass 6×6 m', '60k blades', { pedestal: false });
    const meadow = grassPatch(seed.rng('grass/meadow'), 200_000, 13);
    meadow.position.set(-44, 0, GZ);
    engine.scene.add(meadow);
    exhibit(-44, GZ + 7.5, 'Meadow 13×13 m', '200k blades', { pedestal: false });

    // 2×2 m ground debris square
    const sq = 2.4;
    const cobbleGeo = buildRock('cobble', seed.rng('gc/cob'), 2).geometry;
    const cobInst = new InstancedMesh(cobbleGeo, rockMaterial({ moss: 0.08 }), 42);
    scatterInstances(cobInst, seed.rng('gc/cobs'), sq, 0.02, [0.35, 1.0], true);
    cobInst.position.set(-26, 0.05, GZ);
    engine.scene.add(cobInst);
    const pebInst = new InstancedMesh(
      buildRock('cobble', seed.rng('gc/peb'), 1).geometry,
      rockMaterial({ moss: 0 }),
      260,
    );
    scatterInstances(pebInst, seed.rng('gc/pebs'), sq, 0.02, [0.16, 0.5], true);
    pebInst.position.set(-26, 0.03, GZ);
    engine.scene.add(pebInst);
    for (let v = 0; v < 3; v++) {
      const tw = new InstancedMesh(
        twigGeometry(seed.rng(`gc/twig${v}`)),
        debrisMaterial('twig'),
        36,
      );
      scatterInstances(tw, seed.rng(`gc/twigs${v}`), sq, 0.03, [0.6, 1.5], true);
      tw.position.set(-26, 0.02, GZ);
      engine.scene.add(tw);
      const ch = new InstancedMesh(
        barkChipGeometry(seed.rng(`gc/chip${v}`)),
        debrisMaterial('chip'),
        45,
      );
      scatterInstances(ch, seed.rng(`gc/chips${v}`), sq, 0.02, [0.5, 1.3], true);
      ch.position.set(-26, 0.015, GZ);
      engine.scene.add(ch);
    }
    // leaf litter: quads with the beech atlas, browned
    const beechAtlas = atlases.get('beech');
    if (beechAtlas) {
      const litterGeo = new PlaneGeometry(0.16, 0.16);
      litterGeo.rotateX(-Math.PI / 2);
      // random tile uvs per instance need per-instance offset — bake 4 variants
      for (let v = 0; v < 4; v++) {
        const lg = litterGeo.clone();
        const uvA = lg.getAttribute('uv');
        for (let i = 0; i < uvA.count; i++) {
          uvA.setXY(i, uvA.getX(i) * 0.5 + (v % 2) * 0.5, uvA.getY(i) * 0.5 + Math.floor(v / 2) * 0.5);
        }
        const li = new InstancedMesh(lg, litterMaterial(beechAtlas), 90);
        scatterInstances(li, seed.rng(`gc/lit${v}`), sq, 0.05, [0.7, 1.8], true);
        li.position.set(-26, 0.03, GZ);
        engine.scene.add(li);
      }
    }
    exhibit(-26, GZ + 2, 'Ground square 2.4 m', 'cobbles+twigs+chips+litter', { pedestal: false });

    // ferns
    const fernAtlas = atlases.get('fern');
    if (fernAtlas) {
      for (let i = 0; i < 3; i++) {
        const fern = new Mesh(
          buildFern(seed.rng(`fern/${i}`)),
          foliageCardMaterial(fernAtlas, { color: FERN_CAPTURE.foliageColor }),
        );
        fern.position.set(-12 + i * 3, 0.02, GZ + (i % 2));
        fern.castShadow = true;
        fern.receiveShadow = true;
        engine.scene.add(fern);
      }
      exhibit(-9, GZ + 2.5, 'Ferns ×3', 'frond rosettes', { pedestal: false });
    }

    // flower patches
    const flowerKinds: { kind: FlowerKind; color: { r: number; g: number; b: number }; n: number; label: string }[] = [
      { kind: 'umbel', color: { r: 0.75, g: 0.75, b: 0.7 }, n: 14, label: 'White umbel' },
      { kind: 'bell', color: { r: 0.28, g: 0.14, b: 0.5 }, n: 14, label: 'Bellflower' },
      { kind: 'daisy', color: { r: 0.85, g: 0.72, b: 0.12 }, n: 18, label: 'Yellow daisy' },
    ];
    let fx = 0;
    for (const fk of flowerKinds) {
      const rngF = seed.rng(`flower/${fk.kind}`);
      for (let i = 0; i < fk.n; i++) {
        const fl = new Mesh(buildFlower(fk.kind, rngF.fork(String(i))), flowerMaterial(fk.color));
        fl.position.set(fx + (rngF.float() - 0.5) * 2.4, 0, GZ + (rngF.float() - 0.5) * 2.4);
        fl.rotation.y = rngF.float() * 6.28;
        fl.castShadow = true;
        engine.scene.add(fl);
      }
      exhibit(fx, GZ + 2, fk.label, `×${fk.n}`, { pedestal: false });
      fx += 7;
    }

    // shrubs ×3 (incl. pink flowering)
    let sx = 26;
    for (const sp of UNDERSTORY_SPECIES) {
      const shrub = buildShrub(sp, seed.rng(`shrub/${sp.id}`));
      const bm = new Mesh(shrub.bark, barkTexturedMaterial(barks.get(sp.barkLayer) as BarkTextures));
      bm.position.set(sx, 0, GZ);
      bm.castShadow = true;
      bm.receiveShadow = true;
      engine.scene.add(bm);
      const at = atlases.get(sp.id);
      if (shrub.foliage && at) {
        const fm = new Mesh(shrub.foliage, foliageCardMaterial(at, { color: sp.foliageColor }));
        fm.position.copy(bm.position);
        fm.castShadow = true;
        fm.receiveShadow = true;
        engine.scene.add(fm);
      }
      exhibit(sx, GZ + 2.5, sp.label, 'multi-stem', { pedestal: false });
      sx += 9;
    }
  }
}
