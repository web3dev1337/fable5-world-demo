/**
 * Shared helpers, layout constants, and per-build context for the specimen
 * gallery rows (?scene=gallery). Row builders live in ./*Row.ts and receive a
 * GalleryContext assembled once by buildGalleryScene.
 */

import {
  CanvasTexture,
  CylinderGeometry,
  Mesh,
  PlaneGeometry,
  SRGBColorSpace,
} from 'three';
import type { DataTexture } from 'three';
import { MeshStandardNodeMaterial } from 'three/webgpu';
import { float, positionWorld, vec3 } from 'three/tsl';
import { hash12 } from '../../gpu/noise/NoiseTSL';
import type { BarkTextures } from '../../gpu/passes/BarkSynth';
import type { Engine } from '../../core/Engine';
import type { WorldSeed } from '../../core/Seed';

export const ROW_Z = { hero: -26, trees: 0, rocks: 40, ground: 70, dead: 100 } as const;

export type ExhibitFn = (
  x: number,
  z: number,
  title: string,
  sub: string,
  opts?: { pedestal?: boolean },
) => { x: number; z: number };

/** Shared state every gallery row reads, built once by buildGalleryScene. */
export interface GalleryContext {
  engine: Engine;
  seed: WorldSeed;
  progress: (p: number, msg: string) => void;
  atlases: Map<string, DataTexture>;
  barks: Map<number, BarkTextures>;
  exhibit: ExhibitFn;
}

export function labelSprite(text: string, sub: string): Mesh {
  const cv = document.createElement('canvas');
  cv.width = 512;
  cv.height = 144;
  const c = cv.getContext('2d');
  if (c) {
    c.fillStyle = 'rgba(20,24,28,0.92)';
    c.fillRect(0, 0, 512, 144);
    c.fillStyle = '#e8eef2';
    c.font = '600 44px system-ui, sans-serif';
    c.fillText(text, 18, 58);
    c.fillStyle = '#9fb2bf';
    c.font = '400 32px system-ui, sans-serif';
    c.fillText(sub, 18, 110);
  }
  const tex = new CanvasTexture(cv);
  tex.colorSpace = SRGBColorSpace;
  const mat = new MeshStandardNodeMaterial();
  mat.map = tex;
  mat.roughness = 0.9;
  const m = new Mesh(new PlaneGeometry(2.6, 0.73), mat);
  return m;
}

/** Build the labeled-pedestal placement helper, owning its shared geo/material. */
export function createExhibit(engine: Engine): ExhibitFn {
  const pedestalMat = new MeshStandardNodeMaterial();
  pedestalMat.colorNode = vec3(0.32, 0.31, 0.3).mul(
    hash12(positionWorld.xz.mul(31)).mul(0.15).add(float(0.85)),
  );
  pedestalMat.roughness = 0.88;
  const pedestalGeo = new CylinderGeometry(2.0, 2.3, 0.42, 28);

  return (
    x: number,
    z: number,
    title: string,
    sub: string,
    opts?: { pedestal?: boolean },
  ): { x: number; z: number } => {
    if (opts?.pedestal !== false) {
      const ped = new Mesh(pedestalGeo, pedestalMat);
      ped.position.set(x, 0.21, z);
      ped.receiveShadow = true;
      ped.castShadow = true;
      engine.scene.add(ped);
      const label = labelSprite(title, sub);
      label.position.set(x, 0.62, z + 2.45);
      label.rotation.x = -0.42;
      engine.scene.add(label);
    } else {
      // floating label behind the exhibit (never occludes it)
      const label = labelSprite(title, sub);
      label.position.set(x, 2.3, z - 4.6);
      engine.scene.add(label);
    }
    return { x, z };
  };
}
