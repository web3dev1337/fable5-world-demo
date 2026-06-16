/**
 * ?scene=gallery — specimen gallery (spec §4): every species × 3 seeds on
 * labeled pedestals, rock wall, dressed cliff, debris ground square. Primary
 * review surface for the Phase-4 macro–meso–micro audit. Full lighting/post
 * pipeline (sun/sky, CSM+PCSS, GTAO, TRAA, grade) so review = world shading.
 *
 * ?row=trees|rocks|ground|dead frames the camera on one exhibit row.
 *
 * Each exhibit row is built by a dedicated module in ./gallery; this file is a
 * thin orchestrator that prepares the shared context (sky, ground, atlases,
 * bark) once and runs the rows in order.
 */

import { CircleGeometry, Mesh, PlaneGeometry, Vector3 } from 'three';
import { MeshStandardNodeMaterial } from 'three/webgpu';
import { mix, positionWorld, smoothstep, texture, uv, vec3 } from 'three/tsl';
import type { NF, NV4 } from '../gpu/TSLTypes';
import { hash12 } from '../gpu/noise/NoiseTSL';
import type { DataTexture } from 'three';
import { bakeBarkTextures, type BarkTextures } from '../gpu/passes/BarkSynth';
import { PostStack } from '../render/PostStack';
import { setupSunShadows } from '../render/ShadowSetup';
import { updateSunUniforms } from '../render/VegMaterials';
import { SunSky } from '../sky/SunSky';
import { captureFoliageAtlas } from '../vegetation/FoliageCards';
import { TREE_SPECIES } from '../vegetation/Species';
import { FERN_CAPTURE, UNDERSTORY_SPECIES } from '../vegetation/Understory';
import type { WorldContext } from './Scenes';
import { createExhibit, ROW_Z, type GalleryContext } from './gallery/shared';
import { buildTreesRow } from './gallery/treesRow';
import { buildRocksRow } from './gallery/rocksRow';
import { buildGroundRow } from './gallery/groundRow';
import { buildDeadRow } from './gallery/deadRow';
import { buildHeroRow } from './gallery/heroRow';
import { buildImpostorRow } from './gallery/impostorRow';

export async function buildGalleryScene(ctx: WorldContext): Promise<void> {
  const { engine, params, seed } = ctx;
  const q = new URLSearchParams(window.location.search);

  ctx.progress(0.05, 'gallery: sky');
  const sunSky = new SunSky(engine, params.timeOfDay);
  await sunSky.init(engine.renderer);
  updateSunUniforms(sunSky.sun);

  setupSunShadows(sunSky.sun, engine.camera, undefined, {
    maxFar: 320,
    lightMargin: 90,
  });

  // ---- ground: neutral matte with a faint 5 m scale grid ---------------------
  const groundMat = new MeshStandardNodeMaterial();
  {
    const wxz = positionWorld.xz;
    const n = hash12(wxz.mul(0.71).floor()) as NF;
    const base = mix(
      vec3(0.085, 0.1, 0.06),
      vec3(0.12, 0.125, 0.085),
      n.mul(0.7).add(hash12(wxz.mul(0.093).floor()).mul(0.3)),
    );
    const gx = smoothstep(0.0, 0.06, wxz.x.div(5).fract().sub(0.5).abs());
    const gz = smoothstep(0.0, 0.06, wxz.y.div(5).fract().sub(0.5).abs());
    groundMat.colorNode = base.mul(gx.min(gz).mul(0.12).add(0.88));
    groundMat.roughness = 0.96;
  }
  const ground = new Mesh(new CircleGeometry(420, 64), groundMat);
  ground.rotation.x = -Math.PI / 2;
  ground.receiveShadow = true;
  engine.scene.add(ground);

  // ---- exhibit helper (labeled pedestals) ------------------------------------
  const exhibit = createExhibit(engine);

  // ---- foliage cluster atlases (captured once per species) -------------------
  ctx.progress(0.08, 'gallery: capturing foliage atlases');
  const atlases = new Map<string, DataTexture>();
  for (const sp of [...TREE_SPECIES, ...UNDERSTORY_SPECIES, FERN_CAPTURE]) {
    if (!sp.foliage) continue;
    atlases.set(
      sp.id,
      await captureFoliageAtlas(engine.renderer, sp, seed.rng(`cards/${sp.id}`)),
    );
  }

  // ---- bark textures (synthesized per species layer) -------------------------
  ctx.progress(0.09, 'gallery: synthesizing bark');
  const barks = new Map<number, BarkTextures>();
  for (const sp of TREE_SPECIES) {
    if (barks.has(sp.barkLayer)) continue;
    barks.set(
      sp.barkLayer,
      await bakeBarkTextures(engine.renderer, sp.barkLayer, seed.sub(`bark/${sp.barkLayer}`) % 977),
    );
  }
  if (q.get('view') === 'atlas') {
    // raw atlas inspection row behind the trees
    let ax = -30;
    for (const tex of atlases.values()) {
      const mat = new MeshStandardNodeMaterial();
      const t = texture(tex, uv() as never) as unknown as NV4;
      mat.colorNode = t.rgb.mul(t.rgb);
      mat.opacityNode = t.w;
      mat.alphaTest = 0.1;
      const plane = new Mesh(new PlaneGeometry(10, 10), mat);
      plane.position.set(ax, 6, -22);
      engine.scene.add(plane);
      ax += 12;
    }
  }

  const gallery: GalleryContext = {
    engine,
    seed,
    progress: ctx.progress,
    atlases,
    barks,
    exhibit,
  };

  await buildTreesRow(gallery);
  await buildRocksRow(gallery);
  await buildGroundRow(gallery);
  buildDeadRow(gallery);
  await buildHeroRow(gallery);
  await buildImpostorRow(gallery);

  // ---- post stack (no clouds in the gallery) ----------------------------------
  ctx.progress(0.98, 'gallery: post pipeline');
  const post = new PostStack(engine, sunSky.atmosphere, params.timeOfDay, null);
  engine.post = post;

  ctx.hooks.setTimeOfDay = (t: number) => {
    void (async () => {
      await sunSky.setTimeOfDay(t);
      updateSunUniforms(sunSky.sun);
      post.setTimeOfDay(t);
    })();
  };

  // ---- camera ------------------------------------------------------------------
  if (params.cam === null) {
    const row = (q.get('row') ?? 'trees') as keyof typeof ROW_Z;
    const z = ROW_Z[row] ?? 0;
    engine.camera.position.set(0, 13, z + 64);
    engine.camera.lookAt(new Vector3(0, 9, z));
  }
  engine.onUpdate(() => {
    if (engine.camera.position.y < 0.6) engine.camera.position.y = 0.6;
  });

  ctx.progress(1, 'gallery ready');
}
