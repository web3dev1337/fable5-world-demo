/**
 * Post-processing stack (HDR, in order):
 *   scene pass (MRT: color, view normal, velocity, depth)
 *   → aerial perspective (Hillaire in-scatter from depth — Pillar D haze)
 *   → GTAO multiply (Phase-3 refines to indirect-only)
 *   → TRAA (temporal AA — the geometric density shimmers without it)
 *   → bloom (HDR threshold)
 *   → auto-exposure (GPU histogram-free log-average, smoothed, no readback)
 *   → filmic grade (per-ToD color script: white balance, teal–orange split
 *     toning, saturation, contrast) → AgX via renderer.toneMapping
 *
 * The constructor wires the pass graph; each stage's TSL node graph lives in a
 * dedicated builder under ./post (and the ?cloudview / ?skyveldbg debug-probe
 * ladders in ./post/postProbes).
 */

import { AgXToneMapping, Matrix4, NoToneMapping, Vector2, Vector3 } from 'three';
import type { Renderer, StorageBufferNode } from 'three/webgpu';
import { RenderPipeline } from 'three/webgpu';
import { bloom } from 'three/addons/tsl/display/BloomNode.js';
import { traa } from 'three/addons/tsl/display/TRAANode.js';
import {
  Fn,
  float,
  instancedArray,
  luminance,
  mrt,
  output,
  pass,
  uniform,
  vec3,
  vec4,
  velocity,
} from 'three/tsl';
import type { Engine } from '../core/Engine';
import { tagGpu } from '../core/GpuProfiler';
import type { Froxels } from '../gpu/passes/Froxels';
import type { NV2, NV4 } from '../gpu/TSLTypes';
import type { Atmosphere } from '../sky/Atmosphere';
import type { Clouds } from '../sky/Clouds';
import { GradeUniforms, gradeParamsAt } from './ColorScript';
import { runiform } from '../gpu/RenderUniform';
import { gtaoLayer } from './Gtao';
import { HalfResMrtNode, type HalfResEntry } from './HalfResMrt';
import { buildCloudLayer } from './post/cloudLayer';
import { buildBounceLayer } from './post/bounceLayer';
import { buildAerial } from './post/aerial';
import { buildAoUpsample } from './post/aoUpsample';
import { buildContactShadow } from './post/contactShadow';
import { buildVelReproject } from './post/velReproject';
import { buildExposureInit, buildExposureKernel } from './post/autoExposure';
import { buildGrade } from './post/grade';
import { buildSkyVelDbg } from './post/postProbes';

export class PostStack {
  readonly post: RenderPipeline;
  private grade = new GradeUniforms();
  private exposureBuf: StorageBufferNode<'float'>;
  private exposureKernel: Parameters<Renderer['compute']>[0];

  constructor(
    engine: Engine,
    atmosphere: Atmosphere,
    tod: number,
    clouds: Clouds | null = null,
    froxels: Froxels | null = null,
  ) {
    const { renderer, scene, camera } = engine;
    const q = new URLSearchParams(window.location.search);
    const cloudview = q.get('cloudview');
    // perf attribution: ?ablate=clouds,ao,taa,bloom disables stages
    const ablate = new Set((q.get('ablate') ?? '').split(','));
    // debug probes need raw values — tone mapping would garble them
    const skyveldbg = q.get('skyveldbg') !== null && q.get('skyveldbg') !== '';
    renderer.toneMapping = cloudview || skyveldbg ? NoToneMapping : AgXToneMapping;
    renderer.toneMappingExposure = 1.0;
    const frameU = runiform(0);
    // The post quad pass binds its own orthographic camera: `cameraPosition`,
    // `cameraWorldMatrix` and `cameraProjectionMatrixInverse` all resolve to
    // THAT camera inside outputNode (verified via ?cloudview probes — depth
    // reconstruction gave quad-local ~1 m distances). Feed the scene camera
    // explicitly, like three's own GTAO/TRAA nodes do.
    const uCamPos = runiform(new Vector3());
    const uProjInv = runiform(new Matrix4());
    const uCamWorld = runiform(new Matrix4());
    const uProj = runiform(new Matrix4());
    const uView = runiform(new Matrix4());
    // previous-frame view/projection — sky-pixel velocity for TRAA (see below)
    const uPrevView = runiform(new Matrix4());
    const uPrevProj = runiform(new Matrix4());
    // Synced at RENDER time, not in onUpdate: updateFns run in registration
    // order and the camera movers (FlyCamera, flythrough) mutate the camera
    // AFTER scene-built subsystems registered — an onUpdate copy here read a
    // ONE-FRAME-STALE pose during interactive motion, shifting clouds/aerial/
    // froxels/contact against the freshly-posed geometry every moved frame
    // (the user-visible "clouds shift with the camera" half of the cloud-lag
    // bug). render() runs after ALL updateFns — immune to registration order.
    let firstSync = true;
    this.syncCamera = (): void => {
      frameU.value = (frameU.value + 1) % 1024;
      camera.updateMatrixWorld(); // compose pending pose mutations NOW
      uPrevView.value.copy(firstSync ? camera.matrixWorldInverse : uView.value);
      uPrevProj.value.copy(firstSync ? camera.projectionMatrix : uProj.value);
      firstSync = false;
      uCamPos.value.copy(camera.position);
      uProjInv.value.copy(camera.projectionMatrixInverse);
      uCamWorld.value.copy(camera.matrixWorld);
      uProj.value.copy(camera.projectionMatrix);
      uView.value.copy(camera.matrixWorldInverse);
    };
    const camPosW = vec3(uCamPos);

    const scenePass = pass(scene, camera);
    // per-pass GPU profiler label (texture stays 'output' — getTextureNode
    // looks textures up by name)
    tagGpu(scenePass.renderTarget as object, 'scene');
    // ?postmin=1 — bisect probe: bare scene pass through the pipeline, no
    // MRT/effects. If shadows survive here but not in the full stack, an
    // effect node is the culprit; if they die here, pass() itself is.
    if (q.get('postmin') === '1') {
      if (q.get('postmrt') === '1') {
        scenePass.setMRT(mrt({ output, velocity }));
        this.post = new RenderPipeline(renderer);
        this.post.outputNode = scenePass.getTextureNode('output');
      } else {
        this.post = new RenderPipeline(renderer);
        this.post.outputNode = scenePass;
      }
      this.exposureBuf = instancedArray(2, 'float');
      this.exposureKernel = Fn(() => {})().compute(1);
      return;
    }
    // velocity MRT only for the ?skyveldbg diagnostic: TRAA consumes analytic
    // camera reprojection (see the TRAA section — the buffer is garbage for
    // positionNode-displaced geometry anyway), so the default path would
    // write+clear a full-res rg16f attachment nobody reads
    scenePass.setMRT(mrt(skyveldbg ? { output, velocity } : { output }));
    const beauty = scenePass.getTextureNode('output');
    const depthTex = scenePass.getTextureNode('depth');
    const velocityTex = skyveldbg ? scenePass.getTextureNode('velocity') : null;

    // --- merged half-res MRT pass: clouds march + GTAO + SS bounce -------------
    // These three layers ran as separate half-res passes (two RTTNodes + a
    // GTAONode) — three rasters, three encoders, three RT round-trips over
    // the same depth buffer. One MRT pass renders them together; consumers
    // sample per-attachment texture nodes exactly as before. The cloud march
    // is by far the most expensive screen-space work; half res quarters the
    // ray count and jitter + TRAA absorb the upsample.
    const halfEntries: HalfResEntry[] = [];
    if (clouds && !ablate.has('clouds')) {
      halfEntries.push({
        name: 'clouds',
        node: buildCloudLayer({ clouds, depthTex, uProjInv, uCamWorld, camPosW, frameU }),
      });
    }
    let halfAo: ReturnType<typeof uniform> | null = null;
    if (!ablate.has('ao')) {
      // resolution uniform is patched in below once the pass node exists
      halfAo = runiform(new Vector2(2, 2));
      halfEntries.push({
        name: 'ao',
        node: gtaoLayer(
          depthTex as unknown as Parameters<typeof gtaoLayer>[0],
          camera,
          halfAo,
          // GTAO defaults are mesh-viewer scale: 16 samples cost ~50 ms on
          // terrain vistas (Phase-2 finding) — 8 samples, 1.6 m radius
          { samples: 8, radius: 1.6, distanceFallOff: 0.6 },
        ),
      });
    }
    if (!ablate.has('bounce')) {
      halfEntries.push({
        name: 'bounce',
        node: buildBounceLayer({ depthTex, beauty, uProjInv }),
      });
    }
    let cloudTex: NV4 | null = null;
    let aoTexNode: NV4 | null = null;
    let bounceTex: NV4 | null = null;
    if (halfEntries.length > 0) {
      const halfPass = new HalfResMrtNode(halfEntries, 0.5);
      if (halfAo) {
        // the AO noise tiling must read the pass's true half-res dims
        halfAo.value = halfPass.resolution.value;
      }
      if (clouds && !ablate.has('clouds')) {
        cloudTex = halfPass.getTextureNode('clouds') as unknown as NV4;
      }
      if (!ablate.has('ao')) {
        aoTexNode = halfPass.getTextureNode('ao') as unknown as NV4;
      }
      if (!ablate.has('bounce')) {
        bounceTex = halfPass.getTextureNode('bounce') as unknown as NV4;
      }
    }

    // --- aerial perspective from depth -----------------------------------------
    const aerialNode = buildAerial({
      depthTex,
      beauty,
      uProjInv,
      uCamWorld,
      camPosW,
      atmosphere,
      froxels,
      clouds,
      cloudTex,
      cloudview,
      ablate,
    });

    // --- GTAO upsample (AO itself renders in the merged half-res pass) ----------
    const aoFaded = buildAoUpsample({ aoSrc: aoTexNode, depthTex, uProjInv, beauty });

    // --- screen-space contact shadows (spec §2 floor) ---------------------------
    const contactNode = buildContactShadow({ depthTex, uProjInv, atmosphere, uView, uProj, frameU });

    // aoFaded is null exactly when ablate.has('ao') (no merged-pass entry) —
    // preserving the old ablate semantics: AO ablation also drops contact
    const withAO =
      aoFaded === null
        ? aerialNode
        : aerialNode.mul(aoFaded).mul(ablate.has('contact') ? float(1) : contactNode);

    // --- screen-space bounce composite (layer renders in the merged pass) ------
    // Added back modulated by the receiver's chroma. Subtle by design: probes
    // carry the large-scale bounce; this adds the local green-on-trunk /
    // warm-on-rock bleed that probes are too coarse for. ?ablate=bounce.
    let withBounce = withAO;
    if (bounceTex !== null) {
      // receiver albedo proxy: scene color normalized by its own luminance
      const recLum = luminance(withAO.rgb).add(0.25);
      const recTint = withAO.rgb.div(recLum);
      withBounce = withAO.rgb.add(
        bounceTex.rgb.mul(recTint).mul(bounceTex.a).mul(0.16),
      ) as unknown as typeof withAO;
    }

    // --- TRAA ----------------------------------------------------------------------
    // TRAA's history reprojection consumed the velocity MRT, which is broken
    // here on BOTH ends (user bug "clouds lag the camera"; probe:
    // tools/probe-cloudlag.ts):
    //  - sky pixels rasterize no geometry → velocity = clear value 0 → under
    //    rotation, history blended clouds from the WRONG screen position at
    //    95% weight (sky-band diff 12.2% vs ablate=taa 0.2%);
    //  - geometry velocity is GARBAGE for everything positioned by custom
    //    positionNode shader displacement (terrain CDLOD morph, instanced
    //    veg, canopy shell — i.e. nearly every pixel): three's VelocityNode
    //    projects the raw undisplaced positionLocal, so the buffer reads
    //    |v|≈0.5-1 NDC with a STATIC camera (?skyveldbg=raw paints it) and
    //    TRAA rejected history (weight→1) on most geometry.
    // Fix: feed TRAA full analytic camera reprojection from each pixel's own
    // depth — exact for a static world INCLUDING translation parallax; the
    // far-plane limit covers sky (clouds at quasi-infinity) with no branch.
    // Object self-motion (wind sway, water) isn't captured and falls back to
    // variance clipping — same as before, but now with valid history.
    // Injected through the velocityNode.load() seam (TRAANode samples
    // velocity exactly once, at the closest-depth neighbor texel) and emitted
    // in VelocityNode's convention: ndcCur−ndcPrev in y-up NDC, which TRAA
    // maps to a uv delta via ×(0.5, −0.5). uv space is TOP-LEFT origin:
    // getViewPosition flips v internally (y.oneMinus,
    // PostProcessingUtils.js) so the forward projection must flip back,
    // exactly like three's getScreenPosition — without it the reprojection
    // is vertically MIRRORED (caught by ?skyveldbg: magenta zero-error
    // stripe on the mirror axis).
    const velReproject = buildVelReproject({ depthTex, uProjInv, uCamWorld, uPrevView, uPrevProj });
    const velLoad = (texel: NV2): NV4 =>
      vec4(velReproject(texel), 0, 1) as unknown as NV4;
    const reprojectedVelocity = { load: velLoad } as unknown as typeof depthTex;
    const taaed = ablate.has('taa')
      ? (withBounce as unknown as ReturnType<typeof traa>)
      : traa(withBounce, depthTex, reprojectedVelocity, camera);

    // --- bloom -----------------------------------------------------------------------
    const taaedRgb = (taaed as unknown as NV4).rgb;
    const withBloom = ablate.has('bloom')
      ? taaedRgb
      : taaedRgb.add((bloom(taaed, 0.28, 0.45, 1.5) as unknown as NV4).rgb);

    // --- auto exposure (GPU-only feedback) ----------------------------------------------
    this.exposureBuf = instancedArray(2, 'float');
    void renderer.computeAsync(buildExposureInit(this.exposureBuf));
    const beautyForMeter = scenePass.getTextureNode('output');
    this.exposureKernel = buildExposureKernel(this.exposureBuf, beautyForMeter);

    // --- grade ------------------------------------------------------------------------------
    const { node: graded, refresh } = buildGrade(this.grade, withBloom, this.exposureBuf);
    this.uniformsRefresh = refresh;

    // ?skyveldbg=err|raw|ana — TRAA velocity diagnostics over far geometry
    // (>1.5 km): `ana` paints the analytic camera reprojection (static
    // camera ⇒ black; this is what TRAA consumes), `raw` paints the velocity
    // MRT (broken for positionNode-displaced geometry — reads saturated even
    // static), `err` their difference. R = x ×20, G = y ×20, B = mask.
    const skyVelDbgView =
      skyveldbg && velocityTex
        ? buildSkyVelDbg({ velocityTex, depthTex, uProjInv, velReproject, mode: q.get('skyveldbg') })
        : null;

    this.post = new RenderPipeline(renderer);
    // chain bisect: 9 = constant at pipeline output, 8 = aerial only (no
    // AO/TRAA/bloom/exposure/grade), default = full chain
    this.post.outputNode =
      skyVelDbgView !== null ? skyVelDbgView
      : cloudview === '9' ? vec3(1, 0, 0)
      : cloudview !== null && cloudview !== '' ? aerialNode
      : graded;

    this.setTimeOfDay(tod);
  }

  private uniformsRefresh: () => void = () => undefined;
  private syncCamera: () => void = () => undefined;
  // ?lockexp=1 — freeze auto-exposure at its boot value: motion probes diff
  // frames across runs and the meter's adaptation transient otherwise
  // dominates the signal (probe-cloudlag pitch runs)
  private lockExposure = new URLSearchParams(window.location.search).get('lockexp') === '1';

  setTimeOfDay(tod: number): void {
    this.grade.apply(gradeParamsAt(tod));
    this.uniformsRefresh();
  }

  /** call once per frame after render — updates exposure feedback */
  meter(renderer: Renderer): void {
    if (this.lockExposure) return;
    renderer.compute(this.exposureKernel);
  }

  render(): void {
    this.syncCamera(); // after ALL updateFns — camera pose is final for this frame
    this.post.render();
  }
}
