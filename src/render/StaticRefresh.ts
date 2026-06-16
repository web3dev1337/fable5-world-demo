/**
 * Vegetation refresh control (Phase 8 perf).
 *
 * three's NodeMaterialObserver.needsRefresh() returns true for EVERY draw of a
 * TSL node material EVERY frame (the `hasNode` short-circuit), forcing a full
 * per-draw refresh (updateBefore + nodes.updateForRender + bindings.updateForRender).
 * Measured on an RTX 5090: ~700 color-pass draws, ALL refreshing every frame —
 * the dominant CPU-encode cost (the frame is submit-bound). Skipping ALL veg
 * refreshes lifts a frozen forest interior 74 → 105 fps (CPU submit 27 → 11 ms).
 *
 * Why it is safe to skip: a veg draw has NO per-object DYNAMIC state.
 *  - instance transforms live in storage buffers the GPU cull rewrites each
 *    frame; the draw reads the latest contents with no CPU refresh.
 *  - its object-group uniforms (model matrix = identity, the compacted-list
 *    base offset) are constant — uploaded once at build, never need re-upload.
 *  - everything time/camera varying (the TRAA-jittered camera, vegViewPos,
 *    wind, sun, `time`) is a SHARED frame/render-group uniform: ONE GPU buffer
 *    that three reuses across every draw that references the same uniform set.
 *
 * So per frame we only need ONE draw per shared bind group to refresh — it
 * uploads that buffer (with this frame's jitter/time) and, because it is the
 * FIRST such draw in render order, every later draw sharing the buffer reads the
 * fresh value. We refresh exactly those first-in-order draws and skip the rest.
 * Render order guarantees the flush happens before the readers, so the result
 * is pixel-identical to refreshing all of them — verified by A/B screenshots on
 * the RTX 5090 (diff at the run-to-run noise floor).
 *
 * Correctness does not depend on the renderGroup buffers actually being shared:
 * a draw whose shared buffers are unique (its own) simply always refreshes (it
 * is the sole sentinel for them). The win comes from veg draws SHARING those
 * buffers — see VegInstance/groundring where the per-draw `base` offset lives in
 * the object group so the renderGroup (camera + jitter) stays shared.
 *
 *   ?vrf=off  — disable (baseline three behaviour) for A/B verification.
 */

import type { WebGPURenderer } from 'three/webgpu';
import type { Object3D } from 'three';

const tagged = new WeakSet<object>();
let frameNo = 0;

/** per-frame tally of how many tagged veg/grass draws refreshed vs skipped
 *  (sentinels = first draw per shared buffer; skipped = read the flushed buffer).
 *  Published each tick for the HUD; cheap two-counter accounting. */
const live = { refresh: 0, skip: 0 };
let published = { refresh: 0, skip: 0 };

/** last completed frame's veg refresh/skip counts (HUD readout) */
export function vegRefreshStats(): { refresh: number; skip: number } {
  return published;
}

/** mark a draw object as refresh-managed (uniforms are all shared/static) */
export function markVegRefresh(obj: Object3D): void {
  tagged.add(obj);
}

/** advance the per-frame flush clock (call once per rendered frame, pre-render) */
export function tickVegRefresh(): void {
  frameNo++;
  published = { refresh: live.refresh, skip: live.skip };
  live.refresh = 0;
  live.skip = 0;
}

interface BindingShape {
  groupNode?: { shared?: boolean };
}
interface BindGroupShape {
  bindings?: BindingShape[];
}
interface ROShape {
  object: Object3D;
  // three's RenderObject caches its built bind groups here; null until the
  // first refresh built them. We READ it (never call getBindings(), which would
  // force an expensive node-builder-state build inside needsRefresh).
  _bindings: BindGroupShape[] | null;
  getMonitor(): MonitorShape;
}
interface MonitorShape {
  needsRefresh(ro: unknown, nf: unknown): boolean;
}
interface ObjectsShape {
  createRenderObject(...a: unknown[]): ROShape;
}

export function installVegRefreshControl(renderer: WebGPURenderer): void {
  if (new URLSearchParams(window.location.search).get('vrf') === 'off') return;

  // the SHARED bind groups of each tagged render object (cached after build)
  const sharedOf = new WeakMap<object, BindGroupShape[]>();
  // last frame each shared bind group was flushed by a sentinel
  const flushedAt = new WeakMap<object, number>();
  const built = new WeakSet<object>();

  const sharedGroups = (ro: ROShape): BindGroupShape[] | null => {
    const cached = sharedOf.get(ro);
    if (cached !== undefined) return cached;
    const b = ro._bindings;
    if (b == null) return null; // bindings not built yet — refresh this frame
    const groups = b.filter((g) => g.bindings?.[0]?.groupNode?.shared === true);
    sharedOf.set(ro, groups);
    return groups;
  };

  const objects = (renderer as unknown as { _objects: ObjectsShape })._objects;
  const oProto = Object.getPrototypeOf(objects) as ObjectsShape;
  const origCreate = oProto.createRenderObject;
  let patched = false;

  oProto.createRenderObject = function (
    this: ObjectsShape,
    ...args: unknown[]
  ): ROShape {
    const ro = origCreate.apply(this, args);
    if (!patched) {
      try {
        const mon = ro.getMonitor();
        patched = true;
        const mProto = Object.getPrototypeOf(mon) as MonitorShape;
        const orig = mProto.needsRefresh;
        mProto.needsRefresh = function (
          this: MonitorShape,
          robj: unknown,
          nf: unknown,
        ): boolean {
          const r = robj as ROShape;
          const obj = r.object;
          if (obj === undefined || !tagged.has(obj)) {
            return orig.call(this, robj, nf);
          }
          // always let the very first refresh through (builds node state +
          // bindings), and learn this draw's shared bind groups afterwards.
          if (!built.has(obj)) {
            built.add(obj);
            live.refresh++;
            return orig.call(this, robj, nf);
          }
          const shared = sharedGroups(r);
          if (shared === null || shared.length === 0) {
            live.refresh++;
            return orig.call(this, robj, nf); // unknown sharing → stay correct
          }
          // refresh iff this draw is the first in render order this frame to
          // touch one of its shared buffers (so it flushes the current
          // jitter/time/camera); otherwise the flush already happened — skip.
          let sentinel = false;
          for (const g of shared) {
            if (flushedAt.get(g) !== frameNo) {
              flushedAt.set(g, frameNo);
              sentinel = true;
            }
          }
          if (sentinel) {
            live.refresh++;
            return orig.call(this, robj, nf);
          }
          live.skip++;
          return false;
        };
      } catch {
        /* monitor not ready on this RO; retry next createRenderObject */
      }
    }
    return ro;
  };
}
