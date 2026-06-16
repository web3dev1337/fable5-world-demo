/**
 * FPS soak — boot LAAS at a scene/bookmark (optionally the flythrough for
 * motion), wait until the world is ready, then measure the TRUE presented frame
 * rate by counting requestAnimationFrame ticks over `--soak` seconds.
 *
 * Why a rAF counter and not just window.__laas.stats.fps:
 *  - It measures frames the compositor actually presented, independent of how
 *    the engine computes its own EMA — the ground truth a user perceives.
 *  - It needs NO GPU timestamp-query, so it survives on adapters where the
 *    per-pass profiler (resolveTimestampsAsync) would crash (e.g. SwiftShader).
 * The engine's own stats (frameMs p95, draw calls, triangles, per-pass GPU
 * timings) are sampled alongside so you get both the perceived rate AND the
 * internal attribution in one run.
 *
 * Technique provenance: the rAF-soak loop is the same one used to profile the
 * meme-merge browser port headless (UpsideEngine) — boot → drive into a state →
 * count frames + observe churn/heap over a fixed soak. See docs/FPS_MEASUREMENT.md.
 *
 * Usage:
 *   npx tsx tools/fps-soak.ts --scene world --shot 3 --soak 20
 *   npx tsx tools/fps-soak.ts --scene world --fly --soak 30 --w 1920 --h 1080
 *   npx tsx tools/fps-soak.ts --scene world --cam "10,80,40,1.2,-0.1" --soak 15
 *
 * Note: headless on a GPU-less box falls back to SwiftShader (software) — the
 * number then reflects software rasterization, not real GPU performance. Run on
 * a machine with a real WebGPU adapter (or point a real browser at the dev
 * server) for a representative figure. The printed `adapter` field tells you
 * which backend produced the number.
 */

import { launchWebGPU, laasUrl } from './launch';

interface Args {
  [k: string]: string | boolean;
}

function parseArgs(argv: string[]): Args {
  const out: Args = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i] as string;
    if (a.startsWith('--')) {
      const key = a.slice(2);
      const next = argv[i + 1];
      if (next !== undefined && !next.startsWith('--')) {
        out[key] = next;
        i++;
      } else {
        out[key] = true;
      }
    }
  }
  return out;
}

function str(v: string | boolean | undefined): string | undefined {
  return typeof v === 'string' ? v : undefined;
}

interface SoakResult {
  elapsedSec: number;
  rafFps: number;
  frameMsAvg: number;
  frameMsP50: number;
  frameMsP95: number;
  frameMsP99: number;
  longFrames: number;
  engineFps: number | null;
  engineFrameMsP95: number | null;
  drawCalls: number | null;
  triangles: number | null;
  gpuPasses: Record<string, number> | null;
  heapMB: number | null;
  adapter: string | null;
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const width = Number(str(args['w']) ?? 1600);
  const height = Number(str(args['h']) ?? 900);
  const scene = str(args['scene']) ?? 'world';
  const soakSec = Number(str(args['soak']) ?? 20);
  const warmupSec = Number(str(args['warmup']) ?? 4);
  const timeoutMs = Number(str(args['timeout']) ?? 240000);

  const { browser } = await launchWebGPU();
  const page = await browser.newPage({
    viewport: { width, height },
    deviceScaleFactor: 1,
  });
  page.on('console', (msg) => {
    const t = msg.text();
    if (t.startsWith('[laas]') || msg.type() === 'error') console.log(`[page:${msg.type()}] ${t}`);
  });
  page.on('pageerror', (err) => console.error('[pageerror]', err.message));

  const urlOpts: Parameters<typeof laasUrl>[0] = { scene, width, height };
  if (args['seed'] !== undefined) urlOpts.seed = Number(str(args['seed']));
  if (args['T'] !== undefined) urlOpts.T = Number(str(args['T']));
  const cam = str(args['cam']);
  if (cam) urlOpts.cam = cam;
  const preset = str(args['preset']);
  if (preset) urlOpts.preset = preset;
  // measuring frame rate means the world must MOVE — never freeze
  urlOpts.freeze = false;
  const extra: Record<string, string> = {};
  const shot = str(args['shot']);
  if (shot) extra['shot'] = shot;
  if (args['fly'] === true) extra['fly'] = '1'; // run the 90s flythrough during the soak
  urlOpts.extra = extra;

  const url = laasUrl(urlOpts);
  console.log(`[soak] ${url} → ${soakSec}s soak (${warmupSec}s warmup) @ ${width}x${height}`);
  await page.goto(url, { waitUntil: 'domcontentloaded' });

  // wait for the world to finish generating + first frames
  await page.waitForFunction(
    () => window.__laas && (window.__laas.ready || window.__laas.error !== null),
    undefined,
    { timeout: timeoutMs, polling: 250 },
  );
  const err = await page.evaluate(() => window.__laas.error);
  if (err) {
    console.error(`[soak] FAILED: app reported error:\n${err}`);
    await browser.close();
    process.exit(1);
  }

  // let temporal effects (TRAA, auto-exposure) settle before counting
  await new Promise((r) => setTimeout(r, warmupSec * 1000));

  const result: SoakResult = await page.evaluate(async (sec: number) => {
    const L = window.__laas;
    const frameTimes: number[] = [];
    let frames = 0;
    let running = true;
    let last = performance.now();
    const tick = (): void => {
      const now = performance.now();
      frameTimes.push(now - last);
      last = now;
      frames++;
      if (running) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);

    const t0 = performance.now();
    await new Promise((r) => setTimeout(r, sec * 1000));
    running = false;
    const elapsed = (performance.now() - t0) / 1000;

    const sorted = frameTimes.slice().sort((a, b) => a - b);
    const pct = (p: number): number =>
      sorted.length ? (sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * p))] as number) : 0;
    const avg = frameTimes.length ? frameTimes.reduce((a, b) => a + b, 0) / frameTimes.length : 0;
    const mem = (performance as unknown as { memory?: { usedJSHeapSize: number } }).memory;
    const s = L.stats;
    return {
      elapsedSec: +elapsed.toFixed(2),
      rafFps: +(frames / elapsed).toFixed(1),
      frameMsAvg: +avg.toFixed(2),
      frameMsP50: +pct(0.5).toFixed(2),
      frameMsP95: +pct(0.95).toFixed(2),
      frameMsP99: +pct(0.99).toFixed(2),
      longFrames: frameTimes.filter((m) => m > 50).length,
      engineFps: s ? +s.fps.toFixed(1) : null,
      engineFrameMsP95: s ? +s.frameMsP95.toFixed(2) : null,
      drawCalls: s ? s.drawCalls : null,
      triangles: s ? s.triangles : null,
      gpuPasses: s ? s.gpuPasses : null,
      heapMB: mem ? +(mem.usedJSHeapSize / 1048576).toFixed(1) : null,
      adapter: L.diag ? `${L.diag.vendor ?? '?'} / ${L.diag.architecture ?? L.diag.description ?? '?'}` : null,
    };
  }, soakSec);

  console.log(JSON.stringify(result, null, 2));
  await browser.close();
}

main().catch((e: unknown) => {
  console.error('[soak] FAILED:', e instanceof Error ? e.message : String(e));
  process.exit(1);
});
