import type { Camera } from 'three';

const MATRIX_SIGNATURE_WORDS = 32;

export function createCameraSignature(): Float64Array {
  return new Float64Array(MATRIX_SIGNATURE_WORDS);
}

export function cameraSignatureChanged(camera: Camera, signature: Float64Array): boolean {
  const proj = camera.projectionMatrix.elements;
  const view = camera.matrixWorldInverse.elements;
  let changed = false;
  for (let i = 0; i < 16; i++) {
    const p = proj[i] ?? 0;
    const v = view[i] ?? 0;
    if (signature[i] !== p) {
      signature[i] = p;
      changed = true;
    }
    const j = i + 16;
    if (signature[j] !== v) {
      signature[j] = v;
      changed = true;
    }
  }
  return changed;
}
