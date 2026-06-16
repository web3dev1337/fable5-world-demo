/**
 * Beer–Lambert depth-validated refraction. viewportSharedTexture sampled at a
 * ripple-refracted uv, depth-validated (samples landing on geometry IN FRONT
 * of the water fall back to the straight uv), absorbed by the water column
 * thickness from viewportDepthTexture, plus turbidity in-scatter tied to the
 * sky so it tracks time-of-day.
 *
 * Returns the transmitted color plus the water column thickness it derived —
 * foam/opacity reuse the thickness (see the caller's vDepth).
 */

import {
  cameraFar,
  cameraNear,
  clamp,
  exp,
  float,
  mix,
  perspectiveDepthToViewZ,
  screenUV,
  vec3,
  viewportDepthTexture,
  viewportSharedTexture,
} from 'three/tsl';
import type { Atmosphere } from '../../sky/Atmosphere';
import type { NF, NV3, NV4 } from '../../gpu/TSLTypes';
import { SIGMA } from './constants';

export interface RefractionResult {
  /** Beer–Lambert transmitted color */
  refr: NV3;
  /** water column thickness along the view ray (m) — shared with foam/opacity */
  thick: NF;
}

export function refraction(n: NV3, dist: NF, fragZ: NF, atm: Atmosphere): RefractionResult {
  // refraction uv: ripple-driven, shrinking with distance, depth-validated
  const refrK = clamp(float(9).div(dist.max(1)), 0.04, 1).mul(0.055);
  const ruv = screenUV.add(n.xz.mul(refrK));
  const zR = perspectiveDepthToViewZ(
    (viewportDepthTexture(ruv) as unknown as NV4).x,
    cameraNear,
    cameraFar,
  );
  const leaked = zR.greaterThan(fragZ.add(0.02)); // refr sample in FRONT of water
  const uvF = mix(ruv, screenUV, leaked.select(float(1), float(0)));
  const zScene = mix(
    zR,
    perspectiveDepthToViewZ(
      (viewportDepthTexture(screenUV) as unknown as NV4).x,
      cameraNear,
      cameraFar,
    ),
    leaked.select(float(1), float(0)),
  );
  const thick = fragZ.sub(zScene).max(0); // meters of water along the ray

  // ---- transmitted light --------------------------------------------------------
  const sceneCol = (viewportSharedTexture(uvF) as unknown as NV4).rgb;
  const absorb = thick.mul(1.25);
  const T = vec3(
    exp(absorb.mul(-SIGMA.r)),
    exp(absorb.mul(-SIGMA.g)),
    exp(absorb.mul(-SIGMA.b)),
  );
  // turbidity in-scatter follows the zenith sky → tracks time-of-day
  const inscat = atm.skyColor(vec3(0, 1, 0)).mul(vec3(0.013, 0.036, 0.032));
  const refr = sceneCol.mul(T).add(inscat.mul(vec3(1, 1, 1).sub(T)));
  return { refr, thick };
}
