/**
 * Shared tuning constants for the stream/lake water material builders.
 */

/** clear alpine water: absorption per meter (r dies first → teal depths) */
export const SIGMA = { r: 0.42, g: 0.135, b: 0.095 };

/** flowmap cycles/s — shared by ripples, foam and the caustic advection */
export const FLOW_CYC = 0.45;
