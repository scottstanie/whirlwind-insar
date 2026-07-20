//! Spiral persistent-scatterer phase interpolator - a Rust port of dolphin's
//! `interpolation.interpolate` (numba). For each valid pixel (`ifg != 0`) whose
//! weight is below `weight_cutoff`, search outward in concentric circles
//! (nearest-first) for up to `num_neighbors` high-weight pixels and replace the
//! phase with a Gaussian-distance-weighted average of their unit phasors; the
//! amplitude is preserved. Pixels with `weight >= cutoff` pass through. Masked
//! `ifg == 0` pixels normally remain zero, or can be locally filled when the
//! caller explicitly requests it.

use ndarray::parallel::prelude::*;
use ndarray::{Array2, ArrayView2, Axis};
use num_complex::{Complex32, Complex64};

/// Relative `(dr, dc)` offsets of pixels in concentric circles out to
/// `max_radius` (excluding radii `<= min_radius`), in RADIUS-ASCENDING order via
/// the mid-point circle-drawing algorithm - a faithful port of dolphin's
/// `get_circle_idxs(..., sort_output=False)`. The ascending order is
/// load-bearing: the search collects the *nearest* high-weight pixels first, and
/// the last one collected sets the Gaussian bandwidth (`r2_norm`).
fn circle_idxs(max_radius: usize, min_radius: usize) -> Vec<(i32, i32)> {
    let mut visited = vec![false; max_radius * max_radius];
    let at = |x: usize, y: usize| x * max_radius + y;
    visited[at(0, 0)] = true;
    let mut idx: Vec<(i32, i32)> = Vec::new();
    for r in 1..max_radius {
        let ri = r as i32;
        let mut x = ri;
        let mut y = 0i32;
        let mut p = 1 - ri;
        if r > min_radius {
            idx.push((ri, 0));
            idx.push((-ri, 0));
            idx.push((0, ri));
            idx.push((0, -ri));
        }
        visited[at(r, 0)] = true;
        visited[at(0, r)] = true;
        let mut flag = 0i32;
        while x > y {
            if flag == 0 {
                y += 1;
                if p <= 0 {
                    p += 2 * y + 1;
                } else {
                    x -= 1;
                    p += 2 * y - 2 * x + 1;
                }
            } else {
                flag -= 1;
            }
            if x < y {
                break;
            }
            while !visited[at((x - 1) as usize, y as usize)] {
                x -= 1;
                flag += 1;
            }
            visited[at(x as usize, y as usize)] = true;
            visited[at(y as usize, x as usize)] = true;
            if r > min_radius {
                idx.push((x, y));
                idx.push((-x, -y));
                idx.push((x, -y));
                idx.push((-x, y));
                if x != y {
                    idx.push((y, x));
                    idx.push((-y, -x));
                    idx.push((y, -x));
                    idx.push((-y, x));
                }
            }
            if flag > 0 {
                x += 1;
            }
        }
    }
    idx
}

/// Interpolate `ifg` (complex64, `(m, n)`) using per-pixel `weights` in `[0, 1]`.
/// Returns a complex64 `(m, n)` array with the same amplitude everywhere and
/// interpolated phase at pixels with `weight < weight_cutoff`. See module docs.
///
/// `fill_invalid` controls what happens at pixels whose complex value is exactly
/// zero - the nodata convention shared with dolphin (`ifg != 0`). Normally they
/// are skipped and stay zero. Set it to fill them from surrounding valid phase
/// instead, which is what lets a small water body be smoothed over so the land
/// on either side integrates as one region rather than needing a bridge. Filled
/// pixels get UNIT amplitude: they had none to preserve, and only their phase is
/// meaningful to the unwrapper, but they must not come back as `0+0j` or every
/// downstream `!= 0` nodata test would still call them invalid.
///
/// The fill is self-limiting. A pixel is only filled if the spiral search finds
/// at least one high-weight neighbour within `max_radius`, so a narrow river or
/// a small lake fills while the middle of an ocean finds nothing and stays zero.
#[allow(clippy::too_many_arguments)]
pub fn interpolate(
    ifg: ArrayView2<Complex32>,
    weights: ArrayView2<f32>,
    weight_cutoff: f32,
    num_neighbors: usize,
    max_radius: usize,
    min_radius: usize,
    alpha: f64,
    fill_invalid: bool,
) -> Array2<Complex32> {
    let (nrow, ncol) = ifg.dim();
    assert_eq!(
        weights.dim(),
        (nrow, ncol),
        "ifg and weights must be same shape"
    );
    let nn = num_neighbors.max(1);
    let w = weights.mapv(|x| x.clamp(0.0, 1.0)); // dolphin clips weights to [0, 1]
    let idxs = circle_idxs(max_radius.max(1), min_radius);

    let mut out = Array2::<Complex32>::zeros((nrow, ncol));
    out.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(r0, mut row)| {
            // Per-row scratch, reused across columns (no per-pixel allocation).
            let mut r2 = vec![0f64; nn];
            let mut cphase = vec![Complex64::new(0.0, 0.0); nn];
            for c0 in 0..ncol {
                let v = ifg[(r0, c0)];
                let is_invalid = v.re == 0.0 && v.im == 0.0;
                if is_invalid && !fill_invalid {
                    continue; // masked / invalid pixel stays 0
                }
                if !is_invalid && w[(r0, c0)] >= weight_cutoff {
                    row[c0] = v; // high-weight pixel passes through unchanged
                    continue;
                }
                // Spiral outward (nearest-first) for high-weight neighbors.
                let mut counter = 0usize;
                for &(dr, dc) in &idxs {
                    let r = r0 as i32 + dr;
                    let c = c0 as i32 + dc;
                    if r < 0 || r >= nrow as i32 || c < 0 || c >= ncol as i32 {
                        continue;
                    }
                    if w[(r as usize, c as usize)] >= weight_cutoff {
                        r2[counter] = (dr * dr + dc * dc) as f64;
                        let ang = ifg[(r as usize, c as usize)].arg() as f64;
                        cphase[counter] = Complex64::from_polar(1.0, ang);
                        counter += 1;
                        if counter >= num_neighbors {
                            break;
                        }
                    }
                }
                // Filled nodata pixels have no amplitude to preserve; give them
                // a unit phasor so they read as valid to every downstream
                // `!= 0` nodata test. Otherwise keep the pixel's own amplitude.
                let amp = if is_invalid { 1.0 } else { v.norm() };
                if counter == 0 {
                    // No high-weight neighbour in range. A nodata pixel stays
                    // nodata (this is what keeps the fill local to small gaps);
                    // a valid pixel keeps amplitude with zero phase, as before.
                    if !is_invalid {
                        row[c0] = Complex32::new(amp, 0.0);
                    }
                    continue;
                }
                // Gaussian bandwidth set by the farthest collected neighbor.
                let r2_norm = r2[counter - 1].powf(alpha) / 2.0;
                let mut csum = Complex64::new(0.0, 0.0);
                for i in 0..counter {
                    csum += cphase[i] * (-r2[i] / r2_norm).exp();
                }
                let ang = csum.im.atan2(csum.re); // angle(csum); angle(0) == 0
                row[c0] = Complex32::new(amp * (ang.cos() as f32), amp * (ang.sin() as f32));
            }
        });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn passthrough_and_skip() {
        // 1x3: a masked pixel (0), a high-weight pixel (passes through), and a
        // low-weight pixel with one high-weight neighbor (gets that phase).
        let ifg = array![[
            Complex32::new(0.0, 0.0),
            Complex32::from_polar(2.0, 1.0),
            Complex32::from_polar(3.0, -2.0),
        ]];
        // (0,0) is low-weight too, so the only collected neighbor of (0,2) is the
        // high-weight (0,1); the neighbor check is on WEIGHT, not ifg!=0 (numba).
        let weights = array![[0.1f32, 0.9, 0.1]];
        let out = interpolate(ifg.view(), weights.view(), 0.5, 20, 10, 0, 0.75, false);
        // masked (ifg==0) stays 0
        assert_eq!(out[(0, 0)], Complex32::new(0.0, 0.0));
        // high-weight passes through
        assert!((out[(0, 1)] - ifg[(0, 1)]).norm() < 1e-5);
        // low-weight keeps amplitude 3, takes the neighbor's phase (+1.0 rad)
        assert!((out[(0, 2)].norm() - 3.0).abs() < 1e-4);
        assert!((out[(0, 2)].arg() - 1.0).abs() < 1e-3);
    }

    #[test]
    fn no_neighbors_keeps_amplitude() {
        let ifg = array![[Complex32::from_polar(5.0, 2.5)]];
        let weights = array![[0.1f32]];
        let out = interpolate(ifg.view(), weights.view(), 0.5, 20, 10, 0, 0.75, false);
        assert!((out[(0, 0)].norm() - 5.0).abs() < 1e-4);
        assert!(out[(0, 0)].im.abs() < 1e-5); // phase 0
    }

    #[test]
    fn fill_invalid_is_local_and_reports_support_with_nonzero_output() {
        let ifg = array![[
            Complex32::from_polar(2.0, 1.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
        ]];
        let weights = array![[0.9_f32, 0.0, 0.0, 0.0, 0.0]];
        let out = interpolate(ifg.view(), weights.view(), 0.5, 20, 3, 0, 0.75, true);

        // Search radii are 1..max_radius, so the two nearby nodata pixels fill
        // with unit amplitude and the source phase; farther pixels stay zero.
        for j in 1..=2 {
            assert!((out[(0, j)].norm() - 1.0).abs() < 1e-5);
            assert!((out[(0, j)].arg() - 1.0).abs() < 1e-5);
        }
        assert_eq!(out[(0, 3)], Complex32::new(0.0, 0.0));
        assert_eq!(out[(0, 4)], Complex32::new(0.0, 0.0));
    }

    #[test]
    fn circle_idxs_ring_ascending() {
        let idx = circle_idxs(6, 0);
        let d2: Vec<i32> = idx.iter().map(|(a, b)| a * a + b * b).collect();
        // nearest ring first, nothing beyond max_radius, and a clear upward trend
        // (within a ring d2 wiggles ±1, but rings are emitted in ascending order).
        assert_eq!(d2[0], 1);
        assert!(idx.iter().all(|(a, b)| a.abs() < 6 && b.abs() < 6));
        assert!(*d2.last().unwrap() > d2[0]);
        // ring grouping: the second half is strictly farther than the first ring.
        assert!(d2[d2.len() / 2..].iter().min().unwrap() > &1);
    }
}
