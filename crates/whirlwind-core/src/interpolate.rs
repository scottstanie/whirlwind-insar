//! Spiral persistent-scatterer phase interpolator — a Rust port of dolphin's
//! `interpolation.interpolate` (numba). For each valid pixel (`ifg != 0`) whose
//! weight is below `weight_cutoff`, search outward in concentric circles
//! (nearest-first) for up to `num_neighbors` high-weight pixels and replace the
//! phase with a Gaussian-distance-weighted average of their unit phasors; the
//! amplitude is preserved. Pixels with `weight >= cutoff` (and masked `ifg == 0`)
//! pass through unchanged.

use ndarray::parallel::prelude::*;
use ndarray::{Array2, ArrayView2, Axis};
use num_complex::{Complex32, Complex64};

/// Relative `(dr, dc)` offsets of pixels in concentric circles out to
/// `max_radius` (excluding radii `<= min_radius`), in RADIUS-ASCENDING order via
/// the mid-point circle-drawing algorithm — a faithful port of dolphin's
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
#[allow(clippy::too_many_arguments)]
pub fn interpolate(
    ifg: ArrayView2<Complex32>,
    weights: ArrayView2<f32>,
    weight_cutoff: f32,
    num_neighbors: usize,
    max_radius: usize,
    min_radius: usize,
    alpha: f64,
) -> Array2<Complex32> {
    let (nrow, ncol) = ifg.dim();
    assert_eq!(weights.dim(), (nrow, ncol), "ifg and weights must be same shape");
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
                if v.re == 0.0 && v.im == 0.0 {
                    continue; // masked / invalid pixel stays 0
                }
                if w[(r0, c0)] >= weight_cutoff {
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
                let amp = v.norm(); // f32, == np.abs(complex64)
                if counter == 0 {
                    row[c0] = Complex32::new(amp, 0.0); // no neighbors: amplitude only
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
        let out = interpolate(ifg.view(), weights.view(), 0.5, 20, 10, 0, 0.75);
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
        let out = interpolate(ifg.view(), weights.view(), 0.5, 20, 10, 0, 0.75);
        assert!((out[(0, 0)].norm() - 5.0).abs() < 1e-4);
        assert!(out[(0, 0)].im.abs() < 1e-5); // phase 0
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
