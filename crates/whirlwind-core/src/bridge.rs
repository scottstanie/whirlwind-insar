//! Integration-component gauge bridging post-pass.
//!
//! Sets the relative 2π integer offset between the disconnected valid regions an
//! MCF integrator seeds independently (for example two land slabs separated by a
//! low-coherence river). This is a Rust port of [`whirlwind._bridge`], itself a
//! port of the algorithm isce3's NISAR GUNW workflow uses
//! (`isce3.unwrap.bridge_phase.bridge_unwrapped_phase`):
//!
//!  1. Label the integration regions (4-connected components of the valid mask).
//!  2. For every pair of regions, find the closest boundary-pixel pair.
//!  3. Build a minimum spanning tree of those distances, rooted at the largest
//!     region, so each region is referenced through its nearest neighbour.
//!  4. Walking the tree outward from the root, compare the median unwrapped phase
//!     in a local box around the two bridge endpoints, round the difference to an
//!     integer number of cycles, and shift the child region (and, transitively,
//!     its descendants).
//!
//! Keeping the Rust and Python implementations in lockstep is what lets the CLI
//! reach parity with the Python `unwrap` without shelling out to Python.

use crate::tile::label_components;
use ndarray::{Array2, ArrayView2};
use std::collections::HashMap;

/// Default endpoint-median box half-width (matches `whirlwind._bridge`).
pub const DEFAULT_RADIUS: usize = 500;
/// Default minimum region size, in pixels, to participate in bridging.
pub const DEFAULT_MIN_PX: usize = 500;
/// Default cap on boundary pixels sampled per region for the nearest-pair search.
pub const DEFAULT_MAX_BOUNDARY: usize = 2000;

/// Re-level the disconnected regions of an unwrapped phase image.
///
/// The offset is read straight from the unwrapped phase at the region
/// boundaries, so this needs only the unwrapped phase and a valid mask - no
/// coherence or interferogram.
///
/// * `unw` - unwrapped phase, masked/nodata pixels left at 0 (or NaN).
/// * `mask` - valid-pixel mask defining the integration regions. When `None`,
///   defaults to the finite, nonzero pixels of `unw`.
/// * `radius` - half-width of the box around each bridge endpoint over which the
///   region's local phase level is taken (clamped to a scene-relative size).
/// * `min_px` - ignore integration regions smaller than this many pixels.
/// * `max_boundary` - cap on boundary pixels sampled per region.
///
/// A single-region (or coherently connected) frame is returned unchanged.
pub fn bridge_components(
    unw: ArrayView2<f32>,
    mask: Option<ArrayView2<bool>>,
    radius: usize,
    min_px: usize,
    max_boundary: usize,
) -> Array2<f32> {
    let tau32 = std::f32::consts::TAU;
    let tau64 = std::f64::consts::TAU;
    let (m, n) = unw.dim();

    // Default mask = finite & nonzero (masked/nodata left at 0 or NaN).
    let mask_owned: Array2<bool> = match mask {
        Some(mk) => mk.to_owned(),
        None => unw.mapv(|v| v.is_finite() && v != 0.0),
    };

    // Integration components = 4-connected components of the valid mask, the same
    // partition the MCF integrator walks in `integrate_with_mask`.
    let (region, sizes_vec) = label_components(&mask_owned);
    let n_region = sizes_vec.len();
    if n_region <= 1 {
        return unw.to_owned(); // single integration component -> structural no-op
    }

    // sizes[label] = pixel count; label in 1..=n_region (label 0 = background).
    let mut sizes = vec![0usize; n_region + 1];
    for (lab, &s) in sizes_vec.iter().enumerate() {
        sizes[lab + 1] = s;
    }

    let big: Vec<i32> = (1..=n_region as i32)
        .filter(|&lab| sizes[lab as usize] >= min_px)
        .collect();
    if big.len() <= 1 {
        return unw.to_owned(); // nothing sizeable to bridge
    }

    // Largest region = MST root.
    let ref_lab = *big
        .iter()
        .max_by_key(|&&lab| sizes[lab as usize])
        .expect("big is non-empty");
    let ref_idx = big.iter().position(|&l| l == ref_lab).unwrap();

    let bcoords = boundary_coords(&region, &big, max_boundary);

    // Complete graph of closest-boundary distances; remember the endpoint pair
    // (parent-side first, stored for a<b) for each edge. K is small.
    let k = big.len();
    let mut dist = vec![vec![f64::INFINITY; k]; k];
    let mut endpts: HashMap<(usize, usize), ((f64, f64), (f64, f64))> = HashMap::new();
    for a in 0..k {
        let bi = &bcoords[&big[a]];
        for b in (a + 1)..k {
            let bj = &bcoords[&big[b]];
            // First minimum in (bi-major, bj-minor) order, matching np.argmin.
            let mut best_d2 = f64::INFINITY;
            let mut best_pair = ((0.0, 0.0), (0.0, 0.0));
            for &(yi, xi) in bi {
                for &(yj, xj) in bj {
                    let d2 = (yi - yj) * (yi - yj) + (xi - xj) * (xi - xj);
                    if d2 < best_d2 {
                        best_d2 = d2;
                        best_pair = ((yi, xi), (yj, xj));
                    }
                }
            }
            if best_d2.is_finite() {
                let d = best_d2.sqrt();
                dist[a][b] = d;
                dist[b][a] = d;
                endpts.insert((a, b), best_pair);
            }
        }
    }

    // Prim's MST rooted at the reference; record edges in growth order so a
    // parent is always already corrected when its child is processed.
    let mut in_tree = vec![false; k];
    in_tree[ref_idx] = true;
    let mut edges: Vec<(usize, usize)> = Vec::new(); // (parent_idx, child_idx)
    for _ in 0..(k - 1) {
        let mut best: Option<(f64, usize, usize)> = None;
        for u in 0..k {
            if !in_tree[u] {
                continue;
            }
            for v in 0..k {
                if in_tree[v] || !dist[u][v].is_finite() {
                    continue;
                }
                if best.is_none() || dist[u][v] < best.unwrap().0 {
                    best = Some((dist[u][v], u, v));
                }
            }
        }
        match best {
            Some((_, u, v)) => {
                in_tree[v] = true;
                edges.push((u, v));
            }
            None => break, // graph not fully connected (shouldn't happen for a clique)
        }
    }

    let mut out = unw.to_owned();
    // Cap the endpoint-median box to a scene-relative size so the window is
    // ~500 px on a NISAR-sized frame but shrinks on small frames; a box that
    // grows to a large fraction of the frame reintroduces within-region ramp.
    let r = radius.min((m.min(n) / 8).max(16));
    for &(u_idx, v_idx) in &edges {
        // Recover (parent endpoint, child endpoint) from the a<b storage order.
        let (yx_par, yx_chi) = if u_idx < v_idx {
            let (p, c) = endpts[&(u_idx, v_idx)];
            (p, c)
        } else {
            let (c, p) = endpts[&(v_idx, u_idx)];
            (p, c)
        };
        let par_lab = big[u_idx];
        let chi_lab = big[v_idx];

        let val_par = endpoint_median(&out, &region, par_lab, yx_par, r);
        let val_chi = endpoint_median(&out, &region, chi_lab, yx_chi, r);
        if !(val_par.is_finite() && val_chi.is_finite()) {
            continue;
        }
        // cycles to add to the child (round-half-to-even, matching np.rint).
        let s = -(libm::rint((val_chi - val_par) / tau64) as i64);
        if s != 0 {
            let shift = tau32 * s as f32;
            for ((i, j), &rr) in region.indexed_iter() {
                if rr == chi_lab {
                    out[(i, j)] += shift;
                }
            }
        }
    }
    out
}

/// Strided boundary-pixel `(y, x)` coordinates for each region label.
///
/// A boundary pixel is a valid pixel with a 4-neighbour of a different label.
/// Each set is strided down to at most `max_boundary` points (raster order).
fn boundary_coords(
    region: &Array2<i32>,
    labels: &[i32],
    max_boundary: usize,
) -> HashMap<i32, Vec<(f64, f64)>> {
    let (h, w) = region.dim();
    let mut is_boundary = Array2::<bool>::from_elem((h, w), false);
    for i in 0..h {
        for j in 0..w {
            let r = region[(i, j)];
            if r <= 0 {
                continue;
            }
            let diff = (i + 1 < h && region[(i + 1, j)] != r)
                || (i >= 1 && region[(i - 1, j)] != r)
                || (j + 1 < w && region[(i, j + 1)] != r)
                || (j >= 1 && region[(i, j - 1)] != r);
            is_boundary[(i, j)] = diff;
        }
    }

    let mut coords: HashMap<i32, Vec<(f64, f64)>> = HashMap::new();
    for &lab in labels {
        let mut pts: Vec<(f64, f64)> = Vec::new();
        for i in 0..h {
            for j in 0..w {
                if is_boundary[(i, j)] && region[(i, j)] == lab {
                    pts.push((i as f64, j as f64));
                }
            }
        }
        if pts.len() > max_boundary {
            let step = ((pts.len() as f64) / (max_boundary as f64)).ceil() as usize;
            pts = pts.iter().step_by(step.max(1)).copied().collect();
        }
        coords.insert(lab, pts);
    }
    coords
}

/// Median unwrapped phase of `lab`'s pixels in a square box of half-width
/// `radius` around the endpoint `yx` (the region's local level at the gap).
fn endpoint_median(
    unw: &Array2<f32>,
    region: &Array2<i32>,
    lab: i32,
    yx: (f64, f64),
    radius: usize,
) -> f64 {
    let (m, n) = unw.dim();
    let y = yx.0 as usize;
    let x = yx.1 as usize;
    let y0 = y.saturating_sub(radius);
    let y1 = (y + radius + 1).min(m);
    let x0 = x.saturating_sub(radius);
    let x1 = (x + radius + 1).min(n);
    let mut vals: Vec<f32> = Vec::new();
    for i in y0..y1 {
        for j in x0..x1 {
            if region[(i, j)] == lab {
                let v = unw[(i, j)];
                if v.is_finite() {
                    vals.push(v);
                }
            }
        }
    }
    if vals.is_empty() {
        return f64::NAN;
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let c = vals.len();
    let med = if c % 2 == 1 {
        vals[c / 2]
    } else {
        0.5 * (vals[c / 2 - 1] + vals[c / 2])
    };
    med as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use std::f32::consts::TAU;

    #[test]
    fn single_region_is_noop() {
        let unw = Array2::<f32>::from_elem((8, 8), 1.5);
        let out = bridge_components(unw.view(), None, 4, 1, 100);
        assert_eq!(out, unw);
    }

    #[test]
    fn relevels_two_regions_split_by_a_gap() {
        // Two valid slabs separated by a masked column. The right slab is offset
        // by exactly +1 cycle; bridging should pull it back to agree. The left
        // slab is wider, so it is the (largest) MST root and stays put.
        let (m, n) = (10usize, 12usize);
        let gap = 7usize; // columns 0..7 left (wider), 8..12 right (narrower)
        let mut unw = Array2::<f32>::zeros((m, n));
        let mut mask = Array2::<bool>::from_elem((m, n), true);
        for i in 0..m {
            mask[(i, gap)] = false; // gap column splits the frame in two
            for j in 0..n {
                if j == gap {
                    continue;
                }
                // Left slab at level ~0.2; right slab at 0.2 + one 2π cycle.
                unw[(i, j)] = if j < gap { 0.2 } else { 0.2 + TAU };
            }
        }
        let out = bridge_components(unw.view(), Some(mask.view()), 8, 1, 100);
        // Right slab snapped down by one cycle to match the (larger/ref) left.
        for i in 0..m {
            for j in (gap + 1)..n {
                assert!(
                    (out[(i, j)] - 0.2).abs() < 1e-3,
                    "({i},{j})={}",
                    out[(i, j)]
                );
            }
        }
    }

    /// Cross-check against the Python `whirlwind._bridge` reference on an
    /// identical deterministic 3-slab scene. The Python implementation yields
    /// per-region cycle shifts A:0, B:-1, C:-2 (captured from `py_bridge`); this
    /// port must match exactly.
    #[test]
    fn matches_python_reference_three_slabs() {
        let (m, n) = (20usize, 20usize);
        let tau = TAU;
        let mut unw = Array2::<f32>::zeros((m, n));
        let mut mask = Array2::<bool>::from_elem((m, n), true);
        for j in 0..n {
            // Masked gaps split the frame into A (0..8), B (10..14), C (16..20).
            let masked = (8..10).contains(&j) || (14..16).contains(&j);
            for i in 0..m {
                mask[(i, j)] = !masked;
                // Continuous ramp + integer-cycle offsets per slab; tiny y tilt
                // breaks median ties exactly as in the Python reference.
                let mut v = 0.3 * j as f32 + 0.001 * i as f32;
                if (10..14).contains(&j) {
                    v += tau; // B: +1 cycle
                } else if (16..20).contains(&j) {
                    v += 2.0 * tau; // C: +2 cycles
                }
                unw[(i, j)] = v;
            }
        }
        let out = bridge_components(unw.view(), Some(mask.view()), 8, 1, 2000);

        // Per-slab integer cycle shift = round(mean(out - unw) / tau).
        let slab_shift = |a: usize, b: usize| -> i32 {
            let mut sum = 0.0f64;
            let mut cnt = 0.0f64;
            for i in 0..m {
                for j in a..b {
                    sum += (out[(i, j)] - unw[(i, j)]) as f64;
                    cnt += 1.0;
                }
            }
            (sum / cnt / tau as f64).round() as i32
        };
        assert_eq!(slab_shift(0, 8), 0, "region A");
        assert_eq!(slab_shift(10, 14), -1, "region B");
        assert_eq!(slab_shift(16, 20), -2, "region C");
    }
}
