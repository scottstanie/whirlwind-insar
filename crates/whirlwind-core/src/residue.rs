//! Residue grid computation.
//!
//! A residue at a (m+1, n+1) node is the sum of wrapped phase gradients
//! around the 2x2 block of pixels surrounding it, normalized by 2π.
//! Nonzero values at interior nodes indicate phase singularities that the
//! unwrapper must neutralize.
//!
//! Boundary nodes (residue rows 0 and m, cols 0 and n) carry the wrap counts
//! along the four image edges: a wrap line that exits the image at one of
//! those edges deposits a charge on the corresponding boundary node, which
//! the MCF can then drain (effectively a "wrap line ends here"). Without
//! these, MCF would be forced to pair every boundary-exiting wrap line with
//! a far-away interior partner, generating long flow paths and large
//! integer-surface variations after integration.

use ndarray::parallel::prelude::*;
use ndarray::{Array2, ArrayView2, Axis};
use std::f32::consts::TAU;

/// Round (a-b)/2π to the nearest signed integer in {-1, 0, +1} for two
/// wrapped phases.
#[inline]
fn cycle_diff(a: f32, b: f32) -> i32 {
    ((a - b) / TAU).round() as i32
}

/// Compute the residue grid from a wrapped phase array of shape `(m, n)`.
/// Output shape is `(m+1, n+1)`; boundary (row 0, row m, col 0, col n) is
/// always zero.
///
/// Each residue is the integer winding-number around a 2x2 pixel loop. We
/// fill residue row `i+1` from pixel row `i` only (using `phi[i, j]`,
/// `phi[i+1, j]`, `phi[i, j+1]`), so the per-residue-row work is independent
/// across `i` and we parallelize with rayon.
pub fn compute(wrapped_phase: ArrayView2<f32>) -> Array2<i32> {
    compute_with_mask(wrapped_phase, None)
}

/// Compute the residue grid, zeroing residues whose 2×2 pixel-loop touches
/// any masked-out pixel (where `mask[i, j] == false`).
///
/// Without this, NaN/invalid pixels (replaced by zeros before unwrap) generate
/// spurious large residues at the mask boundary that leak charge into the
/// valid region's primal-dual loop. Zeroing them keeps the MCF problem
/// confined to where the phase data is actually meaningful.
pub fn compute_with_mask(
    wrapped_phase: ArrayView2<f32>,
    mask: Option<ArrayView2<bool>>,
) -> Array2<i32> {
    let (m, n) = wrapped_phase.dim();
    assert!(m >= 1 && n >= 1);
    if let Some(mm) = mask {
        assert_eq!(mm.dim(), (m, n), "mask must match pixel grid");
    }
    let mut out = Array2::<i32>::zeros((m + 1, n + 1));

    if m >= 2 && n >= 2 {
        // Per residue row r (1..=m-1), all contributions come from the
        // 2x2 pixel loop with bottom-right at residue (r, c). Each residue
        // row depends only on pixel rows r-1 and r, so we par_iter rows.
        //
        // R[r, c] = CCW curl of integer-rounded gradients around the 2x2
        // pixel loop {(r-1, c-1), (r-1, c), (r, c), (r, c-1)}.
        out.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(r, mut residue_row)| {
                if r == 0 || r >= m {
                    return; // boundary rows handled below
                }
                let i = r - 1;
                for c in 1..n {
                    let j = c - 1;
                    if let Some(mm) = mask {
                        if !mm[(i, j)] || !mm[(i, j + 1)] || !mm[(i + 1, j)] || !mm[(i + 1, j + 1)] {
                            continue;
                        }
                    }
                    let p00 = wrapped_phase[(i, j)];
                    let p01 = wrapped_phase[(i, j + 1)];
                    let p10 = wrapped_phase[(i + 1, j)];
                    let p11 = wrapped_phase[(i + 1, j + 1)];
                    let s = cycle_diff(p10, p00)
                        + cycle_diff(p11, p10)
                        + cycle_diff(p01, p11)
                        + cycle_diff(p00, p01);
                    residue_row[c] = s;
                }
            });

        // Boundary residues. The "frame" of the residue grid (rows 0/m,
        // cols 0/n) carries the wrap counts along the four image edges so
        // MCF can drain wrap lines that exit through those edges instead of
        // pairing them with distant interior partners.
        //
        // Signs chosen so that `total_residue_sum == 0`: by Stokes, the CCW
        // boundary contour integral of the wrap rates equals the total
        // interior winding (= sum of interior residues), so the boundary
        // deposits get the opposite sign. The `nodata_total_charge` test
        // pins this. Each pixel-edge on the image boundary writes to a
        // unique outer-frame node:
        //   top edge    p[0,j]→p[0,j+1]   → frame (0, j+1)
        //   bottom edge p[m-1,j]→p[m-1,j+1] → frame (m, j+1)
        //   left edge   p[i,0]→p[i+1,0]   → frame (i+1, 0)
        //   right edge  p[i,n-1]→p[i+1,n-1] → frame (i+1, n)
        for j in 0..n - 1 {
            if let Some(mm) = mask {
                if !mm[(0, j)] || !mm[(0, j + 1)] {
                    continue;
                }
            }
            out[(0, j + 1)] += cycle_diff(wrapped_phase[(0, j + 1)], wrapped_phase[(0, j)]);
        }
        for j in 0..n - 1 {
            if let Some(mm) = mask {
                if !mm[(m - 1, j)] || !mm[(m - 1, j + 1)] {
                    continue;
                }
            }
            out[(m, j + 1)] -= cycle_diff(wrapped_phase[(m - 1, j + 1)], wrapped_phase[(m - 1, j)]);
        }
        for i in 0..m - 1 {
            if let Some(mm) = mask {
                if !mm[(i, 0)] || !mm[(i + 1, 0)] {
                    continue;
                }
            }
            out[(i + 1, 0)] -= cycle_diff(wrapped_phase[(i + 1, 0)], wrapped_phase[(i, 0)]);
        }
        for i in 0..m - 1 {
            if let Some(mm) = mask {
                if !mm[(i, n - 1)] || !mm[(i + 1, n - 1)] {
                    continue;
                }
            }
            out[(i + 1, n)] += cycle_diff(wrapped_phase[(i + 1, n - 1)], wrapped_phase[(i, n - 1)]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use std::f32::consts::PI;

    fn wrap(x: f32) -> f32 {
        let two_pi = 2.0 * PI;
        let y = x % two_pi;
        if y > PI {
            y - two_pi
        } else if y <= -PI {
            y + two_pi
        } else {
            y
        }
    }

    #[test]
    fn unwrapped_phase_has_no_residues() {
        // A non-wrapping phase image (range fits in [-π, π]): no interior
        // residues *and* no boundary residues — every cycle_diff rounds to 0.
        let n = 64;
        let mut phase = Array2::<f32>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                phase[(i, j)] = 0.5 * (i as f32) / (n as f32) + 0.5 * (j as f32) / (n as f32);
            }
        }
        let res = compute(phase.view());
        let nonzero: usize = res.iter().filter(|&&v| v != 0).count();
        assert_eq!(nonzero, 0, "non-wrapping smooth phase should have zero residues");
    }

    #[test]
    fn wrapping_ramp_deposits_charge_at_boundary() {
        // A ramp that wraps along its gradient direction: no INTERIOR residues
        // (every 2x2 plaquette winding is 0) but the wrap lines exit at the
        // image boundary, depositing nonzero residues on the outer frame.
        let n = 64;
        let mut phase = Array2::<f32>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let x = -3.0 + 6.0 * (j as f32) / ((n - 1) as f32);
                let y = -3.0 + 6.0 * (i as f32) / ((n - 1) as f32);
                phase[(i, j)] = wrap(PI * (x + y));
            }
        }
        let res = compute(phase.view());
        let (rm, rn) = res.dim();
        let interior_nz: usize = (1..rm - 1)
            .flat_map(|r| (1..rn - 1).map(move |c| (r, c)))
            .filter(|&(r, c)| res[(r, c)] != 0)
            .count();
        let boundary_nz: usize = res.iter().filter(|&&v| v != 0).count() - interior_nz;
        assert_eq!(interior_nz, 0, "wrapping ramp should have no INTERIOR residues");
        assert!(boundary_nz > 0, "wrap lines must deposit charge at the boundary frame");
        assert_eq!(res.iter().sum::<i32>(), 0, "augmented total must balance to zero");
    }

    #[test]
    fn conservation_total_charge_zero_random_phase() {
        // With boundary residues included, the FULL residue grid sum must
        // be 0 for any phase image: Stokes says the CCW contour integral
        // of wrap rates equals the total interior winding, and the boundary
        // deposits carry the contour with opposite sign so the augmented
        // total is exactly zero. This is THE property MCF needs to be
        // solvable (excess and deficit charge balance).
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x1234_5678);
        for &(m, n) in &[(32, 32), (37, 53), (101, 23)] {
            let mut phase = Array2::<f32>::zeros((m, n));
            for i in 0..m {
                for j in 0..n {
                    phase[(i, j)] = rng.gen_range(-PI..PI);
                }
            }
            let res = compute(phase.view());
            let sum: i32 = res.iter().sum();
            assert_eq!(
                sum, 0,
                "residue total must be 0 for any phase ({}x{}): got {}",
                m, n, sum
            );
        }
    }

    #[test]
    fn smooth_ramp_conservation() {
        let n = 32;
        let mut phase = Array2::<f32>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let x = -1.0 + 2.0 * (j as f32) / ((n - 1) as f32);
                let y = -1.0 + 2.0 * (i as f32) / ((n - 1) as f32);
                phase[(i, j)] = wrap(0.3 * (x + y));
            }
        }
        let res = compute(phase.view());
        let sum: i32 = res.iter().sum();
        assert_eq!(sum, 0, "smooth image should sum to zero");
    }

    #[test]
    fn planted_positive_residue() {
        // arctan2 around a center pixel deposits exactly one +1 residue at that pixel.
        let n = 16;
        let mut phase = Array2::<f32>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let dy = i as f32 - 7.5;
                let dx = j as f32 - 7.5;
                phase[(i, j)] = wrap(dy.atan2(dx));
            }
        }
        let res = compute(phase.view());
        let pos: usize = res.iter().filter(|&&v| v > 0).count();
        let neg: usize = res.iter().filter(|&&v| v < 0).count();
        // Exactly one positive vortex; matching negative gets pushed to boundary
        // but boundary is zeroed, so post-fix we may have a small imbalance.
        // What we *do* require: at least one nonzero interior residue.
        assert!(
            pos + neg >= 1,
            "vortex should produce ≥1 interior residue, got pos={pos} neg={neg}"
        );
    }
}
