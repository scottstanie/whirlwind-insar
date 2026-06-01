//! `whirlwind` CLI: simulate synthetic interferograms and unwrap them.

use anyhow::{Context, Result, anyhow};
use clap::{Parser, Subcommand};
use ndarray::Array2;
use num_complex::Complex32;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use tiff::decoder::{Decoder, DecodingResult};
use tiff::encoder::{TiffEncoder, colortype};

#[derive(Parser, Debug)]
#[command(name = "whirlwind", about = "InSAR phase unwrapper (Rust)")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Generate a synthetic interferogram + coherence pair.
    Simulate {
        /// shape as MxN (e.g. 256x256)
        #[arg(long, default_value = "256x256")]
        shape: String,
        /// output directory (writes wrapped.tif, cor.tif, truth.tif)
        #[arg(long)]
        out: PathBuf,
        /// "ramp" or "bump"
        #[arg(long, default_value = "bump")]
        pattern: String,
        /// number of looks for synthetic noise
        #[arg(long, default_value_t = 10)]
        nlooks: usize,
        /// coherence (uniform)
        #[arg(long, default_value_t = 0.85)]
        coherence: f32,
        /// rng seed
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Unwrap an interferogram.
    ///
    /// Takes the wrapped phase as a single float32 TIFF (radians, range
    /// `[-π, π]`). Internally the unwrapper only reads `arg(z)` from the
    /// IG — the magnitude is unused — so wrapped phase is the full input.
    ///
    /// If you have a complex-valued GeoTIFF, you can extract the phase first via
    ///
    ///     gdal_translate DERIVED_SUBDATASET:PHASE:complex.int.tif wrapped.tif
    ///
    Unwrap {
        /// wrapped-phase TIFF (float32, radians)
        #[arg(long)]
        phase: PathBuf,
        /// coherence TIFF (float32)
        #[arg(long)]
        cor: PathBuf,
        /// optional valid-pixel mask (TIFF, u8/u16/i8/i16/f32/f64).
        /// Any nonzero value = valid (SNAPHU convention). Pre-saturates arcs
        /// crossing masked pixels so MCF skips them — critical for large
        /// real scenes with water / shadow / decorrelated regions, where
        /// the unmasked path treats NoData pixels as real residues and
        /// can slow down by 10–100×.
        #[arg(long)]
        mask: Option<PathBuf>,
        /// number of looks
        #[arg(long, default_value_t = 1.0)]
        nlooks: f32,
        /// Goldstein adaptive-filter strength in [0, 1]. Default 0.7.
        /// Set to 0 to skip the prefilter entirely. When > 0, the wrapped
        /// phase is Goldstein-filtered before MCF (≈ 2× faster on noisy
        /// scenes, fewer ±2π errors at wrap-line boundaries), then the
        /// resulting integer cycle field is transferred back to the
        /// *original* wrapped phase (dolphin PR #364 convention — avoids
        /// spurious 2π jumps at fringe boundaries).
        ///
        /// On a 6811×6912 NISAR scene against SNAPHU `ntiles=(9,9)` as the
        /// land-area reference (17 min wall): α=0.5 gave 93.5 % per-pixel
        /// integer-cycle agreement on the cc=1 mainland; α=0.7 gives
        /// 99.90 % — essentially pixel-perfect agreement — while still
        /// running 27× faster than SNAPHU. α=0.75 is marginally worse
        /// (99.87 %), so 0.7 is a good "on" value for typical InSAR scenes.
        /// Default is 0 (off) while the Goldstein-on-vs-off trade-off is
        /// under evaluation; pass `--goldstein-alpha 0.7` to enable.
        #[arg(long, default_value_t = 0.0)]
        goldstein_alpha: f32,
        /// Goldstein FFT patch size (even, ≥ 4). Larger = stronger spatial
        /// smoothing in the filter.
        #[arg(long, default_value_t = 64)]
        goldstein_psize: usize,
        /// Optional connected-components output TIFF (uint16). When set,
        /// runs SNAPHU-style component growing from the same MCF solve and
        /// writes a per-pixel component label (0 = background / unassigned,
        /// 1..N = kept components). Phase is unwrapped consistently within
        /// each component, but the relative 2π·k offset between components
        /// is undefined.
        #[arg(long)]
        conncomp: Option<PathBuf>,
        /// Drop connected components smaller than this fraction of valid
        /// pixels. Default 1e-4 (≈ 5000 px on a 50 Mpx scene), small enough
        /// to keep isolated islands. Only matters when `--conncomp` is set.
        #[arg(long, default_value_t = 1e-4)]
        min_component_frac: f32,
        /// Carballo cost threshold for the conncomp cut rule. Pixel edges
        /// whose min raw forward cost ≤ this are treated as cuts. Lower =
        /// more cuts = more (smaller) components. Default 50 (SNAPHU-equiv).
        #[arg(long, default_value_t = 50)]
        cost_threshold: i32,
        /// output unwrapped phase TIFF
        #[arg(long)]
        out: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Simulate {
            shape,
            out,
            pattern,
            nlooks,
            coherence,
            seed,
        } => cmd_simulate(shape, out, pattern, nlooks, coherence, seed),
        Cmd::Unwrap {
            phase,
            cor,
            mask,
            nlooks,
            goldstein_alpha,
            goldstein_psize,
            conncomp,
            min_component_frac,
            cost_threshold,
            out,
        } => cmd_unwrap(
            phase,
            cor,
            mask,
            nlooks,
            goldstein_alpha,
            goldstein_psize,
            conncomp,
            min_component_frac,
            cost_threshold,
            out,
        ),
    }
}

fn parse_shape(s: &str) -> Result<(usize, usize)> {
    let (m, n) = s
        .split_once('x')
        .ok_or_else(|| anyhow!("shape must be MxN, got {s}"))?;
    Ok((m.parse()?, n.parse()?))
}

fn cmd_simulate(
    shape: String,
    out: PathBuf,
    pattern: String,
    nlooks: usize,
    coherence: f32,
    seed: u64,
) -> Result<()> {
    use rand::SeedableRng;
    let (m, n) = parse_shape(&shape)?;
    let truth = match pattern.as_str() {
        "ramp" => whirlwind_core::simulate::diagonal_ramp((m, n)),
        "bump" => whirlwind_core::simulate::gaussian_bump((m, n), 8.0, (n as f32) / 8.0),
        other => return Err(anyhow!("unknown pattern: {other}")),
    };
    let gamma = Array2::<f32>::from_elem((m, n), coherence);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let (igram, cor) = whirlwind_core::simulate::simulate_ifg(&truth, &gamma, nlooks, &mut rng);

    std::fs::create_dir_all(&out)?;
    write_f32_tiff(&out.join("wrapped.tif"), igram.mapv(|c| c.arg()).view())?;
    write_f32_tiff(&out.join("cor.tif"), cor.view())?;
    write_f32_tiff(&out.join("truth.tif"), truth.view())?;
    eprintln!("wrote {} (shape {m}x{n}, pattern {pattern})", out.display());
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cmd_unwrap(
    phase: PathBuf,
    cor: PathBuf,
    mask: Option<PathBuf>,
    nlooks: f32,
    goldstein_alpha: f32,
    goldstein_psize: usize,
    conncomp: Option<PathBuf>,
    min_component_frac: f32,
    cost_threshold: i32,
    out: PathBuf,
) -> Result<()> {
    let ph = read_f32_tiff(&phase)?;
    let co = read_f32_tiff(&cor)?;
    if ph.dim() != co.dim() {
        return Err(anyhow!(
            "shape mismatch: phase={:?} cor={:?}",
            ph.dim(),
            co.dim()
        ));
    }
    let mk = mask.as_ref().map(|p| read_bool_mask(p)).transpose()?;
    if let Some(m) = &mk
        && m.dim() != ph.dim()
    {
        return Err(anyhow!(
            "shape mismatch: phase={:?} mask={:?}",
            ph.dim(),
            m.dim()
        ));
    }
    // Reject out-of-range / non-finite coherence at unmasked pixels.
    validate_coherence(co.view(), mk.as_ref().map(|m| m.view()))?;

    // The unwrapper consumes complex; reconstruct as unit-magnitude exp(i·phase).
    let igram_orig = ph.mapv(|p| Complex32::from_polar(1.0, p));

    // Optional Goldstein prefilter. The filter denoises the wrapped phase so
    // MCF sees fewer spurious residues at low-coherence pixels; ~2× faster on
    // noisy scenes and visibly cleaner at wrap-line boundaries.
    let (igram_for_unwrap, used_goldstein) = if goldstein_alpha > 0.0 {
        let ig_filt = whirlwind_core::goldstein::goldstein(
            igram_orig.view(),
            goldstein_alpha,
            goldstein_psize,
        );
        // Zero out masked pixels so the filter's spread of energy into them
        // can't leak into the cost computation downstream.
        let ig_filt = if let Some(m) = &mk {
            let mut z = ig_filt;
            for ((i, j), &valid) in m.indexed_iter() {
                if !valid {
                    z[(i, j)] = Complex32::new(0.0, 0.0);
                }
            }
            z
        } else {
            ig_filt
        };
        (ig_filt, true)
    } else {
        (igram_orig.clone(), false)
    };

    // Unwrap. With --conncomp set, use the variant that also grows components.
    let (unw_filt, cc_raster) = if conncomp.is_some() {
        let params = whirlwind_core::ConnCompParams {
            cost_threshold,
            // Absolute floor (≈0.8 km at 80 m) is the real speckle control;
            // `--min-component-frac` only raises it on very large frames.
            min_size_px: 100,
            min_size_frac: min_component_frac,
            // u16 raster supports up to 65535 components; cap below that to
            // keep the conncomp routine from over-fragmenting.
            max_ncomps: 1024,
        };
        // Robust tiled phase + global (solve-free) conncomp; tile_size=0
        // auto-tiles frames > 512 px, multilook=1.
        let (u, c) = whirlwind_core::unwrap_coherence_with_components(
            igram_for_unwrap.view(),
            co.view(),
            nlooks,
            mk.as_ref().map(|m| m.view()),
            0,
            0,
            1,
            params,
        )?;
        (u, Some(c))
    } else {
        let u = whirlwind_core::unwrap_coherence(
            igram_for_unwrap.view(),
            co.view(),
            nlooks,
            mk.as_ref().map(|m| m.view()),
            0,
            0,
            1,
        )?;
        (u, None)
    };

    // K-transfer to original wrapped phase (dolphin PR #364 convention).
    // Rounding against `ph` (the original, *unfiltered* phase) avoids the
    // spurious ±2π jumps at fringe boundaries that the earlier
    // round-against-filtered-phase strategy produced. If Goldstein was
    // skipped, this is a no-op (unw_filt is already congruent with ph).
    let tau = std::f32::consts::TAU;
    let unw = if used_goldstein {
        let mut out_arr = Array2::<f32>::zeros(ph.dim());
        ndarray::Zip::from(&mut out_arr)
            .and(&ph)
            .and(&unw_filt)
            .for_each(|o, &p_orig, &u_filt| {
                let k = ((u_filt - p_orig) / tau).round();
                *o = p_orig + tau * k;
            });
        if let Some(m) = &mk {
            ndarray::Zip::from(&mut out_arr).and(m).for_each(|o, &v| {
                if !v {
                    *o = 0.0;
                }
            });
        }
        out_arr
    } else {
        unw_filt
    };

    write_f32_tiff(&out, unw.view())?;
    eprintln!("wrote {}", out.display());

    if let (Some(cc_path), Some(cc_arr)) = (conncomp.as_ref(), cc_raster.as_ref()) {
        // Summarise components on stderr so callers can see what was found.
        let n_comp = cc_arr.iter().copied().max().unwrap_or(0);
        let total_valid: usize = mk
            .as_ref()
            .map(|m| m.iter().filter(|&&v| v).count())
            .unwrap_or(cc_arr.len());
        let mut sizes = vec![0_usize; (n_comp + 1) as usize];
        for &c in cc_arr.iter() {
            sizes[c as usize] += 1;
        }
        eprintln!("found {n_comp} connected component(s):");
        for (k, &s) in sizes.iter().enumerate().skip(1) {
            let pct = 100.0 * s as f64 / total_valid.max(1) as f64;
            eprintln!("  cc={k:>3}: {s:>10} px  ({pct:5.2}% of valid)");
        }
        let bg = sizes[0];
        let bg_pct = 100.0 * bg as f64 / cc_arr.len().max(1) as f64;
        eprintln!("  bg/dropped: {bg:>10} px  ({bg_pct:5.2}% of total)");

        write_u16_tiff(cc_path, cc_arr.view())?;
        eprintln!("wrote {}", cc_path.display());
    }

    Ok(())
}

fn validate_coherence(
    co: ndarray::ArrayView2<f32>,
    mask: Option<ndarray::ArrayView2<bool>>,
) -> Result<()> {
    // Tiny tolerance for floats that round to 1.0 + ULP from upstream
    // estimator arithmetic. Anything outside [-eps, 1 + eps] is rejected.
    const EPS: f32 = 1e-4;
    let mut bad = 0_usize;
    let mut total = 0_usize;
    let mut sample_min = f32::INFINITY;
    let mut sample_max = f32::NEG_INFINITY;
    for ((i, j), &v) in co.indexed_iter() {
        if let Some(m) = mask
            && !m[(i, j)]
        {
            continue;
        }
        total += 1;
        if !v.is_finite() || !(-EPS..=1.0 + EPS).contains(&v) {
            bad += 1;
            if v.is_finite() {
                sample_min = sample_min.min(v);
                sample_max = sample_max.max(v);
            }
        }
    }
    if bad > 0 {
        let pct = 100.0 * bad as f64 / total.max(1) as f64;
        let extras = if sample_min.is_finite() {
            format!(" (finite out-of-range span: [{sample_min}, {sample_max}])")
        } else {
            String::new()
        };
        return Err(anyhow!(
            "coherence has {bad}/{total} ({pct:.2}%) pixels outside [0, 1] or non-finite{extras}. \
             Either pre-clean the file (e.g. `gdal_calc.py -A coh.tif --calc='where((A>=0)&(A<=1),A,0)'`) \
             or pass `--mask` to exclude these pixels."
        ));
    }
    Ok(())
}

fn read_f32_tiff(path: &Path) -> Result<Array2<f32>> {
    let r = BufReader::new(File::open(path).with_context(|| format!("open {}", path.display()))?);
    // Default `Limits` cap decoding_buffer_size at 256 MiB, which rejects
    // any single-band raster bigger than ~64 Mpx of f32 (or ~32 Mpx of f64).
    // NISAR-scale and full-frame Sentinel-1 inputs routinely exceed that, so
    // lift the limit — this is a local CLI on trusted inputs, not a
    // network-facing decoder.
    let mut d = Decoder::new(r)?.with_limits(tiff::decoder::Limits::unlimited());
    let (w, h) = d.dimensions()?;
    let buf = match d.read_image()? {
        DecodingResult::F32(v) => v,
        DecodingResult::F64(v) => v.into_iter().map(|x| x as f32).collect(),
        other => {
            return Err(anyhow!(
                "unsupported TIFF dtype for {} (need f32): {:?}",
                path.display(),
                std::mem::discriminant(&other)
            ));
        }
    };
    Ok(Array2::from_shape_vec((h as usize, w as usize), buf)?)
}

/// Read a validity mask. Accepts u8/u16/f32/f64 single-band TIFFs and
/// reduces to bool with the SNAPHU convention: any *finite, nonzero*
/// value is valid (`true`), zero / NaN / sentinel = invalid (`false`).
fn read_bool_mask(path: &Path) -> Result<Array2<bool>> {
    let r = BufReader::new(File::open(path).with_context(|| format!("open {}", path.display()))?);
    let mut d = Decoder::new(r)?.with_limits(tiff::decoder::Limits::unlimited());
    let (w, h) = d.dimensions()?;
    let buf: Vec<bool> = match d.read_image()? {
        DecodingResult::U8(v) => v.into_iter().map(|x| x != 0).collect(),
        DecodingResult::U16(v) => v.into_iter().map(|x| x != 0).collect(),
        DecodingResult::I8(v) => v.into_iter().map(|x| x != 0).collect(),
        DecodingResult::I16(v) => v.into_iter().map(|x| x != 0).collect(),
        DecodingResult::F32(v) => v.into_iter().map(|x| x.is_finite() && x != 0.0).collect(),
        DecodingResult::F64(v) => v.into_iter().map(|x| x.is_finite() && x != 0.0).collect(),
        other => {
            return Err(anyhow!(
                "unsupported mask TIFF dtype for {} (need u8/u16/i8/i16/f32/f64): {:?}",
                path.display(),
                std::mem::discriminant(&other)
            ));
        }
    };
    Ok(Array2::from_shape_vec((h as usize, w as usize), buf)?)
}

fn write_f32_tiff(path: &Path, a: ndarray::ArrayView2<f32>) -> Result<()> {
    let (h, w) = a.dim();
    let mut enc = TiffEncoder::new(BufWriter::new(File::create(path)?))?;
    let buf: Vec<f32> = a.iter().copied().collect();
    enc.write_image::<colortype::Gray32Float>(w as u32, h as u32, &buf)?;
    Ok(())
}

fn write_u16_tiff(path: &Path, a: ndarray::ArrayView2<u32>) -> Result<()> {
    let (h, w) = a.dim();
    let mut enc = TiffEncoder::new(BufWriter::new(File::create(path)?))?;
    // Component IDs are emitted as u32 by the core but capped to 1024 by
    // ConnCompParams::max_ncomps above; downcast is lossless. Anything that
    // overflows u16 would be a bug worth catching.
    let buf: Vec<u16> = a
        .iter()
        .map(|&v| {
            assert!(v <= u16::MAX as u32, "component id {v} overflows u16");
            v as u16
        })
        .collect();
    enc.write_image::<colortype::Gray16>(w as u32, h as u32, &buf)?;
    Ok(())
}
