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
        /// optional valid-pixel mask (TIFF, u8/u16/f32/f64).
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
            out,
        } => cmd_unwrap(phase, cor, mask, nlooks, out),
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

fn cmd_unwrap(
    phase: PathBuf,
    cor: PathBuf,
    mask: Option<PathBuf>,
    nlooks: f32,
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
    if let Some(m) = &mk {
        if m.dim() != ph.dim() {
            return Err(anyhow!(
                "shape mismatch: phase={:?} mask={:?}",
                ph.dim(),
                m.dim()
            ));
        }
    }
    // Reject out-of-range / non-finite coherence at unmasked pixels. The
    // Carballo cost LUT is defined for γ ∈ [0, 1]; anything else (NaN,
    // sentinel values like 1e10, negative bias, > 1) silently produces
    // wrong arc costs and bad unwraps. Better to fail loudly with a clear
    // message so the caller cleans up the input or masks the bad pixels.
    validate_coherence(co.view(), mk.as_ref().map(|m| m.view()))?;

    // The unwrapper consumes a complex IG and internally takes arg(z); the
    // magnitude is never used. Reconstruct as unit-magnitude exp(i·phase).
    let igram = ph.mapv(|p| Complex32::from_polar(1.0, p));
    let unw = whirlwind_core::unwrap(
        igram.view(),
        co.view(),
        nlooks,
        mk.as_ref().map(|m| m.view()),
    )?;
    write_f32_tiff(&out, unw.view())?;
    eprintln!("wrote {}", out.display());
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
        if let Some(m) = mask {
            if !m[(i, j)] {
                continue;
            }
        }
        total += 1;
        if !v.is_finite() || v < -EPS || v > 1.0 + EPS {
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
