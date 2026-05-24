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
        /// output directory (writes igram.tif, cor.tif, truth.tif)
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
    /// Unwrap an interferogram. Inputs are plain float TIFFs (complex stored as
    /// 2-channel float32: real, imag) — for proper GeoTIFFs use the Python
    /// binding via rasterio.
    Unwrap {
        /// real-part TIFF (float32)
        #[arg(long)]
        igram_re: PathBuf,
        /// imag-part TIFF (float32)
        #[arg(long)]
        igram_im: PathBuf,
        /// coherence TIFF (float32)
        #[arg(long)]
        cor: PathBuf,
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
            igram_re,
            igram_im,
            cor,
            nlooks,
            out,
        } => cmd_unwrap(igram_re, igram_im, cor, nlooks, out),
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
    write_f32_tiff(&out.join("igram_re.tif"), igram.mapv(|c| c.re).view())?;
    write_f32_tiff(&out.join("igram_im.tif"), igram.mapv(|c| c.im).view())?;
    write_f32_tiff(&out.join("cor.tif"), cor.view())?;
    write_f32_tiff(&out.join("truth.tif"), truth.view())?;
    write_f32_tiff(
        &out.join("wrapped.tif"),
        igram.mapv(|c| c.arg()).view(),
    )?;
    eprintln!("wrote {} (shape {m}x{n}, pattern {pattern})", out.display());
    Ok(())
}

fn cmd_unwrap(
    igram_re: PathBuf,
    igram_im: PathBuf,
    cor: PathBuf,
    nlooks: f32,
    out: PathBuf,
) -> Result<()> {
    let re = read_f32_tiff(&igram_re)?;
    let im = read_f32_tiff(&igram_im)?;
    let co = read_f32_tiff(&cor)?;
    if re.dim() != im.dim() || re.dim() != co.dim() {
        return Err(anyhow!(
            "shape mismatch: re={:?} im={:?} cor={:?}",
            re.dim(),
            im.dim(),
            co.dim()
        ));
    }
    let igram = Array2::from_shape_fn(re.dim(), |(i, j)| Complex32::new(re[(i, j)], im[(i, j)]));
    let unw = whirlwind_core::unwrap(igram.view(), co.view(), nlooks, None)?;
    write_f32_tiff(&out, unw.view())?;
    eprintln!("wrote {}", out.display());
    Ok(())
}

fn read_f32_tiff(path: &Path) -> Result<Array2<f32>> {
    let r = BufReader::new(File::open(path).with_context(|| format!("open {}", path.display()))?);
    let mut d = Decoder::new(r)?;
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

fn write_f32_tiff(path: &Path, a: ndarray::ArrayView2<f32>) -> Result<()> {
    let (h, w) = a.dim();
    let mut enc = TiffEncoder::new(BufWriter::new(File::create(path)?))?;
    let buf: Vec<f32> = a.iter().copied().collect();
    enc.write_image::<colortype::Gray32Float>(w as u32, h as u32, &buf)?;
    Ok(())
}
