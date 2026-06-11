//! `whirlwind` CLI: simulate synthetic interferograms and unwrap them.

mod formats;

use anyhow::{Context, Result, anyhow, bail};
use clap::{Parser, Subcommand, ValueEnum};
use formats::{Endian, FloatLayout};
use ndarray::Array2;
use num_complex::Complex32;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use tiff::decoder::{Decoder, DecodingResult};
use tiff::encoder::{TiffEncoder, colortype};

/// Band layout of a flat-binary `--cor` input.
#[derive(Clone, Copy, PartialEq, Eq, Debug, ValueEnum)]
enum CorFormat {
    /// single- vs two-band (line-interleaved) from the file size
    Auto,
    /// single-band float32 (snaphu FLOAT_DATA)
    Float,
    /// two-band line-interleaved, correlation second (snaphu ALT_LINE_DATA)
    AltLine,
    /// two-band sample-interleaved, correlation second (snaphu ALT_SAMPLE_DATA)
    AltSample,
    /// two-band band-sequential, correlation in the second band (isce2/GDAL BSQ)
    Bsq,
}

/// Format of the unwrapped-phase output.
#[derive(Clone, Copy, PartialEq, Eq, Debug, ValueEnum)]
enum OutFormat {
    /// TIFF for .tif/.tiff, alt-line for .unw, flat float32 otherwise
    Auto,
    /// float32 TIFF
    Tiff,
    /// flat float32, phase only (snaphu FLOAT_DATA)
    Float,
    /// flat two-band line-interleaved amplitude+phase (snaphu ALT_LINE_DATA)
    AltLine,
}

fn is_tiff(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_ascii_lowercase())
            .as_deref(),
        Some("tif" | "tiff")
    )
}

fn resolve_out_format(out_format: OutFormat, out: &Path) -> OutFormat {
    match out_format {
        OutFormat::Auto => {
            let ext = out
                .extension()
                .and_then(|e| e.to_str())
                .map(|s| s.to_ascii_lowercase());
            if is_tiff(out) {
                OutFormat::Tiff
            } else if ext.as_deref() == Some("unw") {
                OutFormat::AltLine
            } else {
                OutFormat::Float
            }
        }
        f => f,
    }
}

fn append_path_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}

fn default_conncomp_path(out: &Path, out_format: OutFormat) -> PathBuf {
    match out_format {
        OutFormat::Tiff => {
            let ext = if is_tiff(out) {
                out.extension().and_then(|e| e.to_str()).unwrap_or("tif")
            } else {
                "tif"
            };
            out.with_extension(format!("conncomp.{ext}"))
        }
        OutFormat::Float | OutFormat::AltLine => append_path_suffix(out, ".conncomp"),
        OutFormat::Auto => default_conncomp_path(out, resolve_out_format(out_format, out)),
    }
}

#[derive(Parser, Debug)]
#[command(name = "whirlwind", version, about = "InSAR phase unwrapper")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

// The Unwrap variant carries the whole flag surface; the size skew vs
// Simulate is irrelevant for a once-parsed CLI enum.
#[allow(clippy::large_enum_variant)]
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
    /// Takes either the complex interferogram (--ifg, flat binary complex64:
    /// snaphu COMPLEX_DATA / ROI_PAC / isce2 .int, GAMMA .int/.diff) or the
    /// wrapped phase (--phase, float32 TIFF or flat binary, radians in
    /// [-pi, pi]).
    ///
    /// Flat-binary (headerless) inputs need the number of columns: pass
    /// --cols (snaphu's "line length" / ROI_PAC WIDTH), or let it be read
    /// from a <file>.rsc / <file>.xml sidecar found next to the data, or
    /// point the matching --ifg-meta / --phase-meta / --cor-meta flag at a
    /// ROI_PAC .rsc, isce2 .xml, or GAMMA .par/.off file. The number of
    /// rows always comes from the file size. GAMMA rasters are big-endian:
    /// use --big-endian (implied when the explicit meta flag is a GAMMA par
    /// file).
    ///
    /// Examples:
    ///
    ///   GeoTIFF (float32 wrapped phase + coherence):
    ///     whirlwind unwrap --phase wrapped.tif --cor coherence.tif \
    ///         --nlooks 10 --out unwrapped.tif
    ///
    ///   snaphu / ROI_PAC flat binary (.int = complex64, .cc = amp+cor
    ///   "rmg"; width from --cols, or from <file>.rsc when present):
    ///     whirlwind unwrap --ifg 20150902_20150914.int \
    ///         --cor 20150902_20150914.cc --cols 840 --nlooks 10 \
    ///         --out 20150902_20150914.unw
    ///
    ///   isce2 stripmapStack / topsStack (geometry, dtype, and byte order
    ///   all come from the <file>.xml sidecars; no extra flags):
    ///     whirlwind unwrap --ifg filt_fine.int --cor filt_fine.cor \
    ///         --nlooks 10 --out filt_fine.unw
    ///
    ///   GAMMA (big-endian; width from the .off/.par; phase-only float32
    ///   .unw output like GAMMA's own):
    ///     whirlwind unwrap --ifg pair.diff --ifg-meta pair.off \
    ///         --cor pair.cc --cor-meta pair.off \
    ///         --nlooks 10 --out-format float --out pair.unw
    ///
    /// If you have a complex-valued GeoTIFF, extract the phase first via
    ///   gdal_translate DERIVED_SUBDATASET:PHASE:complex.int.tif wrapped.tif
    #[command(verbatim_doc_comment)]
    Unwrap {
        /// complex interferogram, flat binary complex64 (interleaved float32
        /// real/imag pairs). Exactly one of --ifg / --phase is required.
        #[arg(long, conflicts_with = "phase")]
        ifg: Option<PathBuf>,
        /// wrapped phase (float32, radians): TIFF by extension (.tif/.tiff),
        /// otherwise flat binary (snaphu FLOAT_DATA)
        #[arg(long)]
        phase: Option<PathBuf>,
        /// coherence (float32): TIFF by extension, otherwise flat binary -
        /// single-band (snaphu FLOAT_DATA, isce2 .cor, GAMMA .cc) or two-band
        /// amplitude+correlation (snaphu ALT_LINE_DATA, ROI_PAC .cc), told
        /// apart by file size; see --cor-format
        #[arg(long)]
        cor: PathBuf,
        /// columns per row for flat-binary inputs (snaphu's "line length",
        /// ROI_PAC WIDTH). Overrides any sidecar
        #[arg(long, visible_alias = "width")]
        cols: Option<usize>,
        /// explicit metadata sidecar for --ifg: ROI_PAC .rsc, isce2 .xml, or
        /// GAMMA .par/.off/.diff_par (GAMMA implies big-endian). Without it,
        /// `<ifg>.rsc` / `<ifg>.xml` is used when present
        #[arg(long, requires = "ifg")]
        ifg_meta: Option<PathBuf>,
        /// explicit metadata sidecar for flat-binary --phase. Without it,
        /// `<phase>.rsc` / `<phase>.xml` is used when present
        #[arg(long, requires = "phase")]
        phase_meta: Option<PathBuf>,
        /// explicit metadata sidecar for flat-binary --cor. Without it,
        /// `<cor>.rsc` / `<cor>.xml` is used when present
        #[arg(long)]
        cor_meta: Option<PathBuf>,
        /// flat-binary inputs/outputs are big-endian (GAMMA convention)
        #[arg(long, action = clap::ArgAction::SetTrue)]
        big_endian: bool,
        /// band layout of a flat-binary --cor file. `auto` picks single- vs
        /// two-band (line-interleaved, correlation in the second channel)
        /// from the file size; `alt-sample` forces snaphu's sample-interleaved
        /// two-band layout, which `auto` cannot distinguish from `alt-line`
        /// by file size alone. isce2 XML `scheme` selects BIL/BIP/BSQ
        /// automatically when present
        #[arg(long, value_enum, default_value_t = CorFormat::Auto)]
        cor_format: CorFormat,
        /// format of the unwrapped-phase output. `auto`: TIFF for
        /// .tif/.tiff, two-band amplitude+phase (snaphu ALT_LINE_DATA / "rmg")
        /// for .unw, flat float32 otherwise. The amplitude band is |igram|
        /// (all ones when the input was --phase)
        #[arg(long, value_enum, default_value_t = OutFormat::Auto)]
        out_format: OutFormat,
        /// optional valid-pixel mask (TIFF, u8/u16/i8/i16/f32/f64).
        /// Any nonzero value = valid (SNAPHU convention). Pre-saturates arcs
        /// crossing masked pixels so MCF skips them - critical for large
        /// real scenes with water / shadow / decorrelated regions, where
        /// the unmasked path treats NoData pixels as real residues and
        /// can slow down by 10-100x.
        #[arg(long)]
        mask: Option<PathBuf>,
        /// number of looks
        #[arg(long, default_value_t = 1.0)]
        nlooks: f32,
        /// Coarse-solve factor for noisy scenes. When > 1, the complex
        /// interferogram is coherently averaged into `downsample x downsample`
        /// blocks and that smaller, smoother frame is unwrapped to decide which
        /// 2π cycle each block sits on; only the integer cycle is borrowed back
        /// onto the full-resolution wrapped phase. `--nlooks` stays the effective
        /// looks of your input coherence (the down-look scaling is internal).
        /// Use it for noisy/moderate-coherence scenes (e.g. Sentinel-1); leave
        /// at 1 for clean scenes.
        #[arg(long, default_value_t = 1)]
        downsample: usize,
        /// Disable the integration-component "bridge" post-pass. By default
        /// (bridge ON, matching the Python API) the relative 2π level of regions
        /// the valid mask splits apart (e.g. two land slabs separated by a
        /// low-coherence river) is re-leveled along a minimum spanning tree
        /// rooted at the largest region. A single coherently-connected frame is
        /// unchanged either way.
        #[arg(long = "no-bridge", action = clap::ArgAction::SetTrue)]
        no_bridge: bool,
        /// Spiral persistent-scatterer interpolation pre-pass. When set, every
        /// valid pixel whose coherence is below `--interp-cutoff` has its phase
        /// replaced by a Gaussian distance-weighted average of nearby
        /// high-coherence pixels before the solve. Like `--goldstein-alpha`, the
        /// fill only INFORMS the MCF: the integer cycle field is applied back to
        /// the original wrapped phase, so every per-pixel value is preserved.
        #[arg(long, action = clap::ArgAction::SetTrue)]
        interpolate: bool,
        /// Coherence below which a valid pixel is interpolated (only with
        /// `--interpolate`).
        #[arg(long, default_value_t = 0.1)]
        interp_cutoff: f32,
        /// Number of nearest high-coherence pixels averaged per interpolated
        /// pixel (only with `--interpolate`).
        #[arg(long, default_value_t = 20)]
        interp_num_neighbors: usize,
        /// Maximum search radius in pixels for the neighbor search (only with
        /// `--interpolate`).
        #[arg(long, default_value_t = 51)]
        interp_max_radius: usize,
        /// Minimum search radius in pixels; closer neighbors are skipped (only
        /// with `--interpolate`).
        #[arg(long, default_value_t = 0)]
        interp_min_radius: usize,
        /// Gaussian distance-weighting falloff for the neighbor average (only
        /// with `--interpolate`).
        #[arg(long, default_value_t = 0.75)]
        interp_alpha: f64,
        /// Goldstein adaptive-filter strength in [0, 1]. Default 0 (off);
        /// pass e.g. `--goldstein-alpha 0.7` to enable. When > 0, the wrapped
        /// phase is Goldstein-filtered before MCF (faster on noisy scenes,
        /// fewer ±2π errors at wrap-line boundaries), then the resulting
        /// integer cycle field is transferred back to the *original* wrapped
        /// phase (avoids spurious 2π jumps at fringe boundaries). α≈0.7 is a
        /// good "on" value for typical InSAR scenes.
        #[arg(long, default_value_t = 0.0)]
        goldstein_alpha: f32,
        /// Goldstein FFT patch size (even, ≥ 4). Larger = stronger spatial
        /// smoothing in the filter.
        #[arg(long, default_value_t = 64)]
        goldstein_psize: usize,
        /// Connected-components output path. By default, writes next to
        /// `--out` (`foo.conncomp.tif` for TIFF output, `foo.unw.conncomp`
        /// for flat `.unw`). TIFF paths write uint16; other paths write one
        /// byte per pixel (the snaphu / isce2 .conncomp convention; labels
        /// above 255 are an error). Phase is unwrapped consistently within
        /// each component, but the relative 2π·k offset between components is
        /// undefined.
        #[arg(long)]
        conncomp: Option<PathBuf>,
        /// Do not compute or write connected components.
        #[arg(long, conflicts_with = "conncomp", action = clap::ArgAction::SetTrue)]
        no_conncomp: bool,
        /// Drop connected components smaller than this fraction of valid
        /// pixels. Default 1e-4 (≈ 5000 px on a 50 Mpx scene), small enough
        /// to keep isolated islands.
        #[arg(long, default_value_t = 1e-4)]
        min_component_frac: f32,
        /// Discard connected components smaller than this many pixels. Absolute
        /// floor (matches the Python API default); `--min-component-frac` raises
        /// it on very large frames.
        #[arg(long, default_value_t = 100)]
        min_size_px: usize,
        /// Maximum number of connected components to keep (largest first).
        #[arg(long, default_value_t = 1024)]
        max_ncomps: u32,
        /// Carballo cost threshold for the conncomp cut rule. Pixel edges
        /// whose min raw forward cost ≤ this are treated as cuts. Lower =
        /// more cuts = more (smaller) components. Default 50 (SNAPHU-equiv).
        #[arg(long, default_value_t = 50)]
        cost_threshold: i32,
        /// Set `--cost-threshold` from a target per-edge one-cycle-correction
        /// probability. Lower is stricter (more boundaries); ~2.4e-4 matches the
        /// default. Takes precedence over `--cost-threshold`.
        #[arg(long)]
        conncomp_cycle_prob: Option<f64>,
        /// Set `--cost-threshold` from a Gaussian-equivalent noise level: an edge
        /// is cut when its one-cycle probability exceeds 0.5*erfc(sigma/sqrt2).
        /// Higher is stricter; ~3.5 reproduces the default. Takes precedence over
        /// both `--cost-threshold` and `--conncomp-cycle-prob`.
        #[arg(long)]
        conncomp_sigma: Option<f64>,
        /// output unwrapped phase; format chosen by --out-format (default:
        /// by extension)
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
            ifg,
            phase,
            cor,
            cols,
            ifg_meta,
            phase_meta,
            cor_meta,
            big_endian,
            cor_format,
            out_format,
            mask,
            nlooks,
            downsample,
            no_bridge,
            interpolate,
            interp_cutoff,
            interp_num_neighbors,
            interp_max_radius,
            interp_min_radius,
            interp_alpha,
            goldstein_alpha,
            goldstein_psize,
            conncomp,
            no_conncomp,
            min_component_frac,
            min_size_px,
            max_ncomps,
            cost_threshold,
            conncomp_cycle_prob,
            conncomp_sigma,
            out,
        } => cmd_unwrap(UnwrapArgs {
            ifg,
            phase,
            cor,
            cols,
            ifg_meta,
            phase_meta,
            cor_meta,
            big_endian,
            cor_format,
            out_format,
            mask,
            nlooks,
            downsample,
            bridge: !no_bridge,
            interpolate,
            interp_cutoff,
            interp_num_neighbors,
            interp_max_radius,
            interp_min_radius,
            interp_alpha,
            goldstein_alpha,
            goldstein_psize,
            conncomp,
            no_conncomp,
            min_component_frac,
            min_size_px,
            max_ncomps,
            cost_threshold,
            conncomp_cycle_prob,
            conncomp_sigma,
            out,
        }),
    }
}

/// Resolved arguments for the `unwrap` subcommand. Mirrors the Python
/// `whirlwind.unwrap` keyword surface so the CLI is at feature parity.
struct UnwrapArgs {
    ifg: Option<PathBuf>,
    phase: Option<PathBuf>,
    cor: PathBuf,
    cols: Option<usize>,
    ifg_meta: Option<PathBuf>,
    phase_meta: Option<PathBuf>,
    cor_meta: Option<PathBuf>,
    big_endian: bool,
    cor_format: CorFormat,
    out_format: OutFormat,
    mask: Option<PathBuf>,
    nlooks: f32,
    downsample: usize,
    bridge: bool,
    interpolate: bool,
    interp_cutoff: f32,
    interp_num_neighbors: usize,
    interp_max_radius: usize,
    interp_min_radius: usize,
    interp_alpha: f64,
    goldstein_alpha: f32,
    goldstein_psize: usize,
    conncomp: Option<PathBuf>,
    no_conncomp: bool,
    min_component_frac: f32,
    min_size_px: usize,
    max_ncomps: u32,
    cost_threshold: i32,
    conncomp_cycle_prob: Option<f64>,
    conncomp_sigma: Option<f64>,
    out: PathBuf,
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

fn cmd_unwrap(args: UnwrapArgs) -> Result<()> {
    let UnwrapArgs {
        ifg,
        phase,
        cor,
        cols,
        ifg_meta,
        phase_meta,
        cor_meta,
        big_endian,
        cor_format,
        out_format,
        mask,
        nlooks,
        downsample,
        bridge,
        interpolate,
        interp_cutoff,
        interp_num_neighbors,
        interp_max_radius,
        interp_min_radius,
        interp_alpha,
        goldstein_alpha,
        goldstein_psize,
        conncomp,
        no_conncomp,
        min_component_frac,
        min_size_px,
        max_ncomps,
        cost_threshold,
        conncomp_cycle_prob,
        conncomp_sigma,
        out,
    } = args;

    let ofmt = resolve_out_format(out_format, &out);
    let conncomp_out = if no_conncomp {
        None
    } else {
        Some(conncomp.unwrap_or_else(|| default_conncomp_path(&out, ofmt)))
    };

    // ---- inputs ---------------------------------------------------------
    // Each input is a TIFF by extension, flat binary otherwise (formats.rs).
    // `in_endian` tracks the interferogram's byte order so flat outputs match
    // the inputs (GAMMA in -> GAMMA out).
    let mut in_endian = if big_endian {
        Endian::Big
    } else {
        Endian::Little
    };
    let (igram_orig, ph): (Array2<Complex32>, Array2<f32>) = match (&ifg, &phase) {
        (Some(p), None) => {
            if is_tiff(p) {
                bail!(
                    "--ifg expects a flat-binary complex64 file; complex TIFF is not \
                     supported. Extract the phase first (gdal_translate \
                     DERIVED_SUBDATASET:PHASE:{} phase.tif) and pass --phase",
                    p.display()
                );
            }
            let m = formats::resolve_flat_meta(p, cols, ifg_meta.as_deref(), big_endian)?;
            formats::check_dtype(&m, p, formats::COMPLEX_DTYPES, "complex64/cfloat")?;
            if let Some(b) = m.bands
                && b != 1
            {
                bail!(
                    "{}: expected a 1-band complex interferogram, sidecar says {b} bands",
                    p.display()
                );
            }
            in_endian = m.endian;
            let ig = formats::read_flat_complex(p, m.cols, m.endian)?;
            formats::check_rows(&m, p, ig.nrows())?;
            let ph = ig.mapv(|z| z.arg());
            (ig, ph)
        }
        (None, Some(p)) => {
            let ph = if is_tiff(p) {
                read_f32_tiff(p)?
            } else {
                let m = formats::resolve_flat_meta(p, cols, phase_meta.as_deref(), big_endian)?;
                formats::check_dtype(&m, p, formats::FLOAT_DTYPES, "float32")?;
                in_endian = m.endian;
                let arr = formats::read_flat_float(p, m.cols, m.endian, FloatLayout::Single, None)?;
                formats::check_rows(&m, p, arr.nrows())?;
                arr
            };
            // The unwrapper consumes complex; reconstruct unit-magnitude
            // exp(i·phase). Internally only arg(z) is read on this path.
            (ph.mapv(|v| Complex32::from_polar(1.0, v)), ph)
        }
        _ => bail!("exactly one of --ifg / --phase is required"),
    };

    let co = if is_tiff(&cor) {
        read_f32_tiff(&cor)?
    } else {
        // The correlation always shares the interferogram's geometry, so its
        // column count never blocks on a sidecar; its own `<cor>.rsc/.xml`
        // (when present) still contributes dtype/band-count/byte-order.
        let m = formats::resolve_flat_meta(
            &cor,
            cols.or(Some(ph.ncols())),
            cor_meta.as_deref(),
            big_endian || in_endian == Endian::Big,
        )?;
        formats::check_dtype(&m, &cor, formats::FLOAT_DTYPES, "float32")?;
        let layout = match cor_format {
            CorFormat::Float => FloatLayout::Single,
            CorFormat::AltLine => FloatLayout::AltLine,
            CorFormat::AltSample => FloatLayout::AltSample,
            CorFormat::Bsq => FloatLayout::Bsq,
            CorFormat::Auto => match m.bands {
                Some(_) => formats::float_layout_from_meta(&m, &cor)?
                    .expect("banded sidecar should resolve to a layout"),
                None => FloatLayout::Auto,
            },
        };
        let arr = formats::read_flat_float(&cor, m.cols, m.endian, layout, Some(ph.nrows()))?;
        formats::check_rows(&m, &cor, arr.nrows())?;
        arr
    };
    if ph.dim() != co.dim() {
        return Err(anyhow!(
            "shape mismatch: phase={:?} cor={:?}",
            ph.dim(),
            co.dim()
        ));
    }
    // Resolve the valid mask. With `--mask` it is read from disk; without it we
    // mirror the Python default `mask = (igram != 0) & (corr > 0)` - on the
    // --phase path the igram is unit-magnitude, so that reduces to `corr > 0`.
    // We only materialize the default when some pixel would actually be masked;
    // an all-valid frame stays `None` to keep the unmasked solver fast path
    let mk: Option<Array2<bool>> = match mask.as_ref() {
        Some(p) => Some(if is_tiff(p) {
            read_bool_mask(p)?
        } else {
            formats::read_flat_mask(p, ph.ncols(), ph.nrows(), in_endian)?
        }),
        None => {
            let zero = Complex32::new(0.0, 0.0);
            let any_zero_ig = ifg.is_some() && igram_orig.iter().any(|&z| z == zero);
            if any_zero_ig || co.iter().any(|&c| c <= 0.0 || c.is_nan()) {
                let mut m = co.mapv(|c| c > 0.0);
                if any_zero_ig {
                    ndarray::Zip::from(&mut m)
                        .and(&igram_orig)
                        .for_each(|mv, &z| {
                            if z == zero {
                                *mv = false;
                            }
                        });
                }
                Some(m)
            } else {
                None
            }
        }
    };
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

    // Resolve the connected-component cost threshold the same way Python does:
    // `--conncomp-sigma` wins over `--conncomp-cycle-prob`, which wins over the
    // explicit `--cost-threshold`.
    let cost_threshold = if let Some(sigma) = conncomp_sigma {
        whirlwind_core::cost_threshold_from_sigma(sigma)
    } else if let Some(p) = conncomp_cycle_prob {
        whirlwind_core::cost_threshold_from_cycle_prob(p)
    } else {
        cost_threshold
    };

    // Build the phase fed to the MCF. Interpolation and Goldstein filtering both
    // only INFORM the solver; the integer 2π·k field they produce is transferred
    // back onto the ORIGINAL wrapped phase below, so every per-pixel value is
    // preserved. Order matches Python: interpolate, then Goldstein.
    let mut ig_solve = igram_orig.clone();
    let mut used_prepass = false;
    if interpolate {
        // Spiral PS interpolator: weights are clamped coherence (NaN -> 0),
        // matching `np.clip(np.nan_to_num(corr), 0, 1)`.
        let weights = co.mapv(|c| if c.is_nan() { 0.0 } else { c.clamp(0.0, 1.0) });
        ig_solve = whirlwind_core::interpolate::interpolate(
            ig_solve.view(),
            weights.view(),
            interp_cutoff,
            interp_num_neighbors,
            interp_max_radius,
            interp_min_radius,
            interp_alpha,
        );
        used_prepass = true;
    }
    if goldstein_alpha > 0.0 {
        ig_solve =
            whirlwind_core::goldstein::goldstein(ig_solve.view(), goldstein_alpha, goldstein_psize);
        used_prepass = true;
    }
    if used_prepass && let Some(m) = &mk {
        // A pre-pass produced a fresh array; zero masked pixels so the solver
        // sees the same nodata convention as the original phase.
        for ((i, j), &valid) in m.indexed_iter() {
            if !valid {
                ig_solve[(i, j)] = Complex32::new(0.0, 0.0);
            }
        }
    }

    // Unwrap. Unless --no-conncomp is set, use the variant that also grows
    // components.
    // `downsample` routes through the coherent-downlook-first path (multilook).
    let (unw_solve, cc_raster) = if conncomp_out.is_some() {
        let params = whirlwind_core::ConnCompParams {
            cost_threshold,
            min_size_px,
            // `--min-component-frac` only raises the absolute px floor on very
            // large frames.
            min_size_frac: min_component_frac,
            max_ncomps,
        };
        // Single-tile linear MCF phase + global (solve-free) conncomp;
        // tile_size=0 means whole-image single-tile (does NOT auto-tile).
        let (u, c) = whirlwind_core::unwrap_coherence_with_components(
            ig_solve.view(),
            co.view(),
            nlooks,
            mk.as_ref().map(|m| m.view()),
            0,
            0,
            downsample,
            params,
        )?;
        (u, Some(c))
    } else {
        let u = whirlwind_core::unwrap_coherence(
            ig_solve.view(),
            co.view(),
            nlooks,
            mk.as_ref().map(|m| m.view()),
            0,
            0,
            downsample,
        )?;
        (u, None)
    };

    // K-transfer to original wrapped phase (dolphin PR #364 convention).
    // Rounding against `ph` (the original, *unfiltered* phase) avoids the
    // spurious ±2π jumps at fringe boundaries. If no pre-pass ran, this is a
    // no-op (unw_solve is already congruent with ph).
    let tau = std::f32::consts::TAU;
    let mut unw = if used_prepass {
        let mut out_arr = Array2::<f32>::zeros(ph.dim());
        ndarray::Zip::from(&mut out_arr)
            .and(&ph)
            .and(&unw_solve)
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
        unw_solve
    };

    // Bridge post-pass (ON by default, matching Python): re-level regions the
    // valid mask splits into disconnected pieces.
    if bridge {
        unw = whirlwind_core::bridge_components(
            unw.view(),
            mk.as_ref().map(|m| m.view()),
            whirlwind_core::bridge::DEFAULT_RADIUS,
            whirlwind_core::bridge::DEFAULT_MIN_PX,
            whirlwind_core::bridge::DEFAULT_MAX_BOUNDARY,
        );
    }

    // ---- outputs --------------------------------------------------------
    match ofmt {
        OutFormat::Auto => unreachable!("Auto resolved above"),
        OutFormat::Tiff => write_f32_tiff(&out, unw.view())?,
        OutFormat::Float => formats::write_flat_float(&out, unw.view(), in_endian)?,
        OutFormat::AltLine => {
            // snaphu's default .unw layout: per row, the magnitude line then
            // the phase line. The magnitude is |igram| - all ones when the
            // input was --phase.
            let mag = igram_orig.mapv(|z| z.norm());
            formats::write_flat_altline(&out, mag.view(), unw.view(), in_endian)?;
        }
    }
    eprintln!("wrote {}", out.display());

    if let (Some(cc_path), Some(cc_arr)) = (conncomp_out.as_ref(), cc_raster.as_ref()) {
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

        if is_tiff(cc_path) {
            write_u16_tiff(cc_path, cc_arr.view())?;
        } else {
            // snaphu / isce2 convention: one byte per pixel.
            formats::write_flat_conncomp_u8(cc_path, cc_arr.view())?;
        }
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
    // lift the limit - this is a local CLI on trusted inputs, not a
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
