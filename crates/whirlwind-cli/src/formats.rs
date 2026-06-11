//! Flat-binary InSAR raster support (snaphu / ROI_PAC / isce2 / GAMMA).
//!
//! All of these are headerless rasters: the only metadata is the number of
//! columns (snaphu's "line length", ROI_PAC's `WIDTH`), the element type, the
//! band interleave, and the byte order. The number of rows is always derived
//! from the file size, exactly as snaphu does. Conventions implemented here
//! follow the reference implementations:
//!
//! - snaphu v2.0.7 `snaphu_io.c`: `COMPLEX_DATA` (interleaved float32
//!   real/imag pairs, the default infile format), `ALT_LINE_DATA` (alternating
//!   full lines from two arrays - the "rmg"/"hgt" format; first line
//!   amplitude, second line data - the default correlation/output format),
//!   `ALT_SAMPLE_DATA` (per-pixel interleaved pairs), and `FLOAT_DATA`
//!   (single-band float32). No headers, platform-native byte order.
//! - ROI_PAC / Stanford: `.int`/`.slc` = complex64, `.cc`/`.cor`/`.unw` =
//!   two-band line-interleaved float32 with amplitude first; geometry in a
//!   whitespace-keyed `<file>.rsc` (`WIDTH`, `FILE_LENGTH`).
//! - isce2: same flat layouts described by a `<file>.xml` Image-API sidecar
//!   (`width`, `length`, `number_bands`, `data_type` of `cfloat`/`float`,
//!   `scheme`, `byte_order` of `l`/`b`). Stack-processor `.cor` files are
//!   single-band float32.
//! - GAMMA: big-endian throughout; `fcomplex` (= complex64) interferograms,
//!   float32 coherence; geometry in colon-keyed `.par`/`.off`/`.diff_par`
//!   text files (`range_samples:`, `interferogram_width:`, ...).

use anyhow::{Context, Result, anyhow, bail};
use ndarray::{Array2, ArrayView2};
use num_complex::Complex32;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Endian {
    Little,
    Big,
}

/// Geometry/dtype metadata recovered from a sidecar file (`.rsc`, isce2
/// `.xml`, or GAMMA `.par`-style). Anything the sidecar does not state is
/// `None`.
#[derive(Debug, Default)]
pub struct Sidecar {
    pub cols: Option<usize>,
    pub rows: Option<usize>,
    /// isce2 `data_type`, lowercased (e.g. "cfloat", "float").
    pub dtype: Option<String>,
    pub bands: Option<usize>,
    /// isce2 band interleave scheme, lowercased (e.g. "bip", "bil", "bsq").
    pub scheme: Option<String>,
    pub big_endian: Option<bool>,
}

/// Fully resolved metadata for one flat-binary raster.
#[derive(Debug)]
pub struct FlatMeta {
    pub cols: usize,
    /// The sidecar's claimed row count, cross-checked against the file size
    /// after reading. `None` when no sidecar states it.
    pub rows: Option<usize>,
    pub endian: Endian,
    pub dtype: Option<String>,
    pub bands: Option<usize>,
    pub scheme: Option<String>,
}

/// Band layout of a flat float32 raster.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FloatLayout {
    /// Pick `Single` or `AltLine` from the file size (needs the expected row
    /// count). A two-band file is assumed line-interleaved - the snaphu and
    /// ROI_PAC default; pass `AltSample` explicitly for the rarer
    /// sample-interleaved variant, which has the same file size.
    Auto,
    /// One band of float32 (snaphu `FLOAT_DATA`, isce2 `.cor`, GAMMA `.cc`).
    Single,
    /// Two bands, alternating full lines: amplitude line then data line
    /// (snaphu `ALT_LINE_DATA`, the ROI_PAC/Stanford "rmg" `.cc`/`.unw`
    /// layout). The data is read from the SECOND line of each pair.
    AltLine,
    /// Two bands, alternating samples: the data is every odd sample
    /// (snaphu `ALT_SAMPLE_DATA`).
    AltSample,
    /// Two bands, band sequential: the data is the second full band
    /// (isce2/GDAL `scheme=BSQ`).
    Bsq,
}

// ---------------------------------------------------------------------------
// Sidecar parsing
// ---------------------------------------------------------------------------

/// Parse a ROI_PAC `.rsc` file: whitespace-separated `KEY value` lines.
pub fn parse_rsc(path: &Path) -> Result<Sidecar> {
    let text = fs::read_to_string(path).with_context(|| format!("read rsc {}", path.display()))?;
    let mut sc = Sidecar::default();
    for line in text.lines() {
        let mut it = line.split_whitespace();
        let (Some(key), Some(val)) = (it.next(), it.next()) else {
            continue;
        };
        match key {
            "WIDTH" => {
                sc.cols = Some(
                    val.parse()
                        .with_context(|| format!("bad WIDTH {val:?} in {}", path.display()))?,
                )
            }
            "FILE_LENGTH" => {
                sc.rows =
                    Some(val.parse().with_context(|| {
                        format!("bad FILE_LENGTH {val:?} in {}", path.display())
                    })?)
            }
            _ => {}
        }
    }
    if sc.cols.is_none() {
        bail!("no WIDTH key in {}", path.display());
    }
    Ok(sc)
}

/// Pull `<property name="{name}"><value>...</value></property>` out of an
/// isce2 Image-API XML sidecar. This is a deliberately small scanner scoped
/// to the very regular XML that isce2's `XmlDumper` emits - not a general
/// XML parser.
fn xml_property<'a>(text: &'a str, name: &str) -> Option<&'a str> {
    let dq = format!("name=\"{name}\"");
    let sq = format!("name='{name}'");
    let pos = text.find(&dq).or_else(|| text.find(&sq))?;
    let rest = &text[pos..];
    let vstart = rest.find("<value>")? + "<value>".len();
    let vend = rest[vstart..].find("</value>")? + vstart;
    Some(rest[vstart..vend].trim())
}

/// Parse an isce2 `<file>.xml` Image-API sidecar.
pub fn parse_isce_xml(path: &Path) -> Result<Sidecar> {
    let text = fs::read_to_string(path).with_context(|| format!("read xml {}", path.display()))?;
    let mut sc = Sidecar::default();
    if let Some(v) = xml_property(&text, "width") {
        sc.cols = Some(
            v.parse()
                .with_context(|| format!("bad width {v:?} in {}", path.display()))?,
        );
    }
    if let Some(v) = xml_property(&text, "length") {
        sc.rows = Some(
            v.parse()
                .with_context(|| format!("bad length {v:?} in {}", path.display()))?,
        );
    }
    if let Some(v) = xml_property(&text, "number_bands") {
        sc.bands = Some(
            v.parse()
                .with_context(|| format!("bad number_bands {v:?} in {}", path.display()))?,
        );
    }
    if let Some(v) = xml_property(&text, "scheme") {
        sc.scheme = Some(v.to_ascii_lowercase());
    }
    if let Some(v) = xml_property(&text, "data_type") {
        sc.dtype = Some(v.to_ascii_lowercase());
    }
    if let Some(v) = xml_property(&text, "byte_order") {
        // isce2 stores 'l' (little) or 'b' (big).
        sc.big_endian = Some(v.to_ascii_lowercase().starts_with('b'));
    }
    if sc.cols.is_none() {
        bail!(
            "no <property name=\"width\"> in {} (is this an isce2 image XML?)",
            path.display()
        );
    }
    Ok(sc)
}

/// Parse a GAMMA parameter file (`.par`, `.off`, `.diff_par`, ...):
/// colon-keyed `key: value [units]` lines. GAMMA rasters are big-endian, so
/// a GAMMA sidecar implies `big_endian = true`.
pub fn parse_gamma_par(path: &Path) -> Result<Sidecar> {
    let text = fs::read_to_string(path).with_context(|| format!("read par {}", path.display()))?;
    let mut sc = Sidecar {
        big_endian: Some(true),
        ..Default::default()
    };
    for line in text.lines() {
        let Some((key, val)) = line.split_once(':') else {
            continue;
        };
        let key = key.trim();
        let Some(val) = val.split_whitespace().next() else {
            continue;
        };
        match key {
            "width" | "range_samp" | "range_samples" | "interferogram_width"
                if sc.cols.is_none() =>
            {
                sc.cols = Some(
                    val.parse()
                        .with_context(|| format!("bad {key} {val:?} in {}", path.display()))?,
                )
            }
            "nlines" | "az_samp" | "azimuth_lines" | "interferogram_azimuth_lines"
                if sc.rows.is_none() =>
            {
                sc.rows = Some(
                    val.parse()
                        .with_context(|| format!("bad {key} {val:?} in {}", path.display()))?,
                )
            }
            _ => {}
        }
    }
    if sc.cols.is_none() {
        bail!(
            "no width key (width/range_samples/interferogram_width) in {}",
            path.display()
        );
    }
    Ok(sc)
}

/// Parse any supported sidecar, dispatching on its extension: `.rsc` ->
/// ROI_PAC, `.xml` -> isce2, anything else (`.par`, `.off`, `.diff_par`,
/// ...) -> GAMMA colon-keyed text.
pub fn parse_sidecar(path: &Path) -> Result<Sidecar> {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())
        .as_deref()
    {
        Some("rsc") => parse_rsc(path),
        Some("xml") => parse_isce_xml(path),
        _ => parse_gamma_par(path),
    }
}

/// Look for an auto-discoverable sidecar next to a data file: `<file>.rsc`
/// (ROI_PAC) or `<file>.xml` (isce2), in that order. GAMMA `.par` files have
/// no per-file naming convention, so they are only reachable via an explicit
/// metadata flag.
pub fn find_sidecar(data_path: &Path) -> Option<PathBuf> {
    for ext in ["rsc", "xml"] {
        let mut name = data_path.as_os_str().to_owned();
        name.push(".");
        name.push(ext);
        let p = PathBuf::from(name);
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

/// Resolve the metadata needed to read one flat-binary raster, combining the
/// CLI flags with any sidecar. Priority: `--cols` flag > sidecar. The
/// sidecar is the matching explicit metadata flag when given, else an
/// auto-discovered `<file>.rsc` / `<file>.xml`. Byte order is big when
/// `--big-endian` is passed OR the sidecar says so (isce2 `byte_order`, or
/// any GAMMA `.par`).
pub fn resolve_flat_meta(
    data_path: &Path,
    cols_flag: Option<usize>,
    meta_flag: Option<&Path>,
    big_endian_flag: bool,
) -> Result<FlatMeta> {
    let sidecar = match meta_flag {
        Some(p) => Some(parse_sidecar(p)?),
        None => find_sidecar(data_path)
            .map(|p| parse_sidecar(&p))
            .transpose()?,
    };
    let sc = sidecar.as_ref();
    let cols = cols_flag
        .or_else(|| sc.and_then(|s| s.cols))
        .ok_or_else(|| {
            anyhow!(
                "cannot determine the number of columns for {}: pass --cols N \
             (snaphu's line length / ROI_PAC WIDTH), or provide a sidecar \
             ({0}.rsc, {0}.xml, or the matching --*-meta <par/rsc/xml>)",
                data_path.display()
            )
        })?;
    let big = big_endian_flag || sc.and_then(|s| s.big_endian).unwrap_or(false);
    Ok(FlatMeta {
        cols,
        rows: sc.and_then(|s| s.rows),
        endian: if big { Endian::Big } else { Endian::Little },
        dtype: sc.and_then(|s| s.dtype.clone()),
        bands: sc.and_then(|s| s.bands),
        scheme: sc.and_then(|s| s.scheme.clone()),
    })
}

/// Sidecar `data_type` strings acceptable for a complex64 interferogram
/// (isce2 `cfloat`, GAMMA `fcomplex`, numpy spellings).
pub const COMPLEX_DTYPES: &[&str] = &["cfloat", "complex64", "fcomplex", "c8"];
/// Sidecar `data_type` strings acceptable for a float32 raster.
pub const FLOAT_DTYPES: &[&str] = &["float", "float32", "f4", "real4", "real*4"];

/// Fail early when a sidecar declares a data type other than what this input
/// must be (e.g. an isce2 `.cor` XML handed to `--ifg`).
pub fn check_dtype(meta: &FlatMeta, path: &Path, ok: &[&str], want: &str) -> Result<()> {
    if let Some(dt) = &meta.dtype
        && !ok.contains(&dt.as_str())
    {
        bail!(
            "{}: sidecar says data_type {dt:?}, but this input must be {want}",
            path.display()
        );
    }
    Ok(())
}

/// Cross-check a sidecar's claimed row count against the size-derived one.
pub fn check_rows(meta: &FlatMeta, path: &Path, got: usize) -> Result<()> {
    if let Some(r) = meta.rows
        && r != got
    {
        bail!(
            "{}: sidecar says {r} rows but the file size gives {got} - \
             wrong --cols or truncated file?",
            path.display()
        );
    }
    Ok(())
}

/// Resolve a float raster's band layout from sidecar band count and isce2
/// interleave scheme. `None` means the sidecar did not say enough and the
/// caller should fall back to size-based auto detection.
pub fn float_layout_from_meta(meta: &FlatMeta, path: &Path) -> Result<Option<FloatLayout>> {
    match meta.bands {
        Some(1) => Ok(Some(FloatLayout::Single)),
        Some(2) => match meta.scheme.as_deref() {
            // isce2/GDAL names.
            Some("bil") => Ok(Some(FloatLayout::AltLine)),
            Some("bip") => Ok(Some(FloatLayout::AltSample)),
            Some("bsq") => Ok(Some(FloatLayout::Bsq)),
            // ROI_PAC/snaphu sidecars often know the band count but not an
            // explicit scheme; their two-band convention is line interleave.
            None => Ok(Some(FloatLayout::AltLine)),
            Some(s) => bail!(
                "{}: unsupported two-band float scheme {s:?} (need BIL, BIP, or BSQ)",
                path.display()
            ),
        },
        Some(b) => bail!(
            "{}: unsupported float band count {b} (need 1 or 2)",
            path.display()
        ),
        None => Ok(None),
    }
}

// ---------------------------------------------------------------------------
// Flat readers
// ---------------------------------------------------------------------------

fn decode_f32(bytes: &[u8], endian: Endian) -> Vec<f32> {
    let conv = match endian {
        Endian::Little => f32::from_le_bytes,
        Endian::Big => f32::from_be_bytes,
    };
    bytes
        .chunks_exact(4)
        .map(|c| conv(c.try_into().unwrap()))
        .collect()
}

fn read_sized(path: &Path, cols: usize, bytes_per_px: usize) -> Result<(Vec<u8>, usize)> {
    let bytes = fs::read(path).with_context(|| format!("open {}", path.display()))?;
    let bpr = cols
        .checked_mul(bytes_per_px)
        .ok_or_else(|| anyhow!("--cols {cols} overflows"))?;
    if bytes.is_empty() || bpr == 0 || bytes.len() % bpr != 0 {
        bail!(
            "{}: file size {} is not a whole number of {}-column rows \
             ({} bytes per row at {} bytes per pixel) - wrong --cols?",
            path.display(),
            bytes.len(),
            cols,
            bpr,
            bytes_per_px,
        );
    }
    let lines = bytes.len() / bpr;
    Ok((bytes, lines))
}

/// Read a flat complex64 interferogram (snaphu `COMPLEX_DATA`, ROI_PAC /
/// isce2 `.int`, GAMMA `fcomplex` `.int`/`.diff`): interleaved float32
/// real/imag pairs, `cols` complex samples per row, rows from the file size.
pub fn read_flat_complex(path: &Path, cols: usize, endian: Endian) -> Result<Array2<Complex32>> {
    let (bytes, rows) = read_sized(path, cols, 8)?;
    let floats = decode_f32(&bytes, endian);
    let data: Vec<Complex32> = floats
        .chunks_exact(2)
        .map(|p| Complex32::new(p[0], p[1]))
        .collect();
    Ok(Array2::from_shape_vec((rows, cols), data)?)
}

/// Read a flat float32 raster in any of the snaphu scalar layouts. For the
/// two-band layouts the returned band is the SECOND channel (snaphu reads
/// correlation from the second channel; ROI_PAC `.cc`/`.unw` store amplitude
/// first). `expected_rows` (from the already-loaded interferogram) drives
/// `Auto` layout selection and is cross-checked for the rest.
pub fn read_flat_float(
    path: &Path,
    cols: usize,
    endian: Endian,
    layout: FloatLayout,
    expected_rows: Option<usize>,
) -> Result<Array2<f32>> {
    let (bytes, lines) = read_sized(path, cols, 4)?;
    let layout = match layout {
        FloatLayout::Auto => match expected_rows {
            Some(er) if lines == er => FloatLayout::Single,
            Some(er) if lines == 2 * er => FloatLayout::AltLine,
            Some(er) => bail!(
                "{}: {} lines of {} float32 columns; expected {} (single-band) \
                 or {} (two-band amp+data) to match the interferogram",
                path.display(),
                lines,
                cols,
                er,
                2 * er,
            ),
            None => FloatLayout::Single,
        },
        other => other,
    };
    let floats = decode_f32(&bytes, endian);
    let arr = match layout {
        FloatLayout::Auto => unreachable!("Auto resolved above"),
        FloatLayout::Single => Array2::from_shape_vec((lines, cols), floats)?,
        FloatLayout::AltLine => {
            if lines % 2 != 0 {
                bail!(
                    "{}: odd line count {} for the two-band line-interleaved layout",
                    path.display(),
                    lines
                );
            }
            let rows = lines / 2;
            // Data is the second line of each (amplitude, data) pair.
            let mut data = Vec::with_capacity(rows * cols);
            for r in 0..rows {
                let start = (2 * r + 1) * cols;
                data.extend_from_slice(&floats[start..start + cols]);
            }
            Array2::from_shape_vec((rows, cols), data)?
        }
        FloatLayout::AltSample => {
            if lines % 2 != 0 {
                bail!(
                    "{}: file size does not fit the two-band sample-interleaved layout",
                    path.display()
                );
            }
            let rows = lines / 2;
            // Data is every odd sample of each 2*cols-float row.
            let mut data = Vec::with_capacity(rows * cols);
            for r in 0..rows {
                let row = &floats[r * 2 * cols..(r + 1) * 2 * cols];
                data.extend(row.iter().skip(1).step_by(2));
            }
            Array2::from_shape_vec((rows, cols), data)?
        }
        FloatLayout::Bsq => {
            if lines % 2 != 0 {
                bail!(
                    "{}: file size does not fit the two-band band-sequential layout",
                    path.display()
                );
            }
            let rows = lines / 2;
            let start = rows * cols;
            Array2::from_shape_vec((rows, cols), floats[start..start + rows * cols].to_vec())?
        }
    };
    if let Some(er) = expected_rows
        && arr.nrows() != er
    {
        bail!(
            "{}: {} rows, but the interferogram has {}",
            path.display(),
            arr.nrows(),
            er
        );
    }
    Ok(arr)
}

/// Read a flat validity mask: either one byte per pixel (snaphu
/// `BYTEMASKFILE`; nonzero = valid) or float32 (nonzero finite = valid),
/// chosen by file size.
pub fn read_flat_mask(
    path: &Path,
    cols: usize,
    expected_rows: usize,
    endian: Endian,
) -> Result<Array2<bool>> {
    let bytes = fs::read(path).with_context(|| format!("open {}", path.display()))?;
    let n = expected_rows * cols;
    let data: Vec<bool> = if bytes.len() == n {
        bytes.iter().map(|&b| b != 0).collect()
    } else if bytes.len() == 4 * n {
        decode_f32(&bytes, endian)
            .into_iter()
            .map(|v| v.is_finite() && v != 0.0)
            .collect()
    } else {
        bail!(
            "{}: mask size {} matches neither {} bytes (u8) nor {} (float32) \
             for a {}x{} frame",
            path.display(),
            bytes.len(),
            n,
            4 * n,
            expected_rows,
            cols,
        );
    };
    Ok(Array2::from_shape_vec((expected_rows, cols), data)?)
}

// ---------------------------------------------------------------------------
// Flat writers
// ---------------------------------------------------------------------------

fn f32_bytes(v: f32, endian: Endian) -> [u8; 4] {
    match endian {
        Endian::Little => v.to_le_bytes(),
        Endian::Big => v.to_be_bytes(),
    }
}

/// Write a single-band flat float32 raster (snaphu `FLOAT_DATA`).
pub fn write_flat_float(path: &Path, a: ArrayView2<f32>, endian: Endian) -> Result<()> {
    let mut buf = Vec::with_capacity(a.len() * 4);
    for &v in a.iter() {
        buf.extend_from_slice(&f32_bytes(v, endian));
    }
    fs::write(path, buf).with_context(|| format!("write {}", path.display()))
}

/// Write a two-band line-interleaved float32 raster (snaphu `ALT_LINE_DATA`,
/// the ROI_PAC "rmg" `.unw` layout): per row, the magnitude line then the
/// data line.
pub fn write_flat_altline(
    path: &Path,
    mag: ArrayView2<f32>,
    data: ArrayView2<f32>,
    endian: Endian,
) -> Result<()> {
    assert_eq!(mag.dim(), data.dim(), "mag/data shape mismatch");
    let (rows, cols) = data.dim();
    let mut buf = Vec::with_capacity(rows * cols * 8);
    for r in 0..rows {
        for c in 0..cols {
            buf.extend_from_slice(&f32_bytes(mag[(r, c)], endian));
        }
        for c in 0..cols {
            buf.extend_from_slice(&f32_bytes(data[(r, c)], endian));
        }
    }
    fs::write(path, buf).with_context(|| format!("write {}", path.display()))
}

/// Write connected-component labels as one byte per pixel (the snaphu and
/// isce2 `.conncomp` convention). Labels above 255 are an error; use a
/// `.tif` output (u16) or a smaller `--max-ncomps` instead.
pub fn write_flat_conncomp_u8(path: &Path, cc: ArrayView2<u32>) -> Result<()> {
    let mut buf = Vec::with_capacity(cc.len());
    for &v in cc.iter() {
        if v > u8::MAX as u32 {
            bail!(
                "component label {v} overflows the 1-byte flat conncomp format; \
                 write a .tif conncomp (u16) or lower --max-ncomps to <= 255"
            );
        }
        buf.push(v as u8);
    }
    fs::write(path, buf).with_context(|| format!("write {}", path.display()))
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use std::io::Write;

    fn tmpfile(name: &str, bytes: &[u8]) -> PathBuf {
        let dir = std::env::temp_dir().join("whirlwind-formats-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join(name);
        let mut f = std::fs::File::create(&p).unwrap();
        f.write_all(bytes).unwrap();
        p
    }

    fn le_floats(vals: &[f32]) -> Vec<u8> {
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn be_floats(vals: &[f32]) -> Vec<u8> {
        vals.iter().flat_map(|v| v.to_be_bytes()).collect()
    }

    #[test]
    fn rsc_parse() {
        let p = tmpfile(
            "t.rsc",
            b"WIDTH         840\nFILE_LENGTH   1200\nX_FIRST       90.1\nPROJECTION    LL\n",
        );
        let sc = parse_rsc(&p).unwrap();
        assert_eq!(sc.cols, Some(840));
        assert_eq!(sc.rows, Some(1200));
        assert_eq!(sc.big_endian, None);
    }

    #[test]
    fn isce_xml_parse() {
        let xml = r#"<imageFile>
    <property name="byte_order"><value>l</value></property>
    <component name="coordinate1">
        <property name="size"><value>9999</value></property>
    </component>
    <property name="data_type"><value>cfloat</value></property>
    <property name="length"><value>200</value></property>
    <property name="number_bands"><value>1</value></property>
    <property name="scheme"><value>BIP</value></property>
    <property name="width"><value>1000</value></property>
</imageFile>"#;
        let p = tmpfile("t.int.xml", xml.as_bytes());
        let sc = parse_isce_xml(&p).unwrap();
        assert_eq!(sc.cols, Some(1000));
        assert_eq!(sc.rows, Some(200));
        assert_eq!(sc.bands, Some(1));
        assert_eq!(sc.dtype.as_deref(), Some("cfloat"));
        assert_eq!(sc.scheme.as_deref(), Some("bip"));
        assert_eq!(sc.big_endian, Some(false));
    }

    #[test]
    fn gamma_par_parse() {
        let par = "Gamma DIFF&GEO Processing Parameters\ntitle: test\n\
                   interferogram_width: 2500\ninterferogram_azimuth_lines: 3000\n\
                   range_looks: 2\n";
        let p = tmpfile("t.off", par.as_bytes());
        let sc = parse_gamma_par(&p).unwrap();
        assert_eq!(sc.cols, Some(2500));
        assert_eq!(sc.rows, Some(3000));
        assert_eq!(
            sc.big_endian,
            Some(true),
            "GAMMA sidecar implies big-endian"
        );
    }

    #[test]
    fn flat_complex_roundtrip_both_endians() {
        // 2x3, value (re, im) = (r+1, 10c)
        let vals = [
            1.0, 0.0, 1.0, 10.0, 1.0, 20.0, //
            2.0, 0.0, 2.0, 10.0, 2.0, 20.0,
        ];
        let p_le = tmpfile("c.int", &le_floats(&vals));
        let p_be = tmpfile("c_be.int", &be_floats(&vals));
        let a = read_flat_complex(&p_le, 3, Endian::Little).unwrap();
        let b = read_flat_complex(&p_be, 3, Endian::Big).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.dim(), (2, 3));
        assert_eq!(a[(1, 2)], Complex32::new(2.0, 20.0));
    }

    #[test]
    fn flat_complex_bad_cols() {
        let p = tmpfile("bad.int", &le_floats(&[0.0; 6])); // 24 bytes
        // 24 % (5*8) != 0
        assert!(read_flat_complex(&p, 5, Endian::Little).is_err());
    }

    #[test]
    fn float_layouts() {
        // 2 rows x 3 cols; amp = 100+, data = correlation-like
        let amp = [100.0, 101.0, 102.0, 200.0, 201.0, 202.0];
        let dat = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];

        // single band
        let p = tmpfile("s.cor", &le_floats(&dat));
        let a = read_flat_float(&p, 3, Endian::Little, FloatLayout::Auto, Some(2)).unwrap();
        assert_eq!(a, array![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]);

        // alt-line (rmg): amp line, data line, per row -> auto-detected 2-band
        let mut rmg = Vec::new();
        for r in 0..2 {
            rmg.extend_from_slice(&amp[r * 3..(r + 1) * 3]);
            rmg.extend_from_slice(&dat[r * 3..(r + 1) * 3]);
        }
        let p = tmpfile("al.cc", &le_floats(&rmg));
        let a = read_flat_float(&p, 3, Endian::Little, FloatLayout::Auto, Some(2)).unwrap();
        assert_eq!(
            a,
            array![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            "second channel"
        );

        // alt-sample: interleaved (amp, data) pairs
        let mut alts = Vec::new();
        for i in 0..6 {
            alts.push(amp[i]);
            alts.push(dat[i]);
        }
        let p = tmpfile("as.cor", &le_floats(&alts));
        let a = read_flat_float(&p, 3, Endian::Little, FloatLayout::AltSample, Some(2)).unwrap();
        assert_eq!(a, array![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], "odd samples");

        // bsq: all amplitude rows, then all data rows.
        let mut bsq = Vec::new();
        bsq.extend_from_slice(&amp);
        bsq.extend_from_slice(&dat);
        let p = tmpfile("bsq.cor", &le_floats(&bsq));
        let a = read_flat_float(&p, 3, Endian::Little, FloatLayout::Bsq, Some(2)).unwrap();
        assert_eq!(a, array![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], "second band");

        // size matching neither 1- nor 2-band is an error
        let p = tmpfile("bad.cor", &le_floats(&[0.0; 9]));
        assert!(read_flat_float(&p, 3, Endian::Little, FloatLayout::Auto, Some(2)).is_err());
    }

    #[test]
    fn isce_scheme_selects_float_layout() {
        let path = Path::new("c.cor");
        let mut meta = FlatMeta {
            cols: 3,
            rows: Some(2),
            endian: Endian::Little,
            dtype: Some("float".to_string()),
            bands: Some(2),
            scheme: Some("bil".to_string()),
        };
        assert_eq!(
            float_layout_from_meta(&meta, path).unwrap(),
            Some(FloatLayout::AltLine)
        );
        meta.scheme = Some("bip".to_string());
        assert_eq!(
            float_layout_from_meta(&meta, path).unwrap(),
            Some(FloatLayout::AltSample)
        );
        meta.scheme = Some("bsq".to_string());
        assert_eq!(
            float_layout_from_meta(&meta, path).unwrap(),
            Some(FloatLayout::Bsq)
        );
    }

    #[test]
    fn mask_byte_and_float() {
        let p = tmpfile("m.msk", &[1u8, 0, 2, 0, 1, 1]);
        let m = read_flat_mask(&p, 3, 2, Endian::Little).unwrap();
        assert_eq!(m, array![[true, false, true], [false, true, true]]);
        let p = tmpfile(
            "m_f32.msk",
            &le_floats(&[1.0, 0.0, f32::NAN, 2.0, 1.0, 1.0]),
        );
        let m = read_flat_mask(&p, 3, 2, Endian::Little).unwrap();
        assert_eq!(m, array![[true, false, false], [true, true, true]]);
    }

    #[test]
    fn altline_write_read_roundtrip() {
        let mag = array![[1.0_f32, 2.0], [3.0, 4.0]];
        let dat = array![[5.0_f32, 6.0], [7.0, 8.0]];
        let p = std::env::temp_dir()
            .join("whirlwind-formats-tests")
            .join("rt.unw");
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        write_flat_altline(&p, mag.view(), dat.view(), Endian::Big).unwrap();
        let back = read_flat_float(&p, 2, Endian::Big, FloatLayout::AltLine, Some(2)).unwrap();
        assert_eq!(back, dat);
    }

    #[test]
    fn conncomp_u8_overflow() {
        let dir = std::env::temp_dir().join("whirlwind-formats-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let ok = array![[0u32, 1], [255, 2]];
        write_flat_conncomp_u8(&dir.join("cc.conncomp"), ok.view()).unwrap();
        assert_eq!(
            std::fs::read(dir.join("cc.conncomp")).unwrap(),
            vec![0u8, 1, 255, 2]
        );
        let bad = array![[0u32, 256]];
        assert!(write_flat_conncomp_u8(&dir.join("cc2.conncomp"), bad.view()).is_err());
    }

    #[test]
    fn resolve_precedence() {
        // data file with an .rsc next to it
        let dir = std::env::temp_dir().join("whirlwind-formats-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let data = dir.join("geom.int");
        std::fs::write(&data, [0u8; 16]).unwrap();
        std::fs::write(dir.join("geom.int.rsc"), b"WIDTH 2\nFILE_LENGTH 1\n").unwrap();
        let r = resolve_flat_meta(&data, None, None, false).unwrap();
        assert_eq!((r.cols, r.rows, r.endian), (2, Some(1), Endian::Little));
        // explicit --cols beats the sidecar
        let r = resolve_flat_meta(&data, Some(4), None, false).unwrap();
        assert_eq!(r.cols, 4);
        // an explicit metadata flag pointing at a GAMMA par flips to big-endian
        let par = dir.join("geom.off");
        std::fs::write(&par, b"interferogram_width: 2\n").unwrap();
        let r = resolve_flat_meta(&data, None, Some(&par), false).unwrap();
        assert_eq!((r.cols, r.endian), (2, Endian::Big));
        // no cols anywhere -> error
        let lone = dir.join("lone.int");
        std::fs::write(&lone, [0u8; 16]).unwrap();
        assert!(resolve_flat_meta(&lone, None, None, false).is_err());
    }
}
