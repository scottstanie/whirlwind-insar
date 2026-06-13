//! End-to-end CLI tests for the flat-binary formats (snaphu /
//! ROI_PAC / isce2 / GAMMA): the same scene fed as TIFF and as each flat
//! layout must produce bit-identical unwrapped output.
//!
//! The TIFF `--phase` path reconstructs `exp(i*phase)`; the flat `--ifg`
//! files below store exactly `(cos(phase), sin(phase))`, so both paths hand
//! the solver identical complex values and the outputs compare with `==`.

use std::f32::consts::PI;
use std::fs;
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::process::Command;
use tiff::encoder::{TiffEncoder, colortype};

const ROWS: usize = 48;
const COLS: usize = 40;

/// A wrapped diagonal ramp (several fringes) and uniform high coherence:
/// deterministic and trivially unwrappable.
fn scene() -> (Vec<f32>, Vec<f32>) {
    let mut ph = Vec::with_capacity(ROWS * COLS);
    for r in 0..ROWS {
        for c in 0..COLS {
            let v = 0.35_f32 * (r as f32 + c as f32);
            ph.push((v + PI).rem_euclid(2.0 * PI) - PI);
        }
    }
    (ph, vec![0.95_f32; ROWS * COLS])
}

fn tdir(name: &str) -> PathBuf {
    let d = std::env::temp_dir()
        .join("whirlwind-cli-flat-tests")
        .join(name);
    fs::create_dir_all(&d).unwrap();
    d
}

fn write_tiff(path: &Path, data: &[f32]) {
    let mut enc = TiffEncoder::new(BufWriter::new(File::create(path).unwrap())).unwrap();
    enc.write_image::<colortype::Gray32Float>(COLS as u32, ROWS as u32, data)
        .unwrap();
}

fn le(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn be(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_be_bytes()).collect()
}

/// Interleaved (re, im) = (cos, sin) of the wrapped phase.
fn complex_pairs(ph: &[f32]) -> Vec<f32> {
    ph.iter().flat_map(|&p| [p.cos(), p.sin()]).collect()
}

/// ROI_PAC/snaphu "rmg": per row, an amplitude line then the data line. The
/// amplitude values are deliberate garbage so a band-order bug shows up.
fn altline(data: &[f32], fill: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(2 * data.len());
    for r in 0..ROWS {
        out.extend(std::iter::repeat_n(fill, COLS));
        out.extend_from_slice(&data[r * COLS..(r + 1) * COLS]);
    }
    out
}

/// Per-pixel interleaved amp/data pairs.
fn altsample(data: &[f32], fill: f32) -> Vec<f32> {
    data.iter().flat_map(|&v| [fill, v]).collect()
}

/// Band-sequential amp band followed by data band.
fn bsq(data: &[f32], fill: f32) -> Vec<f32> {
    let mut out = vec![fill; data.len()];
    out.extend_from_slice(data);
    out
}

fn run(args: &[&str]) {
    let out = Command::new(env!("CARGO_BIN_EXE_whirlwind"))
        .args(args)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "whirlwind {args:?} failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

fn run_expect_fail(args: &[&str]) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_whirlwind"))
        .args(args)
        .output()
        .unwrap();
    assert!(
        !out.status.success(),
        "whirlwind {args:?} unexpectedly succeeded"
    );
    String::from_utf8_lossy(&out.stderr).into_owned()
}

/// Unwrap via the TIFF `--phase` path; the reference all flat paths must hit.
fn baseline(dir: &Path, ph: &[f32], cor: &[f32]) -> Vec<u8> {
    let ph_tif = dir.join("ph.tif");
    let cor_tif = dir.join("cor.tif");
    let out = dir.join("ref.tif");
    write_tiff(&ph_tif, ph);
    write_tiff(&cor_tif, cor);
    run(&[
        "--phase",
        ph_tif.to_str().unwrap(),
        "--cor",
        cor_tif.to_str().unwrap(),
        "--nlooks",
        "10",
        "--out",
        out.to_str().unwrap(),
    ]);
    fs::read(out).unwrap()
}

#[test]
fn flat_complex_and_cor_layouts_match_tiff() {
    let dir = tdir("layouts");
    let (ph, cor) = scene();
    let reference = baseline(&dir, &ph, &cor);

    let int = dir.join("ifg.int");
    fs::write(&int, le(&complex_pairs(&ph))).unwrap();

    // single-band float32 correlation (snaphu FLOAT_DATA / isce2 .cor)
    let cor_flat = dir.join("c.cor");
    fs::write(&cor_flat, le(&cor)).unwrap();
    // two-band line-interleaved with garbage amplitude (snaphu default / .cc)
    let cor_rmg = dir.join("c.cc");
    fs::write(&cor_rmg, le(&altline(&cor, 12345.0))).unwrap();
    // isce/GDAL two-band BIP and BSQ variants. These have the same byte count
    // as BIL, so the XML scheme must drive the layout.
    let cor_bip = dir.join("c_bip.cor");
    fs::write(&cor_bip, le(&altsample(&cor, 12345.0))).unwrap();
    fs::write(
        dir.join("c_bip.cor.xml"),
        r#"<imageFile>
  <property name="byte_order"><value>l</value></property>
  <property name="data_type"><value>float</value></property>
  <property name="length"><value>48</value></property>
  <property name="number_bands"><value>2</value></property>
  <property name="scheme"><value>BIP</value></property>
  <property name="width"><value>40</value></property>
</imageFile>"#,
    )
    .unwrap();
    let cor_bsq = dir.join("c_bsq.cor");
    fs::write(&cor_bsq, le(&bsq(&cor, 12345.0))).unwrap();
    fs::write(
        dir.join("c_bsq.cor.xml"),
        r#"<imageFile>
  <property name="byte_order"><value>l</value></property>
  <property name="data_type"><value>float</value></property>
  <property name="length"><value>48</value></property>
  <property name="number_bands"><value>2</value></property>
  <property name="scheme"><value>BSQ</value></property>
  <property name="width"><value>40</value></property>
</imageFile>"#,
    )
    .unwrap();

    for cor_file in [&cor_flat, &cor_rmg, &cor_bip, &cor_bsq] {
        let out = dir.join("out.tif");
        run(&[
                "--ifg",
            int.to_str().unwrap(),
            "--cols",
            "40",
            "--cor",
            cor_file.to_str().unwrap(),
            "--nlooks",
            "10",
            "--out",
            out.to_str().unwrap(),
        ]);
        assert_eq!(
            fs::read(&out).unwrap(),
            reference,
            "flat path with {cor_file:?} diverged from the TIFF path"
        );
    }
}

#[test]
fn flat_phase_input_matches_tiff() {
    let dir = tdir("flat-phase");
    let (ph, cor) = scene();
    let reference = baseline(&dir, &ph, &cor);

    let ph_flat = dir.join("ph.flat");
    let cor_flat = dir.join("c.cor");
    fs::write(&ph_flat, le(&ph)).unwrap();
    fs::write(&cor_flat, le(&cor)).unwrap();
    let out = dir.join("out.tif");
    run(&[
        "--phase",
        ph_flat.to_str().unwrap(),
        "--cols",
        "40",
        "--cor",
        cor_flat.to_str().unwrap(),
        "--nlooks",
        "10",
        "--out",
        out.to_str().unwrap(),
    ]);
    assert_eq!(fs::read(&out).unwrap(), reference);
    assert!(
        dir.join("out.conncomp.tif").is_file(),
        "TIFF output should get a default conncomp TIFF"
    );

    let out_no_cc = dir.join("out_no_cc.tif");
    let no_cc_default = dir.join("out_no_cc.conncomp.tif");
    let _ = fs::remove_file(&no_cc_default);
    run(&[
        "--phase",
        ph_flat.to_str().unwrap(),
        "--cols",
        "40",
        "--cor",
        cor_flat.to_str().unwrap(),
        "--nlooks",
        "10",
        "--out",
        out_no_cc.to_str().unwrap(),
        "--no-conncomp",
    ]);
    assert!(
        !no_cc_default.exists(),
        "--no-conncomp should suppress the default conncomp output"
    );
}

#[test]
fn big_endian_gamma_style() {
    let dir = tdir("big-endian");
    let (ph, cor) = scene();
    let reference = baseline(&dir, &ph, &cor);

    let int = dir.join("ifg.diff");
    let cc = dir.join("c.cc");
    fs::write(&int, be(&complex_pairs(&ph))).unwrap();
    fs::write(&cc, be(&cor)).unwrap();

    // explicit flag
    let out = dir.join("out.tif");
    run(&[
        "--ifg",
        int.to_str().unwrap(),
        "--cols",
        "40",
        "--big-endian",
        "--cor",
        cc.to_str().unwrap(),
        "--nlooks",
        "10",
        "--out",
        out.to_str().unwrap(),
    ]);
    assert_eq!(fs::read(&out).unwrap(), reference);

    // GAMMA .off sidecar via --ifg-meta: provides the width AND implies big-endian
    let par = dir.join("ifg.off");
    fs::write(
        &par,
        "interferogram_width: 40\ninterferogram_azimuth_lines: 48\n",
    )
    .unwrap();
    let out2 = dir.join("out2.tif");
    run(&[
        "--ifg",
        int.to_str().unwrap(),
        "--ifg-meta",
        par.to_str().unwrap(),
        "--cor",
        cc.to_str().unwrap(),
        "--nlooks",
        "10",
        "--out",
        out2.to_str().unwrap(),
    ]);
    assert_eq!(fs::read(&out2).unwrap(), reference);
}

#[test]
fn cor_meta_is_used_for_flat_correlation() {
    let dir = tdir("cor-meta");
    let (ph, cor) = scene();
    let reference = baseline(&dir, &ph, &cor);

    let ph_tif = dir.join("ph.tif");
    write_tiff(&ph_tif, &ph);
    let cor_flat = dir.join("gamma.cc");
    fs::write(&cor_flat, be(&cor)).unwrap();
    let par = dir.join("gamma.off");
    fs::write(
        &par,
        "interferogram_width: 40\ninterferogram_azimuth_lines: 48\n",
    )
    .unwrap();

    let out = dir.join("out.tif");
    run(&[
        "--phase",
        ph_tif.to_str().unwrap(),
        "--cor",
        cor_flat.to_str().unwrap(),
        "--cor-meta",
        par.to_str().unwrap(),
        "--nlooks",
        "10",
        "--out",
        out.to_str().unwrap(),
    ]);
    assert_eq!(fs::read(&out).unwrap(), reference);
}

#[test]
fn rsc_and_isce_xml_sidecars() {
    let dir = tdir("sidecars");
    let (ph, cor) = scene();
    let reference = baseline(&dir, &ph, &cor);

    // ROI_PAC: <file>.rsc next to the data; no --cols anywhere
    let int = dir.join("ifg.int");
    fs::write(&int, le(&complex_pairs(&ph))).unwrap();
    fs::write(
        dir.join("ifg.int.rsc"),
        "WIDTH         40\nFILE_LENGTH   48\n",
    )
    .unwrap();
    let cc = dir.join("c.cc");
    fs::write(&cc, le(&altline(&cor, 7.0))).unwrap();
    let out = dir.join("out.tif");
    run(&[
        "--ifg",
        int.to_str().unwrap(),
        "--cor",
        cc.to_str().unwrap(),
        "--nlooks",
        "10",
        "--out",
        out.to_str().unwrap(),
    ]);
    assert_eq!(fs::read(&out).unwrap(), reference);

    // isce2: <file>.xml sidecars for both inputs; single-band float .cor
    let int2 = dir.join("isce.int");
    fs::write(&int2, le(&complex_pairs(&ph))).unwrap();
    fs::write(
        dir.join("isce.int.xml"),
        r#"<imageFile>
  <property name="byte_order"><value>l</value></property>
  <property name="data_type"><value>cfloat</value></property>
  <property name="length"><value>48</value></property>
  <property name="number_bands"><value>1</value></property>
  <property name="scheme"><value>BIP</value></property>
  <property name="width"><value>40</value></property>
</imageFile>"#,
    )
    .unwrap();
    let cor2 = dir.join("isce.cor");
    fs::write(&cor2, le(&cor)).unwrap();
    fs::write(
        dir.join("isce.cor.xml"),
        r#"<imageFile>
  <property name="byte_order"><value>l</value></property>
  <property name="data_type"><value>float</value></property>
  <property name="length"><value>48</value></property>
  <property name="number_bands"><value>1</value></property>
  <property name="scheme"><value>BIP</value></property>
  <property name="width"><value>40</value></property>
</imageFile>"#,
    )
    .unwrap();
    let out2 = dir.join("out2.tif");
    run(&[
        "--ifg",
        int2.to_str().unwrap(),
        "--cor",
        cor2.to_str().unwrap(),
        "--nlooks",
        "10",
        "--out",
        out2.to_str().unwrap(),
    ]);
    assert_eq!(fs::read(&out2).unwrap(), reference);

    // an isce2 .cor XML handed to --ifg must fail the dtype check
    let stderr = run_expect_fail(&[
        "--ifg",
        cor2.to_str().unwrap(),
        "--cor",
        cor2.to_str().unwrap(),
        "--nlooks",
        "10",
        "--out",
        dir.join("nope.tif").to_str().unwrap(),
    ]);
    assert!(stderr.contains("data_type"), "stderr: {stderr}");
}

#[test]
fn unw_altline_output_and_flat_conncomp() {
    let dir = tdir("unw-out");
    let (ph, cor) = scene();
    let reference = baseline(&dir, &ph, &cor);
    // decode the reference TIFF pixels for value comparison
    let mut dec = tiff::decoder::Decoder::new(std::io::BufReader::new(
        File::open(dir.join("ref.tif")).unwrap(),
    ))
    .unwrap();
    let ref_px: Vec<f32> = match dec.read_image().unwrap() {
        tiff::decoder::DecodingResult::F32(v) => v,
        other => panic!("unexpected reference dtype: {other:?}"),
    };
    assert!(!reference.is_empty());

    let int = dir.join("ifg.int");
    fs::write(&int, le(&complex_pairs(&ph))).unwrap();
    let cor_flat = dir.join("c.cor");
    fs::write(&cor_flat, le(&cor)).unwrap();

    let out = dir.join("out.unw");
    let cc_out = dir.join("out.unw.conncomp");
    run(&[
        "--ifg",
        int.to_str().unwrap(),
        "--cols",
        "40",
        "--cor",
        cor_flat.to_str().unwrap(),
        "--nlooks",
        "10",
        "--out",
        out.to_str().unwrap(),
    ]);

    // .unw = alt-line (rmg): per row a magnitude line (unit here) then phase
    let bytes = fs::read(&out).unwrap();
    assert_eq!(bytes.len(), ROWS * COLS * 8);
    let floats: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    for r in 0..ROWS {
        let mag = &floats[2 * r * COLS..(2 * r + 1) * COLS];
        let phs = &floats[(2 * r + 1) * COLS..(2 * r + 2) * COLS];
        assert!(
            mag.iter().all(|&m| (m - 1.0).abs() < 1e-6),
            "row {r} magnitude"
        );
        assert_eq!(phs, &ref_px[r * COLS..(r + 1) * COLS], "row {r} phase");
    }

    // flat conncomp: one byte per pixel, the ramp is a single component
    let cc = fs::read(&cc_out).unwrap();
    assert_eq!(cc.len(), ROWS * COLS);
    assert!(cc.iter().all(|&v| v == 1), "expected one full component");
}

#[test]
fn wrong_cols_is_a_clear_error() {
    let dir = tdir("wrong-cols");
    let (ph, cor) = scene();
    let int = dir.join("ifg.int");
    fs::write(&int, le(&complex_pairs(&ph))).unwrap();
    let cor_flat = dir.join("c.cor");
    fs::write(&cor_flat, le(&cor)).unwrap();

    let stderr = run_expect_fail(&[
        "--ifg",
        int.to_str().unwrap(),
        "--cols",
        "39",
        "--cor",
        cor_flat.to_str().unwrap(),
        "--nlooks",
        "10",
        "--out",
        dir.join("out.tif").to_str().unwrap(),
    ]);
    assert!(stderr.contains("whole number"), "stderr: {stderr}");

    // and no geometry at all asks for --cols
    let stderr = run_expect_fail(&[
        "--ifg",
        int.to_str().unwrap(),
        "--cor",
        cor_flat.to_str().unwrap(),
        "--nlooks",
        "10",
        "--out",
        dir.join("out.tif").to_str().unwrap(),
    ]);
    assert!(stderr.contains("--cols"), "stderr: {stderr}");
}

/// The pre-flat interface (`whirlwind unwrap ...`) is still accepted: the
/// leading token is stripped with a deprecation note and the result is
/// identical to the flat invocation.
#[test]
fn deprecated_unwrap_subcommand_still_works() {
    let dir = tdir("deprecated-unwrap");
    let (ph, cor) = scene();
    let reference = baseline(&dir, &ph, &cor);

    let out = dir.join("out.tif");
    let res = Command::new(env!("CARGO_BIN_EXE_whirlwind"))
        .args([
            "unwrap",
            "--phase",
            dir.join("ph.tif").to_str().unwrap(),
            "--cor",
            dir.join("cor.tif").to_str().unwrap(),
            "--nlooks",
            "10",
            "--out",
            out.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&res.stderr);
    assert!(res.status.success(), "stderr: {stderr}");
    assert!(stderr.contains("deprecated"), "stderr: {stderr}");
    assert_eq!(fs::read(&out).unwrap(), reference);
}

#[test]
fn simulate_subcommand_removed_with_pointer() {
    let stderr = run_expect_fail(&["simulate", "--shape", "64x64", "--out", "sim"]);
    assert!(stderr.contains("simulate_ifg"), "stderr: {stderr}");
}
