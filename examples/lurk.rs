use std::{collections::HashMap, fs::{self, File}, io::{BufWriter, Write}, path::PathBuf, time::Instant};

use grumpkin_msm::pasta::utils::{gen_points, gen_scalars};
use num::{BigInt, Num};
use pasta_curves::pallas;
use rayon::{iter::{IntoParallelRefIterator, ParallelIterator}, slice::ParallelSliceMut};
 
use std::fs::OpenOptions;
use std::io::BufReader;

#[cfg(feature = "cuda")]
extern "C" {
    fn cuda_available() -> bool;
}

use serde::de::DeserializeOwned;

/// Path to the directory where Arecibo data will be stored.
pub static ARECIBO_DATA: &str = ".arecibo_data";

/// Reads and deserializes data from a specified section and label.
pub fn read_arecibo_data<T: DeserializeOwned>(
  section: String,
  label: String,
) -> T {
let root_dir = home::home_dir().unwrap().join(ARECIBO_DATA);

  let section_path = root_dir.join(section);
  assert!(section_path.exists(), "Section directory does not exist");

  // Assuming the label uniquely identifies the file, and ignoring the counter for simplicity
  let file_path = section_path.join(label);
  assert!(file_path.exists(), "Data file does not exist");

  let file = File::open(file_path).expect("Failed to open data file");
  let reader = BufReader::new(file);

  bincode::deserialize_from(reader).expect("Failed to read data")
}


struct Stats {
    count: usize,
    total: usize,
}

fn write_stats(scalars: &[pallas::Scalar], root_dir: &str, out_file: &str) -> std::io::Result<()> {
    let root_dir = PathBuf::from(root_dir);
    if !root_dir.exists() {
        fs::create_dir_all(&root_dir).expect("Failed to create stats directory.");
    }
    let stats = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(root_dir.join(out_file))?;

    let mut writer = BufWriter::new(stats);
    let mut hist: HashMap<BigInt, usize> = HashMap::new();

    scalars.iter().for_each(|x| {
        let x = format!("{:?}", x);
        let num = BigInt::from_str_radix(&x[2..], 16).unwrap();
        *hist.entry(num).or_insert(0) += 1;
    });

    let mut entries: Vec<_> = hist.par_iter().collect();
    entries.par_sort_by(|a, b| b.1.cmp(a.1));

    let mut stats_20 = Stats { count: 0, total: 0};
    let mut stats_100 = Stats { count: 0, total: 0};
    let mut stats_500 = Stats { count: 0, total: 0};
    let mut stats_1000 = Stats { count: 0, total: 0};
    for (_, value) in entries.iter() {
        if **value > 1000 {
            stats_1000.count += 1;
            stats_1000.total += *value;
        } 
        if **value > 500 {
            stats_500.count += 1;
            stats_500.total += *value;
        } 
        if **value > 100 {
            stats_100.count += 1;
            stats_100.total += *value;
        } 
        if **value > 20 {
            stats_20.count += 1;
            stats_20.total += *value;
        } 
    }

    writeln!(&mut writer, "scalars length: {}", scalars.len())?;
    let p1000 = 100.0 * (stats_1000.total as f64 / scalars.len() as f64);
    let p500 = 100.0 * (stats_500.total as f64 / scalars.len() as f64);
    let p100 = 100.0 * (stats_100.total as f64 / scalars.len() as f64);
    let p50 = 100.0 * (stats_20.total as f64 / scalars.len() as f64);
    writeln!(&mut writer, "freq>1000 (count, total, %): {:>10}, {:>10}, {:>10.2}%", stats_1000.count, stats_1000.total, p1000)?;
    writeln!(&mut writer, "freq>500  (count, total, %): {:>10}, {:>10}, {:>10.2}%", stats_500.count, stats_500.total, p500)?;
    writeln!(&mut writer, "freq>100  (count, total, %): {:>10}, {:>10}, {:>10.2}%", stats_100.count, stats_100.total, p100)?;
    writeln!(&mut writer, "freq>20   (count, total, %): {:>10}, {:>10}, {:>10.2}%", stats_20.count, stats_20.total, p50)?;
    writeln!(&mut writer, "")?;
    writeln!(&mut writer, "top 100 values:")?;

    let mut count = 0;
    // Print sorted (key, value) pairs
    for (key, value) in entries {
        if count < 100 {
            writeln!(&mut writer, "{:>2}, {:>64}, {:?}", count, key.to_str_radix(16), value)?;
        }
        count += 1;
    }
    writer.flush()?;
    Ok(())
}

fn time_msm(label: &str, points: &[pallas::Affine], scalars: &[pallas::Scalar]) -> pallas::Point {
    let start = Instant::now();
    let res = grumpkin_msm::pasta::pallas(points, scalars);
    println!("{} took:", label);
    println!("\t{:?}", start.elapsed());
    res
}

/// cargo run --release --example lurk
fn main() {
    #[cfg(feature = "cuda")]
    if unsafe { cuda_available() } {
        unsafe { grumpkin_msm::CUDA_OFF = false };
    }

    const LENGTH: usize = 7980043; // 7980043 9722443

    let points = gen_points(LENGTH);
    let scalars = gen_scalars(LENGTH);

    time_msm("baseline", &points, &scalars);

    for i in 0..17 {
        
        let label_i = format!("len_{}_{}", LENGTH, i);
        let witness_i: Vec<pallas::Scalar> = read_arecibo_data("witness_0x00d1b86f4c5cc4f5755819a4bce96e835903637bf451ddb294aa2d65daac9d6e".into(), label_i);
        
        let out_stats = format!("{i}.txt");
        // write_stats(&witness_i, "cross_term_stats", &out_stats).unwrap();

        time_msm(&out_stats, &points, &witness_i);
    }
}