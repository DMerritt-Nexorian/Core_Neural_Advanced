//! Live OpenBCI Cyton serial reader.
//! Opens the serial port, reads raw 33-byte packets, decodes them with the existing
//! OpenBciFrame::decode, and feeds each frame into pipeline::step.
//!
//! Usage:
//!   cargo run -p cyton_reader -- /dev/ttyUSB0
//!   cargo run -p cyton_reader -- COM3 --start
//!
//! Flags:
//!   --start     Send the Cyton 'b' start-streaming command on connect
//!   --log FILE  Append CSV rows (packet,sample_id,traj_x,traj_y,stable,latency_us)
//!
//! Default baud is 115200 (Cyton standard).

use core_neural_core::openbci_hal::OpenBciFrame;
use core_neural_core::pipeline::PipelineState;
use core_neural_core::snn_core::{ELECTRODE_CHANNELS, SNN_NEURONS};
use serialport::SerialPort;
use std::env;
use std::fs::OpenOptions;
use std::io::{self, Read, Write};
use std::time::{Duration, Instant};

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <serial-port> [--start] [--log FILE]", args[0]);
        eprintln!("Example: {} /dev/ttyUSB0 --start --log run.csv", args[0]);
        std::process::exit(1);
    }

    let mut port_name = None;
    let mut send_start = false;
    let mut log_path: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--start" => {
                send_start = true;
                i += 1;
            }
            "--log" => {
                if i + 1 < args.len() {
                    log_path = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    eprintln!("--log requires a filename");
                    std::process::exit(1);
                }
            }
            other if !other.starts_with('-') && port_name.is_none() => {
                port_name = Some(other.to_string());
                i += 1;
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                std::process::exit(1);
            }
        }
    }

    let port_name = port_name.expect("serial port required");

    println!("Opening {} at 115200 baud…", port_name);
    let mut port = serialport::new(&port_name, 115_200)
        .timeout(Duration::from_millis(100))
        .open()
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

    if send_start {
        println!("Sending start-streaming command ('b')");
        let _ = port.write_all(b"b");
        std::thread::sleep(Duration::from_millis(200));
    }

    let mut log_file = if let Some(path) = log_path {
        let mut f = OpenOptions::new().create(true).append(true).open(&path)?;
        writeln!(f, "packet,sample_id,traj_x,traj_y,stable,latency_us")?;
        Some(f)
    } else {
        None
    };

    let mut pipe = PipelineState::new();
    let weights = [[0.001f32; ELECTRODE_CHANNELS]; SNN_NEURONS];
    let i_ext = [0.0; SNN_NEURONS];
    let v_th = [1.0; SNN_NEURONS];
    let mut matrix_a = [
        [0.95, 0.0, 0.0, 0.0],
        [0.0, 0.95, 0.0, 0.0],
        [0.0, 0.0, 0.95, 0.0],
        [0.0, 0.0, 0.0, 0.95],
    ];
    let matrix_b = [[0.0; SNN_NEURONS]; 4];

    let mut buf = [0u8; 33];
    let mut packet_count: u64 = 0;
    let mut decode_errors: u64 = 0;
    let mut unstable_count: u64 = 0;
    let start = Instant::now();
    let mut last_report = Instant::now();

    println!("Reading packets. Ctrl-C to stop.");
    println!("step | sample_id | traj_x | traj_y | stable | latency_us | unstable_pct");

    loop {
        match port.read_exact(&mut buf) {
            Ok(()) => {
                let t0 = Instant::now();
                match OpenBciFrame::decode(&buf) {
                    Ok(frame) => {
                        let (traj, stable) = pipe.step(
                            &frame,
                            &weights,
                            0.9,
                            &i_ext,
                            &v_th,
                            &mut matrix_a,
                            &matrix_b,
                            0.05,
                            0.001,
                            3.0,
                        );
                        let latency_us = t0.elapsed().as_micros() as u64;
                        packet_count += 1;
                        if !stable {
                            unstable_count += 1;
                        }

                        if let Some(f) = log_file.as_mut() {
                            let _ = writeln!(
                                f,
                                "{},{},{:.6},{:.6},{},{}",
                                packet_count,
                                frame.sample_id,
                                traj.state[0],
                                traj.state[1],
                                stable as u8,
                                latency_us
                            );
                        }

                        if last_report.elapsed() >= Duration::from_secs(1) {
                            let pct = if packet_count > 0 {
                                (unstable_count as f64 / packet_count as f64) * 100.0
                            } else {
                                0.0
                            };
                            println!(
                                "{:6} | {:3} | {:7.3} | {:7.3} | {:5} | {:6} us | {:5.1}%",
                                packet_count,
                                frame.sample_id,
                                traj.state[0],
                                traj.state[1],
                                stable,
                                latency_us,
                                pct
                            );
                            last_report = Instant::now();
                        }
                    }
                    Err(e) => {
                        decode_errors += 1;
                        if decode_errors % 50 == 1 {
                            eprintln!("Decode error ({}): {:?}", decode_errors, e);
                        }
                        // Resync: discard one byte
                        let mut discard = [0u8; 1];
                        let _ = port.read(&mut discard);
                    }
                }
            }
            Err(e) if e.kind() == io::ErrorKind::TimedOut => continue,
            Err(e) => {
                eprintln!("Serial read error: {}", e);
                break;
            }
        }

        if start.elapsed() > Duration::from_secs(3600) {
            break;
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    let rate = if elapsed > 0.0 {
        packet_count as f64 / elapsed
    } else {
        0.0
    };
    println!(
        "\nFinished. Packets: {} ({:.1} Hz), Decode errors: {}, Unstable: {}",
        packet_count, rate, decode_errors, unstable_count
    );
    Ok(())
}
