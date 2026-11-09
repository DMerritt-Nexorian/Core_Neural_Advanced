//! Live OpenBCI Cyton serial reader.
//! Opens the serial port, reads raw 33-byte packets, decodes them with the existing
//! `OpenBciFrame::decode`, and feeds each frame into `pipeline::step`.
//!
//! Usage:
//!   cargo run -p cyton_reader -- /dev/ttyUSB0
//!   cargo run -p cyton_reader -- COM3
//!
//! Default baud is 115200 (Cyton standard).

use core_neural_core::openbci_hal::OpenBciFrame;
use core_neural_core::pipeline::{PipelineState, PipelineStepParameters};
use core_neural_core::snn_core::{ELECTRODE_CHANNELS, SNN_NEURONS};
use std::env;
use std::io::{self, Read};
use std::time::{Duration, Instant};

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <serial-port>", args[0]);
        eprintln!("Example: {} /dev/ttyUSB0", args[0]);
        eprintln!("Example: {} COM3", args[0]);
        std::process::exit(1);
    }
    let port_name = &args[1];

    println!("Opening {} at 115200 baud…", port_name);
    let mut port = serialport::new(port_name, 115_200)
        .timeout(Duration::from_millis(100))
        .open()
        .map_err(io::Error::other)?;

    // Optional: send the Cyton start-streaming command ('b')
    // Uncomment if the board is not already streaming.
    // let _ = port.write(b"b");

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
    let start = Instant::now();

    println!("Reading packets. Ctrl-C to stop.");
    println!("step | sample_id | traj_x | traj_y | stable | latency_us");

    loop {
        // Read exactly 33 bytes (Cyton packet size)
        match port.read_exact(&mut buf) {
            Ok(()) => {
                let t0 = Instant::now();
                match OpenBciFrame::decode(&buf) {
                    Ok(frame) => {
                        let mut parameters = PipelineStepParameters {
                            frame: &frame,
                            weights: &weights,
                            alpha: 0.9,
                            i_ext: &i_ext,
                            v_th: &v_th,
                            matrix_a: &mut matrix_a,
                            matrix_b: &matrix_b,
                            contraction_c: 0.05,
                            dt: 0.001,
                            w_max: 3.0,
                        };
                        let (traj, stable) = pipe.step(&mut parameters);
                        let latency_us = t0.elapsed().as_micros();
                        packet_count += 1;

                        if packet_count.is_multiple_of(250) {
                            // ~1 s at 250 Hz
                            println!(
                                "{:6} | {:3} | {:7.3} | {:7.3} | {:5} | {:6} us",
                                packet_count,
                                frame.sample_id,
                                traj.state[0],
                                traj.state[1],
                                stable,
                                latency_us
                            );
                        }
                    }
                    Err(e) => {
                        decode_errors += 1;
                        if decode_errors % 100 == 1 {
                            eprintln!("Decode error ({}): {:?}", decode_errors, e);
                        }
                        // Simple resync: discard one byte and try again on next read
                        let mut discard = [0u8; 1];
                        let _ = port.read(&mut discard);
                    }
                }
            }
            Err(e) if e.kind() == io::ErrorKind::TimedOut => {
                // normal, just continue
                continue;
            }
            Err(e) => {
                eprintln!("Serial read error: {}", e);
                break;
            }
        }

        // Safety: stop after a long run if desired (comment out for continuous)
        if start.elapsed() > Duration::from_secs(3600) {
            break;
        }
    }

    println!(
        "\nFinished. Packets: {}, Decode errors: {}",
        packet_count, decode_errors
    );
    Ok(())
}
