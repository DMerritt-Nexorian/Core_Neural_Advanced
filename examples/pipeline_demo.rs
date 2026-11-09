//! Minimal host-side demo of the end-to-end pipeline.
//! Run with: cargo run --example pipeline_demo

use core_neural_core::openbci_hal::OpenBciFrame;
use core_neural_core::pipeline::PipelineState;
use core_neural_core::snn_core::{ELECTRODE_CHANNELS, SNN_NEURONS};

fn main() {
    let mut pipe = PipelineState::new();
    let weights = [[0.002; ELECTRODE_CHANNELS]; SNN_NEURONS];
    let i_ext = [0.01; SNN_NEURONS];
    let v_th = [0.8; SNN_NEURONS];
    let mut matrix_a = [
        [0.95, 0.01, 0.0, 0.0],
        [0.0, 0.95, 0.0, 0.0],
        [0.0, 0.0, 0.95, 0.0],
        [0.0, 0.0, 0.0, 0.95],
    ];
    let mut matrix_b = [[0.0; SNN_NEURONS]; 4];
    matrix_b[0][0] = 0.1;
    matrix_b[1][1] = 0.1;

    println!("Running 50 synthetic OpenBCI frames through LIF → trajectory pipeline");

    for step in 0..50 {
        let mut channels = [0i32; 8];
        for (i, ch) in channels.iter_mut().enumerate() {
            *ch = ((step as i32 * 100) + (i as i32 * 50)) % 2000 - 1000;
        }
        let frame = OpenBciFrame {
            sample_id: (step % 256) as u8,
            channels,
            aux: [0; 6],
        };

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

        if step % 10 == 0 {
            println!(
                "step {:02} | traj: [{:.3}, {:.3}, {:.3}, {:.3}] | stable: {}",
                step, traj.state[0], traj.state[1], traj.state[2], traj.state[3], stable
            );
        }
    }

    println!("Pipeline demo complete.");
}
