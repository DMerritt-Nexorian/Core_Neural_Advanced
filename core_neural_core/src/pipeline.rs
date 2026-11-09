//! End-to-end deterministic pipeline: `OpenBCI` frame → electrode vector → LIF → trajectory.
//! Bridges the 8-channel Cyton protocol to the 64-channel SNN input by zero-padding
//! and simple spatial replication (documented for later electrode-array expansion).

use crate::openbci_hal::OpenBciFrame;
use crate::safety_core::{check_contractive_stability, project_matrix_a, TrajectoryState};
use crate::snn_core::{SnnState, ELECTRODE_CHANNELS, SNN_NEURONS};

/// Maps an 8-channel `OpenBCI` frame into the fixed 64-channel electrode vector
/// expected by the SNN. Currently zero-pads and tiles the 8 channels.
#[must_use]
#[allow(clippy::cast_precision_loss)] // Decoded ADS1299 24-bit values fit exactly in f32's 24-bit significand.
pub fn expand_openbci_to_electrodes(frame: &OpenBciFrame) -> [f32; ELECTRODE_CHANNELS] {
    let mut electrodes = [0.0; ELECTRODE_CHANNELS];
    // Simple deterministic mapping: place each of the 8 channels into 8 slots
    // and leave the rest at 0.0. Later electrode arrays can replace this.
    for (i, &ch) in frame.channels.iter().enumerate() {
        if i * 8 < ELECTRODE_CHANNELS {
            electrodes[i * 8] = ch as f32 * 1e-6; // scale raw ADC toward microvolt range
        }
    }
    electrodes
}

/// One-step pipeline state.
#[derive(Debug, Clone, Copy)]
pub struct PipelineState {
    pub snn: SnnState,
    pub trajectory: TrajectoryState,
}

/// Explicit inputs and parameters for one deterministic pipeline step.
pub struct PipelineStepParameters<'a> {
    pub frame: &'a OpenBciFrame,
    pub weights: &'a [[f32; ELECTRODE_CHANNELS]; SNN_NEURONS],
    pub alpha: f32,
    pub i_ext: &'a [f32; SNN_NEURONS],
    pub v_th: &'a [f32; SNN_NEURONS],
    pub matrix_a: &'a mut [[f32; 4]; 4],
    pub matrix_b: &'a [[f32; SNN_NEURONS]; 4],
    pub contraction_c: f32,
    pub dt: f32,
    pub w_max: f32,
}

impl PipelineState {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            snn: SnnState::new(),
            trajectory: TrajectoryState::new(),
        }
    }

    /// Process one `OpenBCI` frame through the full chain.
    /// Returns the updated trajectory state and whether the contractive bound held.
    pub fn step(&mut self, parameters: &mut PipelineStepParameters<'_>) -> (TrajectoryState, bool) {
        let electrodes = expand_openbci_to_electrodes(parameters.frame);

        let mut spikes = [0.0; SNN_NEURONS];
        self.snn.update(
            &electrodes,
            parameters.weights,
            parameters.alpha,
            parameters.i_ext,
            parameters.v_th,
            &mut spikes,
        );

        // Project A onto the safe Frobenius ball before use.
        project_matrix_a(&mut *parameters.matrix_a, parameters.w_max);

        let prev = self.trajectory.state;
        self.trajectory
            .update(&mut *parameters.matrix_a, parameters.matrix_b, &spikes);

        let stable = check_contractive_stability(
            &prev,
            &self.trajectory.state,
            parameters.contraction_c,
            parameters.dt,
        );

        (self.trajectory, stable)
    }
}

impl Default for PipelineState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openbci_hal::OpenBciFrame;

    #[test]
    fn test_expand_and_step() {
        let frame = OpenBciFrame {
            sample_id: 1,
            channels: [1000, -500, 0, 2000, -1000, 300, -200, 1500],
            aux: [0; 6],
        };
        // Force a valid checksum-style packet is not required for this unit test
        let electrodes = expand_openbci_to_electrodes(&frame);
        assert_ne!(electrodes[0], 0.0);

        let mut pipe = PipelineState::new();
        let weights = [[0.001; ELECTRODE_CHANNELS]; SNN_NEURONS];
        let i_ext = [0.0; SNN_NEURONS];
        let v_th = [1.0; SNN_NEURONS];
        let mut matrix_a = [
            [0.9, 0.0, 0.0, 0.0],
            [0.0, 0.9, 0.0, 0.0],
            [0.0, 0.0, 0.9, 0.0],
            [0.0, 0.0, 0.0, 0.9],
        ];
        let matrix_b = [[0.0; SNN_NEURONS]; 4];

        let mut parameters = PipelineStepParameters {
            frame: &frame,
            weights: &weights,
            alpha: 0.9,
            i_ext: &i_ext,
            v_th: &v_th,
            matrix_a: &mut matrix_a,
            matrix_b: &matrix_b,
            contraction_c: 0.1,
            dt: 0.001,
            w_max: 2.0,
        };
        let (traj, stable) = pipe.step(&mut parameters);
        assert!(stable || traj.state.iter().all(|&x| x.abs() < 10.0));
    }
}
