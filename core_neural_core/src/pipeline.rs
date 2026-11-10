//! End-to-end deterministic pipeline: OpenBCI frame → electrode vector → LIF → trajectory.
//! Supports a configurable channel map from the 8 Cyton channels into the 64-channel SNN input.

use crate::openbci_hal::OpenBciFrame;
use crate::safety_core::{TrajectoryState, check_contractive_stability, project_matrix_a};
use crate::snn_core::{ELECTRODE_CHANNELS, SNN_NEURONS, SnnState};

/// Channel map: for each of the 64 electrode slots, which Cyton channel (0-7) to use,
/// or `None` to leave it zero.
/// Default identity-style map places the 8 channels at evenly spaced indices.
pub const fn default_channel_map() -> [Option<usize>; ELECTRODE_CHANNELS] {
    let mut map = [None; ELECTRODE_CHANNELS];
    let mut i = 0;
    while i < 8 {
        let idx = i * 8;
        if idx < ELECTRODE_CHANNELS {
            map[idx] = Some(i);
        }
        i += 1;
    }
    map
}

/// Maps an 8-channel OpenBCI frame into the fixed 64-channel electrode vector
/// using the provided channel map. Scales raw ADC counts toward a microvolt-ish range.
pub fn expand_openbci_to_electrodes(
    frame: &OpenBciFrame,
    channel_map: &[Option<usize>; ELECTRODE_CHANNELS],
) -> [f32; ELECTRODE_CHANNELS] {
    let mut electrodes = [0.0; ELECTRODE_CHANNELS];
    for (slot, &src) in channel_map.iter().enumerate() {
        if let Some(ch) = src {
            if ch < 8 {
                electrodes[slot] = frame.channels[ch] as f32 * 1e-6;
            }
        }
    }
    electrodes
}

/// One-step pipeline state.
#[derive(Debug, Clone, Copy)]
pub struct PipelineState {
    pub snn: SnnState,
    pub trajectory: TrajectoryState,
    pub channel_map: [Option<usize>; ELECTRODE_CHANNELS],
}

impl PipelineState {
    pub const fn new() -> Self {
        Self {
            snn: SnnState::new(),
            trajectory: TrajectoryState::new(),
            channel_map: default_channel_map(),
        }
    }

    /// Override the channel map (e.g. from a calibration file or CLI).
    pub fn set_channel_map(&mut self, map: [Option<usize>; ELECTRODE_CHANNELS]) {
        self.channel_map = map;
    }

    /// Process one OpenBCI frame through the full chain.
    /// Returns the updated trajectory state and whether the contractive bound held.
    pub fn step(
        &mut self,
        frame: &OpenBciFrame,
        weights: &[[f32; ELECTRODE_CHANNELS]; SNN_NEURONS],
        alpha: f32,
        i_ext: &[f32; SNN_NEURONS],
        v_th: &[f32; SNN_NEURONS],
        matrix_a: &mut [[f32; 4]; 4],
        matrix_b: &[[f32; SNN_NEURONS]; 4],
        c: f32,
        dt: f32,
        w_max: f32,
    ) -> (TrajectoryState, bool) {
        let electrodes = expand_openbci_to_electrodes(frame, &self.channel_map);

        let mut spikes = [0.0; SNN_NEURONS];
        self.snn.update(&electrodes, weights, alpha, i_ext, v_th, &mut spikes);

        // Project A onto the safe Frobenius ball before use
        project_matrix_a(matrix_a, w_max);

        let prev = self.trajectory.state;
        self.trajectory.update(matrix_a, matrix_b, &spikes);

        let stable = check_contractive_stability(&prev, &self.trajectory.state, c, dt);

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
        let map = default_channel_map();
        let electrodes = expand_openbci_to_electrodes(&frame, &map);
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

        let (traj, stable) = pipe.step(
            &frame, &weights, 0.9, &i_ext, &v_th, &mut matrix_a, &matrix_b, 0.1, 0.001, 2.0,
        );
        assert!(stable || traj.state.iter().all(|&x| x.abs() < 10.0));
    }
}
