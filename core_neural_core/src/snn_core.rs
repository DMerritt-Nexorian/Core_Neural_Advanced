//! Module 2: Pure Static Tensor & SNN Compute Engine (`snn_core`)
//! Zero-allocation matrix-vector operations and Leaky Integrate-and-Fire kernel.

/// The fixed number of input electrode channels.
pub const ELECTRODE_CHANNELS: usize = 64;
/// The fixed number of SNN neurons.
pub const SNN_NEURONS: usize = 32;
/// The fixed number of state dimensions for trajectory tracking.
pub const STATE_DIMENSIONS: usize = 4;

/// Performs matrix-vector multiplication: Out = M * V
/// M is a Row-Major matrix of size R x C, V is a vector of size C, Out is a vector of size R.
/// Assumes all loops are compile-time bounded to guarantee O(1) execution.
#[inline]
pub fn mat_vec_mul<const R: usize, const C: usize>(
    matrix: &[[f32; C]; R],
    vector: &[f32; C],
    out: &mut [f32; R],
) {
    for (r, row) in matrix.iter().enumerate() {
        let mut sum = 0.0;
        for (c, &val) in row.iter().enumerate() {
            sum += val * vector[c];
        }
        out[r] = sum;
    }
}

/// SNN State containing membrane potentials of the 32 SNN neurons.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SnnState {
    /// Membrane potentials of the neurons.
    pub potentials: [f32; SNN_NEURONS],
}

impl SnnState {
    /// Creates a new SNN State initialized to zero potentials.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            potentials: [0.0; SNN_NEURONS],
        }
    }

    /// Reset all potentials to 0.0.
    pub fn reset(&mut self) {
        for v in &mut self.potentials {
            *v = 0.0;
        }
    }

    /// Discrete membrane potential update & deterministic spike emission.
    /// V_{i}[t+1] = α * V_{i}[t] + ∑ (W_{ij} * inputs[j]) + `I_ext`
    /// Returns a boolean array indicating which neurons spiked (S_{i}[t+1] = 1 if V_{i}[t+1] >= `V_th` else 0).
    /// If a neuron spikes, its membrane potential resets to 0.0.
    pub fn update(
        &mut self,
        inputs: &[f32; ELECTRODE_CHANNELS],
        weights: &[[f32; ELECTRODE_CHANNELS]; SNN_NEURONS],
        alpha: f32,
        i_ext: &[f32; SNN_NEURONS],
        v_th: &[f32; SNN_NEURONS],
        spikes: &mut [f32; SNN_NEURONS],
    ) {
        // Compute input contributions: W_ij * inputs[j]
        let mut w_inputs = [0.0; SNN_NEURONS];
        mat_vec_mul(weights, inputs, &mut w_inputs);

        for i in 0..SNN_NEURONS {
            // LIF update equation
            let next_v = alpha * self.potentials[i] + w_inputs[i] + i_ext[i];

            // Deterministic spike emission check
            if next_v >= v_th[i] {
                spikes[i] = 1.0;
                self.potentials[i] = 0.0; // Reset on spike
            } else {
                spikes[i] = 0.0;
                // Clamp to 0.0 if next_v is negative to avoid unstably large negative potential values
                self.potentials[i] = if next_v < 0.0 { 0.0 } else { next_v };
            }
        }
    }
}

impl Default for SnnState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_mat_vec_mul() {
        let matrix = [[1.0, 2.0], [3.0, 4.0]];
        let vector = [5.0, 6.0];
        let mut out = [0.0; 2];
        mat_vec_mul(&matrix, &vector, &mut out);
        assert_eq!(out[0], 17.0); // 1*5 + 2*6 = 17
        assert_eq!(out[1], 39.0); // 3*5 + 4*6 = 39
    }

    #[test]
    fn test_snn_state_reset() {
        let mut s = SnnState::default();
        s.potentials[0] = 12.3;
        s.reset();
        assert_eq!(s.potentials[0], 0.0);
    }

    #[test]
    fn test_snn_state_update_no_spike() {
        let mut s = SnnState::new();
        let inputs = [1.0; ELECTRODE_CHANNELS];
        let weights = [[0.01; ELECTRODE_CHANNELS]; SNN_NEURONS];
        let i_ext = [0.05; SNN_NEURONS];
        let v_th = [2.0; SNN_NEURONS];
        let mut spikes = [0.0; SNN_NEURONS];

        // alpha = 0.9. w_inputs = 64 * 0.01 = 0.64. i_ext = 0.05.
        // next_v = 0.9 * 0.0 + 0.64 + 0.05 = 0.69 < 2.0 (no spike)
        s.update(&inputs, &weights, 0.9, &i_ext, &v_th, &mut spikes);
        assert_eq!(spikes[0], 0.0);
        assert!((s.potentials[0] - 0.69).abs() < 1e-5);
    }

    #[test]
    fn test_snn_state_update_with_spike() {
        let mut s = SnnState::new();
        let inputs = [1.0; ELECTRODE_CHANNELS];
        let weights = [[0.1; ELECTRODE_CHANNELS]; SNN_NEURONS];
        let i_ext = [0.0; SNN_NEURONS];
        let v_th = [2.0; SNN_NEURONS];
        let mut spikes = [0.0; SNN_NEURONS];

        // alpha = 0.9. w_inputs = 64 * 0.1 = 6.4.
        // next_v = 0.9 * 0.0 + 6.4 + 0.0 = 6.4 >= 2.0 (spike)
        s.update(&inputs, &weights, 0.9, &i_ext, &v_th, &mut spikes);
        assert_eq!(spikes[0], 1.0);
        assert_eq!(s.potentials[0], 0.0); // Reset on spike
    }

    #[test]
    fn test_snn_state_update_negative_potential() {
        let mut s = SnnState::new();
        let inputs = [-10.0; ELECTRODE_CHANNELS];
        let weights = [[0.1; ELECTRODE_CHANNELS]; SNN_NEURONS];
        let i_ext = [0.0; SNN_NEURONS];
        let v_th = [2.0; SNN_NEURONS];
        let mut spikes = [0.0; SNN_NEURONS];

        s.update(&inputs, &weights, 0.9, &i_ext, &v_th, &mut spikes);
        assert_eq!(spikes[0], 0.0);
        assert_eq!(s.potentials[0], 0.0); // Clamped to 0.0
    }
}

#[cfg(kani)]
mod proofs {
    use super::*;

    #[kani::proof]
    fn prove_mat_vec_mul_does_not_panic() {
        let matrix: [[f32; 4]; 4] = [
            [kani::any(), kani::any(), kani::any(), kani::any()],
            [kani::any(), kani::any(), kani::any(), kani::any()],
            [kani::any(), kani::any(), kani::any(), kani::any()],
            [kani::any(), kani::any(), kani::any(), kani::any()],
        ];
        let vector: [f32; 4] = [kani::any(), kani::any(), kani::any(), kani::any()];
        let mut out = [0.0; 4];
        mat_vec_mul(&matrix, &vector, &mut out);
    }

    #[kani::proof]
    fn prove_snn_update_does_not_panic() {
        let mut s = SnnState::new();
        let inputs = [0.0; ELECTRODE_CHANNELS];
        let weights = [[0.0; ELECTRODE_CHANNELS]; SNN_NEURONS];
        let i_ext = [0.0; SNN_NEURONS];
        let v_th = [1.0; SNN_NEURONS];
        let mut spikes = [0.0; SNN_NEURONS];
        s.update(&inputs, &weights, 0.9, &i_ext, &v_th, &mut spikes);
    }
}
