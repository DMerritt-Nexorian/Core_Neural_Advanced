//! Module 3: Kinematic Trajectory & Safety Invariant Engine (`safety_core`)
//! State space updates, contractive exponential stability checks, and convex projections onto the Frobenius ball.

use crate::snn_core::mat_vec_mul;

/// Evaluates the Euclidean (L2) norm of a 4D vector.
#[inline]
#[must_use]
pub fn l2_norm(v: &[f32; 4]) -> f32 {
    let sum_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
    libm::sqrtf(sum_sq)
}

/// Evaluates the Frobenius norm of a 4x4 matrix.
#[inline]
#[must_use]
pub fn frobenius_norm(matrix: &[[f32; 4]; 4]) -> f32 {
    let mut sum_sq = 0.0;
    for row in matrix {
        for &val in row {
            sum_sq += val * val;
        }
    }
    libm::sqrtf(sum_sq)
}

/// Invariant 1: Contractive Exponential Stability Check
/// Verify ||`δX_{t+1}`|| <= (1 - c * dt) * ||`δX_t`|| under perturbations (where `c` > 0).
#[inline]
#[must_use]
pub fn check_contractive_stability(delta_x_t: &[f32; 4], delta_x_next: &[f32; 4], c: f32, dt: f32) -> bool {
    let norm_t = l2_norm(delta_x_t);
    let norm_next = l2_norm(delta_x_next);
    let decay = 1.0 - c * dt;
    if decay < 0.0 {
        false
    } else {
        norm_next <= decay * norm_t
    }
}

/// Invariant 2: Convex Projection Operator (`Π_C`)
/// Project weight transition matrix A onto the bounded Frobenius norm ball C:
/// If ||A||_F > `W_max`, set A = (`W_max` / ||A||_F) * A.
#[inline]
pub fn project_matrix_a(matrix: &mut [[f32; 4]; 4], w_max: f32) {
    let norm = frobenius_norm(matrix);
    if norm > w_max && norm > 0.0 {
        let scale = w_max / norm;
        for row in matrix {
            for val in row {
                *val *= scale;
            }
        }
    }
}

/// Trajectory Engine state tracker.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrajectoryState {
    /// 4D trajectory state vector (e.g., position x, position y, velocity x, velocity y).
    pub state: [f32; 4],
}

impl TrajectoryState {
    /// Creates a new `TrajectoryState` initialized to zeros.
    #[must_use]
    pub const fn new() -> Self {
        Self { state: [0.0; 4] }
    }

    /// Reset trajectory state to zeros.
    pub fn reset(&mut self) {
        for val in &mut self.state {
            *val = 0.0;
        }
    }

    /// Trajectory Update: X_{t+1} = A * `X_t` + B * `S_t`
    /// Update the state given transition matrix A (4x4), control matrix B (4x32), and spiking input `S_t` (32).
    pub fn update(
        &mut self,
        matrix_a: &[[f32; 4]; 4],
        matrix_b: &[[f32; 32]; 4],
        spikes: &[f32; 32],
    ) {
        let mut ax = [0.0; 4];
        mat_vec_mul(matrix_a, &self.state, &mut ax);

        let mut bs = [0.0; 4];
        mat_vec_mul(matrix_b, spikes, &mut bs);

        for i in 0..4 {
            self.state[i] = ax[i] + bs[i];
        }
    }
}

impl Default for TrajectoryState {
    fn default() -> Self {
        Self::new()
    }
}

/// Monitors system-level metrics and enforces strict operational safety bounds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SafetyMonitor {
    /// Processing or system latency in milliseconds. Must be <= 5.0 ms.
    pub latency_ms: f32,
    /// Percentage of lost packets. Must be <= 0.1 %.
    pub packet_loss_percent: f32,
    /// Biological tissue impedance in kilo-ohms. Must be < 50.0 kΩ.
    pub tissue_impedance_kohm: f32,
}

impl SafetyMonitor {
    /// Creates a new `SafetyMonitor` with given metrics.
    #[must_use]
    pub const fn new(latency_ms: f32, packet_loss_percent: f32, tissue_impedance_kohm: f32) -> Self {
        Self {
            latency_ms,
            packet_loss_percent,
            tissue_impedance_kohm,
        }
    }

    /// Verifies if all metrics are within safe, allowed limits.
    #[must_use]
    pub fn check_bounds(&self) -> bool {
        self.latency_ms <= 5.0 && self.packet_loss_percent <= 0.1 && self.tissue_impedance_kohm < 50.0
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_norms() {
        let v = [3.0, 4.0, 0.0, 0.0];
        assert_eq!(l2_norm(&v), 5.0);

        let matrix = [
            [1.0, 2.0, 0.0, 0.0],
            [3.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ];
        // Frobenius norm of [[1, 2], [3, 4]] is sqrt(1 + 4 + 9 + 16) = sqrt(30)
        let expected = libm::sqrtf(30.0);
        assert!((frobenius_norm(&matrix) - expected).abs() < 1e-5);
    }

    #[test]
    fn test_stability_check() {
        let delta_x_t = [1.0, 0.0, 0.0, 0.0];
        let delta_x_next = [0.8, 0.0, 0.0, 0.0];
        // c = 0.1, dt = 1.0. decay = 1.0 - 0.1 * 1.0 = 0.9.
        // 0.8 <= 0.9 * 1.0 is true.
        assert!(check_contractive_stability(&delta_x_t, &delta_x_next, 0.1, 1.0));

        let delta_x_next_large = [0.95, 0.0, 0.0, 0.0];
        // 0.95 <= 0.9 * 1.0 is false.
        assert!(!check_contractive_stability(&delta_x_t, &delta_x_next_large, 0.1, 1.0));

        // Negative decay: c * dt > 1.0
        assert!(!check_contractive_stability(&delta_x_t, &delta_x_next, 2.0, 1.0));
    }

    #[test]
    fn test_project_matrix_a() {
        let mut matrix = [
            [3.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ];
        // Frobenius norm = 5.0
        // Case 1: norm <= w_max. w_max = 6.0. No change.
        project_matrix_a(&mut matrix, 6.0);
        assert_eq!(matrix[0][0], 3.0);

        // Case 2: norm > w_max. w_max = 2.5. Scale by 2.5 / 5.0 = 0.5.
        project_matrix_a(&mut matrix, 2.5);
        assert_eq!(matrix[0][0], 1.5);
        assert_eq!(matrix[0][1], 2.0);
    }

    #[test]
    fn test_trajectory_state_update() {
        let mut ts = TrajectoryState::default();
        let matrix_a = [
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.5],
        ];
        let mut matrix_b = [[0.0; 32]; 4];
        matrix_b[0][0] = 10.0;

        let spikes = [0.0; 32];
        ts.state = [1.0, 2.0, 3.0, 4.0];

        // Update with no spikes
        ts.update(&matrix_a, &matrix_b, &spikes);
        assert_eq!(ts.state, [0.5, 1.0, 1.5, 2.0]);

        // Update with spike at index 0
        let mut spikes_active = [0.0; 32];
        spikes_active[0] = 1.0;
        ts.update(&matrix_a, &matrix_b, &spikes_active);
        // ax = [0.25, 0.5, 0.75, 1.0]
        // bs = [10.0, 0.0, 0.0, 0.0]
        assert_eq!(ts.state, [10.25, 0.5, 0.75, 1.0]);

        // Reset
        ts.reset();
        assert_eq!(ts.state, [0.0; 4]);
    }

    #[test]
    fn test_safety_bounds() {
        let monitor_safe = SafetyMonitor::new(4.5, 0.05, 49.0);
        assert!(monitor_safe.check_bounds());

        let monitor_bad_latency = SafetyMonitor::new(5.1, 0.05, 49.0);
        assert!(!monitor_bad_latency.check_bounds());

        let monitor_bad_loss = SafetyMonitor::new(4.5, 0.15, 49.0);
        assert!(!monitor_bad_loss.check_bounds());

        let monitor_bad_impedance = SafetyMonitor::new(4.5, 0.05, 50.0);
        assert!(!monitor_bad_impedance.check_bounds());
    }
}

#[cfg(kani)]
mod proofs {
    use super::*;

    #[kani::proof]
    fn prove_stability_check_does_not_panic() {
        let d_x: [f32; 4] = [kani::any(), kani::any(), kani::any(), kani::any()];
        let d_x_next: [f32; 4] = [kani::any(), kani::any(), kani::any(), kani::any()];
        let c: f32 = kani::any();
        let dt: f32 = kani::any();
        let _ = check_contractive_stability(&d_x, &d_x_next, c, dt);
    }

    #[kani::proof]
    fn prove_projection_does_not_panic() {
        let mut matrix: [[f32; 4]; 4] = [
            [kani::any(), kani::any(), kani::any(), kani::any()],
            [kani::any(), kani::any(), kani::any(), kani::any()],
            [kani::any(), kani::any(), kani::any(), kani::any()],
            [kani::any(), kani::any(), kani::any(), kani::any()],
        ];
        let w_max: f32 = kani::any();
        project_matrix_a(&mut matrix, w_max);
    }
}
