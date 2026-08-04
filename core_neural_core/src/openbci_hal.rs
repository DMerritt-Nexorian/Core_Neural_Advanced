//! Module 1: `OpenBCI` Telemetry Protocol Parser (`openbci_hal`)
//! Support for Cyton/Daisy 33-byte payload decoding under `no_std` environment.

/// Error type for packet decoding failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameError {
    /// Packet header was not 0xA0.
    InvalidHeader,
    /// Packet footer was not 0xC0.
    InvalidStopByte,
    /// Computed 8-bit checksum did not match the checksum byte.
    ChecksumMismatch,
    /// Decoded channel value was out of 24-bit bounds.
    ValueOutOfBounds,
}

/// A decoded `OpenBCI` frame containing the sample ID, 8 ADS1299 EEG channels, and auxiliary data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpenBciFrame {
    /// 1-byte sample identifier.
    pub sample_id: u8,
    /// 8 raw EEG channel values (24-bit signed values).
    pub channels: [i32; 8],
    /// 6 bytes of auxiliary/accelerometer data.
    pub aux: [u8; 6],
}

impl OpenBciFrame {
    /// Decodes a 33-byte `OpenBCI` packet into an `OpenBciFrame`.
    /// Performs framing, checksum, and basic value checks.
    ///
    /// # Errors
    /// Returns `FrameError::InvalidHeader` if the first byte is not `0xA0`.
    /// Returns `FrameError::InvalidStopByte` if the last byte is not `0xC0`.
    /// Returns `FrameError::ChecksumMismatch` if the computed checksum does not match.
    /// Returns `FrameError::ValueOutOfBounds` if any decoded 24-bit channel is out of range.
    #[allow(clippy::cast_possible_wrap)]
    pub fn decode(packet: &[u8; 33]) -> Result<Self, FrameError> {
        // 1. Framing validation
        if packet[0] != 0xA0 {
            return Err(FrameError::InvalidHeader);
        }
        if packet[32] != 0xC0 {
            return Err(FrameError::InvalidStopByte);
        }

        // 2. Checksum validation
        // We use the last byte of auxiliary data (byte 31) as the 8-bit wrapping sum of the preceding 31 bytes (0..31).
        let expected_checksum = packet[31];
        let computed_checksum = packet[0..31]
            .iter()
            .fold(0u8, |acc, &b| acc.wrapping_add(b));
        if computed_checksum != expected_checksum {
            return Err(FrameError::ChecksumMismatch);
        }

        // 3. Decode 24-bit raw ADS1299 ADC channels
        // 8 channels, each 3 bytes. Offset starts at byte 2.
        let mut channels = [0i32; 8];
        for (i, channel) in channels.iter_mut().enumerate() {
            let offset = 2 + i * 3;
            let b0 = packet[offset];
            let b1 = packet[offset + 1];
            let b2 = packet[offset + 2];

            // Reconstruct 24-bit signed integer (MSB first / big-endian)
            let val = (u32::from(b0) << 16) | (u32::from(b1) << 8) | u32::from(b2);
            // Sign extension to 32-bit signed int
            let signed_val = if (val & 0x80_0000) != 0 {
                (val | 0xFF00_0000) as i32
            } else {
                val as i32
            };

            // Spatial and value bounds assertion
            // Raw ADS1299 values must be within [-8_388_600, 8_388_600]
            if !(-8_388_600..=8_388_600).contains(&signed_val) {
                return Err(FrameError::ValueOutOfBounds);
            }

            *channel = signed_val;
        }

        // 4. Extract Auxiliary data (6 bytes)
        let mut aux = [0u8; 6];
        aux.copy_from_slice(&packet[26..32]);

        Ok(OpenBciFrame {
            sample_id: packet[1],
            channels,
            aux,
        })
    }
}

/// A static-memory-bound ring buffer for `OpenBciFrame`s.
pub struct FrameRingBuffer<const N: usize> {
    buffer: [OpenBciFrame; N],
    head: usize,
    tail: usize,
    size: usize,
}

impl<const N: usize> FrameRingBuffer<N> {
    /// Creates a new `FrameRingBuffer` instance.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            buffer: [OpenBciFrame {
                sample_id: 0,
                channels: [0; 8],
                aux: [0; 6],
            }; N],
            head: 0,
            tail: 0,
            size: 0,
        }
    }

    /// Pushes a new frame into the queue.
    ///
    /// # Errors
    /// Returns `Err(frame)` containing the input frame if the queue is full.
    pub fn push(&mut self, frame: OpenBciFrame) -> Result<(), OpenBciFrame> {
        if self.size >= N {
            Err(frame)
        } else {
            self.buffer[self.tail] = frame;
            self.tail = (self.tail + 1) % N;
            self.size += 1;
            Ok(())
        }
    }

    /// Pops a frame from the queue.
    pub fn pop(&mut self) -> Option<OpenBciFrame> {
        if self.size == 0 {
            None
        } else {
            let frame = self.buffer[self.head];
            self.head = (self.head + 1) % N;
            self.size -= 1;
            Some(frame)
        }
    }

    /// Checks if the queue is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Returns the number of elements in the queue.
    #[must_use]
    pub fn len(&self) -> usize {
        self.size
    }
}

impl<const N: usize> Default for FrameRingBuffer<N> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_decode() {
        let mut packet = [0u8; 33];
        packet[0] = 0xA0;
        packet[1] = 0x01;
        packet[32] = 0xC0;
        // Checksum byte: sum of first 31 bytes (index 0 to 30) must match index 31.
        // Sum so far: packet[0] + packet[1] = 0xA0 + 0x01 = 0xA1.
        packet[31] = 0xA1;

        let res = OpenBciFrame::decode(&packet);
        assert!(res.is_ok());
        let frame = res.unwrap();
        assert_eq!(frame.sample_id, 1);
        assert_eq!(frame.channels, [0; 8]);
        assert_eq!(frame.aux, [0, 0, 0, 0, 0, 0xA1]);
    }

    #[test]
    fn test_invalid_header() {
        let mut packet = [0u8; 33];
        packet[0] = 0x00;
        packet[32] = 0xC0;
        assert_eq!(
            OpenBciFrame::decode(&packet),
            Err(FrameError::InvalidHeader)
        );
    }

    #[test]
    fn test_invalid_stop_byte() {
        let mut packet = [0u8; 33];
        packet[0] = 0xA0;
        packet[32] = 0x00;
        assert_eq!(
            OpenBciFrame::decode(&packet),
            Err(FrameError::InvalidStopByte)
        );
    }

    #[test]
    fn test_checksum_mismatch() {
        let mut packet = [0u8; 33];
        packet[0] = 0xA0;
        packet[32] = 0xC0;
        packet[31] = 0x00; // Expected checksum should be 0xA0, but we set 0x00
        assert_eq!(
            OpenBciFrame::decode(&packet),
            Err(FrameError::ChecksumMismatch)
        );
    }

    #[test]
    fn test_value_out_of_bounds() {
        let mut packet = [0u8; 33];
        packet[0] = 0xA0;
        packet[1] = 0x01;
        packet[32] = 0xC0;
        // Set first channel to 0x7F_FFFF (8_388_607), which is > 8_388_600
        packet[2] = 0x7F;
        packet[3] = 0xFF;
        packet[4] = 0xFF;
        // Recompute checksum: 0xA0 + 0x01 + 0x7F + 0xFF + 0xFF = 798 modulo 256 = 230 (0x1E)
        packet[31] = 0x1E;
        assert_eq!(
            OpenBciFrame::decode(&packet),
            Err(FrameError::ValueOutOfBounds)
        );
    }

    #[test]
    fn test_ring_buffer() {
        let mut rb = FrameRingBuffer::<4>::default();
        assert!(rb.is_empty());
        assert_eq!(rb.len(), 0);
        assert!(rb.pop().is_none());

        let frame = OpenBciFrame {
            sample_id: 1,
            channels: [1, 2, 3, 4, 5, 6, 7, 8],
            aux: [0; 6],
        };

        assert!(rb.push(frame).is_ok());
        assert_eq!(rb.len(), 1);
        assert!(!rb.is_empty());

        let popped = rb.pop();
        assert!(popped.is_some());
        assert_eq!(popped.unwrap().sample_id, 1);
        assert!(rb.is_empty());

        // Fill buffer
        assert!(rb.push(frame).is_ok());
        assert!(rb.push(frame).is_ok());
        assert!(rb.push(frame).is_ok());
        assert!(rb.push(frame).is_ok());
        // Queue is full, fifth push should fail
        assert!(rb.push(frame).is_err());
    }
}

#[cfg(kani)]
mod proofs {
    use super::*;

    #[kani::proof]
    #[kani::unwind(35)]
    fn prove_decode_does_not_panic() {
        let mut packet = [0u8; 33];
        for i in 0..33 {
            packet[i] = kani::any();
        }
        let _result = OpenBciFrame::decode(&packet);
    }

    #[kani::proof]
    #[kani::unwind(5)]
    fn prove_ring_buffer_operations() {
        let mut rb = FrameRingBuffer::<3>::new();
        let frame = OpenBciFrame {
            sample_id: kani::any(),
            channels: [kani::any(); 8],
            aux: [kani::any(); 6],
        };
        let _ = rb.push(frame);
        let _ = rb.pop();
    }
}
