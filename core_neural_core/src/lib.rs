#![no_std]
#![deny(unsafe_code)]
#![deny(clippy::pedantic)]
#![allow(unexpected_cfgs)]

pub mod openbci_hal;
pub mod safety_core;
pub mod snn_core;

pub fn init() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lib_init() {
        init();
    }
}
