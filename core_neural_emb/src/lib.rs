#![no_std]
#![deny(unsafe_code)]
#![deny(clippy::pedantic)]

// Library code goes here.
pub fn init() {
    core_neural_core::init();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init() {
        init();
    }
}
