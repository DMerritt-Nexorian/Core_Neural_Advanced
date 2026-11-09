# cyton_reader

Live OpenBCI Cyton serial reader that feeds raw 33-byte packets into the existing `OpenBciFrame::decode` → `pipeline::step` path.

## Usage

```bash
# List ports if needed
cargo run -p cyton_reader -- --help   # or just run to see usage

# Linux
cargo run -p cyton_reader -- /dev/ttyUSB0

# macOS
cargo run -p cyton_reader -- /dev/cu.usbserial-*

# Windows
cargo run -p cyton_reader -- COM3
```

The binary opens the port at 115200 baud, reads 33-byte frames, decodes them, and runs the LIF + trajectory pipeline. It prints a status line roughly once per second.

Uncomment the `port.write(b"b")` line in `main.rs` if you need to send the Cyton start-streaming command.
