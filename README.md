# base32768-rs
The base32768 library based on https://github.com/qntm/base32768

## Features

- `safe-only`: builds the crate from safe Rust alone, enforced with
  `#![forbid(unsafe_code)]`. The SIMD and pointer-based fast paths are left
  out of the build, and a simpler implementation provides the same API,
  output and errors.

```toml
[dependencies]
base32768 = { version = "0.1", features = ["safe-only"] }
```
