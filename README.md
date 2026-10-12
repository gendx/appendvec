# AppendVec: a concurrent append-only container of immutable values

[![Crate](https://img.shields.io/crates/v/appendvec.svg?logo=rust)](https://crates.io/crates/appendvec)
[![Documentation](https://img.shields.io/docsrs/appendvec/0.2.0?logo=rust)](https://docs.rs/appendvec/0.2.0/)
[![Minimum Rust 1.85.0](https://img.shields.io/crates/msrv/appendvec/0.2.0.svg?logo=rust&color=orange)](https://releases.rs/docs/1.85.0/)
[![Lines of Code](https://www.aschey.tech/tokei/github/gendx/appendvec?category=code&branch=0.2.0)](https://github.com/gendx/appendvec/tree/0.2.0)
[![Dependencies](https://deps.rs/crate/appendvec/0.2.0/status.svg)](https://deps.rs/crate/appendvec/0.2.0)
[![License](https://img.shields.io/crates/l/appendvec/0.2.0.svg)](https://github.com/gendx/appendvec/blob/0.2.0/LICENSE)
[![Codecov](https://codecov.io/gh/gendx/appendvec/branch/0.2.0/graph/badge.svg)](https://app.codecov.io/gh/gendx/appendvec/tree/0.2.0)
[![Build Status](https://github.com/gendx/appendvec/actions/workflows/build.yml/badge.svg?branch=0.2.0)](https://github.com/gendx/appendvec/actions/workflows/build.yml)
[![Test Status](https://github.com/gendx/appendvec/actions/workflows/tests.yml/badge.svg?branch=0.2.0)](https://github.com/gendx/appendvec/actions/workflows/tests.yml)

This container data structure ensures that reads return stable references and
can happen concurrently to a write.
