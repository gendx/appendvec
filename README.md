# AppendVec: a concurrent append-only container of immutable values

[![Minimum Rust 1.85.0](https://img.shields.io/badge/rust-1.85.0%2B-orange.svg?logo=rust)](https://releases.rs/docs/1.85.0/)
[![Lines of Code](https://www.aschey.tech/tokei/github/gendx/appendvec?category=code&branch=main)](https://github.com/gendx/appendvec)
[![Dependencies](https://deps.rs/repo/github/gendx/appendvec/status.svg)](https://deps.rs/repo/github/gendx/appendvec)
[![Codecov](https://codecov.io/gh/gendx/appendvec/branch/main/graph/badge.svg)](https://app.codecov.io/gh/gendx/appendvec/tree/main)
[![Build Status](https://github.com/gendx/appendvec/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/gendx/appendvec/actions/workflows/build.yml)
[![Test Status](https://github.com/gendx/appendvec/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/gendx/appendvec/actions/workflows/tests.yml)

This container data structure ensures that reads return stable references and
can happen concurrently to a write.
