# Contributing to ngsolve-sparsesolv

Thank you for your interest in contributing.

## Getting Started

See [docs/development.md](docs/development.md) for:
- Build instructions (CMake, MKL)
- How to run tests
- Known issues and past bugs

## Bug Reports

Please create a [GitHub Issue](https://github.com/ksugahar/ngsolve-sparsesolv/issues) with:
- NGSolve version and OS
- Minimal reproducible example
- Expected behavior vs actual behavior

## Pull Requests

1. Fork the repository and create a feature branch
2. Make your changes
3. Run the full test suite:
   ```bash
   python -m pytest tests/test_sparsesolv.py -v --tb=short
   ```
4. Submit a pull request with a clear description

### Code Style

- C++17, header-only templates (`<Scalar>` for double/complex)
- Follow existing naming conventions (snake_case for functions, CamelCase for classes)
- Keep NGSolve-specific code in `include/sparsesolv/ngsolve/`

## License

By contributing, you agree that your contributions will be licensed under [MPL 2.0](LICENSE).
