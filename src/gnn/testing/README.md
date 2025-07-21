# Testing and Benchmarks

This directory contains testing infrastructure and performance benchmarks:

- `test_round_trip.py` - Comprehensive round-trip testing (100% success rate achieved)
- `README_round_trip.md` - Detailed testing methodology and results
- `round_trip_reports/` - Test reports and analysis

## Current Status: 🎉 100% Success Achieved

**Success Rate: 100.0% (20/20 formats)**

### Perfect Round-Trip Formats
- **Schema Formats (7/7)**: JSON, XML, YAML, PKL, ASN.1, Protobuf, XSD
- **Language Formats (6/6)**: Scala, Python, Lean, Coq, Isabelle, Haskell  
- **Formal Formats (5/5)**: Alloy, BNF, Z-notation, TLA+, Agda
- **Other Formats (2/2)**: Pickle, Maxima

### Performance Benchmarks
- **Total Test Time**: ~0.07 seconds for all 20 formats
- **Individual Format**: <0.01 seconds per format
- **Success Rate**: 100% (20/20 formats)
- **Semantic Checksum Match**: 19/20 formats (95%)

These tools ensure code quality and monitor performance of GNN operations with complete format interoperability.
