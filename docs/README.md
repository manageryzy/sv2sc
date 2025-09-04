# sv2sc Documentation

## Overview

This directory contains comprehensive documentation for the sv2sc SystemVerilog to SystemC translator project. The documentation is organized to provide both high-level overviews and detailed implementation guides.

## Current Documentation

### **Core Status Documents**

#### [CURRENT_PROJECT_STATUS.md](CURRENT_PROJECT_STATUS.md)
**Comprehensive project status and compliance assessment**
- Executive summary of current state
- Detailed compliance metrics
- Real-world validation results
- Outstanding items and next steps
- **Status**: Updated August 26, 2024

### **Feature Documentation**

#### [NBA_SPLITTING_FEATURE.md](NBA_SPLITTING_FEATURE.md)
**Non-Blocking Assignment splitting for performance optimization**
- Complete feature overview and implementation details
- Performance impact analysis (2-5x simulation speedup)
- Technical implementation with code examples
- Usage instructions and configuration options
- **Status**: ✅ Complete and Active

#### [MLIR_INTEGRATION_STATUS.md](MLIR_INTEGRATION_STATUS.md)
**MLIR/CIRCT integration for modern compiler infrastructure**
- Phase 1-3 implementation status
- Comprehensive expression and statement support
- Pass pipeline infrastructure
- Migration path to Phase 4 (CIRCT integration)
- **Status**: Phase 1-3 Complete, Ready for Phase 4

### **Verification and Testing**

#### [PICORV32_VERIFICATION_SUITE.md](PICORV32_VERIFICATION_SUITE.md)
**Real-world validation using PicoRV32 RISC-V CPU**
- Translation success and quality metrics
- Complete verification framework
- Test programs and comparison methodology
- Build instructions and troubleshooting
- **Status**: Translation Complete, Verification Pending

### **User and Developer Guides**

#### [USER_GUIDE.md](USER_GUIDE.md)
**User guide for sv2sc translator**
- Installation and setup instructions
- Command-line usage examples
- Supported SystemVerilog features
- Troubleshooting common issues
- **Status**: Core user documentation

#### [ARCHITECTURE.md](ARCHITECTURE.md)
**System architecture and design overview**
- High-level system design
- Component interactions
- Data flow and processing pipeline
- Extension points and customization
- **Status**: Architecture documentation

### **Advanced Integration**

#### [CIRCT_INTEGRATION_GUIDE.md](CIRCT_INTEGRATION_GUIDE.md)
**Step-by-step guide for CIRCT integration**
- Detailed integration instructions
- Environment setup for different options
- Migration procedures and testing
- Troubleshooting integration issues
- **Status**: Ready for Phase 4 implementation

#### [PHASE4_READINESS_STATUS.md](PHASE4_READINESS_STATUS.md)
**Phase 4 readiness assessment and planning**
- Current readiness evaluation
- Integration options and timelines
- Risk assessment and mitigation
- Success criteria and validation
- **Status**: Ready for CIRCT integration

## Documentation Organization

### **By Audience**
- **Users**: `USER_GUIDE.md`, `CURRENT_PROJECT_STATUS.md`
- **Developers**: `ARCHITECTURE.md`, `MLIR_INTEGRATION_STATUS.md`
- **Integrators**: `CIRCT_INTEGRATION_GUIDE.md`, `PHASE4_READINESS_STATUS.md`
- **Researchers**: `NBA_SPLITTING_FEATURE.md`, `PICORV32_VERIFICATION_SUITE.md`

### **By Status**
- **Complete**: `NBA_SPLITTING_FEATURE.md`, `CURRENT_PROJECT_STATUS.md`
- **In Progress**: `MLIR_INTEGRATION_STATUS.md` (Phase 1-3 complete)
- **Ready for Next Phase**: `CIRCT_INTEGRATION_GUIDE.md`, `PHASE4_READINESS_STATUS.md`
- **Pending External Tools**: `PICORV32_VERIFICATION_SUITE.md`

### **By Type**
- **Status Reports**: `CURRENT_PROJECT_STATUS.md`
- **Feature Documentation**: `NBA_SPLITTING_FEATURE.md`, `MLIR_INTEGRATION_STATUS.md`
- **User Guides**: `USER_GUIDE.md`
- **Technical Guides**: `ARCHITECTURE.md`, `CIRCT_INTEGRATION_GUIDE.md`
- **Verification**: `PICORV32_VERIFICATION_SUITE.md`

## Archive

Outdated and superseded documentation has been moved to the `docs/archive/` directory. This includes:
- Old implementation status documents
- Superseded design documents
- Outdated syntax fix reports
- Consolidated feature documentation

## Getting Started

### **For New Users**
1. Start with `CURRENT_PROJECT_STATUS.md` for project overview
2. Read `USER_GUIDE.md` for usage instructions
3. Try the basic examples in the user guide

### **For Developers**
1. Review `ARCHITECTURE.md` for system design
2. Check `MLIR_INTEGRATION_STATUS.md` for current implementation
3. Read `NBA_SPLITTING_FEATURE.md` for performance features

### **For Integrators**
1. Review `PHASE4_READINESS_STATUS.md` for current readiness
2. Follow `CIRCT_INTEGRATION_GUIDE.md` for integration steps
3. Use `PICORV32_VERIFICATION_SUITE.md` for validation

### **For Researchers**
1. Study `NBA_SPLITTING_FEATURE.md` for performance optimization
2. Review `MLIR_INTEGRATION_STATUS.md` for modern compiler architecture
3. Use `PICORV32_VERIFICATION_SUITE.md` for real-world validation

## Contributing to Documentation

When updating documentation:
1. Update the relevant status document
2. Ensure cross-references are maintained
3. Move outdated documents to `docs/archive/`
4. Update this README.md index

## Documentation Standards

- **Status Indicators**: Use ✅ for complete, ⚠️ for partial, ❌ for not started
- **Date Stamps**: Include last updated dates on all documents
- **Cross-References**: Link related documents where appropriate
- **Code Examples**: Include working code examples and commands
- **Troubleshooting**: Provide common issues and solutions

---

*Last Updated: August 26, 2024*  
*Total Active Documents: 8*  
*Archived Documents: 23*
