This branch of BLIS library provides micro kernels for Arm SVE extension.

Configure with configuration name 'arm64_sve'. For more information on build system,
refer to README.md.

Architectural parameters must be specified in the file config/arm64_sve/sve_architecture.h.

The configuration includes vector-length-agnostic kernels, which work for all SVE vector lengths (128, 256, ..., 2048).
VLA kernels are used if SVE_VECSIZE is set to SVE_VECSIZE_VLA in config/arm64_sve/sve_architecture.h

To use fixed size kernels, set SVE_VECSIZE to one of the other parameters listed in the file. 
As of now, only kernels for 256,512 and 1024 bits are implemented.

Note: 
The VLA microkernel uses vector prefetch instructions, which (to this day) are
not yet supported in gem5. This results in bad performance. If running in gem5,
one should use fixed-size kernels.

