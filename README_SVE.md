This branch of BLIS library provides micro kernels for Arm SVE extension.

Configure with configuration name 'arm64_sve'. For more information on build system,
refer to README.md.

Contrary to other BLIS configurations, this is a semi-generic config, i.e. optimization will be performed according to architectural parameters (cache sizes, associativity, instruction latency, number of SIMD engines) that can be specified in the file config/arm64_sve/sve_architecture.h.

The configuration includes fixed-size kernels as well as vector-length-agnostic (VLA) kernels, which work for all SVE vector lengths (128, 256, ..., 2048). As of now, the only kernels implemented are:
 - 256 bit dgemm and zgemm
 - 512 bit dgemm and zgemm
 - 1024 bit dgemm and zgemm
 - vla dgemm
 - vla zgemm

There are two additional options that can be given to the configure script:
 - --sve-vector-size
   - can be a multiple of 128 up to 2048 or 'vla'
 - --sve-use-fcmla
   - enable the use of the FCMLA instruction in the zgemm microkernel

Note: 
The VLA microkernels use vector prefetch instructions, which (to this day) are
not yet supported in gem5. This negatively impacts measured performance when
running the VLA configuration in the gem5 simulator.

