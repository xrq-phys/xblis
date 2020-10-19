This branch of BLIS library provides micro kernels for Arm SVE extension.

Configure with configuration name 'arm64_sve'. For more information on build system,
refer to README.md.

# Compile/configure-time parameters #

There are two additional options that can be given to the configure script:
 - --sve-vector-size
   - can be a multiple of 128 up to 2048 or 'vla'
 - --sve-use-fcmla
   - enable the use of the FCMLA instruction in the zgemm microkernel


# Run-time parameters #

## Architectural values ##

Contrary to other BLIS configurations, this is a semi-generic config, i.e. optimization will be performed according to architectural parameters (cache sizes, associativity, instruction latency, number of SIMD engines) that can be set at runtime. The defaults are specified in the file config/arm64_sve/sve_architecture.h.

The following environment variables correspond to architectural parameters:
 - BLIS_SVE_N_VFMA - number of SVE engines
 - BLIS_SVE_L_VFMA - latency of an FMA instruction in cycles
 - BLIS_SVE_W_L1 - ways of associativity (L1 data cache)
 - BLIS_SVE_N_L1 - number of cache lines per set (L1 data cache)
 - BLIS_SVE_C_L1 - cache line size in bytes (L1 data cache)
 - BLIS_SVE_W_L2 - ways of associativity (L2 data cache)
 - BLIS_SVE_N_L2 - number of cache lines per set (L2 data cache)
 - BLIS_SVE_C_L2 - cache line size in bytes (L2 data cache)
 - BLIS_SVE_W_L3 - ways of associativity (L3 data cache)
 - BLIS_SVE_N_L3 - number of cache lines per set (L3 data cache)
 - BLIS_SVE_C_L3 - cache line size in bytes (L3 data cache)

## GEMM kernel override ##

You can override the microkernel kernel that will be used by setting the environment variable BLIS_SVE_KERNEL_IDX_X with X being D,S,C or Z (currently only works for VLA DGEMM KERNELS).

The available kernels along with their index are listed in config/arm64_sve/sve_kernels.h

Here's the enum at the time of this writing (2020-10-19):

```
enum kernel_indices
{
    ukr_2vx8                =  1,
    ukr_2vx8_ld1rd          =  2,
    ukr_2vx8_ld1rqd         =  3,
    ukr_2vx9                =  4,
    ukr_2vx10_ld1rd         =  5,
    ukr_2vx10_ld1rd_colwise =  6,
    ukr_2vx10_ld1rqd        =  7,
    ukr_2vx12               =  8,
    ukr_2vx12_dup           =  9,
    ukr_2vx12_ld1rqd        = 10,
    ukr_4vx5                = 11,
    ukr_4vx5_ld1rd_colwise  = 12
};
```

i.e. setting 
```
export BLIS_SVE_KERNEL_IDX_D=6
```
would make BLIS use the kernel bli_dgemm_armv8a_sve_asm_2vx10_ld1rd_colwise for DGEMM

The block sizes m_r and n_r are automatically overridden by overriding a kernel and the calculation of the other block sizes uses the correct m_r and n_r.

## Block size override ##

The block sizes k_c, m_c and n_c can also be directly overridden by specifying the following environment variables:

- BLIS_SVE_KC_X for k_c
- BLIS_SVE_MC_X for m_c
- BLIS_SVE_NC_X for n_c

X being D,S,C or Z. Currently, only the _D variants are supported, specifying the block sizes for DGEMM. Example:

```
export BLIS_SVE_KC_D=512
export BLIS_SVE_MC_D=160
export BLIS_SVE_NC_D=4080
```

would set the block sizes k_c=512, m_c=160 and n_c=4080 for DGEMM

# Notes #

The configuration includes fixed-size kernels as well as vector-length-agnostic (VLA) kernels, which work for all SVE vector lengths (128, 256, ..., 2048). As of now, the only kernels implemented are:
 - 256 bit dgemm and zgemm
 - 512 bit dgemm and zgemm
 - 1024 bit dgemm and zgemm
 - vla dgemm
 - vla zgemm

The VLA microkernels use vector prefetch instructions, which (to this day) are
not yet supported in gem5. This negatively impacts measured performance when
running the VLA configuration in the gem5 simulator.

