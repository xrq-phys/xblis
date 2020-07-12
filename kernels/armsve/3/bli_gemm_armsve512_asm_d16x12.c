/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2020, Dept. Physics, The University of Tokyo

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


*/
#include "blis.h"

/*
   o 16x12 Double precision micro-kernel
   o Runnable on ARMv8a with SVE 512 feature, compiled with aarch64 GCC.
   x Tested on armie for SVE.
   x To be tested & benchmarked on A64fx.

   July 2020.
*/
void bli_dgemm_armsve512_asm_16x12
     (
       dim_t               k0,
       double*    restrict alpha,
       double*    restrict a,
       double*    restrict b,
       double*    restrict beta,
       double*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
  void* a_next = bli_auxinfo_next_a( data );
  void* b_next = bli_auxinfo_next_b( data );

  // Typecast local copies of integers in case dim_t and inc_t are a
  // different size than is expected by load instructions.
  uint64_t k      = k0;
  uint64_t rs_c   = rs_c0;
  uint64_t cs_c   = cs_c0;

__asm__ volatile (
" mov             x9, #16                         \n\t" // Shape M, can be input
" mov             x10, #12                        \n\t" // Shape N, fixed to be 12
" ldr             x8, %[k]                        \n\t" // Shape K to be contracted
"                                                 \n\t"
" ldr             x0, %[alpha]                    \n\t" // Alpha address
" ldr             x1, %[beta]                     \n\t" // Beta address
"                                                 \n\t"
" ldr             x2, %[aaddr]                    \n\t" // Load address of A
" mov             x3, #16                         \n\t" // LdA is 16 from packing, can be input
" ldr             x4, %[baddr]                    \n\t" // Load address of B
" mov             x5, #12                         \n\t" // LdB is 12 from packing, can be input
"                                                 \n\t"
" ldr             x6, %[caddr]                    \n\t" // Load address of C
" ldr             x7, %[cs_c]                     \n\t" // LdC, which is called column-skip in BLIS
" ldr             x20, %[rs_c]                    \n\t" // Row-skip
"                                                 \n\t"
" ldr             x18, %[a_next]                  \n\t" // Pointer to next A pack
" ldr             x19, %[b_next]                  \n\t" // Pointer to next B pack
" mov             x12, #8                         \n\t" // Double in bytes
"                                                 \n\t"
" mov             x11, xzr                        \n\t"
" incd            x11                             \n\t" // Determine vector length, in doubles.
"                                                 \n\t"
" ptrue           p0.d, all                       \n\t" // First half is all-true.
" whilelo         p1.d, x11, x9                   \n\t" // Second half from M argument.
" fmov            d0, #1.0                        \n\t" // Exact floating-point 1.0.
" fmov            x14, d0                         \n\t" // Hard float to avoid conflict with SVE.
"                                                 \n\t"
"                                                 \n\t" // SVE Register configuration:
"                                                 \n\t" // Z[30-31]: A columns
"                                                 \n\t" // Z[0-1]: B elements broadcasted
"                                                 \n\t" // Z[26-29]: Not used
"                                                 \n\t" // Z[2-25]: C change buffer
"                                                 \n\t"
" prfm            PLDL1KEEP, [x2]                 \n\t" // Prefetch A column
" prfm            PLDL1KEEP, [x16]                \n\t" // Prefetch B column
" fmov            z2.d, p0/m, #0.0                \n\t"
" fmov            z3.d, p0/m, #0.0                \n\t"
" fmov            z4.d, p0/m, #0.0                \n\t"
" fmov            z5.d, p0/m, #0.0                \n\t"
" fmov            z6.d, p0/m, #0.0                \n\t"
" fmov            z7.d, p0/m, #0.0                \n\t"
" fmov            z8.d, p0/m, #0.0                \n\t"
" fmov            z9.d, p0/m, #0.0                \n\t"
" fmov            z10.d, p0/m, #0.0               \n\t"
" fmov            z11.d, p0/m, #0.0               \n\t"
" fmov            z12.d, p0/m, #0.0               \n\t"
" fmov            z13.d, p0/m, #0.0               \n\t"
" fmov            z14.d, p0/m, #0.0               \n\t"
" fmov            z15.d, p0/m, #0.0               \n\t"
" fmov            z16.d, p0/m, #0.0               \n\t"
" fmov            z17.d, p0/m, #0.0               \n\t"
" fmov            z18.d, p0/m, #0.0               \n\t"
" fmov            z19.d, p0/m, #0.0               \n\t"
" fmov            z20.d, p0/m, #0.0               \n\t"
" fmov            z21.d, p0/m, #0.0               \n\t"
" fmov            z22.d, p0/m, #0.0               \n\t"
" fmov            z23.d, p0/m, #0.0               \n\t"
" fmov            z24.d, p0/m, #0.0               \n\t"
" fmov            z25.d, p0/m, #0.0               \n\t"
" K_LOOP:                                         \n\t"
"                                                 \n\t" // Load columns from A.
" ld1d            z30.d, p0/z, [x2]               \n\t"
" ld1d            z31.d, p1/z, [x2, x11, lsl 3]   \n\t" // Second vector
" madd            x2, x3, x12, x2                 \n\t" // Move forward
" prfm            PLDL1KEEP, [x2]                 \n\t" // Prefetch next A column
" prfm            PLDL1KEEP, [x2, #64]            \n\t" // Prefetch next A column
"                                                 \n\t"
"                                                 \n\t" // Apply B columns.
"                                                 \n\t"
" madd            x16, x5, x12, x4                \n\t" // Calculate address in advance for prefetching
" prfm            PLDL1KEEP, [x16]                \n\t" // Prefetch next B column
" prfm            PLDL1KEEP, [x16, #64]           \n\t" // Prefetch next B column
"                                                 \n\t"
" mov             x13, x10                        \n\t" // Counter, not used after size fixing.
"                                                 \n\t" // Possible BUG:
"                                                 \n\t" //  When N is odd and loop reaches end,
"                                                 \n\t" //  one more doubleword could be loaded.
"                                                 \n\t" //  Should cause no problem at this moment
"                                                 \n\t" //  as outer frame allocates padding.
"                                                 \n\t"
" ld1rqd          z0.d, p0/z, [x4, #0]            \n\t" // row L column 0 and 1
" ld1rqd          z1.d, p0/z, [x4, #16]           \n\t" // row L column 2 and 3
" fmla            z2.d, z30.d, z0.d[0]            \n\t"
" fmla            z3.d, z31.d, z0.d[0]            \n\t"
" fmla            z4.d, z30.d, z0.d[1]            \n\t"
" fmla            z5.d, z31.d, z0.d[1]            \n\t"
" fmla            z6.d, z30.d, z1.d[0]            \n\t"
" fmla            z7.d, z31.d, z1.d[0]            \n\t"
" fmla            z8.d, z30.d, z1.d[1]            \n\t"
" fmla            z9.d, z31.d, z1.d[1]            \n\t"
" ld1rqd          z0.d, p0/z, [x4, #32]           \n\t" // row L column 4 and 5
" ld1rqd          z1.d, p0/z, [x4, #48]           \n\t" // row L column 6 and 7
" fmla            z10.d, z30.d, z0.d[0]           \n\t"
" fmla            z11.d, z31.d, z0.d[0]           \n\t"
" fmla            z12.d, z30.d, z0.d[1]           \n\t"
" fmla            z13.d, z31.d, z0.d[1]           \n\t"
" fmla            z14.d, z30.d, z1.d[0]           \n\t"
" fmla            z15.d, z31.d, z1.d[0]           \n\t"
" fmla            z16.d, z30.d, z1.d[1]           \n\t"
" fmla            z17.d, z31.d, z1.d[1]           \n\t"
" ld1rqd          z0.d, p0/z, [x4, #64]           \n\t" // row L column 8 and 9
" ld1rqd          z1.d, p0/z, [x4, #80]           \n\t" // row L column 10 and 11
" fmla            z18.d, z30.d, z0.d[0]           \n\t"
" fmla            z19.d, z31.d, z0.d[0]           \n\t"
" fmla            z20.d, z30.d, z0.d[1]           \n\t"
" fmla            z21.d, z31.d, z0.d[1]           \n\t"
" fmla            z22.d, z30.d, z1.d[0]           \n\t"
" fmla            z23.d, z31.d, z1.d[0]           \n\t"
" fmla            z24.d, z30.d, z1.d[1]           \n\t"
" fmla            z25.d, z31.d, z1.d[1]           \n\t"
"                                                 \n\t"
" NEXT_ROW:                                       \n\t"
" mov             x4, x16                         \n\t" // Move forward
" subs            x8, x8, #1                      \n\t"
" b.ne            K_LOOP                          \n\t" // Next column / row.
" WRITE_MEM:                                      \n\t"
"                                                 \n\t" // Override A and B buffers:
"                                                 \n\t" // Z[30-31]: extended alpha and beta.
"                                                 \n\t" // Z[0-1]: C memory buffer.
"                                                 \n\t"
" ldr             x15, [x0]                       \n\t" // Alpha, as 64-bits
" ld1rd           z30.d, p0/z, [x0]               \n\t" // Alpha, to the vector.
" ld1rd           z31.d, p0/z, [x1]               \n\t" // Beta, to the vector.
"                                                 \n\t"
" prfm            PLDL2KEEP, [x18]                \n\t" // Prefetch next A and B.
" prfm            PLDL2KEEP, [x19]                \n\t"
"                                                 \n\t"
" cmp             x14, x15                        \n\t" // (R&)Write data back to C memory.
" b.eq            UNIT_ALPHA                      \n\t"
"                                                 \n\t"
" fmul            z2.d, z2.d, z30.d               \n\t" // Non-unit alpha case.
" fmul            z3.d, z3.d, z30.d               \n\t" // Scale all C change buffers.
" fmul            z4.d, z4.d, z30.d               \n\t"
" fmul            z5.d, z5.d, z30.d               \n\t"
" fmul            z6.d, z6.d, z30.d               \n\t"
" fmul            z7.d, z7.d, z30.d               \n\t"
" fmul            z8.d, z8.d, z30.d               \n\t"
" fmul            z9.d, z9.d, z30.d               \n\t"
" fmul            z10.d, z10.d, z30.d             \n\t"
" fmul            z11.d, z11.d, z30.d             \n\t"
" fmul            z12.d, z12.d, z30.d             \n\t"
" fmul            z13.d, z13.d, z30.d             \n\t"
" fmul            z14.d, z14.d, z30.d             \n\t"
" fmul            z15.d, z15.d, z30.d             \n\t"
" fmul            z16.d, z16.d, z30.d             \n\t"
" fmul            z17.d, z17.d, z30.d             \n\t"
" fmul            z18.d, z18.d, z30.d             \n\t"
" fmul            z19.d, z19.d, z30.d             \n\t"
" fmul            z20.d, z20.d, z30.d             \n\t"
" fmul            z21.d, z21.d, z30.d             \n\t"
" fmul            z22.d, z22.d, z30.d             \n\t"
" fmul            z23.d, z23.d, z30.d             \n\t"
" fmul            z24.d, z24.d, z30.d             \n\t"
" fmul            z25.d, z25.d, z30.d             \n\t"
"                                                 \n\t"
" UNIT_ALPHA:                                     \n\t" // Unit alpha case.
" cmp             x20, #1                         \n\t"
" b.ne            CS_CCOL                         \n\t"
"                                                 \n\t"
" CT_CCOL:                                        \n\t" // Contiguous columns.
"                                                 \n\t" // X10 counter no longer used.
"                                                 \n\t"
" madd            x16, x7, x12, x6                \n\t" // next column
" prfm            PSTL1KEEP, [x16]                \n\t" // prefetch next column
" ld1d            z0.d, p0/z, [x6]                \n\t" // column vector 0
" ld1d            z1.d, p1/z, [x6, x11, lsl #3]   \n\t"
" fmad            z0.d, p0/m, z31.d, z2.d         \n\t"
" fmad            z1.d, p1/m, z31.d, z3.d         \n\t"
" st1d            z0.d, p0, [x6]                  \n\t"
" st1d            z1.d, p1, [x6, x11, lsl #3]     \n\t"
" mov             x6, x16                         \n\t" // move forward
" madd            x16, x7, x12, x6                \n\t" // next column
" prfm            PSTL1KEEP, [x16]                \n\t" // prefetch next column
" ld1d            z0.d, p0/z, [x6]                \n\t" // column vector 1
" ld1d            z1.d, p1/z, [x6, x11, lsl #3]   \n\t"
" fmad            z0.d, p0/m, z31.d, z4.d         \n\t"
" fmad            z1.d, p1/m, z31.d, z5.d         \n\t"
" st1d            z0.d, p0, [x6]                  \n\t"
" st1d            z1.d, p1, [x6, x11, lsl #3]     \n\t"
" mov             x6, x16                         \n\t" // move forward
" madd            x16, x7, x12, x6                \n\t" // next column
" prfm            PSTL1KEEP, [x16]                \n\t" // prefetch next column
" ld1d            z0.d, p0/z, [x6]                \n\t" // column vector 2
" ld1d            z1.d, p1/z, [x6, x11, lsl #3]   \n\t"
" fmad            z0.d, p0/m, z31.d, z6.d         \n\t"
" fmad            z1.d, p1/m, z31.d, z7.d         \n\t"
" st1d            z0.d, p0, [x6]                  \n\t"
" st1d            z1.d, p1, [x6, x11, lsl #3]     \n\t"
" mov             x6, x16                         \n\t" // move forward
" madd            x16, x7, x12, x6                \n\t" // next column
" prfm            PSTL1KEEP, [x16]                \n\t" // prefetch next column
" ld1d            z0.d, p0/z, [x6]                \n\t" // column vector 3
" ld1d            z1.d, p1/z, [x6, x11, lsl #3]   \n\t"
" fmad            z0.d, p0/m, z31.d, z8.d         \n\t"
" fmad            z1.d, p1/m, z31.d, z9.d         \n\t"
" st1d            z0.d, p0, [x6]                  \n\t"
" st1d            z1.d, p1, [x6, x11, lsl #3]     \n\t"
" mov             x6, x16                         \n\t" // move forward
" madd            x16, x7, x12, x6                \n\t" // next column
" prfm            PSTL1KEEP, [x16]                \n\t" // prefetch next column
" ld1d            z0.d, p0/z, [x6]                \n\t" // column vector 4
" ld1d            z1.d, p1/z, [x6, x11, lsl #3]   \n\t"
" fmad            z0.d, p0/m, z31.d, z10.d        \n\t"
" fmad            z1.d, p1/m, z31.d, z11.d        \n\t"
" st1d            z0.d, p0, [x6]                  \n\t"
" st1d            z1.d, p1, [x6, x11, lsl #3]     \n\t"
" mov             x6, x16                         \n\t" // move forward
" madd            x16, x7, x12, x6                \n\t" // next column
" prfm            PSTL1KEEP, [x16]                \n\t" // prefetch next column
" ld1d            z0.d, p0/z, [x6]                \n\t" // column vector 5
" ld1d            z1.d, p1/z, [x6, x11, lsl #3]   \n\t"
" fmad            z0.d, p0/m, z31.d, z12.d        \n\t"
" fmad            z1.d, p1/m, z31.d, z13.d        \n\t"
" st1d            z0.d, p0, [x6]                  \n\t"
" st1d            z1.d, p1, [x6, x11, lsl #3]     \n\t"
" mov             x6, x16                         \n\t" // move forward
" madd            x16, x7, x12, x6                \n\t" // next column
" prfm            PSTL1KEEP, [x16]                \n\t" // prefetch next column
" ld1d            z0.d, p0/z, [x6]                \n\t" // column vector 6
" ld1d            z1.d, p1/z, [x6, x11, lsl #3]   \n\t"
" fmad            z0.d, p0/m, z31.d, z14.d        \n\t"
" fmad            z1.d, p1/m, z31.d, z15.d        \n\t"
" st1d            z0.d, p0, [x6]                  \n\t"
" st1d            z1.d, p1, [x6, x11, lsl #3]     \n\t"
" mov             x6, x16                         \n\t" // move forward
" madd            x16, x7, x12, x6                \n\t" // next column
" prfm            PSTL1KEEP, [x16]                \n\t" // prefetch next column
" ld1d            z0.d, p0/z, [x6]                \n\t" // column vector 7
" ld1d            z1.d, p1/z, [x6, x11, lsl #3]   \n\t"
" fmad            z0.d, p0/m, z31.d, z16.d        \n\t"
" fmad            z1.d, p1/m, z31.d, z17.d        \n\t"
" st1d            z0.d, p0, [x6]                  \n\t"
" st1d            z1.d, p1, [x6, x11, lsl #3]     \n\t"
" mov             x6, x16                         \n\t" // move forward
" madd            x16, x7, x12, x6                \n\t" // next column
" prfm            PSTL1KEEP, [x16]                \n\t" // prefetch next column
" ld1d            z0.d, p0/z, [x6]                \n\t" // column vector 8
" ld1d            z1.d, p1/z, [x6, x11, lsl #3]   \n\t"
" fmad            z0.d, p0/m, z31.d, z18.d        \n\t"
" fmad            z1.d, p1/m, z31.d, z19.d        \n\t"
" st1d            z0.d, p0, [x6]                  \n\t"
" st1d            z1.d, p1, [x6, x11, lsl #3]     \n\t"
" mov             x6, x16                         \n\t" // move forward
" madd            x16, x7, x12, x6                \n\t" // next column
" prfm            PSTL1KEEP, [x16]                \n\t" // prefetch next column
" ld1d            z0.d, p0/z, [x6]                \n\t" // column vector 9
" ld1d            z1.d, p1/z, [x6, x11, lsl #3]   \n\t"
" fmad            z0.d, p0/m, z31.d, z20.d        \n\t"
" fmad            z1.d, p1/m, z31.d, z21.d        \n\t"
" st1d            z0.d, p0, [x6]                  \n\t"
" st1d            z1.d, p1, [x6, x11, lsl #3]     \n\t"
" mov             x6, x16                         \n\t" // move forward
" madd            x16, x7, x12, x6                \n\t" // next column
" prfm            PSTL1KEEP, [x16]                \n\t" // prefetch next column
" ld1d            z0.d, p0/z, [x6]                \n\t" // column vector 10
" ld1d            z1.d, p1/z, [x6, x11, lsl #3]   \n\t"
" fmad            z0.d, p0/m, z31.d, z22.d        \n\t"
" fmad            z1.d, p1/m, z31.d, z23.d        \n\t"
" st1d            z0.d, p0, [x6]                  \n\t"
" st1d            z1.d, p1, [x6, x11, lsl #3]     \n\t"
" mov             x6, x16                         \n\t" // move forward
" madd            x16, x7, x12, x6                \n\t" // next column
" prfm            PSTL1KEEP, [x16]                \n\t" // prefetch next column
" ld1d            z0.d, p0/z, [x6]                \n\t" // column vector 11
" ld1d            z1.d, p1/z, [x6, x11, lsl #3]   \n\t"
" fmad            z0.d, p0/m, z31.d, z24.d        \n\t"
" fmad            z1.d, p1/m, z31.d, z25.d        \n\t"
" st1d            z0.d, p0, [x6]                  \n\t"
" st1d            z1.d, p1, [x6, x11, lsl #3]     \n\t"
"                                                 \n\t"
"                                                 \n\t"
" b               END_WRITE_MEM                   \n\t"
"                                                 \n\t"
" CS_CCOL:                                        \n\t" // C has row-strides.
" mul             x21, x20, x12                   \n\t" // Column stride in bytes
" mul             x17, x21, x11                   \n\t" // Vector length in memory
"                                                 \n\t"
"                                                 \n\t" // Z30: index for loading C columns.
" index           z30.d, xzr, x21                 \n\t" // Generate indices.
"                                                 \n\t"
" add             x16, x17, x6                    \n\t"
" ld1d            z0.d, p0/z, [x6, z30.d]         \n\t" // column vector 0
" ld1d            z0.d, p1/z, [x16, z30.d]        \n\t"
" fmad            z0.d, p0/m, z31.d, z2.d         \n\t"
" fmad            z0.d, p1/m, z31.d, z3.d         \n\t"
" st1d            z0.d, p0, [x6, z30.d]           \n\t"
" st1d            z0.d, p1, [x16, z30.d]          \n\t"
" madd            x6, x7, x12, x6                 \n\t"
" add             x16, x17, x6                    \n\t"
" ld1d            z0.d, p0/z, [x6, z30.d]         \n\t" // column vector 1
" ld1d            z0.d, p1/z, [x16, z30.d]        \n\t"
" fmad            z0.d, p0/m, z31.d, z4.d         \n\t"
" fmad            z0.d, p1/m, z31.d, z5.d         \n\t"
" st1d            z0.d, p0, [x6, z30.d]           \n\t"
" st1d            z0.d, p1, [x16, z30.d]          \n\t"
" madd            x6, x7, x12, x6                 \n\t"
" add             x16, x17, x6                    \n\t"
" ld1d            z0.d, p0/z, [x6, z30.d]         \n\t" // column vector 2
" ld1d            z0.d, p1/z, [x16, z30.d]        \n\t"
" fmad            z0.d, p0/m, z31.d, z6.d         \n\t"
" fmad            z0.d, p1/m, z31.d, z7.d         \n\t"
" st1d            z0.d, p0, [x6, z30.d]           \n\t"
" st1d            z0.d, p1, [x16, z30.d]          \n\t"
" madd            x6, x7, x12, x6                 \n\t"
" add             x16, x17, x6                    \n\t"
" ld1d            z0.d, p0/z, [x6, z30.d]         \n\t" // column vector 3
" ld1d            z0.d, p1/z, [x16, z30.d]        \n\t"
" fmad            z0.d, p0/m, z31.d, z8.d         \n\t"
" fmad            z0.d, p1/m, z31.d, z9.d         \n\t"
" st1d            z0.d, p0, [x6, z30.d]           \n\t"
" st1d            z0.d, p1, [x16, z30.d]          \n\t"
" madd            x6, x7, x12, x6                 \n\t"
" add             x16, x17, x6                    \n\t"
" ld1d            z0.d, p0/z, [x6, z30.d]         \n\t" // column vector 4
" ld1d            z0.d, p1/z, [x16, z30.d]        \n\t"
" fmad            z0.d, p0/m, z31.d, z10.d        \n\t"
" fmad            z0.d, p1/m, z31.d, z11.d        \n\t"
" st1d            z0.d, p0, [x6, z30.d]           \n\t"
" st1d            z0.d, p1, [x16, z30.d]          \n\t"
" madd            x6, x7, x12, x6                 \n\t"
" add             x16, x17, x6                    \n\t"
" ld1d            z0.d, p0/z, [x6, z30.d]         \n\t" // column vector 5
" ld1d            z0.d, p1/z, [x16, z30.d]        \n\t"
" fmad            z0.d, p0/m, z31.d, z12.d        \n\t"
" fmad            z0.d, p1/m, z31.d, z13.d        \n\t"
" st1d            z0.d, p0, [x6, z30.d]           \n\t"
" st1d            z0.d, p1, [x16, z30.d]          \n\t"
" madd            x6, x7, x12, x6                 \n\t"
" add             x16, x17, x6                    \n\t"
" ld1d            z0.d, p0/z, [x6, z30.d]         \n\t" // column vector 6
" ld1d            z0.d, p1/z, [x16, z30.d]        \n\t"
" fmad            z0.d, p0/m, z31.d, z14.d        \n\t"
" fmad            z0.d, p1/m, z31.d, z15.d        \n\t"
" st1d            z0.d, p0, [x6, z30.d]           \n\t"
" st1d            z0.d, p1, [x16, z30.d]          \n\t"
" madd            x6, x7, x12, x6                 \n\t"
" add             x16, x17, x6                    \n\t"
" ld1d            z0.d, p0/z, [x6, z30.d]         \n\t" // column vector 7
" ld1d            z0.d, p1/z, [x16, z30.d]        \n\t"
" fmad            z0.d, p0/m, z31.d, z16.d        \n\t"
" fmad            z0.d, p1/m, z31.d, z17.d        \n\t"
" st1d            z0.d, p0, [x6, z30.d]           \n\t"
" st1d            z0.d, p1, [x16, z30.d]          \n\t"
" madd            x6, x7, x12, x6                 \n\t"
" add             x16, x17, x6                    \n\t"
" ld1d            z0.d, p0/z, [x6, z30.d]         \n\t" // column vector 8
" ld1d            z0.d, p1/z, [x16, z30.d]        \n\t"
" fmad            z0.d, p0/m, z31.d, z18.d        \n\t"
" fmad            z0.d, p1/m, z31.d, z19.d        \n\t"
" st1d            z0.d, p0, [x6, z30.d]           \n\t"
" st1d            z0.d, p1, [x16, z30.d]          \n\t"
" madd            x6, x7, x12, x6                 \n\t"
" add             x16, x17, x6                    \n\t"
" ld1d            z0.d, p0/z, [x6, z30.d]         \n\t" // column vector 9
" ld1d            z0.d, p1/z, [x16, z30.d]        \n\t"
" fmad            z0.d, p0/m, z31.d, z20.d        \n\t"
" fmad            z0.d, p1/m, z31.d, z21.d        \n\t"
" st1d            z0.d, p0, [x6, z30.d]           \n\t"
" st1d            z0.d, p1, [x16, z30.d]          \n\t"
" madd            x6, x7, x12, x6                 \n\t"
" add             x16, x17, x6                    \n\t"
" ld1d            z0.d, p0/z, [x6, z30.d]         \n\t" // column vector 10
" ld1d            z0.d, p1/z, [x16, z30.d]        \n\t"
" fmad            z0.d, p0/m, z31.d, z22.d        \n\t"
" fmad            z0.d, p1/m, z31.d, z23.d        \n\t"
" st1d            z0.d, p0, [x6, z30.d]           \n\t"
" st1d            z0.d, p1, [x16, z30.d]          \n\t"
" madd            x6, x7, x12, x6                 \n\t"
" add             x16, x17, x6                    \n\t"
" ld1d            z0.d, p0/z, [x6, z30.d]         \n\t" // column vector 11
" ld1d            z0.d, p1/z, [x16, z30.d]        \n\t"
" fmad            z0.d, p0/m, z31.d, z24.d        \n\t"
" fmad            z0.d, p1/m, z31.d, z25.d        \n\t"
" st1d            z0.d, p0, [x6, z30.d]           \n\t"
" st1d            z0.d, p1, [x16, z30.d]          \n\t"
"                                                 \n\t"
"                                                 \n\t"
" END_WRITE_MEM:                                  \n\t" // End of computation.
" mov             x0, #0                          \n\t" // Return normal.
" b               END_EXEC                        \n\t"
" END_ERROR:                                      \n\t"
" mov             x0, #1                          \n\t" // Return error.
" END_EXEC:                                       \n\t"
:// output operands (none)
:// input operands
 [aaddr]  "m" (a),      // 0
 [baddr]  "m" (b),      // 1
 [caddr]  "m" (c),      // 2
 [k]      "m" (k),      // 3
 [alpha]  "m" (alpha),  // 5
 [beta]   "m" (beta),   // 6
 [rs_c]   "m" (rs_c),   // 6
 [cs_c]   "m" (cs_c),   // 7
 [a_next] "m" (a_next), // 8
 [b_next] "m" (b_next)  // 9
:// Register clobber list
 "x0","x1","x2","x3","x4","x5","x6","x7","x8",
 "x9","x10","x11","x12","x13","x14","x15",
 "x16","x17","x18","x19","x20","x21","x22","x23","x24","x25","x26","x27",
 "z0","z1","z2","z3","z4","z5","z6","z7",
 "z8","z9","z10","z11","z12","z13","z14","z15",
 "z16","z17","z18","z19",
 "z20","z21","z22","z23",
 "z24","z25","z26","z27",
 "z28","z29","z30","z31" );

}
