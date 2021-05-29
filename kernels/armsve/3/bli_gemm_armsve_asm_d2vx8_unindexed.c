/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2019, Forschunszentrum Juelich
   Copyright (C) 2020, The University of Tokyo

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

// Double-precision composite instructions.
#include "armsve_asm_macros_double.h"

// 2vx8 microkernels.
#include "armsve_asm_2vx8.h"

void bli_dgemm_armsve_asm_2vx8_unindexed
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
  uint64_t k_mker = k0 / 15;
  uint64_t k_left = k0 % 15;
  uint64_t rs_c   = rs_c0;
  uint64_t cs_c   = cs_c0;

  __asm__ volatile (
" ldr             x0, %[a]                        \n\t"
" ldr             x1, %[b]                        \n\t"
" mov             x2, xzr                         \n\t"
" incd            x2, ALL, MUL #2                 \n\t" // Column-skip of A.
" mov             x3, #8                          \n\t" // Row-skip of B.
"                                                 \n\t"
" ldr             x5, %[c]                        \n\t"
" ldr             x6, %[rs_c]                     \n\t" // Row-skip of C.
" ldr             x7, %[cs_c]                     \n\t" // Column-skip of C.
#ifdef _A64FX
" mov             x8, 0x3                         \n\t" // Tag C address.
" lsl             x8, x8, #56                     \n\t"
" orr             x5, x5, x8                      \n\t"
" mov             x8, 0x2                         \n\t" // Tag B address.
" lsl             x8, x8, #56                     \n\t"
" orr             x1, x1, x8                      \n\t"
" mov             x8, 0x1                         \n\t" // Tag A address.
" lsl             x8, x8, #56                     \n\t"
" orr             x0, x0, x8                      \n\t"
#endif
"                                                 \n\t"
" mov             x8, #8                          \n\t" // Multiply some address skips by sizeof(double).
" madd            x2, x8, x2, xzr                 \n\t" // cs_a
" madd            x3, x8, x3, xzr                 \n\t" // rs_b
" madd            x7, x8, x7, xzr                 \n\t" // cs_c
" ptrue           p0.d                            \n\t"
"                                                 \n\t"
" ldr             x4, %[k_mker]                   \n\t" // Number of loops.
" ldr             x8, %[k_left]                   \n\t"
"                                                 \n\t"
" LOAD_ABC:                                       \n\t"
" cmp             x4, #0                          \n\t" // Don't preload if no microkernel there.
" b.eq            END_CCOL_PRFM                   \n\t"

" ld1rd           z16.d, p0/z, [x1, 8*0]          \n\t" // Load first B row.
" ld1rd           z17.d, p0/z, [x1, 8*1]          \n\t"
" ld1rd           z18.d, p0/z, [x1, 8*2]          \n\t"
" ld1rd           z19.d, p0/z, [x1, 8*3]          \n\t"
" ld1rd           z20.d, p0/z, [x1, 8*4]          \n\t"
" ld1rd           z21.d, p0/z, [x1, 8*5]          \n\t"
" ld1rd           z22.d, p0/z, [x1, 8*6]          \n\t"
" ld1rd           z23.d, p0/z, [x1, 8*7]          \n\t"
" add             x1, x1, x3                      \n\t" // Forward B's address to the next row.
" ld1rd           z24.d, p0/z, [x1, 8*0]          \n\t" // Load 2/8 second B row.
" ld1rd           z25.d, p0/z, [x1, 8*1]          \n\t"
"                                                 \n\t"
GEMM_ACOL_CONTIGUOUS_LOAD(z26,z27,p0,p0,x0)
" add             x0, x0, x2                      \n\t" // Forward A's address to the next column.
GEMM_ACOL_CONTIGUOUS_LOAD(z28,z29,p0,p0,x0)
" add             x0, x0, x2                      \n\t" // Forward A's address to the next column.
GEMM_ACOL_CONTIGUOUS_LOAD(z30,z31,p0,p0,x0)
" add             x0, x0, x2                      \n\t" // Forward A's address to the next column.
"                                                 \n\t"
" CCOL_PRFM:                                      \n\t"
" cmp             x6, #1                          \n\t"
" b.ne            END_CCOL_PRFM                   \n\t" // Do not prefetch for generic C storage.
" mov             x16, x5                         \n\t"
" prfm            PLDL1KEEP, [x16]                \n\t"
" add             x16, x16, x7                    \n\t"
" prfm            PLDL1KEEP, [x16]                \n\t"
" add             x16, x16, x7                    \n\t"
" prfm            PLDL1KEEP, [x16]                \n\t"
" add             x16, x16, x7                    \n\t"
" prfm            PLDL1KEEP, [x16]                \n\t"
" add             x16, x16, x7                    \n\t"
" prfm            PLDL1KEEP, [x16]                \n\t"
" add             x16, x16, x7                    \n\t"
" prfm            PLDL1KEEP, [x16]                \n\t"
" add             x16, x16, x7                    \n\t"
" prfm            PLDL1KEEP, [x16]                \n\t"
" add             x16, x16, x7                    \n\t"
" prfm            PLDL1KEEP, [x16]                \n\t"
" END_CCOL_PRFM:                                  \n\t"
"                                                 \n\t"
CLEAR_COL16(z0,z1,z2,z3,z4,z5,z6,z7,z8,z9,z10,z11,z12,z13,z14,z15)
"                                                 \n\t"
" cmp             x4, #0                          \n\t" // If no 5-microkernel can be applied
" b.eq            K_LEFT_LOOP                     \n\t"
"                                                 \n\t"
" K_MKER_LOOP:                                    \n\t"
"                                                 \n\t"
GEMM_2VX8_MKER_LOOP_PLAIN_C_1(z0,z2,z4,z6,z8,z10,z12,z14,z1,z3,z5,z7,z9,z11,z13,z15,p0,z26,z27,z16,z17,z18,z19,z20,z21,z22,z23,z24,z25,x1,x3)
GEMM_ACOL_CONTIGUOUS_LOAD(z26,z27,p0,p0,x0)
" add             x0, x0, x2                      \n\t"
GEMM_2VX8_MKER_LOOP_PLAIN_C_2(z0,z2,z4,z6,z8,z10,z12,z14,z1,z3,z5,z7,z9,z11,z13,z15,p0,z28,z29,z16,z17,z18,z19,z20,z21,z22,z23,z24,z25,x1,x3)
GEMM_ACOL_CONTIGUOUS_LOAD(z28,z29,p0,p0,x0)
" add             x0, x0, x2                      \n\t"
GEMM_2VX8_MKER_LOOP_PLAIN_C_3(z0,z2,z4,z6,z8,z10,z12,z14,z1,z3,z5,z7,z9,z11,z13,z15,p0,z30,z31,z16,z17,z18,z19,z20,z21,z22,z23,z24,z25,x1,x3)
GEMM_ACOL_CONTIGUOUS_LOAD(z30,z31,p0,p0,x0)
" add             x0, x0, x2                      \n\t"
GEMM_2VX8_MKER_LOOP_PLAIN_C_4(z0,z2,z4,z6,z8,z10,z12,z14,z1,z3,z5,z7,z9,z11,z13,z15,p0,z26,z27,z16,z17,z18,z19,z20,z21,z22,z23,z24,z25,x1,x3)
GEMM_ACOL_CONTIGUOUS_LOAD(z26,z27,p0,p0,x0)
" add             x0, x0, x2                      \n\t"
GEMM_2VX8_MKER_LOOP_PLAIN_C_5(z0,z2,z4,z6,z8,z10,z12,z14,z1,z3,z5,z7,z9,z11,z13,z15,p0,z28,z29,z16,z17,z18,z19,z20,z21,z22,z23,z24,z25,x1,x3)
GEMM_ACOL_CONTIGUOUS_LOAD(z28,z29,p0,p0,x0)
" add             x0, x0, x2                      \n\t"
"                                                 \n\t"
"                                                 \n\t"
GEMM_2VX8_MKER_LOOP_PLAIN_C_1(z0,z2,z4,z6,z8,z10,z12,z14,z1,z3,z5,z7,z9,z11,z13,z15,p0,z30,z31,z16,z17,z18,z19,z20,z21,z22,z23,z24,z25,x1,x3)
GEMM_ACOL_CONTIGUOUS_LOAD(z30,z31,p0,p0,x0)
" add             x0, x0, x2                      \n\t" // Forward A's address to the next column.
GEMM_2VX8_MKER_LOOP_PLAIN_C_2(z0,z2,z4,z6,z8,z10,z12,z14,z1,z3,z5,z7,z9,z11,z13,z15,p0,z26,z27,z16,z17,z18,z19,z20,z21,z22,z23,z24,z25,x1,x3)
GEMM_ACOL_CONTIGUOUS_LOAD(z26,z27,p0,p0,x0)
" add             x0, x0, x2                      \n\t" // Forward A's address to the next column.
GEMM_2VX8_MKER_LOOP_PLAIN_C_3(z0,z2,z4,z6,z8,z10,z12,z14,z1,z3,z5,z7,z9,z11,z13,z15,p0,z28,z29,z16,z17,z18,z19,z20,z21,z22,z23,z24,z25,x1,x3)
GEMM_ACOL_CONTIGUOUS_LOAD(z28,z29,p0,p0,x0)
" add             x0, x0, x2                      \n\t" // Forward A's address to the next column.
GEMM_2VX8_MKER_LOOP_PLAIN_C_4(z0,z2,z4,z6,z8,z10,z12,z14,z1,z3,z5,z7,z9,z11,z13,z15,p0,z30,z31,z16,z17,z18,z19,z20,z21,z22,z23,z24,z25,x1,x3)
GEMM_ACOL_CONTIGUOUS_LOAD(z30,z31,p0,p0,x0)
" add             x0, x0, x2                      \n\t" // Forward A's address to the next column.
GEMM_2VX8_MKER_LOOP_PLAIN_C_5(z0,z2,z4,z6,z8,z10,z12,z14,z1,z3,z5,z7,z9,z11,z13,z15,p0,z26,z27,z16,z17,z18,z19,z20,z21,z22,z23,z24,z25,x1,x3)
GEMM_ACOL_CONTIGUOUS_LOAD(z26,z27,p0,p0,x0)
" add             x0, x0, x2                      \n\t" // Forward A's address to the next column.
"                                                 \n\t"
"                                                 \n\t"
GEMM_2VX8_MKER_LOOP_PLAIN_C_1(z0,z2,z4,z6,z8,z10,z12,z14,z1,z3,z5,z7,z9,z11,z13,z15,p0,z28,z29,z16,z17,z18,z19,z20,z21,z22,z23,z24,z25,x1,x3)
GEMM_ACOL_CONTIGUOUS_LOAD(z28,z29,p0,p0,x0)
" add             x0, x0, x2                      \n\t" // Forward A's address to the next column.
GEMM_2VX8_MKER_LOOP_PLAIN_C_2(z0,z2,z4,z6,z8,z10,z12,z14,z1,z3,z5,z7,z9,z11,z13,z15,p0,z30,z31,z16,z17,z18,z19,z20,z21,z22,z23,z24,z25,x1,x3)
GEMM_ACOL_CONTIGUOUS_LOAD(z30,z31,p0,p0,x0)
" add             x0, x0, x2                      \n\t" // Forward A's address to the next column.
GEMM_2VX8_MKER_LOOP_PLAIN_C_3(z0,z2,z4,z6,z8,z10,z12,z14,z1,z3,z5,z7,z9,z11,z13,z15,p0,z26,z27,z16,z17,z18,z19,z20,z21,z22,z23,z24,z25,x1,x3)
" subs            x4, x4, #1                      \n\t" // Decrease counter before final replica.
" b.eq            FIN_MKER_LOOP                   \n\t" // Branch early to avoid reading excess mem.
"                                                 \n\t"
GEMM_ACOL_CONTIGUOUS_LOAD(z26,z27,p0,p0,x0)
" add             x0, x0, x2                      \n\t" // Forward A's address to the next column.
GEMM_2VX8_MKER_LOOP_PLAIN_C_4(z0,z2,z4,z6,z8,z10,z12,z14,z1,z3,z5,z7,z9,z11,z13,z15,p0,z28,z29,z16,z17,z18,z19,z20,z21,z22,z23,z24,z25,x1,x3)
GEMM_ACOL_CONTIGUOUS_LOAD(z28,z29,p0,p0,x0)
" add             x0, x0, x2                      \n\t" // Forward A's address to the next column.
GEMM_2VX8_MKER_LOOP_PLAIN_C_5(z0,z2,z4,z6,z8,z10,z12,z14,z1,z3,z5,z7,z9,z11,z13,z15,p0,z30,z31,z16,z17,z18,z19,z20,z21,z22,z23,z24,z25,x1,x3)
GEMM_ACOL_CONTIGUOUS_LOAD(z30,z31,p0,p0,x0)
" add             x0, x0, x2                      \n\t" // Forward A's address to the next column.
" b               K_MKER_LOOP                     \n\t"
"                                                 \n\t"
" FIN_MKER_LOOP:                                  \n\t"
GEMM_2VX8_MKER_LOOP_PLAIN_C_4_RESIDUAL(z0,z2,z4,z6,z8,z10,z12,z14,z1,z3,z5,z7,z9,z11,z13,z15,p0,z28,z29,z16,z17,z18,z19,z20,z21,z22,z23,z24,z25,x1,x3)
GEMM_2VX8_MKER_LOOP_PLAIN_C_5_RESIDUAL(z0,z2,z4,z6,z8,z10,z12,z14,z1,z3,z5,z7,z9,z11,z13,z15,p0,z30,z31,z16,z17,z18,z19,z20,z21,z22,z23,z24,z25,x1,x3)
"                                                 \n\t"
" K_LEFT_LOOP:                                    \n\t"
" cmp             x8, #0                          \n\t" // End of execution.
" b.eq            WRITE_MEM_PREP                  \n\t"
"                                                 \n\t"
GEMM_ACOL_CONTIGUOUS_LOAD(z30,z31,p0,p0,x0)
" ld1rd           z16.d, p0/z, [x1, 8*0]          \n\t" // Load B row.
" ld1rd           z17.d, p0/z, [x1, 8*1]          \n\t"
" ld1rd           z18.d, p0/z, [x1, 8*2]          \n\t"
" ld1rd           z19.d, p0/z, [x1, 8*3]          \n\t"
" ld1rd           z20.d, p0/z, [x1, 8*4]          \n\t"
" ld1rd           z21.d, p0/z, [x1, 8*5]          \n\t"
" ld1rd           z22.d, p0/z, [x1, 8*6]          \n\t"
" ld1rd           z23.d, p0/z, [x1, 8*7]          \n\t"
GEMM_FMLA2(z0,z1,p0,z30,z31,z16)
GEMM_FMLA2(z2,z3,p0,z30,z31,z17)
GEMM_FMLA2(z4,z5,p0,z30,z31,z18)
GEMM_FMLA2(z6,z7,p0,z30,z31,z19)
GEMM_FMLA2(z8,z9,p0,z30,z31,z20)
GEMM_FMLA2(z10,z11,p0,z30,z31,z21)
GEMM_FMLA2(z12,z13,p0,z30,z31,z22)
GEMM_FMLA2(z14,z15,p0,z30,z31,z23)
" add             x0, x0, x2                      \n\t" // Forward A.
" add             x1, x1, x3                      \n\t" // Forward B.
" sub             x8, x8, #1                      \n\t"
" b               K_LEFT_LOOP                     \n\t" // Next column / row.
"                                                 \n\t"
" WRITE_MEM_PREP:                                 \n\t"
"                                                 \n\t"
" ldr             x4, %[alpha]                    \n\t" // Load alpha & beta (address).
" ldr             x8, %[beta]                     \n\t"
" ld1rd           z30.d, p0/z, [x4]               \n\t" // Load alpha & beta (value).
" ld1rd           z31.d, p0/z, [x8]               \n\t"
" fmov            d28, #1.0                       \n\t" // Prepare FP 1.0.
"                                                 \n\t"
" PREFETCH_ABNEXT:                                \n\t"
" ldr             x0, %[a_next]                   \n\t"
" ldr             x1, %[b_next]                   \n\t"
#ifdef _A64FX
" mov             x8, 0x2                         \n\t" // Tag B address.
" lsl             x8, x8, #56                     \n\t"
" orr             x1, x1, x8                      \n\t"
" mov             x8, 0x1                         \n\t" // Tag A address.
" lsl             x8, x8, #56                     \n\t"
" orr             x0, x0, x8                      \n\t"
#endif
" prfm            PLDL1STRM, [x0]                 \n\t"
" prfm            PLDL1STRM, [x0, 256*1]          \n\t"
" prfm            PLDL1STRM, [x1]                 \n\t"
" prfm            PLDL1STRM, [x1, 256*1]          \n\t"
"                                                 \n\t"
"                                                 \n\t"
" WRITE_MEM:                                      \n\t"
"                                                 \n\t"
" fcmp            d30, d28                        \n\t"
" b.eq            UNIT_ALPHA                      \n\t"
"                                                 \n\t"
SCALE_COL16(z0,z1,z2,z3,z4,z5,z6,z7,z8,z9,z10,z11,z12,z13,z14,z15,z30)
"                                                 \n\t"
" UNIT_ALPHA:                                     \n\t"
" mov             x9, x5                          \n\t" // C address for loading.
"                                                 \n\t" // C address for storing is x5 itself.
" cmp             x6, #1                          \n\t"
" b.ne            WRITE_MEM_G                     \n\t"
"                                                 \n\t"
" WRITE_MEM_C:                                    \n\t"
"                                                 \n\t"
GEMM_C_LOAD_UKER_7COLS_C(z16,z18,z20,z22,z24,z26,z28,z17,z19,z21,z23,z25,z27,z29,p0,p0,x9,x7)
GEMM_CCOL_FMAD(z16,z17,p0,p0,z0,z1,z31)
GEMM_CCOL_CONTIGUOUS_LOAD_FWD(z0,z1,p0,p0,x9,x7)
GEMM_C_FMAD_UKER_7COLS(z18,z20,z22,z24,z26,z28,z0,z19,z21,z23,z25,z27,z29,z1,p0,p0,z2,z4,z6,z8,z10,z12,z14,z3,z5,z7,z9,z11,z13,z15,z31)
GEMM_C_STORE_UKER_8COLS_C(z16,z18,z20,z22,z24,z26,z28,z0,z17,z19,z21,z23,z25,z27,z29,z1,p0,p0,x5,x7)
" b               END_WRITE_MEM                   \n\t"
"                                                 \n\t"
" WRITE_MEM_G:                                    \n\t"
"                                                 \n\t"
" mov             x8, xzr                         \n\t"
" incb            x8                              \n\t"
" madd            x8, x8, x6, xzr                 \n\t" // C-column's logical 1-vector skip.
" index           z30.d, xzr, x6                  \n\t" // Skips passed to index is not multiplied by 8.
GEMM_C_LOAD_UKER_7COLS_G(z16,z18,z20,z22,z24,z26,z28,z17,z19,z21,z23,z25,z27,z29,z30,p0,p0,x9,x7,x8,x16)
GEMM_CCOL_FMAD(z16,z17,p0,p0,z0,z1,z31)
GEMM_CCOL_GATHER_LOAD_FWD(z0,z1,z30,p0,p0,x9,x7,x8,x16)
GEMM_C_FMAD_UKER_7COLS(z18,z20,z22,z24,z26,z28,z0,z19,z21,z23,z25,z27,z29,z1,p0,p0,z2,z4,z6,z8,z10,z12,z14,z3,z5,z7,z9,z11,z13,z15,z31)
GEMM_C_STORE_UKER_8COLS_G(z16,z18,z20,z22,z24,z26,z28,z0,z17,z19,z21,z23,z25,z27,z29,z1,z30,p0,p0,x5,x7,x8,x16)
"                                                 \n\t"
" END_WRITE_MEM:                                  \n\t"
" b               END_EXEC                        \n\t"
"                                                 \n\t"
" END_ERROR:                                      \n\t"
" mov             x0, #1                          \n\t" // Return error.
" END_EXEC:                                       \n\t"
" mov             x0, #0                          \n\t" // Return normal.
:
: [a]      "m" (a),
  [b]      "m" (b),
  [c]      "m" (c),
  [rs_c]   "m" (rs_c),
  [cs_c]   "m" (cs_c),
  [k_mker] "m" (k_mker),
  [k_left] "m" (k_left),
  [alpha]  "m" (alpha),
  [beta]   "m" (beta),
  [a_next] "m" (a_next),
  [b_next] "m" (b_next)
: "x0","x1","x2","x3","x4","x5","x6","x7","x8",
  "x9","x16",
  "z0","z1","z2","z3","z4","z5","z6","z7",
  "z8","z9","z10","z11","z12","z13","z14","z15",
  "z16","z17","z18","z19",
  "z20","z21","z22","z23",
  "z24","z25","z26","z27",
  "z28","z29","z30","z31"
   );
}

