/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2019, Forschunszentrum Juelich

   Author(s): Stepan Nassyr, s.nassyr@fz-juelich.de

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
#include "bli_dgemm_sve_asm_macros.h"
#include "bli_sve_asm_debug.h"


#define BLOCK_4VX5_Z12_Z31(cur_bvec0,cur_bvec1,cur_bvec2,cur_bvec3,cur_bvec4, avec0,avec1,avec2,avec3, next_bvec0,next_bvec1,next_bvec2,next_bvec3,next_bvec4, aoff0,aoff1,aoff2,aoff3)\
MLA4COL_D(z12,z16,z20,z24,z28, cur_avec0, cur_bvec0,cur_bvec1,cur_bvec2,cur_bvec3,cur_bvec4, p0)\
LOAD1VEC_VOFF_D(next_avec0, p0, x0, aoff0)\
MLA4COL_D(z13,z17,z21,z25,z29, cur_avec1, cur_bvec0,cur_bvec1,cur_bvec2,cur_bvec3,cur_bvec4, p0)\
LOAD1VEC_VOFF_D(next_avec1, p0, x0, aoff1)\
MLA4COL_D(z14,z18,z22,z26,z30, cur_avec2, cur_bvec0,cur_bvec1,cur_bvec2,cur_bvec3,cur_bvec4, p0)\
LOAD1VEC_VOFF_D(next_avec2, p0, x0, aoff2)\
/*MLA4COL_D(z15,z19,z23,z27,z31, cur_avec3, cur_bvec0,cur_bvec1,cur_bvec2,cur_bvec3,cur_bvec4, p0)\*/\
MLA1COL_D(z15, cur_avec3, cur_bvec0, p0)\
LOAD1VEC_DIST_OFF_D(next_bvec0, p0, x0, 0)\
MLA1COL_D(z19, cur_avec3, cur_bvec1, p0)\
LOAD1VEC_DIST_OFF_D(next_bvec1, p0, x0, 8)\
MLA1COL_D(z23, cur_avec3, cur_bvec2, p0)\
LOAD1VEC_DIST_OFF_D(next_bvec2, p0, x0, 16)\
MLA1COL_D(z27, cur_avec3, cur_bvec3, p0)\
LOAD1VEC_DIST_OFF_D(next_bvec3, p0, x0, 24)\
MLA1COL_D(z31, cur_avec3, cur_bvec4, p0)\
LOAD1VEC_DIST_OFF_D(next_bvec4, p0, x0, 32)\
LOAD1VEC_VOFF_D(next_avec3, p0, x0, aoff3)\
" add x1, x1, #40 \n\t"

#define ENDBLOCK_4VX5_Z12_Z31(bvec0,bvec1,bvec2,bvec3,bvec4, cur_avec0,cur_avec1,cur_avec2,cur_avec3)\
MLA4COL_D(z12,z16,z20,z24,z28, cur_avec0, bvec0,bvec1,bvec2,bvec3,bvec4, p0)\
MLA4COL_D(z13,z17,z21,z25,z29, cur_avec1, bvec0,bvec1,bvec2,bvec3,bvec4, p0)\
MLA4COL_D(z14,z18,z22,z26,z30, cur_avec2, bvec0,bvec1,bvec2,bvec3,bvec4, p0)\
MLA4COL_D(z15,z19,z23,z27,z31, cur_avec3, bvec0,bvec1,bvec2,bvec3,bvec4, p0)


/* 4 vectors in m_r, n_r = 5
*/
void bli_dgemm_armv8a_sve_asm_4vx5_ld1rd_colwise
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

    // 8 k iterations in unrolled loop
	uint64_t k_iter = k0 / 8;
    // rest is handled separately
    uint64_t k_left = k0 % 8;

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.

	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

__asm__ volatile
(
"                                            \n\t" 
" ldr x0,%[aaddr]                            \n\t" // Load address of A 
" ldr x1,%[baddr]                            \n\t" // Load address of B
" ldr x2,%[caddr]                            \n\t" // Load address of C
"                                            \n\t"
" ldr x3,%[a_next]                           \n\t" // Move pointer
" ldr x4,%[b_next]                           \n\t" // Move pointer
"                                            \n\t"
" ldr x5,%[k_iter]                           \n\t" // number of 8xk iterations
" ldr x6,%[k_left]                           \n\t" // number of k iterations afterward
"                                            \n\t" 
" ldr x7,%[alpha]                            \n\t" // Alpha address      
" ldr x8,%[beta]                             \n\t" // Beta address      
"                                            \n\t" 
" ldr x9,%[cs_c]                             \n\t" // Load cs_c
" lsl x10,x9,#3                              \n\t" // cs_c * sizeof(double)
"                                            \n\t"
" ldr x13,%[rs_c]                            \n\t" // Load rs_c.
//" lsl x14,x13,#3                             \n\t" // rs_c * sizeof(double). 
"                                            \n\t"
" mov x11,#0                                 \n\t"
" incd x11                                   \n\t"
"                                            \n\t"
" add x20,x2,x10                             \n\t" //Load address Column 1 of C
" add x21,x20,x10                            \n\t" //Load address Column 2 of C
" add x22,x21,x10                            \n\t" //Load address Column 3 of C
" add x23,x22,x10                            \n\t" //Load address Column 4 of C
"                                            \n\t"
#if defined(PREFETCH64) || defined(PREFETCH256)
" prfm pstl1keep,[x2]                        \n\t" // Prefetch c column 0.
" prfm pstl1keep,[x20]                       \n\t" // Prefetch c column 1.
" prfm pstl1keep,[x21]                       \n\t" // Prefetch c column 2.
" prfm pstl1keep,[x22]                       \n\t" // Prefetch c column 3.
" prfm pstl1keep,[x23]                       \n\t" // Prefetch c column 4.

#endif
#if defined(PREFETCHSVE1) || defined(PREFETCHSVE2)
" prfd pldl1keep,p0, [x0, #0, MUL VL]        \n\t" // 1/8 | 4/8
#endif
#if defined(PREFETCHSVE1)
" prfd pldl1keep,p0, [x0, #1, MUL VL]        \n\t" // 2/8 | 4/8
" prfd pldl1keep,p0, [x0, #2, MUL VL]        \n\t" // 3/8 | 4/8
" prfd pldl1keep,p0, [x0, #3, MUL VL]        \n\t" // 4/8 | 4/8
#endif
#if defined(PREFETCHSVE1) || defined(PREFETCHSVE2)
" prfd pldl1keep,p0, [x0, #4, MUL VL]        \n\t" // 5/8 | 8/8
#endif
#if defined(PREFETCHSVE1)
" prfd pldl1keep,p0, [x0, #5, MUL VL]        \n\t" // 6/8 | 8/8
" prfd pldl1keep,p0, [x0, #6, MUL VL]        \n\t" // 7/8 | 8/8
" prfd pldl1keep,p0, [x0, #7, MUL VL]        \n\t" // 8/8 | 8/8
#endif
"                                            \n\t"
" ptrue p0.d                                 \n\t" // Creating all true predicate
"                                            \n\t"
/*************************************************
 *
 * Store A vectors in z0,z1,z2,z3
 *
 * load B values in z4-z11, rotating starting with (4,5,6,7,8)
 * LD1 has 11 cycles latency
 *
 *************************************************/
LOAD4VEC_D(z0,z1,z2,z3,p0,x0)
"                                            \n\t"
"                                            \n\t"
LOAD5VEC_DIST_D(z4,z5,z6,z7,z8, p0,x1)
"                                            \n\t"
ZERO4VEC_D(z12,z13,z14,z15)                          // c columns 0
ZERO4VEC_D(z16,z17,z18,z19)                          // c columns 1
ZERO4VEC_D(z20,z21,z22,z23)                          // c columns 2
ZERO4VEC_D(z24,z25,z26,z27)                          // c columns 3
ZERO4VEC_D(z28,z29,z30,z31)                          // c columns 4

"                                            \n\t"
"                                            \n\t"
" cmp x5,#0                                  \n\t" // If k_iter == 0, jump to k_left.
" beq .D4VX5CONSIDERKLEFT                   \n\t"
"                                            \n\t"
" incb x0, ALL, MUL #2                       \n\t" // A = A+vecsize*2
" add x1, x1, #64                            \n\t" // B = B+8*sizeof(double)
"                                            \n\t"
" cmp x5,1                                   \n\t" // If there is just one k_iter, jump to that one. 
" beq .D4VX5LASTITER                        \n\t" // (as loop is do-while-like).
"                                            \n\t" 
"                                            \n\t"
" .D4VX5LOOP:                               \n\t" // Body
BLOCK_4VX5_Z12_Z31(z4,z5,z6,z7,z8,   z0,z1,z2,z3, z9,z10,z11,z4,z5, 0,1,2,3)
"                                            \n\t"
BLOCK_4VX5_Z12_Z31(z9,z10,z11,z4,z5, z0,z1,z2,z3, z6,z7,z8,z9,z10,  4,5,6,7)
" incb x0, ALL, MUL #8                       \n\t" // Next 8 A vectors
BLOCK_4VX5_Z12_Z31(z6,z7,z8,z9,z10,  z0,z1,z2,z3, z11,z4,z5,z6,z7,  0,1,2,3)
"                                            \n\t"
BLOCK_4VX5_Z12_Z31(z11,z4,z5,z6,z7,  z0,z1,z2,z3, z8,z9,z10,z11,z4, 4,5,6,7)
" incb x0, ALL, MUL #8                       \n\t" // Next 8 A vectors
BLOCK_4VX5_Z12_Z31(z8,z9,z10,z11,z4, z0,z1,z2,z3, z5,z6,z7,z8,z9,   0,1,2,3)
"                                            \n\t"
BLOCK_4VX5_Z12_Z31(z5,z6,z7,z8,z9,   z0,z1,z2,z3, z10,z11,z4,z5,z6, 4,5,6,7)
" incb x0, ALL, MUL #8                       \n\t" // Next 8 A vectors
BLOCK_4VX5_Z12_Z31(z10,z11,z4,z5,z6, z0,z1,z2,z3, z7,z8,z9,z10,z11, 0,1,2,3)
"                                            \n\t"
BLOCK_4VX5_Z12_Z31(z7,z8,z9,z10,z11, z0,z1,z2,z3, z4,z5,z6,z7,z8,   4,5,6,7)
" incb x0, ALL, MUL #8                       \n\t" // Next 8 A vectors
"                                            \n\t"
" sub x5,x5,1                                \n\t" // i-=1
" cmp x5,1                                   \n\t" // Iterate again if we are not in k_iter == 1.
" bne .D4VX5LOOP                             \n\t"
" .D4VX5LASTITER:                            \n\t" // Body
BLOCK_4VX5_Z12_Z31(z4,z5,z6,z7,z8,   z0,z1,z2,z3, z9,z10,z11,z4,z5, 0,1,2,3)
"                                            \n\t"
BLOCK_4VX5_Z12_Z31(z9,z10,z11,z4,z5, z0,z1,z2,z3, z6,z7,z8,z9,z10,  4,5,6,7)
" incb x0, ALL, MUL #8                       \n\t" // Next 8 A vectors
BLOCK_4VX5_Z12_Z31(z6,z7,z8,z9,z10,  z0,z1,z2,z3, z11,z4,z5,z6,z7,  0,1,2,3)
"                                            \n\t"
BLOCK_4VX5_Z12_Z31(z11,z4,z5,z6,z7,  z0,z1,z2,z3, z8,z9,z10,z11,z4, 4,5,6,7)
" incb x0, ALL, MUL #8                       \n\t" // Next 8 A vectors
BLOCK_4VX5_Z12_Z31(z8,z9,z10,z11,z4, z0,z1,z2,z3, z5,z6,z7,z8,z9,   0,1,2,3)
"                                            \n\t"
BLOCK_4VX5_Z12_Z31(z5,z6,z7,z8,z9,   z0,z1,z2,z3, z10,z11,z4,z5,z6, 4,5,6,7)
" incb x0, ALL, MUL #8                       \n\t" // Next 8 A vectors
BLOCK_4VX5_Z12_Z31(z10,z11,z4,z5,z6, z0,z1,z2,z3, z7,z8,z9,z10,z11, 0,1,2,3)
"                                            \n\t"
ENDBLOCK_4VX5_Z12_Z31(z7,z8,z9,z10,z11, z0,z1,z2,z3)
"                                            \n\t"
" incb x0, ALL, MUL #4                       \n\t" // 4 more Vectors loaded
"                                            \n\t"
" .D4VX5CONSIDERKLEFT:                       \n\t" 
" cmp x6,0                                   \n\t" // If k_left == 0, we are done.
" beq .D4VX5POSTACCUM                        \n\t" // else, we enter the k_left loop.
"                                            \n\t"
".D4VX5LOOPKLEFT:                            \n\t"
"                                            \n\t"
LOAD4VEC_D(z0,z1,z2,z3,p0,x0)
" incb x0, ALL, MUL #2                       \n\t" // Advance a pointer by 2 vectors
"                                            \n\t"
LOAD5VEC_DIST_D(z4,z5,z6,z7,z8,  p0, x1)
" add x1, x1, #40                            \n\t" // advance b pointer by 10 doubles
"                                            \n\t"
" sub x6,x6,1                                \n\t"
"                                            \n\t"
ENDBLOCK_4VX5_Z12_Z31(z4,z5,z6,z7,z8, z0,z1,z2,z3)
"                                            \n\t"
" cmp x6,0                                   \n\t" // Iterate again.
" bne .D4VX5LOOPKLEFT                       \n\t" // if i!=0.
"                                            \n\t"
" .D4VX5POSTACCUM:                          \n\t"

#if defined(PREFETCH64) || defined(PREFETCH256)
" prfm PLDL2KEEP, [x3]                       \n\t"
" prfm PLDL2KEEP, [x4]                       \n\t"
#endif
"                                            \n\t"
" ld1rd  z0.d, p0/z, [x7]                    \n\t" // Load alpha
" ld1rd  z1.d, p0/z, [x8]                   \n\t" // Load beta

"                                            \n\t"
" cmp x13,#1                                 \n\t" // If rs_c != 1 (column-major)
" bne .D4VX5GENSTORED                       \n\t"
"                                            \n\t"
" .D4VX5COLSTORED:                          \n\t" // C is column-major.
"                                            \n\t"

// Accumulated results are stored in z12-z31
// alpha in z0, beta in z1 - put alpha is in z0.d[0], beta in z1.d[1]
// z2-z11 are free
"                                        \n\t"

" fcmp d1,#0.0                           \n\t"
" beq .D4VX5BETAZEROCONTCOLSTOREDS      \n\t" // multiply with beta if beta isn't zero
"                                        \n\t"
LOAD4VEC_D(z2,z3,z4,z5, p0,x2)                   // Load Column 0
LOAD4VEC_D(z6,z7,z8,z9, p0,x20)                  // Load Column 1
MUL4ROW_D(z12,z13,z14,z15, z12,z13,z14,z15, z0, p0)
MLA4ROW_D(z12,z13,z14,z15, z2,z3,z4,z5,     z1, p0)

LOAD4VEC_D(z2,z3,z4,z5, p0,x21)                  // Load Column 2
MUL4ROW_D(z16,z17,z18,z19, z16,z17,z18,z19, z0, p0)
MLA4ROW_D(z16,z17,z18,z19, z6,z7,z8,z9,     z1, p0)

LOAD4VEC_D(z6,z7,z8,z9, p0,x22)                  // Load Column 3
MUL4ROW_D(z20,z21,z22,z23, z20,z21,z22,z23, z0, p0)
MLA4ROW_D(z20,z21,z22,z23, z2,z3,z4,z5,     z1, p0)

LOAD4VEC_D(z2,z3,z4,z5, p0,x23)                  // Load Column 4
MUL4ROW_D(z24,z25,z26,z27, z24,z25,z26,z27, z0, p0)
MLA4ROW_D(z24,z25,z26,z27, z6,z7,z8,z9,     z1, p0)

MUL4ROW_D(z28,z29,z30,z31, z28,z29,z30,z31, z0, p0)
MLA4ROW_D(z28,z29,z30,z31, z2,z3,z4,z5,     z1, p0)

STOR4VEC_D(z12,z13,z14,z15, p0,x2)               // Store Column 0
STOR4VEC_D(z16,z17,z18,z19, p0,x20)              // Store Column 1
STOR4VEC_D(z20,z21,z22,z23, p0,x21)              // Store Column 2
STOR4VEC_D(z24,z25,z26,z27, p0,x22)              // Store Column 3
STOR4VEC_D(z28,z29,z30,z31, p0,x23)              // Store Column 4

" b .D4VX5END                              \n\t"       // Duplicate code for stores required due to lack of registers
"                                           \n\t"
" .D4VX5BETAZEROCONTCOLSTOREDS:            \n\t"
// No need to zero anything as we are storing the scaled accumulated A*B values
MUL4ROW_D(z12,z13,z14,z15, z12,z13,z14,z15, z0, p0)
MUL4ROW_D(z16,z17,z18,z19, z16,z17,z18,z19, z0, p0)
MUL4ROW_D(z20,z21,z22,z23, z20,z21,z22,z23, z0, p0)
MUL4ROW_D(z24,z25,z26,z27, z24,z25,z26,z27, z0, p0)
MUL4ROW_D(z28,z29,z30,z31, z28,z29,z30,z31, z0, p0)

STOR4VEC_D(z12,z13,z14,z15, p0,x2)               // Store Column 0
STOR4VEC_D(z16,z17,z18,z19, p0,x20)              // Store Column 1
STOR4VEC_D(z20,z21,z22,z23, p0,x21)              // Store Column 2
STOR4VEC_D(z24,z25,z26,z27, p0,x22)              // Store Column 3
STOR4VEC_D(z28,z29,z30,z31, p0,x23)              // Store Column 4

"                                           \n\t"
" b .D4VX5END                              \n\t"
"                                           \n\t"
" .D4VX5GENSTORED:                         \n\t" // C is general-stride stored.

"                                           \n\t" // Creating index for stride load&store access
" index z2.d, xzr, x13                       \n\t" // 0, stride*double, 2*stride*double, ...
" mul x3, x13, x11                           \n\t" // x3 <- stride*double*vecsize
" index z3.d, x3, x13                        \n\t" // stride*double*vecsize, (vecsize+1)*stride*double, (vecsize+2)*stride*double, ...
" lsl x4, x3, #1                             \n\t" // x4 <- 2*stride*double*vecsize
" index z4.d, x4, x13                        \n\t" // stride*double*(2*vecsize), (2*vecsize+1)*stride*double, (2*vecsize+2)*stride*double, ...
" add x5, x4, x3                             \n\t" // x5 <- 3*stride*double*vecsize
" index z5.d, x5, x13                       \n\t" // stride*double*(3*vecsize), (3*vecsize+1)*stride*double, (3*vecsize+2)*stride*double, ...
"                                           \n\t"
" fcmp d1,#0.0                              \n\t"
" beq .D4VX5BETAZEROGENSTOREDS             \n\t" // multiply with beta if beta isn't zero
"                                        \n\t"
LOAD4VEC_GENI_D(z6,z7,z8,z9, p0,x2, z2,z3,z4,z5)
MUL4ROW_D(z12,z13,z14,z15,  z12,z13,z14,z15,  z32, p0)
MLA4ROW_D(z12,z13,z14,z15,  z6,z7,z8,z9,     z33, p0)
LOAD4VEC_GENI_D(z6,z7,z8,z9, p0,x20, z2,z3,z4,z5)
MUL4ROW_D(z16,z17,z18,z19, z16,z17,z18,z19, z32, p0)
MLA4ROW_D(z16,z17,z18,z19, z6,z7,z8,z9,     z33, p0)
LOAD4VEC_GENI_D(z6,z7,z8,z9, p0,x21, z2,z3,z4,z5)
MUL4ROW_D(z20,z21,z22,z23, z20,z21,z22,z23, z32, p0)
MLA4ROW_D(z20,z21,z22,z23, z6,z7,z8,z9,     z33, p0)
LOAD4VEC_GENI_D(z6,z7,z8,z9, p0,x22, z2,z3,z4,z5)
MUL4ROW_D(z24,z25,z26,z27, z24,z25,z26,z27, z32, p0)
MLA4ROW_D(z24,z25,z26,z27, z6,z7,z8,z9,     z33, p0)
LOAD4VEC_GENI_D(z6,z7,z8,z9, p0,x23, z2,z3,z4,z5)
MUL4ROW_D(z28,z29,z30,z31, z28,z29,z30,z31, z32, p0)
MLA4ROW_D(z28,z29,z30,z31, z6,z7,z8,z9,     z33, p0)
"                                            \n\t"
STOR4VEC_GENI_D(z12,z13,z14,z15,  p0,x2, z2,z3,z4,z5)
STOR4VEC_GENI_D(z16,z17,z18,z19, p0,x20, z2,z3,z4,z5)
STOR4VEC_GENI_D(z20,z21,z22,z23, p0,x21, z2,z3,z4,z5)
STOR4VEC_GENI_D(z24,z25,z26,z27, p0,x22, z2,z3,z4,z5)
STOR4VEC_GENI_D(z28,z29,z30,z31, p0,x23, z2,z3,z4,z5)

" b .D4VX5END                                \n\t" // Duplicate code for stores required due to lack of registers
"                                            \n\t"
" .D4VX5BETAZEROGENSTOREDS:                  \n\t"
// No need to zero anything as we are storing the scaled accumulated A*B values
MUL4ROW_D(z12,z13,z14,z15, z12,z13,z14,z15, z32, p0)
MUL4ROW_D(z16,z17,z18,z19, z16,z17,z18,z19, z32, p0)
MUL4ROW_D(z20,z21,z22,z23, z20,z21,z22,z23, z32, p0)
MUL4ROW_D(z24,z25,z26,z27, z24,z25,z26,z27, z32, p0)
MUL4ROW_D(z28,z29,z30,z31, z28,z29,z30,z31, z32, p0)

STOR4VEC_GENI_D(z12,z13,z14,z15,  p0,x2, z2,z3,z4,z5)
STOR4VEC_GENI_D(z16,z17,z18,z19, p0,x20, z2,z3,z4,z5)
STOR4VEC_GENI_D(z20,z21,z22,z23, p0,x21, z2,z3,z4,z5)
STOR4VEC_GENI_D(z24,z25,z26,z27, p0,x22, z2,z3,z4,z5)
STOR4VEC_GENI_D(z28,z29,z30,z31, p0,x23, z2,z3,z4,z5)
"                                           \n\t"
"                                           \n\t"
"                                            \n\t"
" .D4VX5END:                                \n\t"     // Done!
"                                            \n\t"
:// output operands (none)
:// input operands
 [aaddr]  "m" (a),      // 0
 [baddr]  "m" (b),      // 1
 [caddr]  "m" (c),      // 2
 [k_iter] "m" (k_iter), // 3
 [k_left] "m" (k_left), // 4
 [alpha]  "m" (alpha),  // 5
 [beta]   "m" (beta),   // 6
 [rs_c]   "m" (rs_c),   // 6
 [cs_c]   "m" (cs_c),   // 7
 [a_next] "m" (a_next), // 8
 [b_next] "m" (b_next)  // 9
:// Register clobber list
 "x0","x1","x2","x3",
 "x4","x5","x6",
 "x7","x8","x9",
 "x10","x11","x13",
 "x20","x21","x22","x23","x24","x25","x26",
 "z0","z1","z2",
 "z3","z4","z5",
 "z6","z7","z8",
 "z9","z10","z11",
 "z12","z13","z14",
 "z15","z16","z17","z18","z19",
 "z20","z21","z22","z23",
 "z24","z25","z26","z27",
 "z28","z29","z30","z31", 
 "p0"
);

}
