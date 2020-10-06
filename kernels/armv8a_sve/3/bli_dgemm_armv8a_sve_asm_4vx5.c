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
#include "bli_sve_asm_mla_d.h"

/* 4 vectors in m_r, n_r = 5
*/
void bli_dgemm_armv8a_sve_asm_4vx5
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

    // 4 k iterations in unrolled loop
	uint64_t k_iter = k0 / 4;
    // rest is handled separately
    uint64_t k_left = k0 % 4;

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
" ldr x5,%[k_iter]                           \n\t" // number of 4xk iterations
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
" add x24,x23,x10                            \n\t" //Load address Column 5 of C
"                                            \n\t"
#if defined(PREFETCH64) || defined(PREFETCH256)
" prfm pstl1keep,[x2]                        \n\t" // Prefetch c.
" prfm pstl1keep,[x20]                       \n\t" // Prefetch c.
" prfm pstl1keep,[x21]                       \n\t" // Prefetch c.
" prfm pstl1keep,[x22]                       \n\t" // Prefetch c.
" prfm pstl1keep,[x23]                       \n\t" // Prefetch c.
" prfm pstl1keep,[x24]                         \n\t" // Prefetch c.
" prfm pstl1keep,[x25]                       \n\t" // Prefetch c.
" prfm pstl1keep,[x26]                       \n\t" // Prefetch c.
#endif
"                                            \n\t"
" ptrue p0.d                                 \n\t" // Creating all true predicate
"                                            \n\t"
LOAD4VEC_D(z0,z1,z2,z3, p0,x0)
"                                            \n\t"
LOAD5VEC_DIST_D(z4,z5,z6,z7,z8,p0,x1)
"                                            \n\t"
"                                            \n\t"
ZERO4VEC_D(z9,z10,z11,z12)                          // c column 0
#if defined(PREFETCH64)
" prfm PLDL1KEEP, [x1, #64]                  \n\t" // 128/160 (0/160) | 160/160 (96/160) (from load)
#endif
ZERO4VEC_D(z13,z14,z15,z16)                          // c column 1
#if defined(PREFETCH64)
" prfm PLDL1KEEP, [x1, #128]                  \n\t" // 160/160 (
#endif
#if defined(PREFETCHSVE1) || defined(PREFETCHSVE2)
" prfd pldl1keep,p0, [x0, #4, MUL VL]        \n\t" // 1/32 (0/32) | 4/32 (0/32)
#endif
ZERO4VEC_D(z17,z18,z19,z20)                          // c column 2
#if defined(PREFETCHSVE1)
" prfd pldl1keep,p0, [x0, #5, MUL VL]        \n\t" // 2/32 (0/32)
#endif
ZERO4VEC_D(z21,z22,z23,z24)                          // c column 3
#if defined(PREFETCHSVE1)
" prfd pldl1keep,p0, [x0, #6, MUL VL]        \n\t" // 3/32 (0/32)
#endif
ZERO4VEC_D(z25,z26,z27,z28)                          // c column 4
#if defined(PREFETCHSVE1)
" prfd pldl1keep,p0, [x0, #7, MUL VL]        \n\t" // 4/32 (0/32)
#endif
"                                            \n\t"
"                                            \n\t"
" cmp x5,#0                                  \n\t" // If k_iter == 0, jump to k_left.
" beq .D4VX5CONSIDERKLEFT                    \n\t"
"                                            \n\t"
" incb x0, ALL, MUL #4                       \n\t" // A = A+vecsize*4
" add x1, x1, #40                            \n\t" // B = B+5*sizeof(double)
"                                            \n\t"
" cmp x5,1                                   \n\t" // If there is just one k_iter, jump to that one. 
" beq .D4VX5LASTITER                         \n\t" // (as loop is do-while-like).
"                                            \n\t" 
"                                            \n\t"
" .D4VX5LOOP:                                \n\t" // Body
MLA4ROW_LA_LB_D(z9,z10,z11,z12, z0,z1,z2,z3, z4, p0, z29, x0,0,x1,0) 
#if defined(PREFETCHSVE1) || defined (PREFETCHSVE2)
" prfd pldl1keep,p0, [x0, #4, MUL VL]        \n\t" // 5/16 (0/16) | 8/16 (0/16)
#endif
MLA4ROW_LA_LB_D(z13,z14,z15,z16, z0,z1,z2,z3, z5, p0, z30, x0,1,x1,8) 
#if defined(PREFETCHSVE1)
" prfd pldl1keep,p0, [x0, #5, MUL VL]        \n\t" // 6/16 (0/16)
#endif
MLA4ROW_LA_LB_D(z17,z18,z19,z20, z0,z1,z2,z3, z6, p0, z31, x0,2,x1,16) 
#if defined(PREFETCHSVE1)
" prfd pldl1keep,p0, [x0, #6, MUL VL]        \n\t" // 7/16 (0/16)
#endif
MLA4ROW_LB_D(z21,z22,z23,z24, z0,z1,z2,z3, z7, p0, x1, 24)
#if defined(PREFETCHSVE1)
" prfd pldl1keep,p0, [x0, #7, MUL VL]        \n\t" // 8/16 (0/16)
#endif 
MLA4ROW_LA_LB_D(z25,z26,z27,z28, z0,z1,z2,z3, z8, p0, z3, x0,3,x1,32)
#if defined(PREFETCHSVE1)
" prfd pldl1keep,p0, [x0, #8, MUL VL]        \n\t" // 9/16 (0/16) | 12/16 (0/16)
#endif 

" incb x0, ALL, MUL #4                       \n\t" // Next 4 A vectors

MLA4ROW_LA_LB_D(z9,z10,z11,z12, z29,z30,z31,z3, z4, p0, z0, x0,0,x1,40) 
#if defined(PREFETCHSVE1)
" prfd pldl1keep,p0, [x0, #5, MUL VL]        \n\t" // 10/16 (0/16)
#endif
MLA4ROW_LA_LB_D(z13,z14,z15,z16, z29,z30,z31,z3, z5, p0, z1, x0,1,x1,48) 
#if defined(PREFETCHSVE1)
" prfd pldl1keep,p0, [x0, #6, MUL VL]        \n\t" // 11/16 (0/16)
#endif
MLA4ROW_LA_LB_D(z17,z18,z19,z20, z29,z30,z31,z3, z6, p0, z2, x0,2,x1,56) 
#if defined(PREFETCHSVE1)
" prfd pldl1keep,p0, [x0, #7, MUL VL]        \n\t" // 12/16 (0/16)
#endif
MLA4ROW_LB_D(z21,z22,z23,z24, z29,z30,z31,z3, z7, p0, x1, 64) 
#if defined(PREFETCHSVE1) || defined(PREFETCHSVE2)
" prfd pldl1keep,p0, [x0, #8, MUL VL]        \n\t" // 13/16 (0/16) | 16/16 (0/16)
#endif
MLA4ROW_LA_LB_D(z25,z26,z27,z28, z29,z30,z31,z3, z8, p0, z3, x0,3,x1,72) 
#if defined(PREFETCHSVE1)
" prfd pldl1keep,p0, [x0, #9, MUL VL]        \n\t" // 14/16 (0/16)
#endif

" incb x0, ALL, MUL #4                       \n\t" // Next 4 A vectors

MLA4ROW_LA_LB_D(z9,z10,z11,z12, z0,z1,z2,z3, z4, p0, z29, x0,0,x1,80) 
#if defined(PREFETCHSVE1)
" prfd pldl1keep,p0, [x0, #6, MUL VL]        \n\t" // 15/16 (0/16)
#endif
MLA4ROW_LA_LB_D(z13,z14,z15,z16, z0,z1,z2,z3, z5, p0, z30, x0,1,x1,88) 
#if defined(PREFETCHSVE1)
" prfd pldl1keep,p0, [x0, #7, MUL VL]        \n\t" // 16/16 (0/16)
#endif
MLA4ROW_LA_LB_D(z17,z18,z19,z20, z0,z1,z2,z3, z6, p0, z31, x0,2,x1,96) 
#if defined(PREFETCHSVE1) || defined(PREFETCHSVE2)
" prfd pldl1keep,p0, [x0, #8, MUL VL]        \n\t" // 16/16 (1/16) | 16/16 (4/16)
#endif
MLA4ROW_LB_D(z21,z22,z23,z24, z0,z1,z2,z3, z7, p0, x1, 104) 
#if defined(PREFETCHSVE1)
" prfd pldl1keep,p0, [x0, #9, MUL VL]        \n\t" // 16/16 (2/16) 
#endif
MLA4ROW_LA_LB_D(z25,z26,z27,z28, z0,z1,z2,z3, z8, p0, z3, x0,3,x1,112) 
#if defined(PREFETCHSVE1)
" prfd pldl1keep,p0, [x0, #10, MUL VL]        \n\t" // 16/16 (3/16) 
#endif

" incb x0, ALL, MUL #4                       \n\t" // Next 4 A vectors

MLA4ROW_LA_LB_D(z9,z10,z11,z12, z29,z30,z31,z3, z4, p0, z0, x0,0,x1,120) 
#if defined(PREFETCHSVE1)
" prfd pldl1keep,p0, [x0, #7, MUL VL]        \n\t" // 16/16 (4/16)
#endif
MLA4ROW_LA_LB_D(z13,z14,z15,z16, z29,z30,z31,z3, z5, p0, z1, x0,1,x1,128) 
MLA4ROW_LA_LB_D(z17,z18,z19,z20, z29,z30,z31,z3, z6, p0, z2, x0,2,x1,136)
MLA4ROW_LB_D(z21,z22,z23,z24, z29,z30,z31,z3, z7, p0, x1, 144)
MLA4ROW_LA_LB_D(z25,z26,z27,z28, z29,z30,z31,z3, z8, p0, z3, x0,3,x1,152)



" incb x0, ALL, MUL #4                       \n\t" // Advancing 16 vectors, 12 already loaded
" add x1, x1, #160                           \n\t" // B = B+4*5*sizeof(double)
"                                            \n\t"
" sub x5,x5,1                                \n\t" // i-=1
" cmp x5,1                                   \n\t" // Iterate again if we are not in k_iter == 1.
" bne .D4VX5LOOP                             \n\t"
" .D4VX5LASTITER:                            \n\t" // Body
MLA4ROW_LA_LB_D(z9,z10,z11,z12, z0,z1,z2,z3, z4, p0, z29, x0,0,x1,0) 
#if defined(PREFETCHSVE1) || defined (PREFETCHSVE2)
" prfd pldl1keep,p0, [x0, #4, MUL VL]        \n\t" // 5/16 (0/16) | 8/16 (0/16)
#endif
MLA4ROW_LA_LB_D(z13,z14,z15,z16, z0,z1,z2,z3, z5, p0, z30, x0,1,x1,8) 
#if defined(PREFETCHSVE1)
" prfd pldl1keep,p0, [x0, #5, MUL VL]        \n\t" // 6/16 (0/16)
#endif
MLA4ROW_LA_LB_D(z17,z18,z19,z20, z0,z1,z2,z3, z6, p0, z31, x0,2,x1,16) 
#if defined(PREFETCHSVE1)
" prfd pldl1keep,p0, [x0, #6, MUL VL]        \n\t" // 7/16 (0/16)
#endif
MLA4ROW_LB_D(z21,z22,z23,z24, z0,z1,z2,z3, z7, p0, x1, 24)
#if defined(PREFETCHSVE1)
" prfd pldl1keep,p0, [x0, #7, MUL VL]        \n\t" // 8/16 (0/16)
#endif 
MLA4ROW_LA_LB_D(z25,z26,z27,z28, z0,z1,z2,z3, z8, p0, z3, x0,3,x1,32)
#if defined(PREFETCHSVE1)
" prfd pldl1keep,p0, [x0, #8, MUL VL]        \n\t" // 9/16 (0/16) | 12/16 (0/16)
#endif 

" incb x0, ALL, MUL #4                       \n\t" // Next 4 A vectors

MLA4ROW_LA_LB_D(z9,z10,z11,z12, z29,z30,z31,z3, z4, p0, z0, x0,0,x1,40) 
#if defined(PREFETCHSVE1)
" prfd pldl1keep,p0, [x0, #5, MUL VL]        \n\t" // 10/16 (0/16)
#endif
MLA4ROW_LA_LB_D(z13,z14,z15,z16, z29,z30,z31,z3, z5, p0, z1, x0,1,x1,48) 
#if defined(PREFETCHSVE1)
" prfd pldl1keep,p0, [x0, #6, MUL VL]        \n\t" // 11/16 (0/16)
#endif
MLA4ROW_LA_LB_D(z17,z18,z19,z20, z29,z30,z31,z3, z6, p0, z2, x0,2,x1,56) 
#if defined(PREFETCHSVE1)
" prfd pldl1keep,p0, [x0, #7, MUL VL]        \n\t" // 12/16 (0/16)
#endif
MLA4ROW_LB_D(z21,z22,z23,z24, z29,z30,z31,z3, z7, p0, x1, 64) 
#if defined(PREFETCHSVE1) || defined(PREFETCHSVE2)
" prfd pldl1keep,p0, [x0, #8, MUL VL]        \n\t" // 13/16 (0/16) | 16/16 (0/16)
#endif
MLA4ROW_LA_LB_D(z25,z26,z27,z28, z29,z30,z31,z3, z8, p0, z3, x0,3,x1,72) 
#if defined(PREFETCHSVE1)
" prfd pldl1keep,p0, [x0, #9, MUL VL]        \n\t" // 14/16 (0/16)
#endif

" incb x0, ALL, MUL #4                       \n\t" // Next 4 A vectors

MLA4ROW_LA_LB_D(z9,z10,z11,z12, z0,z1,z2,z3, z4, p0, z29, x0,0,x1,80) 
#if defined(PREFETCHSVE1)
" prfd pldl1keep,p0, [x0, #6, MUL VL]        \n\t" // 15/16 (0/16)
#endif
MLA4ROW_LA_LB_D(z13,z14,z15,z16, z0,z1,z2,z3, z5, p0, z30, x0,1,x1,88) 
#if defined(PREFETCHSVE1)
" prfd pldl1keep,p0, [x0, #7, MUL VL]        \n\t" // 16/16 (0/16)
#endif
MLA4ROW_LA_LB_D(z17,z18,z19,z20, z0,z1,z2,z3, z6, p0, z31, x0,2,x1,96) 
MLA4ROW_LB_D(z21,z22,z23,z24, z0,z1,z2,z3, z7, p0, x1, 104) 
MLA4ROW_LA_LB_D(z25,z26,z27,z28, z0,z1,z2,z3, z8, p0, z3, x0,3,x1,112) 

MLA4ROW_D(z9,z10,z11,z12, z29,z30,z31,z3, z4, p0) 
MLA4ROW_D(z13,z14,z15,z16, z29,z30,z31,z3, z5, p0) 
MLA4ROW_D(z17,z18,z19,z20, z29,z30,z31,z3, z6, p0)
MLA4ROW_D(z21,z22,z23,z24, z29,z30,z31,z3, z7, p0)
MLA4ROW_D(z25,z26,z27,z28, z29,z30,z31,z3, z8, p0)
" incb x0, ALL, MUL #4                       \n\t" // Advancing 12 vectors, 8 already loaded
" add x1, x1, #120                           \n\t" // B = B+3*5*sizeof(double)
"                                            \n\t"
" .D4VX5CONSIDERKLEFT:                       \n\t" 
" cmp x6,0                                   \n\t" // If k_left == 0, we are done.
" beq .D4VX5POSTACCUM                        \n\t" // else, we enter the k_left loop.
"                                            \n\t"
".D4VX5LOOPKLEFT:                            \n\t"
"                                            \n\t"
LOAD4VEC_D(z0,z1,z2,z3,p0,x0)
" incb x0, ALL, MUL #4                       \n\t" // Advance a pointer by 4 vectors
"                                            \n\t"
LOAD5VEC_DIST_D(z4,z5,z6,z7,z8, p0,x1)
" add x1, x1, #40                            \n\t" // advance b pointer by 5 doubles
"                                            \n\t"
" sub x6,x6,1                                \n\t"
"                                            \n\t"
MLA4ROW_D(z9,z10,z11,z12,  z0,z1,z2,z3,     z4,p0)
MLA4ROW_D(z13,z14,z15,z16, z0,z1,z2,z3,     z5,p0)
MLA4ROW_D(z17,z18,z19,z20, z0,z1,z2,z3,     z6,p0)
MLA4ROW_D(z21,z22,z23,z24, z0,z1,z2,z3,     z7,p0)
MLA4ROW_D(z25,z26,z27,z28, z0,z1,z2,z3,     z8,p0)
"                                            \n\t"
" cmp x6,0                                   \n\t" // Iterate again.
" bne .D4VX5LOOPKLEFT                        \n\t" // if i!=0.
"                                            \n\t"
" .D4VX5POSTACCUM:                           \n\t"
#if defined(PREFETCH64) || defined(PREFETCH256)
" prfm PLDL2KEEP, [x3]                       \n\t"
" prfm PLDL2KEEP, [x4]                       \n\t"
#endif
"                                            \n\t"
" ld1rd  z29.d, p0/z, [x7]                   \n\t" // Load alpha
" ld1rd  z30.d, p0/z, [x8]                   \n\t" // Load beta
"                                            \n\t"
" cmp x13,#1                                 \n\t" // If rs_c != 1 (column-major)
" bne .D4VX5GENSTORED                        \n\t"
"                                            \n\t"
" .D4VX5COLSTORED:                           \n\t" // C is column-major.
"                                            \n\t"
// Don't use FINC macro, do it by hand because interleaving is required
//FINC_4COL(4VX5,CONT, z0,z1,z2,z3,z4,z5,z6,z7, x2,x20,x21,x22, no,no, 29,30, z10,z11,z12,z13,z14,z15,z16,z17, 1)
//FINC_4COL(4VX5,CONT, z8,z9,z10,z11,z12,z13,z14,z15, x23,x24,x25,x26, no,no, 29,30, z18,z19,z20,z21,z22,z23,z24,z25, 2)   

// Accumulated results are stored in z9-z28
// alpha is in z29, beta in z30
// z0-z8 and z31 are free
// Keep fmas on same registers min. 9 fma instructions apart (A64FX latency)
"                                        \n\t"
" fcmp d30,#0.0                          \n\t"
" beq .D4VX5BETAZEROCONTCOLSTOREDS       \n\t" // multiply with beta if beta isn't zero
"                                        \n\t"
LOAD4VEC_D(z0,z1,z2,z3, p0,x2)
LOAD4VEC_D(z4,z5,z6,z7, p0,x20)
"                                            \n\t"

// Let's do Accum=Accum*alpha, then Accum=Accum+C*beta and add some interleaving
MUL4ROW_D(z9,z10,z11,z12,  z9,z10,z11,z12,  z29, p0)
MUL4ROW_D(z13,z14,z15,z16, z13,z14,z15,z16, z29, p0)
MUL4ROW_D(z17,z18,z19,z20, z17,z18,z19,z20, z29, p0)
MLA4ROW_D(z9,z10,z11,z12,  z0,z1,z2,z3,     z30, p0)
MUL4ROW_D(z21,z22,z23,z24, z21,z22,z23,z24, z29, p0)
MLA4ROW_D(z13,z14,z15,z16, z4,z5,z6,z7,     z30, p0)
LOAD4VEC_D(z0,z1,z2,z3,    p0,x21)
MUL4ROW_D(z25,z26,z27,z28, z25,z26,z27,z28, z29, p0)
LOAD4VEC_D(z4,z5,z6,z7,    p0,x22)
MLA4ROW_D(z17,z18,z19,z20, z0,z1,z2,z3,     z30, p0)
LOAD4VEC_D(z0,z1,z2,z3,    p0,x23)
MLA4ROW_D(z21,z22,z23,z24, z4,z5,z6,z7,     z30, p0)
MLA4ROW_D(z25,z26,z27,z28, z0,z1,z2,z3,     z30, p0)

STOR4VEC_D(z9,z10,z11,z12, p0,x2)
STOR4VEC_D(z13,z14,z15,z16, p0,x20)
STOR4VEC_D(z17,z18,z19,z20, p0,x21)
STOR4VEC_D(z21,z22,z23,z24, p0,x22)
STOR4VEC_D(z25,z26,z27,z28, p0,x23)

" b .D4VX5END                                \n\t" // Duplicate code for stores required due to lack of registers
"                                            \n\t"
" .D4VX5BETAZEROCONTCOLSTOREDS:              \n\t"
// No need to zero anything as we are storing the scaled accumulated A*B values
MUL4ROW_D(z9,z10,z11,z12, z9,z10,z11,z12, z29, p0)
MUL4ROW_D(z13,z14,z15,z16, z13,z14,z15,z16, z29, p0)
MUL4ROW_D(z17,z18,z19,z20, z17,z18,z19,z20, z29, p0)
MUL4ROW_D(z21,z22,z23,z24, z21,z22,z23,z24, z29, p0)
MUL4ROW_D(z25,z26,z27,z28, z25,z26,z27,z28, z29, p0)

STOR4VEC_D(z9,z10,z11,z12, p0,x2)
STOR4VEC_D(z13,z14,z15,z16, p0,x20)
STOR4VEC_D(z17,z18,z19,z20, p0,x21)
STOR4VEC_D(z21,z22,z23,z24, p0,x22)
STOR4VEC_D(z25,z26,z27,z28, p0,x23)

"                                            \n\t"
" b .D4VX5END                                \n\t"
"                                            \n\t"
" .D4VX5GENSTORED:                           \n\t" // C is general-stride stored.
"                                            \n\t" // Creating index for stride load&store access
" index z6.d, xzr, x13                       \n\t" // 0, stride*double, 2*stride*double, ...
" mul x3, x13, x11                           \n\t" // x3 <- stride*double*vecsize
" index z7.d, x3, x13                        \n\t" // stride*double*vecsize, (vecsize+1)*stride*double, (vecsize+2)*stride*double, ...
" lsl x4, x3, #1                             \n\t" // x4 <- 2*stride*double*vecsize
" index z8.d, x4, x13                        \n\t" // stride*double*(2*vecsize), (2*vecsize+1)*stride*double, (2*vecsize+2)*stride*double, ...
" add x5, x4, x3                             \n\t" // x5 <- 3*stride*double*vecsize
" index z31.d, x5, x13                       \n\t" // stride*double*(3*vecsize), (3*vecsize+1)*stride*double, (3*vecsize+2)*stride*double, ...
"                                            \n\t"
//FINC_2COL(4VX5,GENI,z0,z1,z2,z3,     x2,x20,  z8,z9, 29, 30, z10,z11,z12,z13,1)
//FINC_2COL(4VX5,GENI,z4,z5,z6,z7,     x21,x22, z8,z9, 29, 30, z14,z15,z16,z17,2)
//FINC_2COL(4VX5,GENI,z10,z11,z12,z13, x23,x24, z8,z9, 29, 30, z18,z19,z20,z21,3)
//FINC_2COL(4VX5,GENI,z14,z15,z16,z17, x25,x26, z8,z9, 29, 30, z22,z23,z24,z25,4)
//FINC_4COL(4VX5,GENI, z0,z1,z2,z3,z4,z5,z6,z7, x2,x20,x21,x22, z8,z9, 26,27, z10,z11,z12,z13,z14,z15,z16,z17, 1)
//FINC_4COL(4VX5,GENI, z10,z11,z12,z13,z14,z15,z16,z17, x23,x24,x25,x26, z8,z9, 26,27, z18,z19,z20,z21,z22,z23,z24,z25, 2)
// Accumulated results are stored in z9-z28
// alpha is in z29, beta in z30
// index is stored in z6,z7,z8,z31
// z0-z5 are free
// Have to go column by column and hope for register renaming
"                                        \n\t"
" fcmp d30,#0.0                          \n\t"
" beq .D4VX5BETAZEROGENICOLSTOREDS       \n\t" // multiply with beta if beta isn't zero
"                                        \n\t"
LOAD4VEC_GENI_D (z0,z1,z2,z3, p0,x2, z6,z7,z8,z31)
MUL4ROW_D(z9,z10,z11,z12,  z9,z10,z11,z12,  z29, p0)
MLA4ROW_D(z9,z10,z11,z12,  z0,z1,z2,z3,     z30, p0)
LOAD4VEC_GENI_D (z0,z1,z2,z3, p0,x20, z6,z7,z8,z31)
MUL4ROW_D(z13,z14,z15,z16, z13,z14,z15,z16, z29, p0)
MLA4ROW_D(z13,z14,z15,z16, z0,z1,z2,z3,     z30, p0)
LOAD4VEC_GENI_D (z0,z1,z2,z3, p0,x21, z6,z7,z8,z31)
MUL4ROW_D(z17,z18,z19,z20, z17,z18,z19,z20, z29, p0)
MLA4ROW_D(z17,z18,z19,z20, z0,z1,z2,z3,     z30, p0)
LOAD4VEC_GENI_D (z0,z1,z2,z3, p0,x22, z6,z7,z8,z31)
MUL4ROW_D(z21,z22,z23,z24, z21,z22,z23,z24, z29, p0)
MLA4ROW_D(z21,z22,z23,z24, z0,z1,z2,z3,     z30, p0)
LOAD4VEC_GENI_D (z0,z1,z2,z3, p0,x23, z6,z7,z8,z31)
MUL4ROW_D(z25,z26,z27,z28, z25,z26,z27,z28, z29, p0)
MLA4ROW_D(z25,z26,z27,z28, z0,z1,z2,z3,     z30, p0)
"                                            \n\t"
STOR4VEC_GENI_D(z9,z10,z11,z12,  p0,x2, z6,z7,z8,z31)
STOR4VEC_GENI_D(z13,z14,z15,z16, p0,x20, z6,z7,z8,z31)
STOR4VEC_GENI_D(z17,z18,z19,z20, p0,x21, z6,z7,z8,z31)
STOR4VEC_GENI_D(z21,z22,z23,z24, p0,x22, z6,z7,z8,z31)
STOR4VEC_GENI_D(z25,z26,z27,z28, p0,x23, z6,z7,z8,z31)

" b .D4VX5END                                \n\t" // Duplicate code for stores required due to lack of registers
"                                            \n\t"
" .D4VX5BETAZEROGENICOLSTOREDS:              \n\t"
// No need to zero anything as we are storing the scaled accumulated A*B values
MUL4ROW_D(z9,z10,z11,z12, z9,z10,z11,z12, z29, p0)
MUL4ROW_D(z13,z14,z15,z16, z13,z14,z15,z16, z29, p0)
MUL4ROW_D(z17,z18,z19,z20, z17,z18,z19,z20, z29, p0)
MUL4ROW_D(z21,z22,z23,z24, z21,z22,z23,z24, z29, p0)
MUL4ROW_D(z25,z26,z27,z28, z25,z26,z27,z28, z29, p0)

STOR4VEC_GENI_D(z9,z10,z11,z12,  p0,x2, z6,z7,z8,z31)
STOR4VEC_GENI_D(z13,z14,z15,z16, p0,x20, z6,z7,z8,z31)
STOR4VEC_GENI_D(z17,z18,z19,z20, p0,x21, z6,z7,z8,z31)
STOR4VEC_GENI_D(z21,z22,z23,z24, p0,x22, z6,z7,z8,z31)
STOR4VEC_GENI_D(z25,z26,z27,z28, p0,x23, z6,z7,z8,z31)

"                                            \n\t"
" .D4VX5END:                                 \n\t" // Done!
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
