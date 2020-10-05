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

/* 2 vectors in m_r, n_r = 12
*/
void bli_dgemm_armv8a_sve_asm_2vx12
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
"                                            \n\t"
#if defined(PREFETCH64) || defined(PREFETCH256)
" prfm pstl1keep,[x2]                        \n\t" // Prefetch c.
" prfm pstl1keep,[x20]                       \n\t" // Prefetch c.
" prfm pstl1keep,[x21]                       \n\t" // Prefetch c.
" prfm pstl1keep,[x22]                       \n\t" // Prefetch c.
//" prfm pstl1keep,[x23]                       \n\t" // Prefetch c.
//" prfm pstl1keep,[x24]                       \n\t" // Prefetch c.
//" prfm pstl1keep,[x25]                       \n\t" // Prefetch c.
//" prfm pstl1keep,[x26]                       \n\t" // Prefetch c.
#endif
"                                            \n\t"
" ptrue p0.d                                 \n\t" // Creating all true predicate
"                                            \n\t"
/*************************************************
 *
 * Store A vectors in z0,z1,z2,z3, alternating between (z0,z1) and (z2,z3)
 *
 * Store B values in z4, qword-duplicate to z5 and z6
 * LD1 has 11 cycles latency
 * DUP has 6 cycles latency, but uses FLA pipeline, might slow down
 *
 *************************************************/
LOAD2VEC_D(z0,z1,p0,x0)
"                                            \n\t"
LOAD1VEC_D(z4,p0,x1)
"                                            \n\t"
" dup z5.q, z4.q[0] \n\t"
" dup z6.q, z4.q[1] \n\t"
"                                            \n\t"
ZERO4VEC_D(z7,z8,z9,z10)                             // c columns 0-1
ZERO4VEC_D(z11,z12,z13,z14)                          // c columns 2-3
ZERO4VEC_D(z15,z16,z17,z18)                          // c columns 4-5
ZERO4VEC_D(z19,z20,z21,z22)                          // c columns 6-7
ZERO4VEC_D(z23,z24,z25,z26)                          // c columns 8-9
ZERO4VEC_D(z27,z28,z29,z30)                          // c columns 10-11

"                                            \n\t"
"                                            \n\t"
" cmp x5,#0                                  \n\t" // If k_iter == 0, jump to k_left.
" beq .D2VX12CONSIDERKLEFT                   \n\t"
"                                            \n\t"
" incb x0, ALL, MUL #2                       \n\t" // A = A+vecsize*2
" add x1, x1, #64                            \n\t" // B = B+8*sizeof(double)
"                                            \n\t"
" cmp x5,1                                   \n\t" // If there is just one k_iter, jump to that one. 
" beq .D2VX12LASTITER                        \n\t" // (as loop is do-while-like).
"                                            \n\t" 
"                                            \n\t"
" .D2VX12LOOP:                               \n\t" // Body
MLA2ROW_I_LA_D(z7,z8,  z0,z1, z5, 0, p0, z2, x0,0)    // first 2vx12 block
MLA2ROW_I_LA_D(z9,z10, z0,z1, z5, 1, p0, z3, x0,1)
" dup z5.q, z4.q[2] \n\t"
MLA2ROW_I_D(z11,z12, z0,z1, z6, 0, p0)
MLA2ROW_I_D(z13,z14, z0,z1, z6, 1, p0)
" dup z6.q, z4.q[3] \n\t"
LOAD1VEC_D(z4,p0,x1)
" add x1, x1, #64 \n\t"
MLA2ROW_I_D(z15,z16, z0,z1, z5, 0, p0)
MLA2ROW_I_D(z17,z18, z0,z1, z5, 1, p0)
" dup z5.q, z4.q[0] \n\t"
MLA2ROW_I_D(z19,z20, z0,z1, z6, 0, p0)
MLA2ROW_I_D(z21,z22, z0,z1, z6, 1, p0)
" dup z6.q, z4.q[1] \n\t"
MLA2ROW_I_D(z23,z24, z0,z1, z5, 0, p0)
MLA2ROW_I_D(z25,z26, z0,z1, z5, 1, p0)
" dup z5.q, z4.q[2] \n\t"
MLA2ROW_I_D(z27,z28, z0,z1, z6, 0, p0)
MLA2ROW_I_D(z29,z30, z0,z1, z6, 1, p0)
" dup z6.q, z4.q[3] \n\t"
LOAD1VEC_D(z4,p0,x1)
" add x1, x1, #64 \n\t"                                // end first 2vx12 block
"                                            \n\t"
MLA2ROW_I_LA_D(z7,z8,  z2,z3, z5, 0, p0, z0, x0,2)    // second 2vx12 block
MLA2ROW_I_LA_D(z9,z10, z2,z3, z5, 1, p0, z1, x0,3)
" dup z5.q, z4.q[0] \n\t"
MLA2ROW_I_D(z11,z12, z2,z3, z6, 0, p0)
MLA2ROW_I_D(z13,z14, z2,z3, z6, 1, p0)
" dup z6.q, z4.q[1] \n\t"
MLA2ROW_I_D(z15,z16, z2,z3, z5, 0, p0)
MLA2ROW_I_D(z17,z18, z2,z3, z5, 1, p0)
" dup z5.q, z4.q[2] \n\t"
MLA2ROW_I_D(z19,z20, z2,z3, z6, 0, p0)
MLA2ROW_I_D(z21,z22, z2,z3, z6, 1, p0)
" dup z6.q, z4.q[3] \n\t"
LOAD1VEC_D(z4,p0,x1)
" add x1, x1, #64 \n\t"
MLA2ROW_I_D(z23,z24, z2,z3, z5, 0, p0)
MLA2ROW_I_D(z25,z26, z2,z3, z5, 1, p0)
" dup z5.q, z4.q[0] \n\t"
MLA2ROW_I_D(z27,z28, z2,z3, z6, 0, p0)
MLA2ROW_I_D(z29,z30, z2,z3, z6, 1, p0)
" dup z6.q, z4.q[1] \n\t"                              // end second 2vx12 block
" incb x0, ALL, MUL #4                       \n\t" // Next 4 A vectors
MLA2ROW_I_LA_D(z7,z8,  z0,z1, z5, 0, p0, z2, x0,0)    // third 2vx12 block
MLA2ROW_I_LA_D(z9,z10, z0,z1, z5, 1, p0, z3, x0,1)
" dup z5.q, z4.q[2] \n\t"
MLA2ROW_I_D(z11,z12, z0,z1, z6, 0, p0)
MLA2ROW_I_D(z13,z14, z0,z1, z6, 1, p0)
" dup z6.q, z4.q[3] \n\t"
LOAD1VEC_D(z4,p0,x1)
" add x1, x1, #64 \n\t"
MLA2ROW_I_D(z15,z16, z0,z1, z5, 0, p0)
MLA2ROW_I_D(z17,z18, z0,z1, z5, 1, p0)
" dup z5.q, z4.q[0] \n\t"
MLA2ROW_I_D(z19,z20, z0,z1, z6, 0, p0)
MLA2ROW_I_D(z21,z22, z0,z1, z6, 1, p0)
" dup z6.q, z4.q[1] \n\t"
MLA2ROW_I_D(z23,z24, z0,z1, z5, 0, p0)
MLA2ROW_I_D(z25,z26, z0,z1, z5, 1, p0)
" dup z5.q, z4.q[2] \n\t"
MLA2ROW_I_D(z27,z28, z0,z1, z6, 0, p0)
MLA2ROW_I_D(z29,z30, z0,z1, z6, 1, p0)
" dup z6.q, z4.q[3] \n\t"
LOAD1VEC_D(z4,p0,x1)
" add x1, x1, #64 \n\t"                                // end third 2vx12 block
"                                            \n\t"
MLA2ROW_I_LA_D(z7,z8,  z2,z3, z5, 0, p0, z0, x0,2)    // fourth 2vx12 block
MLA2ROW_I_LA_D(z9,z10, z2,z3, z5, 1, p0, z1, x0,3)
" dup z5.q, z4.q[0] \n\t"
MLA2ROW_I_D(z11,z12, z2,z3, z6, 0, p0)
MLA2ROW_I_D(z13,z14, z2,z3, z6, 1, p0)
" dup z6.q, z4.q[1] \n\t"
MLA2ROW_I_D(z15,z16, z2,z3, z5, 0, p0)
MLA2ROW_I_D(z17,z18, z2,z3, z5, 1, p0)
" dup z5.q, z4.q[2] \n\t"
MLA2ROW_I_D(z19,z20, z2,z3, z6, 0, p0)
MLA2ROW_I_D(z21,z22, z2,z3, z6, 1, p0)
" dup z6.q, z4.q[3] \n\t"
LOAD1VEC_D(z4,p0,x1)
" add x1, x1, #64 \n\t"
MLA2ROW_I_D(z23,z24, z2,z3, z5, 0, p0)
MLA2ROW_I_D(z25,z26, z2,z3, z5, 1, p0)
" dup z5.q, z4.q[0] \n\t"
MLA2ROW_I_D(z27,z28, z2,z3, z6, 0, p0)
MLA2ROW_I_D(z29,z30, z2,z3, z6, 1, p0)
" dup z6.q, z4.q[1] \n\t"                              // end fourth 2vx12 block
" incb x0, ALL, MUL #4                       \n\t" // Next 4 A vectors
// " add x1, x1, #256                           \n\t" // B = B+4*8*sizeof(double)
"                                            \n\t"
" sub x5,x5,1                                \n\t" // i-=1
" cmp x5,1                                   \n\t" // Iterate again if we are not in k_iter == 1.
" bne .D2VX12LOOP                             \n\t"
" .D2VX12LASTITER:                            \n\t" // Body

MLA2ROW_I_LA_D(z7,z8,  z0,z1, z5, 0, p0, z2, x0,0)    // first 2vx12 block
MLA2ROW_I_LA_D(z9,z10, z0,z1, z5, 1, p0, z3, x0,1)
" dup z5.q, z4.q[2] \n\t"
MLA2ROW_I_D(z11,z12, z0,z1, z6, 0, p0)
MLA2ROW_I_D(z13,z14, z0,z1, z6, 1, p0)
" dup z6.q, z4.q[3] \n\t"
LOAD1VEC_D(z4,p0,x1)
" add x1, x1, #64 \n\t"
MLA2ROW_I_D(z15,z16, z0,z1, z5, 0, p0)
MLA2ROW_I_D(z17,z18, z0,z1, z5, 1, p0)
" dup z5.q, z4.q[0] \n\t"
MLA2ROW_I_D(z19,z20, z0,z1, z6, 0, p0)
MLA2ROW_I_D(z21,z22, z0,z1, z6, 1, p0)
" dup z6.q, z4.q[1] \n\t"
MLA2ROW_I_D(z23,z24, z0,z1, z5, 0, p0)
MLA2ROW_I_D(z25,z26, z0,z1, z5, 1, p0)
" dup z5.q, z4.q[2] \n\t"
MLA2ROW_I_D(z27,z28, z0,z1, z6, 0, p0)
MLA2ROW_I_D(z29,z30, z0,z1, z6, 1, p0)
" dup z6.q, z4.q[3] \n\t"
LOAD1VEC_D(z4,p0,x1)
" add x1, x1, #64 \n\t"                                // end first 2vx12 block
"                                            \n\t"
MLA2ROW_I_LA_D(z7,z8,  z2,z3, z5, 0, p0, z0, x0,2)    // second 2vx12 block
MLA2ROW_I_LA_D(z9,z10, z2,z3, z5, 1, p0, z1, x0,3)
" dup z5.q, z4.q[0] \n\t"
MLA2ROW_I_D(z11,z12, z2,z3, z6, 0, p0)
MLA2ROW_I_D(z13,z14, z2,z3, z6, 1, p0)
" dup z6.q, z4.q[1] \n\t"
MLA2ROW_I_D(z15,z16, z2,z3, z5, 0, p0)
MLA2ROW_I_D(z17,z18, z2,z3, z5, 1, p0)
" dup z5.q, z4.q[2] \n\t"
MLA2ROW_I_D(z19,z20, z2,z3, z6, 0, p0)
MLA2ROW_I_D(z21,z22, z2,z3, z6, 1, p0)
" dup z6.q, z4.q[3] \n\t"
LOAD1VEC_D(z4,p0,x1)
" add x1, x1, #64 \n\t"
MLA2ROW_I_D(z23,z24, z2,z3, z5, 0, p0)
MLA2ROW_I_D(z25,z26, z2,z3, z5, 1, p0)
" dup z5.q, z4.q[0] \n\t"
MLA2ROW_I_D(z27,z28, z2,z3, z6, 0, p0)
MLA2ROW_I_D(z29,z30, z2,z3, z6, 1, p0)
" dup z6.q, z4.q[1] \n\t"                              // end second 2vx12 block
"                                                \n\t"
MLA2ROW_I_LA_D(z7,z8,  z0,z1, z5, 0, p0, z2, x0,4)    // third 2vx12 block
MLA2ROW_I_LA_D(z9,z10, z0,z1, z5, 1, p0, z3, x0,5)
" dup z5.q, z4.q[2] \n\t"
MLA2ROW_I_D(z11,z12, z0,z1, z6, 0, p0)
MLA2ROW_I_D(z13,z14, z0,z1, z6, 1, p0)
" dup z6.q, z4.q[3] \n\t"
LOAD1VEC_D(z4,p0,x1)
" add x1, x1, #64 \n\t"
MLA2ROW_I_D(z15,z16, z0,z1, z5, 0, p0)
MLA2ROW_I_D(z17,z18, z0,z1, z5, 1, p0)
" dup z5.q, z4.q[0] \n\t"
MLA2ROW_I_D(z19,z20, z0,z1, z6, 0, p0)
MLA2ROW_I_D(z21,z22, z0,z1, z6, 1, p0)
" dup z6.q, z4.q[1] \n\t"
MLA2ROW_I_D(z23,z24, z0,z1, z5, 0, p0)
MLA2ROW_I_D(z25,z26, z0,z1, z5, 1, p0)
" dup z5.q, z4.q[2] \n\t"
MLA2ROW_I_D(z27,z28, z0,z1, z6, 0, p0)
MLA2ROW_I_D(z29,z30, z0,z1, z6, 1, p0)
" dup z6.q, z4.q[3] \n\t"
LOAD1VEC_D(z4,p0,x1)
" add x1, x1, #64 \n\t"                                // end third 2vx12 block
"                                            \n\t"
MLA2ROW_I_D(z7,z8,  z2,z3, z5, 0, p0)    // fourth 2vx12 block
MLA2ROW_I_D(z9,z10, z2,z3, z5, 1, p0)
" dup z5.q, z4.q[0] \n\t"
MLA2ROW_I_D(z11,z12, z2,z3, z6, 0, p0)
MLA2ROW_I_D(z13,z14, z2,z3, z6, 1, p0)
" dup z6.q, z4.q[1] \n\t"
MLA2ROW_I_D(z15,z16, z2,z3, z5, 0, p0)
MLA2ROW_I_D(z17,z18, z2,z3, z5, 1, p0)
" dup z5.q, z4.q[2] \n\t"
MLA2ROW_I_D(z19,z20, z2,z3, z6, 0, p0)
MLA2ROW_I_D(z21,z22, z2,z3, z6, 1, p0)
" dup z6.q, z4.q[3] \n\t"
//LOAD1VEC_D(z4,p0,x1)
//" add x1, x1, #64 \n\t"
MLA2ROW_I_D(z23,z24, z2,z3, z5, 0, p0)
MLA2ROW_I_D(z25,z26, z2,z3, z5, 1, p0)
MLA2ROW_I_D(z27,z28, z2,z3, z6, 0, p0)
MLA2ROW_I_D(z29,z30, z2,z3, z6, 1, p0)             // end fourth 2vx12 block
"                                            \n\t"

" incb x0, ALL, MUL #6                       \n\t" // 6 Vectors loaded
//" add x1, x1, #192                           \n\t" // B = B+3*8*sizeof(double)
"                                            \n\t"
" .D2VX12CONSIDERKLEFT:                       \n\t" 
" cmp x6,0                                   \n\t" // If k_left == 0, we are done.
" beq .D2VX12POSTACCUM                        \n\t" // else, we enter the k_left loop.
"                                            \n\t"
".D2VX12LOOPKLEFT:                            \n\t"
"                                            \n\t"
LOAD2VEC_D(z0,z1,p0,x0)
" incb x0, ALL, MUL #2                       \n\t" // Advance a pointer by 2 vectors
"                                            \n\t"
LOAD1VEC_D(z4,p0,x1)
" dup z5.q, z4.q[0]                          \n\t"
" dup z6.q, z4.q[1]                          \n\t"
" add x1, x1, #64                            \n\t" // advance b pointer by 8 doubles
LOAD2VEC_QDIST(z2,z3,p0,x1)
" add x1, x1, #32                            \n\t" // advance b pointer by 8 doubles
"                                            \n\t"
" sub x6,x6,1                                \n\t"
"                                            \n\t"
MLA2ROW_I_D(z7,z8,    z0,z1, z5, 0, p0)
MLA2ROW_I_D(z9,z10,   z0,z1, z5, 1, p0)
" dup z5.q, z4.q[2]                          \n\t"
MLA2ROW_I_D(z11,z12,  z0,z1, z6, 0, p0)
MLA2ROW_I_D(z13,z14,  z0,z1, z6, 1, p0)
" dup z6.q, z4.q[3]                          \n\t"
MLA2ROW_I_D(z15,z16,  z0,z1, z5, 0, p0)
MLA2ROW_I_D(z17,z18,  z0,z1, z5, 1, p0)
MLA2ROW_I_D(z19,z20,  z0,z1, z6, 0, p0)
MLA2ROW_I_D(z21,z22,  z0,z1, z6, 1, p0)
MLA2ROW_I_D(z23,z24,  z0,z1, z2, 0, p0)
MLA2ROW_I_D(z25,z26,  z0,z1, z2, 1, p0)
MLA2ROW_I_D(z27,z28,  z0,z1, z3, 0, p0)
MLA2ROW_I_D(z29,z30,  z0,z1, z3, 1, p0)

"                                            \n\t"
" cmp x6,0                                   \n\t" // Iterate again.
" bne .D2VX12LOOPKLEFT                       \n\t" // if i!=0.
"                                            \n\t"
" .D2VX12POSTACCUM:                          \n\t"

#if defined(PREFETCH64) || defined(PREFETCH256)
" prfm PLDL2KEEP, [x3]                       \n\t"
" prfm PLDL2KEEP, [x4]                       \n\t"
#endif
"                                            \n\t"
" ld1rd  z0.d, p0/z, [x7]                    \n\t" // Load alpha
" ld1rd  z31.d, p0/z, [x8]                   \n\t" // Load beta

"                                            \n\t"
" cmp x13,#1                                 \n\t" // If rs_c != 1 (column-major)
" bne .D2VX12GENSTORED                       \n\t"
"                                            \n\t"
" .D2VX12COLSTORED:                          \n\t" // C is column-major.
"                                            \n\t"

// Accumulated results are stored in z7-z30
// alpha in z0, beta in z31 - (IMPOSSIBLE) put alpha is in z31.d[0], beta in z31.d[1]
// So this is incredibly annoying, but indexed fmla/fmul only works if the third operand is
// z0-z15, so put alpha and beta in z0 instead...
// z1-z6 are free
"                                        \n\t"

" fcmp d31,#0.0                          \n\t"
" trn1 z0.d, z0.d, z31.d                 \n\t" // splice here
" beq .D2VX12BETAZEROCONTCOLSTOREDS      \n\t" // multiply with beta if beta isn't zero
"                                        \n\t"
LOAD2VEC_D(z1,z2,p0,x2)                         // Load Column 0
LOAD2VEC_D(z3,z4,p0,x20)                        // Load Column 1
LOAD2VEC_D(z5,z6,p0,x21)                        // Load Column 2
"                                            \n\t"

// Let's do Accum=Accum*alpha, then Accum=Accum+C*beta and add some interleaving
MUL4ROW_I_D(z7,z8,z9,z10,    z7,z8,z9,z10,    z0, 0, p0)

//  0   1   2   3   4   5   6   7   8   9  10  11
//  ^   ^   ^   ^
// x2 x20 x21 x22

MUL4ROW_I_D(z11,z12,z13,z14, z11,z12,z13,z14, z0, 0, p0)
MLA4ROW_I_D(z7,z8,z9,z10,    z1,z2,z3,z4,     z0, 1, p0)
STOR2VEC_D(z7,z8,  p0, x2)                                // Store Column 0
" add x2,x22,x10                            \n\t"       // Load address Column 4 of C
STOR2VEC_D(z9,z10, p0, x20)                               // Store Column 1
" add x20,x2,x10                            \n\t"       // Load address Column 5 of C
LOAD2VEC_D(z7,z8, p0,x22)                                 // Load Column 3
LOAD2VEC_D(z9,z10, p0,x2)                                 // Load Column 4

//  0   1   2   3   4   5   6   7   8   9  10  11
//          ^   ^   ^   ^
//        x21 x22  x2 x20

MUL4ROW_I_D(z15,z16,z17,z18, z15,z16,z17,z18, z0, 0, p0)
MLA4ROW_I_D(z11,z12,z13,z14, z5,z6,z7,z8,     z0, 1, p0)
STOR2VEC_D(z11,z12, p0, x21)                                  // Store Column 2
" add x21,x20,x10                           \n\t"       // Load address Column 6 of C
STOR2VEC_D(z13,z14, p0, x22)                                  // Store Column 3
" add x22,x21,x10                           \n\t"       // Load address Column 7 of C
LOAD2VEC_D(z11,z12, p0,x20)                               // Load Column 5 
LOAD2VEC_D(z13,z14, p0,x21)                               // Load Column 6

//  0   1   2   3   4   5   6   7   8   9  10  11
//                  ^   ^   ^   ^
//                 x2 x20 x21 x22

MUL4ROW_I_D(z19,z20,z21,z22, z19,z20,z21,z22, z0, 0, p0)
MLA4ROW_I_D(z15,z16,z17,z18, z9,z10,z11,z12,  z0, 1, p0)
STOR2VEC_D(z15,z16, p0, x2)                                   // Store Column 4
" add x2,x22,x10                            \n\t"       // Load address Column 8 of C
STOR2VEC_D(z17,z18, p0, x20)                                  // Store Column 5
" add x20,x2,x10                            \n\t"       // Load address Column 9 of C
LOAD2VEC_D(z15,z16,p0,x22)                                // Load Column 7
LOAD2VEC_D(z17,z18,p0,x2)                                 // Load Column 8

//  0   1   2   3   4   5   6   7   8   9  10  11
//                          ^   ^   ^   ^
//                        x21 x22  x2 x20

MUL4ROW_I_D(z23,z24,z25,z26, z23,z24,z25,z26, z0, 0, p0)
MLA4ROW_I_D(z19,z20,z21,z22, z13,z14,z15,z16, z0, 1, p0)
STOR2VEC_D(z19,z20, p0, x21)                                  // Store Column 6
" add x21,x20,x10                           \n\t"       // Load address Column 10 of C
STOR2VEC_D(z21,z22, p0, x22)                                  // Store Column 7
" add x22,x21,x10                           \n\t"       // Load address Column 11 of C
LOAD2VEC_D(z19,z20,p0,x20)                                // Load Column 9
LOAD2VEC_D(z21,z22,p0,x21)                                // Load Column 10

//  0   1   2   3   4   5   6   7   8   9  10  11
//                                  ^   ^   ^   ^
//                                 x2 x20 x21 x22

MUL4ROW_I_D(z27,z28,z29,z30, z27,z28,z29,z30, z0, 0, p0)
MLA4ROW_I_D(z23,z24,z25,z26, z17,z18,z19,z20, z0, 1, p0)
STOR2VEC_D(z23,z24, p0, x2)                                   // Store Column 8
STOR2VEC_D(z25,z26, p0, x20)                                  // Store Column 9
LOAD2VEC_D(z23,z24, p0, x22)                                // Load Column 11

MLA4ROW_I_D(z27,z28,z29,z30, z21,z22,z23,z24, z0, 1, p0)
STOR2VEC_D(z27,z28, p0, x21)                                   // Store Column 10
STOR2VEC_D(z29,z30, p0, x22)                                  // Store Column 11

" b .D2VX12END                              \n\t"       // Duplicate code for stores required due to lack of registers
"                                           \n\t"
" .D2VX12BETAZEROCONTCOLSTOREDS:            \n\t"
// No need to zero anything as we are storing the scaled accumulated A*B values
MUL4ROW_I_D(z7,z8,z9,z10,    z7,z8,z9,z10,    z0, 0, p0)
MUL4ROW_I_D(z11,z12,z13,z14, z11,z12,z13,z14, z0, 0, p0)
MUL4ROW_I_D(z15,z16,z17,z18, z15,z16,z17,z18, z0, 0, p0)
MUL4ROW_I_D(z19,z20,z21,z22, z19,z20,z21,z22, z0, 0, p0)
MUL4ROW_I_D(z23,z24,z25,z26, z23,z24,z25,z26, z0, 0, p0)
MUL4ROW_I_D(z27,z28,z29,z30, z27,z28,z29,z30, z0, 0, p0)
STOR2VEC_D(z7,z8,p0,x2)                                   // Store Column 0
" add x2,x22,x10                            \n\t"       // Load address Column 4 of C
STOR2VEC_D(z9,z10,p0,x20)                                 // Store Column 1
" add x20,x2,x10                            \n\t"       // Load address Column 5 of C
STOR2VEC_D(z11,z12,p0,x21)                                // Store Column 2
" add x21,x20,x10                           \n\t"       // Load address Column 6 of C
STOR2VEC_D(z13,z14,p0,x22)                                // Store Column 3
" add x22,x21,x10                           \n\t"       // Load address Column 7 of C
STOR2VEC_D(z15,z16,p0,x2)                                 // Store Column 4
" add x2,x22,x10                            \n\t"       // Load address Column 8 of C
STOR2VEC_D(z17,z18,p0,x20)                                // Store Column 5
" add x20,x2,x10                            \n\t"       // Load address Column 9 of C
STOR2VEC_D(z19,z20,p0,x21)                                // Store Column 6
" add x21,x20,x10                           \n\t"       // Load address Column 10 of C
STOR2VEC_D(z21,z22,p0,x22)                                // Store Column 7
" add x22,x21,x10                           \n\t"       // Load address Column 11 of C
STOR2VEC_D(z23,z24,p0,x2)                                 // Store Column 8
STOR2VEC_D(z25,z26,p0,x20)                                // Store Column 9
STOR2VEC_D(z27,z28,p0,x21)                                // Store Column 10
STOR2VEC_D(z29,z30,p0,x22)                                // Store Column 11

"                                            \n\t"
" b .D2VX12END                                \n\t"
"                                            \n\t"
" .D2VX12GENSTORED:                           \n\t" // C is general-stride stored.

"                                            \n\t" // Creating index for stride load&store access
" index z1.d, xzr, x13                       \n\t" // 0, stride*double, 2*stride*double, ...
" mul x3, x13, x11                           \n\t" // x3 <- stride*double*vecsize
" index z2.d, x3, x13                        \n\t" // stride*double*vecsize, (vecsize+1)*stride*double, (vecsize+2)*stride*double, ...
"                                            \n\t"
" fcmp d31,#0.0                          \n\t"
" trn1 z0.d, z0.d, z31.d                 \n\t" // splice here
" beq .D2VX12BETAZEROGENSTOREDS      \n\t" // multiply with beta if beta isn't zero
"                                            \n\t"
LOAD2VEC_GENI_D (z3,z4, p0, x2, z1,z2)                    // Load Column 0
LOAD2VEC_GENI_D (z5,z6, p0,x20, z1,z2)                    // Load Column 1
"                                            \n\t"

// Let's do Accum=Accum*alpha, then Accum=Accum+C*beta and add some interleaving

//  0   1   2   3   4   5   6   7   8   9  10  11
//  ^   ^   ^   ^
// x2 x20 x21 x22

MUL4ROW_I_D(z7,z8,z9,z10,    z7,z8,z9,z10,    z0, 0, p0)
MLA4ROW_I_D(z7,z8,z9,z10,    z3,z4,z5,z6,     z0, 1, p0)
STOR2VEC_GENI_D(z7,z8,  p0, x2,  z1,z2)                   // Store Column 0
" add x2,x22,x10                            \n\t"       // Load address Column 4 of C
STOR2VEC_GENI_D(z9,z10, p0, x20, z1,z2)                   // Store Column 1
" add x20,x2,x10                            \n\t"       // Load address Column 5 of C
LOAD2VEC_GENI_D(z7,z8,  p0,x21, z1,z2)                    // Load Column 2
LOAD2VEC_GENI_D(z9,z10, p0,x22, z1,z2)                    // Load Column 3

//  0   1   2   3   4   5   6   7   8   9  10  11
//          ^   ^   ^   ^
//        x21 x22  x2 x20

MUL4ROW_I_D(z11,z12,z13,z14, z11,z12,z13,z14, z0, 0, p0)
MLA4ROW_I_D(z11,z12,z13,z14, z7,z8,z9,z10,    z0, 1, p0)
STOR2VEC_GENI_D(z11,z12, p0, x21, z1,z2)                  // Store Column 2
" add x21,x20,x10                           \n\t"       // Load address Column 6 of C
STOR2VEC_GENI_D(z13,z14, p0, x22, z1,z2)                  // Store Column 3
" add x22,x21,x10                           \n\t"       // Load address Column 7 of C
LOAD2VEC_GENI_D(z11,z12, p0,x2,  z1,z2)                   // Load Column 4
LOAD2VEC_GENI_D(z13,z14, p0,x20, z1,z2)                   // Load Column 5


//  0   1   2   3   4   5   6   7   8   9  10  11
//                  ^   ^   ^   ^
//                 x2 x20 x21 x22

MUL4ROW_I_D(z15,z16,z17,z18, z15,z16,z17,z18, z0, 0, p0)
MLA4ROW_I_D(z15,z16,z17,z18, z11,z12,z13,z14, z0, 1, p0)
STOR2VEC_GENI_D(z15,z16, p0, x2,  z1,z2)                  // Store Column 4
" add x2,x22,x10                            \n\t"       // Load address Column 8 of C
STOR2VEC_GENI_D(z17,z18, p0, x20, z1,z2)                  // Store Column 5
" add x20,x2,x10                            \n\t"       // Load address Column 9 of C
LOAD2VEC_GENI_D(z15,z16, p0,x21, z1,z2)                   // Load Column 6
LOAD2VEC_GENI_D(z17,z18, p0,x22, z1,z2)                   // Load Column 7

//  0   1   2   3   4   5   6   7   8   9  10  11
//                          ^   ^   ^   ^
//                        x21 x22  x2 x20

MUL4ROW_I_D(z19,z20,z21,z22, z19,z20,z21,z22, z0, 0, p0)
MLA4ROW_I_D(z19,z20,z21,z22, z15,z16,z17,z18, z0, 1, p0)
STOR2VEC_GENI_D(z19,z20, p0, x21, z1,z2)                  // Store Column 6
" add x21,x20,x10                           \n\t"       // Load address Column 10 of C
STOR2VEC_GENI_D(z21,z22, p0, x22, z1,z2)                  // Store Column 7
" add x22,x21,x10                           \n\t"       // Load address Column 11 of C
LOAD2VEC_GENI_D(z19,z20,p0,x2,  z1,z2)                    // Load Column 8
LOAD2VEC_GENI_D(z21,z22,p0,x20, z1,z2)                    // Load Column 9

//  0   1   2   3   4   5   6   7   8   9  10  11
//                                  ^   ^   ^   ^
//                                 x2 x20 x21 x22

MUL4ROW_I_D(z23,z24,z25,z26, z23,z24,z25,z26, z0, 0, p0)
MLA4ROW_I_D(z23,z24,z25,z26, z19,z20,z21,z22, z0, 1, p0)
STOR2VEC_GENI_D(z23,z24,p0, x2,  z1,z2)                   // Store Column 8
STOR2VEC_GENI_D(z25,z26,p0, x20, z1,z2)                   // Store Column 9
LOAD2VEC_GENI_D(z23,z24,p0, x21, z1,z2)                   // Load Column 10
LOAD2VEC_GENI_D(z25,z26,p0, x22, z1,z2)                   // Load Column 11

MUL4ROW_I_D(z27,z28,z29,z30, z27,z28,z29,z30, z0, 0, p0)
MLA4ROW_I_D(z27,z28,z29,z30, z23,z24,z25,z26, z0, 1, p0)
STOR2VEC_GENI_D(z27,z28, p0, x21, z1,z2)                              // Store Column 10
STOR2VEC_GENI_D(z29,z30, p0, x22, z1,z2)                              // Store Column 11

" b .D2VX12END                          \n\t"
" .D2VX12BETAZEROGENSTOREDS:            \n\t"
// No need to zero anything as we are storing the scaled accumulated A*B values
MUL4ROW_I_D(z7,z8,z9,z10,    z7,z8,z9,z10,    z0, 0, p0)
MUL4ROW_I_D(z11,z12,z13,z14, z11,z12,z13,z14, z0, 0, p0)
MUL4ROW_I_D(z15,z16,z17,z18, z15,z16,z17,z18, z0, 0, p0)
MUL4ROW_I_D(z19,z20,z21,z22, z19,z20,z21,z22, z0, 0, p0)
MUL4ROW_I_D(z23,z24,z25,z26, z23,z24,z25,z26, z0, 0, p0)
MUL4ROW_I_D(z27,z28,z29,z30, z27,z28,z29,z30, z0, 0, p0)
STOR2VEC_GENI_D(z7,z8,  p0,x2,  z1,z2)                                   // Store Column 0
" add x2,x22,x10                            \n\t"       // Load address Column 4 of C
STOR2VEC_GENI_D(z9,z10, p0,x20, z1,z2)                                 // Store Column 1
" add x20,x2,x10                            \n\t"       // Load address Column 5 of C
STOR2VEC_GENI_D(z11,z12,p0,x21, z1,z2)                                // Store Column 2
" add x21,x20,x10                           \n\t"       // Load address Column 6 of C
STOR2VEC_GENI_D(z13,z14,p0,x22, z1,z2)                                // Store Column 3
" add x22,x21,x10                           \n\t"       // Load address Column 7 of C
STOR2VEC_GENI_D(z15,z16,p0,x2,  z1,z2)                                 // Store Column 4
" add x2,x22,x10                            \n\t"       // Load address Column 8 of C
STOR2VEC_GENI_D(z17,z18,p0,x20, z1,z2)                                // Store Column 5
" add x20,x2,x10                            \n\t"       // Load address Column 9 of C
STOR2VEC_GENI_D(z19,z20,p0,x21, z1,z2)                                // Store Column 6
" add x21,x20,x10                           \n\t"       // Load address Column 10 of C
STOR2VEC_GENI_D(z21,z22,p0,x22, z1,z2)                                // Store Column 7
" add x22,x21,x10                           \n\t"       // Load address Column 11 of C
STOR2VEC_GENI_D(z23,z24,p0,x2,  z1,z2)                                 // Store Column 8
STOR2VEC_GENI_D(z25,z26,p0,x20, z1,z2)                                // Store Column 9
STOR2VEC_GENI_D(z27,z28,p0,x21, z1,z2)                                // Store Column 10
STOR2VEC_GENI_D(z29,z30,p0,x22, z1,z2)                                // Store Column 11
"                                            \n\t"
" .D2VX12END:                                \n\t" // Done!
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
