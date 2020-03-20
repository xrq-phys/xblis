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


/***********************************************
 * SIGNIFICANT DIFFERENCES FROM OTHER KERNELS: *
 * - 8xk unroll instead of 4xk                 *
 ***********************************************/


#include "blis.h"
#include "bli_dgemm_sve_asm_macros.h"

/* 2 vectors in m_r, n_r = 9
*/
void bli_dgemm_armv8a_sve_asm_2vx9
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
" add x24,x23,x10                            \n\t" //Load address Column 5 of C
" add x25,x24,x10                            \n\t" //Load address Column 6 of C
" add x26,x25,x10                            \n\t" //Load address Column 7 of C
" add x27,x26,x10                            \n\t" //Load address Column 8 of C
"                                            \n\t"
" prfm pldl1keep,[x2]                        \n\t" // Prefetch c column 0.
" prfm pldl1keep,[x20]                       \n\t" // Prefetch c column 1.
" prfm pldl1keep,[x21]                       \n\t" // Prefetch c column 2.
" prfm pldl1keep,[x22]                       \n\t" // Prefetch c column 3.
" prfm pldl1keep,[x23]                       \n\t" // Prefetch c column 4.
" prfm pldl1keep,[x24]                       \n\t" // Prefetch c column 5.
" prfm pldl1keep,[x25]                       \n\t" // Prefetch c column 6.
" prfm pldl1keep,[x26]                       \n\t" // Prefetch c column 7.
" prfm pldl1keep,[x27]                       \n\t" // Prefetch c column 8.
"                                            \n\t"
" ptrue p0.d                                 \n\t" // Creating all true predicate
"                                            \n\t"
LOAD2VEC(z0,z1,p0,x0)
"                                            \n\t"
LOAD8VEC_DIST(z2,z3,z4,z5,z6,z7,z8,z9,p0,x1)
LDR_NOADDR(z10,preg)OA(x1,64)"\n\t"
"                                            \n\t"
"                                            \n\t"
ZERO4VEC(z11,z12,z13,z14)                          // c columns 0-1
" prfm PLDL1KEEP, [x1, #64]                  \n\t"
ZERO4VEC(z15,z16,z17,z18)                          // c columns 2-3
" prfm PLDL1KEEP, [x1, #128]                  \n\t"
ZERO4VEC(z19,z20,z21,z22)                          // c columns 4-5
" prfd pldl1keep,p0, [x0, #2, MUL VL]        \n\t" // prefetch next a vector
ZERO4VEC(z23,z24,z25,z26)                          // c columns 6-7
ZERO2VEC(z27,z28)                                  // c column 8
" prfd pldl1keep,p0, [x0, #3, MUL VL]        \n\t" // prefetch next a vector
"                                            \n\t"
"                                            \n\t"
" cmp x5,#0                                  \n\t" // If k_iter == 0, jump to k_left.
" beq .D2VX9CONSIDERKLEFT                    \n\t"
"                                            \n\t"
" incb x0, ALL, MUL #2                       \n\t" // A = A+vecsize*2
" add x1, x1, #64                            \n\t" // B = B+8*sizeof(double) (keep in mind: additional 1 double offset - we are using 9)
"                                            \n\t"
" cmp x5,1                                   \n\t" // If there is just one k_iter, jump to that one. 
" beq .D2VX9LASTITER                         \n\t" // (as loop is do-while-like).
"                                            \n\t" 
"                                            \n\t"
" .D2VX9LOOP:                                \n\t" // Body
MLA2ROW_LA_LB(z11, z12, z0, z1, z2, p0, z29, x0,0,x1,8)
" prfd pldl1keep,p0, [x0, #2, MUL VL]        \n\t" // prefetch state: this: 3/16, next: 0/16
MLA2ROW_LA_LB(z13, z14, z0, z1, z3, p0, z30, x0,1,x1,16)
" prfm PLDL1KEEP, [x1, #128]                 \n\t" // prefetch state: this: 184/576, next: 0/576
MLA2ROW_LB(z15, z16, z0, z1, z4, p0, x1,24)
" prfd pldl1keep,p0, [x0, #3, MUL VL]        \n\t" // prefetch state: this: 4/16, next: 0/16
MLA2ROW_LB(z17, z18, z0, z1, z5, p0, x1,32)
" prfm PLDL1KEEP, [x1, #192]                 \n\t" // prefetch state: this: 248/576, next: 0/576
MLA2ROW_LB(z19, z20, z0, z1, z6, p0, x1,40)
" prfd pldl1keep,p0, [x0, #4, MUL VL]        \n\t" // prefetch state: this: 5/16, next: 0/16
MLA2ROW_LB(z21, z22, z0, z1, z7, p0, x1,48)
" prfm PLDL1KEEP, [x1, #256]                 \n\t" // prefetch state: this: 312/576, next: 0/576
MLA2ROW_LB(z23, z24, z0, z1, z8, p0, x1,56)
" prfd pldl1keep,p0, [x0, #5, MUL VL]        \n\t" // prefetch state: this: 6/16, next: 0/16
MLA2ROW_LB(z25, z26, z0, z1, z9, p0, x1,64)
" prfm PLDL1KEEP, [x1, #320]                 \n\t" // prefetch state: this: 376/576, next: 0/576
MLA2ROW_LB(z27, z28, z0, z1, z10, p0, x1,72)
" prfd pldl1keep,p0, [x0, #6, MUL VL]        \n\t" // prefetch state: this: 7/16, next: 0/16
"                                            \n\t"
MLA2ROW_LA_LB(z11, z12, z29, z30, z2, p0, z0, x0,2,x1,80) 
" prfm PLDL1KEEP, [x1, #384]                 \n\t" // prefetch state: this: 440/576, next: 0/576
MLA2ROW_LA_LB(z13, z14, z29, z30, z3, p0, z1, x0,3,x1,88) 
" prfd pldl1keep,p0, [x0, #7, MUL VL]        \n\t" // prefetch state: this: 8/16, next: 0/16
MLA2ROW_LB(z15, z16, z29, z30, z4, p0, x1,96)
" prfm PLDL1KEEP, [x1, #448]                 \n\t" // prefetch state: this: 504/576, next: 0/576
MLA2ROW_LB(z17, z18, z29, z30, z5, p0, x1,104)
" prfd pldl1keep,p0, [x0, #8, MUL VL]        \n\t" // prefetch state: this: 9/16, next: 0/16
MLA2ROW_LB(z19, z20, z29, z30, z6, p0, x1,112)
" prfm PLDL1KEEP, [x1, #512]                 \n\t" // prefetch state: this: 568/576, next: 0/576
MLA2ROW_LB(z21, z22, z29, z30, z7, p0, x1,120)
" prfd pldl1keep,p0, [x0, #9, MUL VL]        \n\t" // prefetch state: this: 10/16, next: 0/16
MLA2ROW_LB(z23, z24, z29, z30, z8, p0, x1,128)
" prfm PLDL1KEEP, [x1, #576]                 \n\t" // prefetch state: this: 576/576, next: 56/576
MLA2ROW_LB(z25, z26, z29, z30, z9, p0, x1,136)
" prfd pldl1keep,p0, [x0, #10, MUL VL]       \n\t" // prefetch state: this: 11/16, next: 0/16
MLA2ROW_LB(z27, z28, z29, z30, z10, p0, x1,144)
" incb x0, ALL, MUL #4                       \n\t" // Next 4 A vectors
MLA2ROW_LA_LB(z11, z12, z0, z1, z2, p0, z29, x0,0,x1,152) 
" prfm PLDL1KEEP, [x1, #640]                 \n\t" // prefetch state: this: 576/576, next: 120/576
MLA2ROW_LA_LB(z13, z14, z0, z1, z3, p0, z30, x0,1,x1,160) 
" prfd pldl1keep,p0, [x0, #7, MUL VL]        \n\t" // prefetch state: this: 12/16, next: 0/16
MLA2ROW_LB(z15, z16, z0, z1, z4, p0, x1,168)
" prfd pldl1keep,p0, [x0, #8, MUL VL]        \n\t" // prefetch state: this: 13/16, next: 0/16
MLA2ROW_LB(z17, z18, z0, z1, z5, p0, x1,176)
" prfd pldl1keep,p0, [x0, #9, MUL VL]        \n\t" // prefetch state: this: 14/16, next: 0/16
MLA2ROW_LB(z19, z20, z0, z1, z6, p0, x1,184)
" prfd pldl1keep,p0, [x0, #10, MUL VL]       \n\t" // prefetch state: this: 15/16, next: 0/16
MLA2ROW_LB(z21, z22, z0, z1, z7, p0, x1,192)
" prfd pldl1keep,p0, [x0, #11, MUL VL]       \n\t" // prefetch state: this: 16/16, next: 0/16
MLA2ROW_LB(z23, z24, z0, z1, z8, p0, x1,200)
" prfd pldl1keep,p0, [x0, #12, MUL VL]       \n\t" // prefetch state: this: 16/16, next: 1/16
MLA2ROW_LB(z25, z26, z0, z1, z9, p0, x1,208)
" prfd pldl1keep,p0, [x0, #13, MUL VL]       \n\t" // prefetch state: this: 16/16, next: 2/16
MLA2ROW_LB(z27, z28, z0, z1, z10, p0, x1,216)
"                                            \n\t"
MLA2ROW_LA_LB(z11, z12, z29, z30, z2, p0, z0, x0,2,x1,224) 
MLA2ROW_LA_LB(z13, z14, z29, z30, z3, p0, z1, x0,3,x1,232) 
MLA2ROW_LB(z15, z16, z29, z30, z4, p0, x1,240)
MLA2ROW_LB(z17, z18, z29, z30, z5, p0, x1,248)
MLA2ROW_LB(z19, z20, z29, z30, z6, p0, x1,256)
MLA2ROW_LB(z21, z22, z29, z30, z7, p0, x1,264)
MLA2ROW_LB(z23, z24, z29, z30, z8, p0, x1,272)
MLA2ROW_LB(z25, z26, z29, z30, z9, p0, x1,280)
MLA2ROW_LB(z27, z28, z29, z30, z10, p0, x1,288)
" incb x0, ALL, MUL #4                       \n\t" // Next 4 A vectors
MLA2ROW_LA_LB(z11, z12, z0, z1, z2, p0, z29, x0,0,x1,296) 
MLA2ROW_LA_LB(z13, z14, z0, z1, z3, p0, z30, x0,1,x1,304) 
MLA2ROW_LB(z15, z16, z0, z1, z4, p0, x1,312)
MLA2ROW_LB(z17, z18, z0, z1, z5, p0, x1,320)
MLA2ROW_LB(z19, z20, z0, z1, z6, p0, x1,328)
MLA2ROW_LB(z21, z22, z0, z1, z7, p0, x1,336)
MLA2ROW_LB(z23, z24, z0, z1, z8, p0, x1,344)
MLA2ROW_LB(z25, z26, z0, z1, z9, p0, x1,352)
MLA2ROW_LB(z27, z28, z0, z1, z10, p0, x1,360)
"                                            \n\t"
MLA2ROW_LA_LB(z11, z12, z29, z30, z2, p0, z0, x0,2,x1,368) 
MLA2ROW_LA_LB(z13, z14, z29, z30, z3, p0, z1, x0,3,x1,376) 
MLA2ROW_LB(z15, z16, z29, z30, z4, p0, x1,384)
MLA2ROW_LB(z17, z18, z29, z30, z5, p0, x1,392)
MLA2ROW_LB(z19, z20, z29, z30, z6, p0, x1,400)
MLA2ROW_LB(z21, z22, z29, z30, z7, p0, x1,408)
MLA2ROW_LB(z23, z24, z29, z30, z8, p0, x1,416)
MLA2ROW_LB(z25, z26, z29, z30, z9, p0, x1,424)
MLA2ROW_LB(z27, z28, z29, z30, z10, p0, x1,432)
" incb x0, ALL, MUL #4                       \n\t" // Next 4 A vectors
MLA2ROW_LA_LB(z11, z12, z0, z1, z2, p0, z29, x0,0,x1,440) 
MLA2ROW_LA_LB(z13, z14, z0, z1, z3, p0, z30, x0,1,x1,448) 
MLA2ROW_LB(z15, z16, z0, z1, z4, p0, x1,456)
MLA2ROW_LB(z17, z18, z0, z1, z5, p0, x1,464)
MLA2ROW_LB(z19, z20, z0, z1, z6, p0, x1,472)
MLA2ROW_LB(z21, z22, z0, z1, z7, p0, x1,480)
MLA2ROW_LB(z23, z24, z0, z1, z8, p0, x1,488)
MLA2ROW_LB(z25, z26, z0, z1, z9, p0, x1,496)
MLA2ROW_LB(z27, z28, z0, z1, z10, p0, x1,504)
"                                            \n\t"
MLA2ROW_LA_LB(z11, z12, z29, z30, z2, p0, z0, x0,2,x1,512) 
MLA2ROW_LA_LB(z13, z14, z29, z30, z3, p0, z1, x0,3,x1,520) 
MLA2ROW_LB(z15, z16, z29, z30, z4, p0, x1,528)
MLA2ROW_LB(z17, z18, z29, z30, z5, p0, x1,536)
MLA2ROW_LB(z19, z20, z29, z30, z6, p0, x1,544)
MLA2ROW_LB(z21, z22, z29, z30, z7, p0, x1,552)
MLA2ROW_LB(z23, z24, z29, z30, z8, p0, x1,560)
MLA2ROW_LB(z25, z26, z29, z30, z9, p0, x1,568)
MLA2ROW_LB(z27, z28, z29, z30, z10, p0, x1,576)
" incb x0, ALL, MUL #4                       \n\t" // Next 4 A vectors
" add x1, x1, #576                           \n\t" // B = B+8*9*sizeof(double)
"                                            \n\t"
" sub x5,x5,1                                \n\t" // i-=1
" cmp x5,1                                   \n\t" // Iterate again if we are not in k_iter == 1.
" bne .D2VX9LOOP                             \n\t"
" .D2VX9LASTITER:                            \n\t" // Body
MLA2ROW_LA_LB(z11, z12, z0, z1, z2, p0, z29, x0,0,x1,8)
" prfd pldl1keep,p0, [x0, #2, MUL VL]        \n\t" // prefetch state: this: 3/16, next: 0/16
MLA2ROW_LA_LB(z13, z14, z0, z1, z3, p0, z30, x0,1,x1,16)
" prfm PLDL1KEEP, [x1, #128]                 \n\t" // prefetch state: this: 184/576, next: 0/576
MLA2ROW_LB(z15, z16, z0, z1, z4, p0, x1,24)
" prfd pldl1keep,p0, [x0, #3, MUL VL]        \n\t" // prefetch state: this: 4/16, next: 0/16
MLA2ROW_LB(z17, z18, z0, z1, z5, p0, x1,32)
" prfm PLDL1KEEP, [x1, #192]                 \n\t" // prefetch state: this: 248/576, next: 0/576
MLA2ROW_LB(z19, z20, z0, z1, z6, p0, x1,40)
" prfd pldl1keep,p0, [x0, #4, MUL VL]        \n\t" // prefetch state: this: 5/16, next: 0/16
MLA2ROW_LB(z21, z22, z0, z1, z7, p0, x1,48)
" prfm PLDL1KEEP, [x1, #256]                 \n\t" // prefetch state: this: 312/576, next: 0/576
MLA2ROW_LB(z23, z24, z0, z1, z8, p0, x1,56)
" prfd pldl1keep,p0, [x0, #5, MUL VL]        \n\t" // prefetch state: this: 6/16, next: 0/16
MLA2ROW_LB(z25, z26, z0, z1, z9, p0, x1,64)
" prfm PLDL1KEEP, [x1, #320]                 \n\t" // prefetch state: this: 376/576, next: 0/576
MLA2ROW_LB(z27, z28, z0, z1, z10, p0, x1,72)
" prfd pldl1keep,p0, [x0, #6, MUL VL]        \n\t" // prefetch state: this: 7/16, next: 0/16
"                                            \n\t"
MLA2ROW_LA_LB(z11, z12, z29, z30, z2, p0, z0, x0,2,x1,80) 
" prfm PLDL1KEEP, [x1, #384]                 \n\t" // prefetch state: this: 440/576, next: 0/576
MLA2ROW_LA_LB(z13, z14, z29, z30, z3, p0, z1, x0,3,x1,88) 
" prfd pldl1keep,p0, [x0, #7, MUL VL]        \n\t" // prefetch state: this: 8/16, next: 0/16
MLA2ROW_LB(z15, z16, z29, z30, z4, p0, x1,96)
" prfm PLDL1KEEP, [x1, #448]                 \n\t" // prefetch state: this: 504/576, next: 0/576
MLA2ROW_LB(z17, z18, z29, z30, z5, p0, x1,104)
" prfd pldl1keep,p0, [x0, #8, MUL VL]        \n\t" // prefetch state: this: 9/16, next: 0/16
MLA2ROW_LB(z19, z20, z29, z30, z6, p0, x1,112)
" prfm PLDL1KEEP, [x1, #512]                 \n\t" // prefetch state: this: 568/576, next: 0/576
MLA2ROW_LB(z21, z22, z29, z30, z7, p0, x1,120)
" prfd pldl1keep,p0, [x0, #9, MUL VL]        \n\t" // prefetch state: this: 10/16, next: 0/16
MLA2ROW_LB(z23, z24, z29, z30, z8, p0, x1,128)
// TODO: Look whether prefetching here has strong negative impact on performance
" prfm PLDL1KEEP, [x1, #576]                 \n\t" // prefetch state: this: 576/576, next: 56/576
MLA2ROW_LB(z25, z26, z29, z30, z9, p0, x1,136)
" prfd pldl1keep,p0, [x0, #10, MUL VL]        \n\t" // prefetch state: this: 11/16, next: 0/16
MLA2ROW_LB(z27, z28, z29, z30, z10, p0, x1,144)
" incb x0, ALL, MUL #4                       \n\t" // Next 4 A vectors
MLA2ROW_LA_LB(z11, z12, z0, z1, z2, p0, z29, x0,0,x1,152) 
MLA2ROW_LA_LB(z13, z14, z0, z1, z3, p0, z30, x0,1,x1,160) 
" prfd pldl1keep,p0, [x0, #7, MUL VL]        \n\t" // prefetch state: this: 12/16, next: 0/16
MLA2ROW_LB(z15, z16, z0, z1, z4, p0, x1,168)
" prfd pldl1keep,p0, [x0, #8, MUL VL]        \n\t" // prefetch state: this: 13/16, next: 0/16
MLA2ROW_LB(z17, z18, z0, z1, z5, p0, x1,176)
" prfd pldl1keep,p0, [x0, #9, MUL VL]        \n\t" // prefetch state: this: 14/16, next: 0/16
MLA2ROW_LB(z19, z20, z0, z1, z6, p0, x1,184)
" prfd pldl1keep,p0, [x0, #10, MUL VL]       \n\t" // prefetch state: this: 15/16, next: 0/16
MLA2ROW_LB(z21, z22, z0, z1, z7, p0, x1,192)
" prfd pldl1keep,p0, [x0, #11, MUL VL]       \n\t" // prefetch state: this: 16/16, next: 0/16
MLA2ROW_LB(z23, z24, z0, z1, z8, p0, x1,200)
MLA2ROW_LB(z25, z26, z0, z1, z9, p0, x1,208)
MLA2ROW_LB(z27, z28, z0, z1, z10, p0, x1,216)
"                                            \n\t"
MLA2ROW_LA_LB(z11, z12, z29, z30, z2, p0, z0, x0,2,x1,224) 
MLA2ROW_LA_LB(z13, z14, z29, z30, z3, p0, z1, x0,3,x1,232) 
MLA2ROW_LB(z15, z16, z29, z30, z4, p0, x1,240)
MLA2ROW_LB(z17, z18, z29, z30, z5, p0, x1,248)
MLA2ROW_LB(z19, z20, z29, z30, z6, p0, x1,256)
MLA2ROW_LB(z21, z22, z29, z30, z7, p0, x1,264)
MLA2ROW_LB(z23, z24, z29, z30, z8, p0, x1,272)
MLA2ROW_LB(z25, z26, z29, z30, z9, p0, x1,280)
MLA2ROW_LB(z27, z28, z29, z30, z10, p0, x1,288)
" incb x0, ALL, MUL #4                       \n\t" // Next 4 A vectors
MLA2ROW_LA_LB(z11, z12, z0, z1, z2, p0, z29, x0,0,x1,296) 
MLA2ROW_LA_LB(z13, z14, z0, z1, z3, p0, z30, x0,1,x1,304) 
MLA2ROW_LB(z15, z16, z0, z1, z4, p0, x1,312)
MLA2ROW_LB(z17, z18, z0, z1, z5, p0, x1,320)
MLA2ROW_LB(z19, z20, z0, z1, z6, p0, x1,328)
MLA2ROW_LB(z21, z22, z0, z1, z7, p0, x1,336)
MLA2ROW_LB(z23, z24, z0, z1, z8, p0, x1,344)
MLA2ROW_LB(z25, z26, z0, z1, z9, p0, x1,352)
MLA2ROW_LB(z27, z28, z0, z1, z10, p0, x1,360)
"                                            \n\t"
MLA2ROW_LA_LB(z11, z12, z29, z30, z2, p0, z0, x0,2,x1,368) 
MLA2ROW_LA_LB(z13, z14, z29, z30, z3, p0, z1, x0,3,x1,376) 
MLA2ROW_LB(z15, z16, z29, z30, z4, p0, x1,384)
MLA2ROW_LB(z17, z18, z29, z30, z5, p0, x1,392)
MLA2ROW_LB(z19, z20, z29, z30, z6, p0, x1,400)
MLA2ROW_LB(z21, z22, z29, z30, z7, p0, x1,408)
MLA2ROW_LB(z23, z24, z29, z30, z8, p0, x1,416)
MLA2ROW_LB(z25, z26, z29, z30, z9, p0, x1,424)
MLA2ROW_LB(z27, z28, z29, z30, z10, p0, x1,432)
" incb x0, ALL, MUL #4                       \n\t" // Next 4 A vectors
MLA2ROW_LA_LB(z11, z12, z0, z1, z2, p0, z29, x0,0,x1,440) 
MLA2ROW_LA_LB(z13, z14, z0, z1, z3, p0, z30, x0,1,x1,448) 
MLA2ROW_LB(z15, z16, z0, z1, z4, p0, x1,456)
MLA2ROW_LB(z17, z18, z0, z1, z5, p0, x1,464)
MLA2ROW_LB(z19, z20, z0, z1, z6, p0, x1,472)
MLA2ROW_LB(z21, z22, z0, z1, z7, p0, x1,480)
MLA2ROW_LB(z23, z24, z0, z1, z8, p0, x1,488)
MLA2ROW_LB(z25, z26, z0, z1, z9, p0, x1,496)
MLA2ROW_LB(z27, z28, z0, z1, z10, p0, x1,504)
"                                            \n\t"
MLA2ROW(z11, z12, z29, z30, z2, p0) 
MLA2ROW(z13, z14, z29, z30, z3, p0) 
MLA2ROW(z15, z16, z29, z30, z4, p0)
MLA2ROW(z17, z18, z29, z30, z5, p0)
MLA2ROW(z19, z20, z29, z30, z6, p0)
MLA2ROW(z21, z22, z29, z30, z7, p0)
MLA2ROW(z23, z24, z29, z30, z8, p0)
MLA2ROW(z25, z26, z29, z30, z9, p0)
MLA2ROW(z27, z28, z29, z30, z10, p0)
" incb x0, ALL, MUL #2                       \n\t" // 14 Vectors loaded, 3x4 vecs already added to address
" add x1, x1, #504                           \n\t" // B = B+7*9*sizeof(double)
"                                            \n\t"
" .D2VX9CONSIDERKLEFT:                       \n\t" 
" cmp x6,0                                   \n\t" // If k_left == 0, we are done.
" beq .D2VX9POSTACCUM                        \n\t" // else, we enter the k_left loop.
"                                            \n\t"
".D2VX9LOOPKLEFT:                            \n\t"
"                                            \n\t"
LOAD2VEC(z0,z1,p0,x0)
" incb x0, ALL, MUL #2                       \n\t" // Advance a pointer by 2 vectors
"                                            \n\t"
LOAD8VEC_DIST(z2,z3,z4,z5,z6,z7,z8,z9,p0,x1)
LDR_NOADDR(z10,preg)OA(x1,64)"\n\t"
" add x1, x1, #72                            \n\t" // advance b pointer by 9 doubles
"                                            \n\t"
" sub x6,x6,1                                \n\t"
"                                            \n\t"
MLA2ROW(z11,z12,z0,z1,z2,p0)
MLA2ROW(z13,z14,z0,z1,z3,p0)
MLA2ROW(z15,z16,z0,z1,z4,p0)
MLA2ROW(z17,z18,z0,z1,z5,p0)
MLA2ROW(z19,z20,z0,z1,z6,p0)
MLA2ROW(z21,z22,z0,z1,z7,p0)
MLA2ROW(z23,z24,z0,z1,z8,p0)
MLA2ROW(z25,z26,z0,z1,z9,p0)
MLA2ROW(z27,z28,z0,z1,z10,p0)
"                                            \n\t"
" cmp x6,0                                   \n\t" // Iterate again.
" bne .D2VX9LOOPKLEFT                        \n\t" // if i!=0.
"                                            \n\t"
" .D2VX9POSTACCUM:                           \n\t"
" prfm PLDL2KEEP, [x3]                       \n\t" // prefetch next A address into L2
" prfm PLDL2KEEP, [x4]                       \n\t" // prefetch next B address into L2
"                                            \n\t"
" ld1rd  z29.d, p0/z, [x7]                   \n\t" // Load alpha
" ld1rd  z30.d, p0/z, [x8]                   \n\t" // Load beta
"                                            \n\t"
" cmp x13,#1                                 \n\t" // If rs_c != 1 (column-major)
" bne .D2VX9GENSTORED                        \n\t"
"                                            \n\t"
" .D2VX9COLSTORED:                           \n\t" // C is column-major.
"                                            \n\t"
// Don't use FINC macro, do it by hand because interleaving is required
//FINC_4COL(2VX9,CONT, z0,z1,z2,z3,z4,z5,z6,z7, x2,x20,x21,x22, no,no, 29,30, z10,z11,z12,z13,z14,z15,z16,z17, 1)
//FINC_4COL(2VX9,CONT, z8,z9,z10,z11,z12,z13,z14,z15, x23,x24,x25,x26, no,no, 29,30, z18,z19,z20,z21,z22,z23,z24,z25, 2)   

// Accumulated results are stored in z11-z28
// alpha is in z29, beta in z30
// z0-z10 and z31 are free
// Keep fmas on same registers min. 9 fma instructions apart (A64FX latency)
"                                        \n\t"
" fcmp d30,#0.0                          \n\t"
" beq .D2VX9BETAZEROCONTCOLSTOREDS       \n\t" // multiply with beta if beta isn't zero
"                                        \n\t"
LOAD2VEC (z0,z1,p0,x2)
LOAD2VEC (z2,z3,p0,x20)
LOAD2VEC (z4,z5,p0,x21)
LOAD2VEC (z6,z7,p0,x22)
LOAD2VEC (z8,z9,p0,x23)
LOAD2VEC (z10,z31,p0,x24)
"                                            \n\t"
// This is C=C*beta, then C+=Accum*alpha (Accum being accumulated A*B sum), spacing instructions 9 fmas apart won't work with this
// MUL4ROW(z0,z1,z2,z3,z0,z1,z2,z3,z30,p0)
// MUL4ROW(z4,z5,z6,z7,z4,z5,z6,z7,z30,p0)
// MUL4ROW(z8,z9,z10,z31,z8,z9,z10,z31,z30,p0)

// MLA4ROW(z0,z1,z2,z3,z11,z12,z13,z14,z29,p0)
// MLA4ROW(z4,z5,z6,z7,z15,z16,z17,z18,z29,p0)
// MLA4ROW(z8,z9,z10,z31,z19,z20,z21,z22,z29,p0)

// Let's do Accum=Accum*alpha, then Accum=Accum+C*beta and add some interleaving
MUL4ROW(z11,z12,z13,z14, z11,z12,z13,z14, z29, p0)
MUL4ROW(z15,z16,z17,z18, z15,z16,z17,z18, z29, p0)
MUL4ROW(z19,z20,z21,z22, z19,z20,z21,z22, z29, p0)
MLA4ROW(z11,z12,z13,z14, z0,z1,z2,z3,     z30, p0)
MUL4ROW(z23,z24,z25,z26, z23,z24,z25,z26, z29, p0)
MLA4ROW(z15,z16,z17,z18, z4,z5,z6,z7,     z30, p0)
LOAD2VEC(z0,z1,p0,x25)
STOR2VEC(z11,z12,p0,x2)
MUL2ROW(z27,z28, z27,z28, z29, p0)
LOAD2VEC(z2,z3,p0,x26)
STOR2VEC(z13,z14,p0,x20)
//MLA4ROW(z19,z20,z21,z22, z8,z9,z10,z31, z30, p0) // do 2x2 instead
MLA2ROW(z19,z20, z8,z9, z30, p0)
LOAD2VEC(z4,z5,p0,x27)
STOR2VEC(z15,z16,p0,x21)
MLA2ROW(z21,z22, z10,z31, z30, p0)

MLA4ROW(z23,z24,z25,z26, z0,z1,z2,z3,   z30, p0)
MLA2ROW(z27,z28, z4,z5, z30, p0)

STOR2VEC(z17,z18,p0,x22)
STOR2VEC(z19,z20,p0,x23)
STOR2VEC(z21,z22,p0,x24)
STOR2VEC(z23,z24,p0,x25)
STOR2VEC(z25,z26,p0,x26)
STOR2VEC(z27,z28,p0,x27)

" b .D2VX9END                                \n\t" // Duplicate code for stores required due to lack of registers
"                                            \n\t"
" .D2VX9BETAZEROCONTCOLSTOREDS:              \n\t"
// No need to zero anything as we are storing the scaled accumulated A*B values
MUL4ROW(z11,z12,z13,z14, z11,z12,z13,z14, z29, p0)
MUL4ROW(z15,z16,z17,z18, z15,z16,z17,z18, z29, p0)
MUL4ROW(z19,z20,z21,z22, z19,z20,z21,z22, z29, p0)
STOR2VEC(z11,z12,p0,x2)
MUL4ROW(z23,z24,z25,z26, z23,z24,z25,z26, z29, p0)
STOR2VEC(z13,z14,p0,x20)
MUL2ROW(z27,z28, z27,z28, z29, p0)
STOR2VEC(z15,z16,p0,x21)
STOR2VEC(z17,z18,p0,x22)
STOR2VEC(z19,z20,p0,x23)
STOR2VEC(z21,z22,p0,x24)
STOR2VEC(z23,z24,p0,x25)
STOR2VEC(z25,z26,p0,x26)
STOR2VEC(z27,z28,p0,x27)
"                                            \n\t"
"                                            \n\t"
" b .D2VX9END                                \n\t"
"                                            \n\t"
" .D2VX9GENSTORED:                           \n\t" // C is general-stride stored.
"                                            \n\t" // Creating index for stride load&store access
" index z10.d, xzr, x13                       \n\t" // 0, stride*double, 2*stride*double, ...
" mul x3, x13, x11                           \n\t" // x3 <- stride*double*vecsize
" index z31.d, x3, x13                        \n\t" // stride*double*vecsize, (vecsize+1)*stride*double, (vecsize+2)*stride*double, ...
"                                            \n\t"
// Additional restriction: z10 and z31 are used for the general stride index
"                                        \n\t"
" fcmp d30,#0.0                          \n\t"
" beq .D2VX9BETAZEROGENICOLSTOREDS       \n\t" // multiply with beta if beta isn't zero
"                                        \n\t"
LOAD2VEC_GENI (z0,z1,p0,x2, z10, z31)
LOAD2VEC_GENI (z2,z3,p0,x20, z10, z31)
LOAD2VEC_GENI (z4,z5,p0,x21, z10, z31)
LOAD2VEC_GENI (z6,z7,p0,x22, z10, z31)
LOAD2VEC_GENI (z8,z9,p0,x23, z10, z31)
"                                            \n\t"

// Let's do Accum=Accum*alpha, then Accum=Accum+C*beta and add some interleaving
MUL4ROW(z11,z12,z13,z14, z11,z12,z13,z14, z29, p0)
MUL4ROW(z15,z16,z17,z18, z15,z16,z17,z18, z29, p0)
MUL4ROW(z19,z20,z21,z22, z19,z20,z21,z22, z29, p0)
MLA4ROW(z11,z12,z13,z14, z0,z1,z2,z3,     z30, p0)
MUL4ROW(z23,z24,z25,z26, z23,z24,z25,z26, z29, p0)
MLA4ROW(z15,z16,z17,z18, z4,z5,z6,z7,     z30, p0)
LOAD2VEC_GENI(z0,z1,p0,x24, z10, z31)    // Starting to reuse registers for C here
STOR2VEC_GENI(z11,z12,p0,x2, z10, z31)
MUL2ROW(z27,z28, z27,z28, z29, p0)
LOAD2VEC_GENI(z2,z3,p0,x25, z10, z31)
STOR2VEC_GENI(z13,z14,p0,x20, z10, z31)
MLA2ROW(z19,z20, z8,z9, z30, p0)
LOAD2VEC_GENI(z4,z5,p0,x26, z10, z31)
STOR2VEC_GENI(z15,z16,p0,x21, z10, z31)
MLA2ROW(z21,z22, z0,z1, z30, p0)
LOAD2VEC_GENI(z6,z7,p0,x27, z10, z31)
MLA4ROW(z23,z24,z25,z26, z2,z3,z4,z5,   z30, p0)
MLA2ROW(z27,z28, z6,z7, z30, p0)

STOR2VEC_GENI(z17,z18,p0,x22, z10, z31)
STOR2VEC_GENI(z19,z20,p0,x23, z10, z31)
STOR2VEC_GENI(z21,z22,p0,x24, z10, z31)
STOR2VEC_GENI(z23,z24,p0,x25, z10, z31)
STOR2VEC_GENI(z25,z26,p0,x26, z10, z31)
STOR2VEC_GENI(z27,z28,p0,x27, z10, z31)

" b .D2VX9END                                \n\t" // Duplicate code for stores required due to lack of registers
"                                            \n\t"
" .D2VX9BETAZEROGENICOLSTOREDS:              \n\t"
// No need to zero anything as we are storing the scaled accumulated A*B values
MUL4ROW(z11,z12,z13,z14, z11,z12,z13,z14, z29, p0)
MUL4ROW(z15,z16,z17,z18, z15,z16,z17,z18, z29, p0)
MUL4ROW(z19,z20,z21,z22, z19,z20,z21,z22, z29, p0)
STOR2VEC_GENI(z11,z12,p0,x2, z10, z31)
MUL4ROW(z23,z24,z25,z26, z23,z24,z25,z26, z29, p0)
STOR2VEC_GENI(z13,z14,p0,x20, z10, z31)
MUL2ROW(z27,z28, z27,z28, z29, p0)
STOR2VEC_GENI(z15,z16,p0,x21, z10, z31)
STOR2VEC_GENI(z17,z18,p0,x22, z10, z31)
STOR2VEC_GENI(z19,z20,p0,x23, z10, z31)
STOR2VEC_GENI(z21,z22,p0,x24, z10, z31)
STOR2VEC_GENI(z23,z24,p0,x25, z10, z31)
STOR2VEC_GENI(z25,z26,p0,x26, z10, z31)
STOR2VEC_GENI(z27,z28,p0,x27, z10, z31)
"                                            \n\t"
" .D2VX9END:                                 \n\t" // Done!
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
 "x20","x21","x22","x23","x24","x25","x26","x27",
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
