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
#include "bli_sve_finc_d.h"

#if defined(DEBUG)
// seen a compiler use the stack instead when it's inside the function
void* debug_cache;
#endif

/* 2 vectors in m_r, n_r = 8
*/
void bli_zgemm_armv8a_sve_asm_2vx4
     (
       dim_t               k0,
       dcomplex*    restrict alpha,
       dcomplex*    restrict a,
       dcomplex*    restrict b,
       dcomplex*    restrict beta,
       dcomplex*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	void* a_next = bli_auxinfo_next_a( data );
	void* b_next = bli_auxinfo_next_b( data );

#if defined(DEBUG)
    debug_cache=malloc(256);
    memset(debug_cache,0,256);
#endif
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
#if defined(DEBUG)
INASM_START_TRACE
#endif
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
" ldr x10,%[cs_c]                            \n\t" // Load cs_c
" lsl x10,x10,#4                             \n\t" // cs_c * sizeof(dcomplex)
"                                            \n\t"
" ldr x13,%[rs_c]                            \n\t" // Load rs_c.
//" lsl x14,x13,#4                             \n\t" // rs_c * sizeof(dcomplex). 
"                                            \n\t"
" mov x11,#0                                 \n\t"
" incd x11                                   \n\t"
"                                            \n\t"
" add x20,x2,x10                             \n\t" //Load address Column 1 of C
" add x21,x20,x10                            \n\t" //Load address Column 2 of C
" add x22,x21,x10                            \n\t" //Load address Column 3 of C
"                                            \n\t"
" prfm pldl1keep,[x2]                        \n\t" // Prefetch c.
" prfm pldl1keep,[x20]                       \n\t" // Prefetch c.
" prfm pldl1keep,[x21]                       \n\t" // Prefetch c.
" prfm pldl1keep,[x22]                       \n\t" // Prefetch c.
"                                            \n\t"
" ptrue p0.d                                 \n\t" // Creating all true predicate
"                                            \n\t"
LOAD2VEC_Z(z0,z1,p0,x0)
"                                            \n\t"
LOAD8VEC_DIST_Z(z2,z3,z4,z5,z6,z7,z8,z9,p0,x1)
"                                            \n\t"
"                                            \n\t"
ZERO2VEC_D(z10,z11)                                  // c column 0
" prfm PLDL1KEEP, [x1, #64]                  \n\t"
ZERO2VEC_D(z12,z13)                                  // c column 1
" prfm PLDL1KEEP, [x1, #128]                  \n\t"
ZERO2VEC_D(z14,z15)                                  // c column 2
PFL1(x0, p0, 2)                                    // prefetch next a vector
ZERO2VEC_D(z16,z17)                                  // c column 3
PFL1(x0, p0, 3)
"                                            \n\t"
"                                            \n\t"
" cmp x5,#0                                  \n\t" // If k_iter == 0, jump to k_left.
" beq .Z2VX4CONSIDERKLEFT                    \n\t"
"                                            \n\t"
" incb x0, ALL, MUL #2                       \n\t" // A = A+vecsize*2
" add x1, x1, #64                            \n\t" // B = B+8*sizeof(double)
"                                            \n\t"
" cmp x5,1                                   \n\t" // If there is just one k_iter, jump to that one. 
" beq .Z2VX4LASTITER                         \n\t" // (as loop is do-while-like).
"                                            \n\t" 
"                                            \n\t"
" .Z2VX4LOOP:                                \n\t" // Body
MLA1ROW_ILV_LA_LB_Z(z10, z11, z0, z1, z2, z3, p0, PFL1(x0, p0, 2), PFL1(x0, p0, 3), PFL1(x0, p0, 4), PFL1(x0, p0, 5), z26, z27, x0,0,1, x1,0,8) 
MLA1ROW_ILV_LB_Z   (z12, z13, z0, z1, z4, z5, p0, PFL1(x0, p0, 6), PFL1(x0, p0, 7), PFL1(x0, p0, 8), PFL1(x0, p0, 9), x1,16,24) 
MLA1ROW_LB_Z       (z14, z15, z0, z1, z6, z7, p0, x1,32,40)
MLA1ROW_LB_Z       (z16, z17, z0, z1, z8, z9, p0, x1,48,56)
"                                            \n\t"
" add x1, x1, #64                            \n\t" // B = B+4*sizeof(dcomplex)
MLA1ROW_LA_LB_Z(z10, z11, z26, z27, z2, z3, p0, z0, z1, x0, 2,3, x1,0,8) 
" prfm PLDL1KEEP, [x1, #64]                  \n\t"
MLA1ROW_LB_Z   (z12, z13, z26, z27, z4, z5, p0, x1,16,24) 
" prfm PLDL1KEEP, [x1, #128]                 \n\t"
MLA1ROW_LB_Z   (z14, z15, z26, z27, z6, z7, p0, x1,32,40)
" prfm PLDL1KEEP, [x1, #192]                 \n\t"
MLA1ROW_LB_Z   (z16, z17, z26, z27, z8, z9, p0, x1,48,56)
" prfm PLDL1KEEP, [x1, #256]                 \n\t"

" incb x0, ALL, MUL #4                       \n\t" // Next 4 A vectors
" add x1, x1, #64                            \n\t" // B = B+4*sizeof(dcomplex)
MLA1ROW_LA_LB_Z(z10, z11, z0, z1, z2, z3, p0, z26, z27, x0,0,1, x1,0,8) 
MLA1ROW_LB_Z   (z12, z13, z0, z1, z4, z5, p0, x1,16,24) 
MLA1ROW_LB_Z   (z14, z15, z0, z1, z6, z7, p0, x1,32,40)
MLA1ROW_LB_Z   (z16, z17, z0, z1, z8, z9, p0, x1,48,56)
"                                            \n\t"
" add x1, x1, #64                            \n\t" // B = B+4*sizeof(dcomplex)
MLA1ROW_LA_LB_Z(z10, z11, z26, z27, z2, z3, p0, z0, z1, x0,2,3, x1,0,8) 
MLA1ROW_LB_Z   (z12, z13, z26, z27, z4, z5, p0, x1,16,24) 
MLA1ROW_LB_Z   (z14, z15, z26, z27, z6, z7, p0, x1,32,40)
MLA1ROW_LB_Z   (z16, z17, z26, z27, z8, z9, p0, x1,48,56)
" incb x0, ALL, MUL #4                       \n\t" // Next 4 A vectors
" add x1, x1, #64                           \n\t" // B = B+4*sizeof(dcomplex)
"                                            \n\t"
" sub x5,x5,1                                \n\t" // i-=1
" cmp x5,1                                   \n\t" // Iterate again if we are not in k_iter == 1.
" bne .Z2VX4LOOP                             \n\t"
" .Z2VX4LASTITER:                            \n\t" // Body
MLA1ROW_ILV_LA_LB_Z(z10, z11, z0, z1, z2, z3, p0, PFL1(x0, p0, 2), PFL1(x0, p0, 3), PFL1(x0, p0, 4), PFL1(x0, p0, 5), z26, z27, x0,0,1, x1,0,8) 
MLA1ROW_LB_Z       (z12, z13, z0, z1, z4, z5, p0, x1,16,24) 
MLA1ROW_LB_Z       (z14, z15, z0, z1, z6, z7, p0, x1,32,40)
MLA1ROW_LB_Z       (z16, z17, z0, z1, z8, z9, p0, x1,48,56)
"                                            \n\t"
" add x1, x1, #64                            \n\t" // B = B+8*sizeof(double)
MLA1ROW_LA_LB_Z(z10, z11, z26, z27, z2, z3, p0, z0, z1, x0,2,3, x1,0,8) 
MLA1ROW_LB_Z   (z12, z13, z26, z27, z4, z5, p0, x1,16,24) 
MLA1ROW_LB_Z   (z14, z15, z26, z27, z6, z7, p0, x1,32,40)
MLA1ROW_LB_Z   (z16, z17, z26, z27, z8, z9, p0, x1,48,56)
" add x1, x1, #64                            \n\t" // B = B+8*sizeof(double)
MLA1ROW_LA_LB_Z(z10, z11, z0, z1, z2, z3, p0, z26, z27, x0,4,5, x1,0,8) 
MLA1ROW_LB_Z   (z12, z13, z0, z1, z4, z5, p0, x1,16,24) 
MLA1ROW_LB_Z   (z14, z15, z0, z1, z6, z7, p0, x1,32,40)
MLA1ROW_LB_Z   (z16, z17, z0, z1, z8, z9, p0, x1,48,56)
"                                            \n\t"
MLA1ROW_Z(z10, z11, z26, z27, z2, z3, p0) 
MLA1ROW_Z(z12, z13, z26, z27, z4, z5, p0) 
MLA1ROW_Z(z14, z15, z26, z27, z6, z7, p0)
MLA1ROW_Z(z16, z17, z26, z27, z8, z9, p0)
" incb x0, ALL, MUL #6                       \n\t" // 6 Vectors loaded
" add x1, x1, #64                           \n\t" // B = B+8*sizeof(double)
"                                            \n\t"
" .Z2VX4CONSIDERKLEFT:                       \n\t" 
" cmp x6,0                                   \n\t" // If k_left == 0, we are done.
" beq .Z2VX4POSTACCUM                        \n\t" // else, we enter the k_left loop.
"                                            \n\t"
".Z2VX4LOOPKLEFT:                            \n\t"
"                                            \n\t"
LOAD2VEC_Z(z0,z1,p0,x0)
" incb x0, ALL, MUL #2                       \n\t" // Advance a pointer by 2 vectors
"                                            \n\t"
LOAD8VEC_DIST_Z(z2,z3,z4,z5,z6,z7,z8,z9,p0,x1)
" add x1, x1, #64                            \n\t" // advance b pointer by 8 doubles
"                                            \n\t"
" sub x6,x6,1                                \n\t"
"                                            \n\t"
MLA1ROW_Z(z10, z11, z0, z1, z2, z3, p0) 
MLA1ROW_Z(z12, z13, z0, z1, z4, z5, p0) 
MLA1ROW_Z(z14, z15, z0, z1, z6, z7, p0)
MLA1ROW_Z(z16, z17, z0, z1, z8, z9, p0)
"                                            \n\t"
" cmp x6,0                                   \n\t" // Iterate again.
" bne .Z2VX4LOOPKLEFT                        \n\t" // if i!=0.
"                                            \n\t"
" .Z2VX4POSTACCUM:                           \n\t"
" prfm PLDL2KEEP, [x3]                       \n\t"
" prfm PLDL2KEEP, [x4]                       \n\t"
"                                            \n\t"
#if defined(USE_SVE_CMLA_INSTRUCTION)
" ld1rqd  {z26.d}, p0/z, [x7]                \n\t" // Load alpha
" ld1rqd  {z28.d}, p0/z, [x8]                \n\t" // Load beta
#else
" ld1rd  z26.d, p0/z, [x7]                   \n\t" // Load alpha
" ld1rd  z27.d, p0/z, [x7,#8]                \n\t" // Load alpha
" ld1rd  z28.d, p0/z, [x8]                   \n\t" // Load beta
" ld1rd  z29.d, p0/z, [x8,#8]                \n\t" // Load beta
#endif
"                                            \n\t"
" cmp x13,#1                                 \n\t" // If rs_c != 1 (column-major)
" bne .Z2VX4GENSTORED                        \n\t"
"                                            \n\t"
" .Z2VX4COLSTORED:                           \n\t" // C is column-major.
"                                            \n\t"
CFINC_4COL(2VX4,CONT_Z, z0,z1,z2,z3,z4,z5,z6,z7, z18,z19,z20,z21,z22,z23,z24,z25, x2,x20,x21,x22, z8,z9, 26,27,28,29, z10,z11,z12,z13,z14,z15,z16,z17, 1)
"                                            \n\t"
" b .Z2VX4END                                \n\t"
"                                            \n\t"
" .Z2VX4GENSTORED:                           \n\t" // C is general-stride stored.
"                                            \n\t" // Creating index for stride load&store access
MKINDC_2VEC_D(z8,z9,x13,x11,x3)
"                                            \n\t"
CFINC_4COL(2VX4,GENI_Z, z0,z1,z2,z3,z4,z5,z6,z7, z18,z19,z20,z21,z22,z23,z24,z25, x2,x20,x21,x22, z8,z9, 26,27,28,29, z10,z11,z12,z13,z14,z15,z16,z17, 1)
"                                            \n\t"
" .Z2VX4END:                                 \n\t" // Done!
"                                            \n\t"
#if defined(DEBUG)
INASM_STOP_TRACE
#endif
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
#if defined(DEBUG)
 [b_next] "m" (b_next), // 9
 [debug_cache] "m" (debug_cache) // 10
#else
 [b_next] "m" (b_next) // 9
#endif
:// Register clobber list
 "x0","x1","x2","x3",
 "x4","x5","x6","x7",
 "x8","x10","x11","x13",
 "x20","x21","x22",
 "z0","z1","z2","z3",
 "z4","z5","z6","z7",
 "z8", "z9","z10","z11",
 "z12","z13","z14", "z15",
 "z16","z17","z18","z19",
 "z20","z21","z22","z23",
 "z24","z25","z26","z27",
 "z28","z29","z30","z31", 
 "p0","p1","p2"
);
#if defined(DEBUG)
free(debug_cache);
#endif

}
