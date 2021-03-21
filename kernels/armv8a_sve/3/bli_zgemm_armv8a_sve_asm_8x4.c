/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2019, Forschunszentrum Juelich

   Author(s): Bine Brank

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


/*
   o 8x4 Double precision complex micro-kernel 
   o Runnable on ARMv8+SVE, compiled with aarch64 GCC.
   o Use it together with the armv8_sve BLIS configuration.
   o Tested on Juawei arm nodes with ARMIE emulator.
   o Only for vector size of 512

 * tests still need to be done to check performance
*/



void bli_zgemm_armv8a_sve_asm_8x4
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
	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;
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
" ldr x5,%[k_iter]                           \n\t" // Init guard (k_iter)
" ldr x6,%[k_left]                           \n\t" // Init guard (k_iter)
"                                            \n\t" 
"                                            \n\t" 
" ldr x10,%[cs_c]                             \n\t" // Load cs_c
" lsl x10,x10,#4                              \n\t" // cs_c * sizeof(complex double)
"                                            \n\t"
" ldr x13,%[rs_c]                            \n\t" // Load rs_c.
//" lsl x14,x13,#3                             \n\t" // rs_c * sizeof(double). 
"                                            \n\t"
" mov x11, #8                                \n\t"
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
LD_AVEC_D(z0, z1, p0, x0)
"                                            \n\t"
LDR_BVEC_D(z2, z3, p0, x1, 0, 8)
LDR_BVEC_D(z4, z5, p0, x1, 16, 24)
LDR_BVEC_D(z6, z7, p0, x1, 32, 40)
LDR_BVEC_D(z8, z9, p0, x1, 48, 56)
"                                            \n\t"
"                                            \n\t"
" dup z12.d, #0                              \n\t" // Vector for accummulating column 0
" dup z13.d, #0                              \n\t" // Vector for accummulating column 0
" dup z14.d, #0                              \n\t" // Vector for accummulating column 1
" prfm PLDL1KEEP, [x1, #64]                  \n\t"
" dup z15.d, #0                              \n\t" // Vector for accummulating column 1
" prfm PLDL1KEEP, [x1, #128]                  \n\t"
" dup z16.d, #0                              \n\t" // Vector for accummulating column 2
" prfm PLDL1KEEP, [x1, #192]                  \n\t"
" dup z17.d, #0                              \n\t" // Vector for accummulating column 2
" prfm PLDL1KEEP, [x1, #256]                  \n\t"
" dup z18.d, #0                              \n\t" // Vector for accummulating column 3
" prfm PLDL1KEEP, [x0, #128]                  \n\t"
" dup z19.d, #0                              \n\t" // Vector for accummulating column 3
" prfm PLDL1KEEP, [x0, #192]                  \n\t"
"                                            \n\t"
" prfm PLDL1KEEP, [x0, #256]                  \n\t"
"                                            \n\t"
" prfm PLDL1KEEP, [x0, #320]                  \n\t"
" prfm PLDL1KEEP, [x0, #384]                  \n\t"
" prfm PLDL1KEEP, [x0, #448]                  \n\t"
" prfm PLDL1KEEP, [x0, #512]                  \n\t"
" prfm PLDL1KEEP, [x0, #576]                  \n\t"
"                                            \n\t"
" cmp x5,#0                                  \n\t" // If k_iter == 0, jump to k_left.
" beq .C256CONSIDERKLEFT                     \n\t"
"                                            \n\t"
" add x0, x0, #128                            \n\t" //update address of A
" add x1, x1, #64                            \n\t" //update address of B
"                                            \n\t"
" cmp x5,1                                   \n\t" // If there is just one k_iter, jump to that one. 
" beq .C256LASTITER                          \n\t" // (as loop is do-while-like).
"                                            \n\t"
" D256LOOP:                                  \n\t" // Body
"                                            \n\t"
MLA1ROW_Z(z12, z13, z0, z1, z2, z3, p0)
LDR_BVEC_D(z2, z3, p0, x1, 0, 8)
"                                            \n\t"
MLA1ROW_Z(z14, z15, z0, z1, z4, z5, p0)
LDR_BVEC_D(z4, z5, p0, x1, 16, 24)
"                                            \n\t"
MLA1ROW_Z(z16, z17, z0, z1, z6, z7, p0)
LDR_BVEC_D(z6, z7, p0, x1, 32, 40)
"                                            \n\t"
MLA1ROW_Z(z18, z19, z0, z1, z8, z9, p0)
LDR_BVEC_D(z8, z9, p0, x1, 48, 56)
" prfm PLDL1KEEP, [x1, #256]                  \n\t"
" add x1, x1, #64                            \n\t" //update address of B
"                                            \n\t"
"                                            \n\t"
LD_AVEC_D(z0, z1, p0, x0)
" prfm PLDL1KEEP, [x0, #512]                  \n\t"
" prfm PLDL1KEEP, [x0, #576]                  \n\t"
" add x0, x0, #128                            \n\t" //update address of A
"                                            \n\t"	//End it 1.
"                                            \n\t"
MLA1ROW_Z(z12, z13, z0, z1, z2, z3, p0)
LDR_BVEC_D(z2, z3, p0, x1, 0, 8)
"                                            \n\t"
MLA1ROW_Z(z14, z15, z0, z1, z4, z5, p0)
LDR_BVEC_D(z4, z5, p0, x1, 16, 24)
"                                            \n\t"
MLA1ROW_Z(z16, z17, z0, z1, z6, z7, p0)
LDR_BVEC_D(z6, z7, p0, x1, 32, 40)
"                                            \n\t"
MLA1ROW_Z(z18, z19, z0, z1, z8, z9, p0)
LDR_BVEC_D(z8, z9, p0, x1, 48, 56)
" prfm PLDL1KEEP, [x1, #256]                  \n\t"
" add x1, x1, #64                            \n\t" //update address of B
"                                            \n\t"
"                                            \n\t"
LD_AVEC_D(z0, z1, p0, x0)
" prfm PLDL1KEEP, [x0, #512]                  \n\t"
" prfm PLDL1KEEP, [x0, #576]                  \n\t"
" add x0, x0, #128                            \n\t" //update address of A
"                                            \n\t"	//End it 2.
"                                            \n\t"
"                                            \n\t"
MLA1ROW_Z(z12, z13, z0, z1, z2, z3, p0)
LDR_BVEC_D(z2, z3, p0, x1, 0, 8)
"                                            \n\t"
MLA1ROW_Z(z14, z15, z0, z1, z4, z5, p0)
LDR_BVEC_D(z4, z5, p0, x1, 16, 24)
"                                            \n\t"
MLA1ROW_Z(z16, z17, z0, z1, z6, z7, p0)
LDR_BVEC_D(z6, z7, p0, x1, 32, 40)
"                                            \n\t"
MLA1ROW_Z(z18, z19, z0, z1, z8, z9, p0)
LDR_BVEC_D(z8, z9, p0, x1, 48, 56)
" prfm PLDL1KEEP, [x1, #256]                  \n\t"
" add x1, x1, #64                            \n\t" //update address of B
"                                            \n\t"
"                                            \n\t"
LD_AVEC_D(z0, z1, p0, x0)
" prfm PLDL1KEEP, [x0, #512]                  \n\t"
" prfm PLDL1KEEP, [x0, #576]                  \n\t"
" add x0, x0, #128                            \n\t" //update address of A
"                                            \n\t"	//End it 3.
"                                            \n\t"
MLA1ROW_Z(z12, z13, z0, z1, z2, z3, p0)
LDR_BVEC_D(z2, z3, p0, x1, 0, 8)
"                                            \n\t"
MLA1ROW_Z(z14, z15, z0, z1, z4, z5, p0)
LDR_BVEC_D(z4, z5, p0, x1, 16, 24)
"                                            \n\t"
MLA1ROW_Z(z16, z17, z0, z1, z6, z7, p0)
LDR_BVEC_D(z6, z7, p0, x1, 32, 40)
"                                            \n\t"
MLA1ROW_Z(z18, z19, z0, z1, z8, z9, p0)
LDR_BVEC_D(z8, z9, p0, x1, 48, 56)
" prfm PLDL1KEEP, [x1, #256]                  \n\t"
" add x1, x1, #64                            \n\t" //update address of B
"                                            \n\t"
"                                            \n\t"
LD_AVEC_D(z0, z1, p0, x0)
" prfm PLDL1KEEP, [x0, #512]                  \n\t"
" prfm PLDL1KEEP, [x0, #576]                  \n\t"
" add x0, x0, #128                            \n\t" //update address of A
"                                            \n\t"
"                                            \n\t"	//End it 4.
"                                            \n\t"
"                                            \n\t"
" sub x5,x5,1                                \n\t" // i-=1
" cmp x5,1                                   \n\t" // Iterate again if we are not in k_iter == 1.
" bne D256LOOP                               \n\t"
"                                            \n\t"
" .C256LASTITER:                             \n\t"
"                                            \n\t"
"                                            \n\t"
MLA1ROW_Z(z12, z13, z0, z1, z2, z3, p0)
LDR_BVEC_D(z2, z3, p0, x1, 0, 8)
MLA1ROW_Z(z14, z15, z0, z1, z4, z5, p0)
LDR_BVEC_D(z4, z5, p0, x1, 16, 24)
MLA1ROW_Z(z16, z17, z0, z1, z6, z7, p0)
LDR_BVEC_D(z6, z7, p0, x1, 32, 40)
MLA1ROW_Z(z18, z19, z0, z1, z8, z9, p0)
LDR_BVEC_D(z8, z9, p0, x1, 48, 56)
" prfm PLDL1KEEP, [x1, #256]                  \n\t"
" add x1, x1, #64                            \n\t" //update address of B
"                                            \n\t"
LD_AVEC_D(z0, z1, p0, x0)
" prfm PLDL1KEEP, [x0, #512]                  \n\t"
" prfm PLDL1KEEP, [x0, #576]                  \n\t"
" add x0, x0, #128                            \n\t" // incremenenting by 64  
"                                            \n\t"	//End it 1.
"                                            \n\t"
"                                            \n\t"
MLA1ROW_Z(z12, z13, z0, z1, z2, z3, p0)
LDR_BVEC_D(z2, z3, p0, x1, 0, 8)
MLA1ROW_Z(z14, z15, z0, z1, z4, z5, p0)
LDR_BVEC_D(z4, z5, p0, x1, 16, 24)
MLA1ROW_Z(z16, z17, z0, z1, z6, z7, p0)
LDR_BVEC_D(z6, z7, p0, x1, 32, 40)
MLA1ROW_Z(z18, z19, z0, z1, z8, z9, p0)
LDR_BVEC_D(z8, z9, p0, x1, 48, 56)
" add x1, x1, #64                            \n\t" //update address of B
"                                            \n\t"
LD_AVEC_D(z0, z1, p0, x0)
" add x0, x0, #128                            \n\t" // incremenenting by 64  
"                                            \n\t"	//End it 2.
"                                            \n\t"
"                                            \n\t"
MLA1ROW_Z(z12, z13, z0, z1, z2, z3, p0)
LDR_BVEC_D(z2, z3, p0, x1, 0, 8)
MLA1ROW_Z(z14, z15, z0, z1, z4, z5, p0)
LDR_BVEC_D(z4, z5, p0, x1, 16, 24)
MLA1ROW_Z(z16, z17, z0, z1, z6, z7, p0)
LDR_BVEC_D(z6, z7, p0, x1, 32, 40)
MLA1ROW_Z(z18, z19, z0, z1, z8, z9, p0)
LDR_BVEC_D(z8, z9, p0, x1, 48, 56)
" add x1, x1, #64                            \n\t" //update address of B
"                                            \n\t"
LD_AVEC_D(z0, z1, p0, x0)
" add x0, x0, #128                            \n\t" // incremenenting by 64  
"                                            \n\t"	//End it 3.
"                                            \n\t"
MLA1ROW_Z(z12, z13, z0, z1, z2, z3, p0)
MLA1ROW_Z(z14, z15, z0, z1, z4, z5, p0)
MLA1ROW_Z(z16, z17, z0, z1, z6, z7, p0)
MLA1ROW_Z(z18, z19, z0, z1, z8, z9, p0)
"                                            \n\t"
"                                            \n\t"	//End it 4.
"                                            \n\t"
"                                            \n\t"
" .C256CONSIDERKLEFT:                        \n\t" 
" cmp x6,0                                   \n\t" // If k_left == 0, we are done.
" beq .C256POSTACCUM                         \n\t" // else, we enter the k_left loop.
"                                            \n\t"
".C256LOOPKLEFT:                             \n\t"
"                                            \n\t"
LD_AVEC_D(z0, z1, p0, x0)
" prfm PLDL1KEEP, [x0, #128]                  \n\t"
" prfm PLDL1KEEP, [x0, #192]                  \n\t"
" add x0, x0, #128                            \n\t"
"                                            \n\t"
LDR_BVEC_D(z2, z3, p0, x1, 0, 8)
LDR_BVEC_D(z4, z5, p0, x1, 16, 24)
LDR_BVEC_D(z6, z7, p0, x1, 32, 40)
LDR_BVEC_D(z8, z9, p0, x1, 48, 56)
" prfm PLDL1KEEP, [x1, #64]                  \n\t"
" add x1, x1, #64                            \n\t"
"                                            \n\t"
" sub x6,x6,1                                \n\t"
"                                            \n\t"
MLA1ROW_Z(z12, z13, z0, z1, z2, z3, p0)
MLA1ROW_Z(z14, z15, z0, z1, z4, z5, p0)
MLA1ROW_Z(z16, z17, z0, z1, z6, z7, p0)
MLA1ROW_Z(z18, z19, z0, z1, z8, z9, p0)
"                                            \n\t"
"                                            \n\t"
" cmp x6,0                                   \n\t" // Iterate again.
" bne .C256LOOPKLEFT                         \n\t" // if i!=0.
"                                            \n\t"
" .C256POSTACCUM:                            \n\t"
"                                            \n\t"
" ldr x0,%[alpha]                            \n\t" // Alpha address      
" ldr x1,%[beta]                             \n\t" // Beta address      
"                                            \n\t"
LDR_BVEC_D(z8, z9, p0, x0, 0, 8)                     // Load alpha
LDR_BVEC_D(z10, z11, p0, x1, 0, 8)                   // Load beta
"                                            \n\t"
" cmp x13,#1                                 \n\t" // If rs_c != 1 (column-major)
" bne .C256GENSTORED                         \n\t"
"                                            \n\t"
" .C256COLSTORED:                            \n\t" // C is column-major.
"                                            \n\t"
" dup z0.d, #0                               \n\t" // for loading C
" dup z1.d, #0                               \n\t" 
" dup z2.d, #0                               \n\t" 
" dup z3.d, #0                               \n\t" 
" dup z4.d, #0                               \n\t" 
" dup z5.d, #0                               \n\t" 
" dup z6.d, #0                               \n\t" 
" dup z7.d, #0                               \n\t" 
"                                            \n\t"
" dup z20.d, #0                              \n\t" // for zeros because of no fcmul instruction
" dup z21.d, #0                              \n\t" 
" dup z22.d, #0                              \n\t" 
" dup z23.d, #0                              \n\t" 
" dup z24.d, #0                              \n\t" 
" dup z25.d, #0                              \n\t" 
" dup z26.d, #0                              \n\t" 
" dup z27.d, #0                              \n\t" 
"                                            \n\t"
CMPCZB_D(z10, z11, ".C256BETAZEROCOLSTOREDS1")
"                                            \n\t"
LD_AVEC_D(z0, z1, p0, x2)  // Load column 0 of C
LD_AVEC_D(z2, z3, p0, x20) // Load column 1 of C
LD_AVEC_D(z4, z5, p0, x21) // Load column 2 of C
LD_AVEC_D(z6, z7, p0, x22) // Load column 3 of C
"                                            \n\t"
"                                            \n\t" // Scale by beta
MLA1ROW_Z(z20, z21, z0, z1, z10, z11, p0)
MLA1ROW_Z(z22, z23, z2, z3, z10, z11, p0)
MLA1ROW_Z(z24, z25, z4, z5, z10, z11, p0)
MLA1ROW_Z(z26, z27, z6, z7, z10, z11, p0)
"                                            \n\t"
" .C256BETAZEROCOLSTOREDS1:                  \n\t"
"                                            \n\t"
MLA1ROW_Z(z20, z21, z12, z13, z8, z9, p0)
MLA1ROW_Z(z22, z23, z14, z15, z8, z9, p0)
MLA1ROW_Z(z24, z25, z16, z17, z8, z9, p0)
MLA1ROW_Z(z26, z27, z18, z19, z8, z9, p0)
"                                            \n\t"
ST_AVEC_D(z20, z21, p0, x2)  // Store column 0 of C
ST_AVEC_D(z22, z23, p0, x20) // Store column 1 of C
ST_AVEC_D(z24, z25, p0, x21) // Store column 2 of C
ST_AVEC_D(z26, z27, p0, x22) // Store column 3 of C
"                                            \n\t"
"                                            \n\t"
" b .C256END                                 \n\t"
"                                            \n\t"
" .C256GENSTORED:                            \n\t" // C is general-stride stored.
"                                            \n\t"
MKINDC_2VEC_D(z28, z29, x13, x11, x3)
" dup z0.d, #0                               \n\t" // for loading C
" dup z1.d, #0                               \n\t" 
" dup z2.d, #0                               \n\t" 
" dup z3.d, #0                               \n\t" 
" dup z4.d, #0                               \n\t" 
" dup z5.d, #0                               \n\t" 
" dup z6.d, #0                               \n\t" 
" dup z7.d, #0                               \n\t" 
"                                            \n\t"
" dup z20.d, #0                              \n\t" // for zeros because of no fcmul instruction
" dup z21.d, #0                              \n\t" 
" dup z22.d, #0                              \n\t" 
" dup z23.d, #0                              \n\t" 
" dup z24.d, #0                              \n\t" 
" dup z25.d, #0                              \n\t" 
" dup z26.d, #0                              \n\t" 
" dup z27.d, #0                              \n\t" 
"                                            \n\t"
CMPCZB_D(z10, z11, ".C256BETAZEROGENSTOREDS1")
"                                            \n\t"
" ld1d {z0.d}, p0/z, [x2, z28.d, LSL #3]     \n\t" // Load column 0 of C
" ld1d {z1.d}, p0/z, [x2, z29.d, LSL #3]     \n\t"
"                                            \n\t"
" ld1d {z2.d}, p0/z, [x20, z28.d, LSL #3]    \n\t" // Load column 1 of C
" ld1d {z3.d}, p0/z, [x20, z29.d, LSL #3]    \n\t"
"                                            \n\t"
" ld1d {z4.d}, p0/z, [x21, z28.d, LSL #3]    \n\t" // Load column 2 of C
" ld1d {z5.d}, p0/z, [x21, z29.d, LSL #3]    \n\t"
"                                            \n\t"
" ld1d {z6.d}, p0/z, [x22, z28.d, LSL #3]    \n\t" // Load column 3 of C
" ld1d {z7.d}, p0/z, [x22, z29.d, LSL #3]    \n\t"
"                                            \n\t"
MLA1ROW_Z(z20, z21, z0, z1, z10, z11, p0)
MLA1ROW_Z(z22, z23, z2, z3, z10, z11, p0)
MLA1ROW_Z(z24, z25, z4, z5, z10, z11, p0)
MLA1ROW_Z(z26, z27, z6, z7, z10, z11, p0)
"                                            \n\t"
" .C256BETAZEROGENSTOREDS1:                  \n\t"
"                                            \n\t"
MLA1ROW_Z(z20, z21, z12, z13, z8, z9, p0)
MLA1ROW_Z(z22, z23, z14, z15, z8, z9, p0)
MLA1ROW_Z(z24, z25, z16, z17, z8, z9, p0)
MLA1ROW_Z(z26, z27, z18, z19, z8, z9, p0)
"                                            \n\t"
" st1d {z20.d}, p0, [x2, z28.d, LSL #3]      \n\t" // Store column 0 of C
" st1d {z21.d}, p0, [x2, z29.d, LSL #3]      \n\t"
"                                            \n\t"
" st1d {z22.d}, p0, [x20, z28.d, LSL #3]     \n\t" // Store column 1 of C
" st1d {z23.d}, p0, [x20, z29.d, LSL #3]     \n\t"
"                                            \n\t"
" st1d {z24.d}, p0, [x21, z28.d, LSL #3]     \n\t" // Store column 2 of C
" st1d {z25.d}, p0, [x21, z29.d, LSL #3]     \n\t"
"                                            \n\t"
" st1d {z26.d}, p0, [x22, z28.d, LSL #3]     \n\t" // Store column 3 of C
" st1d {z27.d}, p0, [x22, z29.d, LSL #3]     \n\t"
"                                            \n\t"
" .C256END:                                  \n\t" // Done!
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
 "x10","x11","x13",
 "x20","x21","x22","x23","x24","x25","x26",
 "x27", "x28",       
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


