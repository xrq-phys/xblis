    /* 

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

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
   o 12x6 Double precision micro-kernel 
   o Runnable on ARMv8+SVE, compiled with aarch64 GCC.
   o Use it together with the armv8_sve BLIS configuration.
   o Tested on Juawei arm nodes with ARMIE emulator.
   o Due to the fact that mr and nr depend on size of the vector
     registers, this kernel only works when size=256!!

   February 2019.
   Bine Brank 

 * tests still need to be done to check performance
*/
void bli_dgemm_armv8a_sve256bits_asm_12x6
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
" ldr x7,%[alpha]                            \n\t" // Alpha address      
" ldr x8,%[beta]                             \n\t" // Beta address      
"                                            \n\t" 
" ldr x9,%[cs_c]                             \n\t" // Load cs_c
" lsl x10,x9,#3                              \n\t" // cs_c * sizeof(double)
"                                            \n\t"
" ldr x13,%[rs_c]                            \n\t" // Load rs_c.
" lsl x14,x13,#3                             \n\t" // rs_c * sizeof(double). 
"                                            \n\t"
" mov x11, #4                                \n\t"
" mov x12, #8                                \n\t"
"                                            \n\t"
" add x20,x2,x10                             \n\t" //Load address Column 1 of C
" add x21,x20,x10                            \n\t" //Load address Column 2 of C
" add x22,x21,x10                            \n\t" //Load address Column 3 of C
" add x23,x22,x10                            \n\t" //Load address Column 4 of C
" add x24,x23,x10                            \n\t" //Load address Column 5 of C
"                                            \n\t"
" prfm pldl1keep,[x2]                        \n\t" // Prefetch c.
" prfm pldl1keep,[x20]                       \n\t" // Prefetch c.
" prfm pldl1keep,[x21]                       \n\t" // Prefetch c.
" prfm pldl1keep,[x22]                       \n\t" // Prefetch c.
" prfm pldl1keep,[x23]                       \n\t" // Prefetch c.
" prfm pldl1keep,[x24]                       \n\t" // Prefetch c.
"                                            \n\t"
" whilelt p0.d, xzr, x11                     \n\t" // Creating all true predicate
"                                            \n\t"
" ld1d  z0.d, p0/z, [x0]                     \n\t" // Load a
" ld1d  z1.d, p0/z, [x0, #1, MUL VL]         \n\t"
" ld1d  z2.d, p0/z, [x0, #2, MUL VL]         \n\t"
"                                            \n\t"
" ld1rd  z3.d, p0/z, [x1]                    \n\t" // Load b
" ld1rd  z4.d, p0/z, [x1, #8]                \n\t"
" ld1rd  z5.d, p0/z, [x1, #16]               \n\t"
" ld1rd  z6.d, p0/z, [x1, #24]               \n\t"
" ld1rd  z7.d, p0/z, [x1, #32]               \n\t"
" ld1rd  z8.d, p0/z, [x1, #40]               \n\t"
"                                            \n\t"
"                                            \n\t"
" dup z14.d, #0                              \n\t" // Vector for accummulating column 0
" dup z15.d, #0                              \n\t" // Vector for accummulating column 0
" prfm PLDL1KEEP, [x1, #192]                 \n\t"
" dup z16.d, #0                              \n\t" // Vector for accummulating column 0
" prfm PLDL1KEEP, [x1, #256]                 \n\t"
"                                            \n\t"
" dup z17.d, #0                              \n\t" // Vector for accummulating column 1
" prfm PLDL1KEEP, [x1, #320]                 \n\t"
" dup z18.d, #0                              \n\t" // Vector for accummulating column 1
" dup z19.d, #0                              \n\t" // Vector for accummulating column 1
" prfm PLDL1KEEP, [x0, #384]                 \n\t"
"                                            \n\t"
" dup z20.d, #0                              \n\t" // Vector for accummulating column 2
" prfm PLDL1KEEP, [x0, #448]                 \n\t"
" dup z21.d, #0                              \n\t" // Vector for accummulating column 2
" prfm PLDL1KEEP, [x0, #512]                 \n\t"
" dup z22.d, #0                              \n\t" // Vector for accummulating column 2
"                                            \n\t"
" dup z23.d, #0                              \n\t" // Vector for accummulating column 3
" prfm PLDL1KEEP, [x0, #576]                 \n\t"
" dup z24.d, #0                              \n\t" // Vector for accummulating column 3
" prfm PLDL1KEEP, [x0, #640]                 \n\t"
" dup z25.d, #0                              \n\t" // Vector for accummulating column 3
" prfm PLDL1KEEP, [x0, #704]                 \n\t"
"                                            \n\t"
" dup z26.d, #0                              \n\t" // Vector for accummulating column 4
" dup z27.d, #0                              \n\t" // Vector for accummulating column 4
" dup z28.d, #0                              \n\t" // Vector for accummulating column 4
"                                            \n\t"
" dup z29.d, #0                              \n\t" // Vector for accummulating column 5
" dup z30.d, #0                              \n\t" // Vector for accummulating column 5
" dup z31.d, #0                              \n\t" // Vector for accummulating column 5
"                                            \n\t"
" cmp x5,#0                                  \n\t" // If k_iter == 0, jump to k_left.
" beq .DCONSIDERKLEFT                        \n\t"
"                                            \n\t"
" add x0, x0, #96                            \n\t" //update address of A
" add x1, x1, #48                            \n\t" //update address of B
"                                            \n\t"
" cmp x5,1                                   \n\t" // If there is just one k_iter, jump to that one. 
" beq .DLASTITER                             \n\t" // (as loop is do-while-like).
"                                            \n\t"
" DLOOP:                                     \n\t" // Body
"                                            \n\t"
" fmla z14.d, p0/m, z0.d, z3.d               \n\t"
" prfm PLDL1KEEP, [x0, #672]                 \n\t" // 704 + 64 - 96 = 672
" fmla z15.d, p0/m, z1.d, z3.d               \n\t"
" prfm PLDL1KEEP, [x0, #736]                 \n\t" 
" fmla z16.d, p0/m, z2.d, z3.d               \n\t"
" prfm PLDL1KEEP, [x0, #800]                 \n\t" 
" ld1d   z9.d, p0/z, [x0]                    \n\t" 
" ld1rd  z3.d, p0/z, [x1]                    \n\t"
"                                            \n\t"
" fmla z17.d, p0/m, z0.d, z4.d               \n\t"
" prfm PLDL1KEEP, [x0, #864]                 \n\t" 
" fmla z18.d, p0/m, z1.d, z4.d               \n\t"
" prfm PLDL1KEEP, [x0, #928]                 \n\t" 
" fmla z19.d, p0/m, z2.d, z4.d               \n\t"
" prfm PLDL1KEEP, [x0, #992]                 \n\t" 
" ld1d   z10.d, p0/z, [x0, #1, MUL VL]       \n\t" 
" ld1rd  z4.d, p0/z, [x1, #8]                \n\t"
"                                            \n\t"
" fmla z20.d, p0/m, z0.d, z5.d               \n\t"
" fmla z21.d, p0/m, z1.d, z5.d               \n\t"
" fmla z22.d, p0/m, z2.d, z5.d               \n\t"
" ld1d   z11.d, p0/z, [x0, #2, MUL VL]       \n\t" 
" ld1rd  z5.d, p0/z, [x1, #16]               \n\t"
"                                            \n\t"
" fmla z23.d, p0/m, z0.d, z6.d               \n\t"
" fmla z24.d, p0/m, z1.d, z6.d               \n\t"
" fmla z25.d, p0/m, z2.d, z6.d               \n\t"
" ld1rd  z6.d, p0/z, [x1, #24]               \n\t"
"                                            \n\t"
" fmla z26.d, p0/m, z0.d, z7.d               \n\t"
" fmla z27.d, p0/m, z1.d, z7.d               \n\t"
" fmla z28.d, p0/m, z2.d, z7.d               \n\t"
" ld1rd  z7.d, p0/z, [x1, #32]               \n\t"
"                                            \n\t"
" fmla z29.d, p0/m, z0.d, z8.d               \n\t"
" fmla z30.d, p0/m, z1.d, z8.d               \n\t"
" fmla z31.d, p0/m, z2.d, z8.d               \n\t"
" ld1rd  z8.d, p0/z, [x1, #40]               \n\t"
"                                            \n\t"	//End it 1.
"                                            \n\t"
" fmla z14.d, p0/m, z9.d, z3.d               \n\t"
" fmla z15.d, p0/m, z10.d, z3.d              \n\t"
" fmla z16.d, p0/m, z11.d, z3.d              \n\t"
" ld1d   z0.d, p0/z, [x0, #3, MUL VL]        \n\t" 
" ld1rd  z3.d, p0/z, [x1, #48]               \n\t"
"                                            \n\t"
" fmla z17.d, p0/m, z9.d, z4.d               \n\t"
" fmla z18.d, p0/m, z10.d, z4.d              \n\t"
" fmla z19.d, p0/m, z11.d, z4.d              \n\t"
" ld1d   z1.d, p0/z, [x0, #4, MUL VL]        \n\t" 
" ld1rd  z4.d, p0/z, [x1, #56]               \n\t"
"                                            \n\t"
" fmla z20.d, p0/m, z9.d, z5.d               \n\t"
" fmla z21.d, p0/m, z10.d, z5.d              \n\t"
" fmla z22.d, p0/m, z11.d, z5.d              \n\t"
" prfm PLDL1KEEP, [x1, #336]                 \n\t"
" ld1d   z2.d, p0/z, [x0, #5, MUL VL]        \n\t" 
" prfm PLDL1KEEP, [x1, #400]                 \n\t"
" ld1rd  z5.d, p0/z, [x1, #64]               \n\t"
" prfm PLDL1KEEP, [x1, #464]                 \n\t"
"                                            \n\t"
" fmla z23.d, p0/m, z9.d, z6.d               \n\t"
" fmla z24.d, p0/m, z10.d, z6.d              \n\t"
" fmla z25.d, p0/m, z11.d, z6.d              \n\t"
" ld1rd  z6.d, p0/z, [x1, #72]               \n\t"
"                                            \n\t"
" fmla z26.d, p0/m, z9.d, z7.d               \n\t"
" fmla z27.d, p0/m, z10.d, z7.d              \n\t"
" fmla z28.d, p0/m, z11.d, z7.d              \n\t"
" ld1rd  z7.d, p0/z, [x1, #80]               \n\t"
"                                            \n\t"
" fmla z29.d, p0/m, z9.d, z8.d               \n\t"
" fmla z30.d, p0/m, z10.d, z8.d              \n\t"
" fmla z31.d, p0/m, z11.d, z8.d              \n\t"
" ld1rd  z8.d, p0/z, [x1, #88]               \n\t"
"                                            \n\t"	//End it 2.
"                                            \n\t"
" add x0, x0, #192                           \n\t" // Increasing x0 by 384/2
"                                            \n\t"
" fmla z14.d, p0/m, z0.d, z3.d               \n\t"
" fmla z15.d, p0/m, z1.d, z3.d               \n\t"
" fmla z16.d, p0/m, z2.d, z3.d               \n\t"
" ld1d   z9.d, p0/z, [x0]                    \n\t" 
" ld1rd  z3.d, p0/z, [x1, #96]               \n\t"
"                                            \n\t"
" fmla z17.d, p0/m, z0.d, z4.d               \n\t"
" fmla z18.d, p0/m, z1.d, z4.d               \n\t"
" fmla z19.d, p0/m, z2.d, z4.d               \n\t"
" ld1d   z10.d, p0/z, [x0, #1, MUL VL]       \n\t" 
" ld1rd  z4.d, p0/z, [x1, #104]              \n\t"
"                                            \n\t"
" fmla z20.d, p0/m, z0.d, z5.d               \n\t"
" fmla z21.d, p0/m, z1.d, z5.d               \n\t"
" fmla z22.d, p0/m, z2.d, z5.d               \n\t"
" ld1d   z11.d, p0/z, [x0, #2, MUL VL]       \n\t" 
" ld1rd  z5.d, p0/z, [x1, #112]              \n\t"
"                                            \n\t"
" fmla z23.d, p0/m, z0.d, z6.d               \n\t"
" fmla z24.d, p0/m, z1.d, z6.d               \n\t"
" fmla z25.d, p0/m, z2.d, z6.d               \n\t"
" ld1rd  z6.d, p0/z, [x1, #120]              \n\t"
"                                            \n\t"
" fmla z26.d, p0/m, z0.d, z7.d               \n\t"
" fmla z27.d, p0/m, z1.d, z7.d               \n\t"
" fmla z28.d, p0/m, z2.d, z7.d               \n\t"
" ld1rd  z7.d, p0/z, [x1, #128]              \n\t"
"                                            \n\t"
" fmla z29.d, p0/m, z0.d, z8.d               \n\t"
" fmla z30.d, p0/m, z1.d, z8.d               \n\t"
" fmla z31.d, p0/m, z2.d, z8.d               \n\t"
" ld1rd  z8.d, p0/z, [x1, #136]              \n\t"
"                                            \n\t"	//End it 3.
"                                            \n\t"
" fmla z14.d, p0/m, z9.d, z3.d               \n\t"
" fmla z15.d, p0/m, z10.d, z3.d              \n\t"
" fmla z16.d, p0/m, z11.d, z3.d              \n\t"
" ld1d   z0.d, p0/z, [x0, #3, MUL VL]        \n\t" 
" ld1rd  z3.d, p0/z, [x1, #144]              \n\t"
"                                            \n\t"
" fmla z17.d, p0/m, z9.d, z4.d               \n\t"
" fmla z18.d, p0/m, z10.d, z4.d              \n\t"
" fmla z19.d, p0/m, z11.d, z4.d              \n\t"
" ld1d   z1.d, p0/z, [x0, #4, MUL VL]        \n\t" 
" ld1rd  z4.d, p0/z, [x1, #152]              \n\t"
"                                            \n\t"
" fmla z20.d, p0/m, z9.d, z5.d               \n\t"
" fmla z21.d, p0/m, z10.d, z5.d              \n\t"
" fmla z22.d, p0/m, z11.d, z5.d              \n\t"
" ld1d   z2.d, p0/z, [x0, #5, MUL VL]        \n\t" 
" ld1rd  z5.d, p0/z, [x1, #160]              \n\t"
"                                            \n\t"
" fmla z23.d, p0/m, z9.d, z6.d               \n\t"
" fmla z24.d, p0/m, z10.d, z6.d              \n\t"
" fmla z25.d, p0/m, z11.d, z6.d              \n\t"
" ld1rd  z6.d, p0/z, [x1, #168]              \n\t"
"                                            \n\t"
" fmla z26.d, p0/m, z9.d, z7.d               \n\t"
" fmla z27.d, p0/m, z10.d, z7.d              \n\t"
" fmla z28.d, p0/m, z11.d, z7.d              \n\t"
" ld1rd  z7.d, p0/z, [x1, #176]              \n\t"
"                                            \n\t"
" fmla z29.d, p0/m, z9.d, z8.d               \n\t"
" fmla z30.d, p0/m, z10.d, z8.d              \n\t"
" fmla z31.d, p0/m, z11.d, z8.d              \n\t"
" ld1rd  z8.d, p0/z, [x1, #184]              \n\t"
"                                            \n\t"	//End it 4.
"                                            \n\t"
" add x0, x0, #192                           \n\t" // incremenenting by 384/2 
" add x1, x1, #192                           \n\t"
"                                            \n\t"
" sub x5,x5,1                                \n\t" // i-=1
" cmp x5,1                                   \n\t" // Iterate again if we are not in k_iter == 1.
" bne DLOOP                                  \n\t"
"                                            \n\t"
".DLASTITER:                                 \n\t"
"                                            \n\t"
" fmla z14.d, p0/m, z0.d, z3.d               \n\t"
" fmla z15.d, p0/m, z1.d, z3.d               \n\t"
" fmla z16.d, p0/m, z2.d, z3.d               \n\t"
" ld1d   z9.d, p0/z, [x0]                    \n\t" 
" ld1rd  z3.d, p0/z, [x1]                    \n\t"
"                                            \n\t"
" fmla z17.d, p0/m, z0.d, z4.d               \n\t"
" fmla z18.d, p0/m, z1.d, z4.d               \n\t"
" fmla z19.d, p0/m, z2.d, z4.d               \n\t"
" ld1d   z10.d, p0/z, [x0, #1, MUL VL]       \n\t" 
" ld1rd  z4.d, p0/z, [x1, #8]                \n\t"
"                                            \n\t"
" fmla z20.d, p0/m, z0.d, z5.d               \n\t"
" fmla z21.d, p0/m, z1.d, z5.d               \n\t"
" fmla z22.d, p0/m, z2.d, z5.d               \n\t"
" ld1d   z11.d, p0/z, [x0, #2, MUL VL]       \n\t" 
" ld1rd  z5.d, p0/z, [x1, #16]               \n\t"
"                                            \n\t"
" fmla z23.d, p0/m, z0.d, z6.d               \n\t"
" fmla z24.d, p0/m, z1.d, z6.d               \n\t"
" fmla z25.d, p0/m, z2.d, z6.d               \n\t"
" ld1rd  z6.d, p0/z, [x1, #24]               \n\t"
"                                            \n\t"
" fmla z26.d, p0/m, z0.d, z7.d               \n\t"
" fmla z27.d, p0/m, z1.d, z7.d               \n\t"
" fmla z28.d, p0/m, z2.d, z7.d               \n\t"
" ld1rd  z7.d, p0/z, [x1, #32]               \n\t"
"                                            \n\t"
" fmla z29.d, p0/m, z0.d, z8.d               \n\t"
" fmla z30.d, p0/m, z1.d, z8.d               \n\t"
" fmla z31.d, p0/m, z2.d, z8.d               \n\t"
" ld1rd  z8.d, p0/z, [x1, #40]               \n\t"
"                                            \n\t"	//End it 1.
"                                            \n\t"
" fmla z14.d, p0/m, z9.d, z3.d               \n\t"
" fmla z15.d, p0/m, z10.d, z3.d              \n\t"
" fmla z16.d, p0/m, z11.d, z3.d              \n\t"
" ld1d   z0.d, p0/z, [x0, #3, MUL VL]        \n\t" 
" ld1rd  z3.d, p0/z, [x1, #48]               \n\t"
"                                            \n\t"
" fmla z17.d, p0/m, z9.d, z4.d               \n\t"
" fmla z18.d, p0/m, z10.d, z4.d              \n\t"
" fmla z19.d, p0/m, z11.d, z4.d              \n\t"
" ld1d   z1.d, p0/z, [x0, #4, MUL VL]        \n\t" 
" ld1rd  z4.d, p0/z, [x1, #56]               \n\t"
"                                            \n\t"
" fmla z20.d, p0/m, z9.d, z5.d               \n\t"
" fmla z21.d, p0/m, z10.d, z5.d              \n\t"
" fmla z22.d, p0/m, z11.d, z5.d              \n\t"
" ld1d   z2.d, p0/z, [x0, #5, MUL VL]        \n\t" 
" ld1rd  z5.d, p0/z, [x1, #64]               \n\t"
"                                            \n\t"
" fmla z23.d, p0/m, z9.d, z6.d               \n\t"
" fmla z24.d, p0/m, z10.d, z6.d              \n\t"
" fmla z25.d, p0/m, z11.d, z6.d              \n\t"
" ld1rd  z6.d, p0/z, [x1, #72]               \n\t"
"                                            \n\t"
" fmla z26.d, p0/m, z9.d, z7.d               \n\t"
" fmla z27.d, p0/m, z10.d, z7.d              \n\t"
" fmla z28.d, p0/m, z11.d, z7.d              \n\t"
" ld1rd  z7.d, p0/z, [x1, #80]               \n\t"
"                                            \n\t"
" fmla z29.d, p0/m, z9.d, z8.d               \n\t"
" fmla z30.d, p0/m, z10.d, z8.d              \n\t"
" fmla z31.d, p0/m, z11.d, z8.d              \n\t"
" ld1rd  z8.d, p0/z, [x1, #88]               \n\t"
"                                            \n\t"	//End it 2.
" add x0, x0, #192                           \n\t"
"                                            \n\t"
" fmla z14.d, p0/m, z0.d, z3.d               \n\t"
" fmla z15.d, p0/m, z1.d, z3.d               \n\t"
" fmla z16.d, p0/m, z2.d, z3.d               \n\t"
" ld1d   z9.d, p0/z, [x0]                    \n\t" 
" ld1rd  z3.d, p0/z, [x1, #96]               \n\t"
"                                            \n\t"
" fmla z17.d, p0/m, z0.d, z4.d               \n\t"
" fmla z18.d, p0/m, z1.d, z4.d               \n\t"
" fmla z19.d, p0/m, z2.d, z4.d               \n\t"
" ld1d   z10.d, p0/z, [x0, #1, MUL VL]       \n\t" 
" ld1rd  z4.d, p0/z, [x1, #104]              \n\t"
"                                            \n\t"
" fmla z20.d, p0/m, z0.d, z5.d               \n\t"
" fmla z21.d, p0/m, z1.d, z5.d               \n\t"
" fmla z22.d, p0/m, z2.d, z5.d               \n\t"
" ld1d   z11.d, p0/z, [x0, #2, MUL VL]       \n\t" 
" ld1rd  z5.d, p0/z, [x1, #112]              \n\t"
"                                            \n\t"
" fmla z23.d, p0/m, z0.d, z6.d               \n\t"
" fmla z24.d, p0/m, z1.d, z6.d               \n\t"
" fmla z25.d, p0/m, z2.d, z6.d               \n\t"
" ld1rd  z6.d, p0/z, [x1, #120]              \n\t"
"                                            \n\t"
" fmla z26.d, p0/m, z0.d, z7.d               \n\t"
" fmla z27.d, p0/m, z1.d, z7.d               \n\t"
" fmla z28.d, p0/m, z2.d, z7.d               \n\t"
" ld1rd  z7.d, p0/z, [x1, #128]              \n\t"
"                                            \n\t"
" fmla z29.d, p0/m, z0.d, z8.d               \n\t"
" fmla z30.d, p0/m, z1.d, z8.d               \n\t"
" fmla z31.d, p0/m, z2.d, z8.d               \n\t"
" ld1rd  z8.d, p0/z, [x1, #136]              \n\t"
"                                            \n\t"	//End it 3.
"                                            \n\t"
" fmla z14.d, p0/m, z9.d, z3.d               \n\t"
" fmla z15.d, p0/m, z10.d, z3.d              \n\t"
" fmla z16.d, p0/m, z11.d, z3.d              \n\t"
"                                            \n\t"
" fmla z17.d, p0/m, z9.d, z4.d               \n\t"
" fmla z18.d, p0/m, z10.d, z4.d              \n\t"
" fmla z19.d, p0/m, z11.d, z4.d              \n\t"
"                                            \n\t"
" fmla z20.d, p0/m, z9.d, z5.d               \n\t"
" fmla z21.d, p0/m, z10.d, z5.d              \n\t"
" fmla z22.d, p0/m, z11.d, z5.d              \n\t"
"                                            \n\t"
" fmla z23.d, p0/m, z9.d, z6.d               \n\t"
" fmla z24.d, p0/m, z10.d, z6.d              \n\t"
" fmla z25.d, p0/m, z11.d, z6.d              \n\t"
"                                            \n\t"
" add x0, x0, #96                            \n\t"
"                                            \n\t"
" fmla z26.d, p0/m, z9.d, z7.d               \n\t"
" fmla z27.d, p0/m, z10.d, z7.d              \n\t"
" fmla z28.d, p0/m, z11.d, z7.d              \n\t"
"                                            \n\t"
" fmla z29.d, p0/m, z9.d, z8.d               \n\t"
" fmla z30.d, p0/m, z10.d, z8.d              \n\t"
" fmla z31.d, p0/m, z11.d, z8.d              \n\t"
"                                            \n\t"	//End it 4.
"                                            \n\t"
" add x1, x1, #144                           \n\t"
"                                            \n\t"
" .DCONSIDERKLEFT:                           \n\t" 
" cmp x6,0                                   \n\t" // If k_left == 0, we are done.
" beq .DPOSTACCUM                            \n\t" // else, we enter the k_left loop.
"                                            \n\t"
".DLOOPKLEFT:                                \n\t"
"                                            \n\t"
" ld1d  z0.d, p0/z, [x0]                     \n\t" // Load a
" ld1d  z1.d, p0/z, [x0, #1, MUL VL]         \n\t"
" ld1d  z2.d, p0/z, [x0, #2, MUL VL]         \n\t"
" add x0, x0, #96                            \n\t"
"                                            \n\t"
" ld1rd  z3.d, p0/z, [x1]                    \n\t" // Load b
" ld1rd  z4.d, p0/z, [x1, #8]                \n\t"
" ld1rd  z5.d, p0/z, [x1, #16]               \n\t"
" ld1rd  z6.d, p0/z, [x1, #24]               \n\t"
" ld1rd  z7.d, p0/z, [x1, #32]               \n\t"
" ld1rd  z8.d, p0/z, [x1, #40]               \n\t"
" add x1, x1, #48                            \n\t"
"                                            \n\t"
" sub x6,x6,1                                \n\t"
"                                            \n\t"
" fmla z14.d, p0/m, z0.d, z3.d               \n\t"
" fmla z15.d, p0/m, z1.d, z3.d               \n\t"
" fmla z16.d, p0/m, z2.d, z3.d               \n\t"
"                                            \n\t"
" fmla z17.d, p0/m, z0.d, z4.d               \n\t"
" fmla z18.d, p0/m, z1.d, z4.d               \n\t"
" fmla z19.d, p0/m, z2.d, z4.d               \n\t"
"                                            \n\t"
" fmla z20.d, p0/m, z0.d, z5.d               \n\t"
" fmla z21.d, p0/m, z1.d, z5.d               \n\t"
" fmla z22.d, p0/m, z2.d, z5.d               \n\t"
"                                            \n\t"
" fmla z23.d, p0/m, z0.d, z6.d               \n\t"
" fmla z24.d, p0/m, z1.d, z6.d               \n\t"
" fmla z25.d, p0/m, z2.d, z6.d               \n\t"
"                                            \n\t"
" fmla z26.d, p0/m, z0.d, z7.d               \n\t"
" fmla z27.d, p0/m, z1.d, z7.d               \n\t"
" fmla z28.d, p0/m, z2.d, z7.d               \n\t"
"                                            \n\t"
" fmla z29.d, p0/m, z0.d, z8.d               \n\t"
" fmla z30.d, p0/m, z1.d, z8.d               \n\t"
" fmla z31.d, p0/m, z2.d, z8.d               \n\t"
"                                            \n\t"
" cmp x6,0                                   \n\t" // Iterate again.
" bne .DLOOPKLEFT                            \n\t" // if i!=0.
"                                            \n\t"
" .DPOSTACCUM:                               \n\t"
"                                            \n\t"
" ld1rd  z6.d, p0/z, [x7]                    \n\t" // Load alpha
" ld1rd  z7.d, p0/z, [x8]                    \n\t" // Load beta
"                                            \n\t"
" cmp x13,#1                                 \n\t" // If rs_c != 1 (column-major)
" bne .DGENSTORED                            \n\t"
"                                            \n\t"
" .DCOLSTORED:                               \n\t" // C is column-major.
"                                            \n\t"
" dup z0.d, #0                               \n\t" 
" dup z1.d, #0                               \n\t" 
" dup z2.d, #0                               \n\t" 
" dup z3.d, #0                               \n\t" 
" dup z4.d, #0                               \n\t" 
" dup z5.d, #0                               \n\t" 
"                                            \n\t"
" fcmp d7,#0.0                               \n\t"
" beq .DBETAZEROCOLSTOREDS1                  \n\t" // Taking care of the beta==0 case.
"                                            \n\t"
" ld1d  {z0.d}, p0/z, [x2]                   \n\t" // Load column 0 of C
" ld1d  {z1.d}, p0/z, [x2, #1, MUL VL]       \n\t"
" ld1d  {z2.d}, p0/z, [x2, #2, MUL VL]       \n\t"
"                                            \n\t"
" ld1d  {z3.d}, p0/z, [x20]                  \n\t" // Load column 1 of C
" ld1d  {z4.d}, p0/z, [x20, #1, MUL VL]      \n\t"
" ld1d  {z5.d}, p0/z, [x20, #2, MUL VL]      \n\t"
"                                            \n\t"
" fmul z0.d, p0/m, z0.d, z7.d                \n\t" // Scale by beta
" fmul z1.d, p0/m, z1.d, z7.d                \n\t" // Scale by beta
" fmul z2.d, p0/m, z2.d, z7.d                \n\t" // Scale by beta
" fmul z3.d, p0/m, z3.d, z7.d                \n\t" // Scale by beta
" fmul z4.d, p0/m, z4.d, z7.d                \n\t" // Scale by beta
" fmul z5.d, p0/m, z5.d, z7.d                \n\t" // Scale by beta
"                                            \n\t"
" .DBETAZEROCOLSTOREDS1:                     \n\t"
"                                            \n\t"
" fmla z0.d, p0/m, z14.d, z6.d               \n\t" // Scale by alpha
" fmla z1.d, p0/m, z15.d, z6.d               \n\t" // Scale by alpha
" fmla z2.d, p0/m, z16.d, z6.d               \n\t" // Scale by alpha
" fmla z3.d, p0/m, z17.d, z6.d               \n\t" // Scale by alpha
" fmla z4.d, p0/m, z18.d, z6.d               \n\t" // Scale by alpha
" fmla z5.d, p0/m, z19.d, z6.d               \n\t" // Scale by alpha
"                                            \n\t"
" st1d  {z0.d}, p0, [x2]                     \n\t" // Store column 0 of C
" st1d  {z1.d}, p0, [x2, #1, MUL VL]         \n\t"
" st1d  {z2.d}, p0, [x2, #2, MUL VL]         \n\t"
"                                            \n\t"
" st1d  {z3.d}, p0, [x20]                    \n\t" // Store column 1 of C
" st1d  {z4.d}, p0, [x20, #1, MUL VL]        \n\t"
" st1d  {z5.d}, p0, [x20, #2, MUL VL]        \n\t"
"                                            \n\t"
" dup z8.d, #0                               \n\t" 
" dup z9.d, #0                               \n\t" 
" dup z10.d, #0                              \n\t" 
" dup z11.d, #0                              \n\t" 
" dup z12.d, #0                              \n\t" 
" dup z13.d, #0                              \n\t" 
"                                            \n\t"
" fcmp d7,#0.0                               \n\t"
" beq .DBETAZEROCOLSTOREDS2                  \n\t" // Taking care of the beta==0 case.
"                                            \n\t"
" ld1d  {z8.d}, p0/z, [x21]                  \n\t" // Load column 2 of C
" ld1d  {z9.d}, p0/z, [x21, #1, MUL VL]      \n\t"
" ld1d  {z10.d}, p0/z, [x21, #2, MUL VL]     \n\t"
"                                            \n\t"
" ld1d  {z11.d}, p0/z, [x22]                 \n\t" // Load column 3 of C
" ld1d  {z12.d}, p0/z, [x22, #1, MUL VL]     \n\t"
" ld1d  {z13.d}, p0/z, [x22, #2, MUL VL]     \n\t"
"                                            \n\t"
" fmul z8.d, p0/m, z8.d, z7.d                \n\t" // Scale by beta
" fmul z9.d, p0/m, z9.d, z7.d                \n\t" // Scale by beta
" fmul z10.d, p0/m, z10.d, z7.d              \n\t" // Scale by beta
" fmul z11.d, p0/m, z11.d, z7.d              \n\t" // Scale by beta
" fmul z12.d, p0/m, z12.d, z7.d              \n\t" // Scale by beta
" fmul z13.d, p0/m, z13.d, z7.d              \n\t" // Scale by beta
"                                            \n\t"
" .DBETAZEROCOLSTOREDS2:                     \n\t"
"                                            \n\t"
" fmla z8.d, p0/m, z20.d, z6.d               \n\t" // Scale by alpha
" fmla z9.d, p0/m, z21.d, z6.d               \n\t" // Scale by alpha
" fmla z10.d, p0/m, z22.d, z6.d              \n\t" // Scale by alpha
" fmla z11.d, p0/m, z23.d, z6.d              \n\t" // Scale by alpha
" fmla z12.d, p0/m, z24.d, z6.d              \n\t" // Scale by alpha
" fmla z13.d, p0/m, z25.d, z6.d              \n\t" // Scale by alpha
"                                            \n\t"
" st1d  {z8.d}, p0, [x21]                    \n\t" // Store column 2 of C
" st1d  {z9.d}, p0, [x21, #1, MUL VL]        \n\t"
" st1d  {z10.d}, p0, [x21, #2, MUL VL]       \n\t"
"                                            \n\t"
" st1d  {z11.d}, p0, [x22]                   \n\t" // Store column 3 of C
" st1d  {z12.d}, p0, [x22, #1, MUL VL]       \n\t"
" st1d  {z13.d}, p0, [x22, #2, MUL VL]       \n\t"
"                                            \n\t"
" dup z14.d, #0                              \n\t" 
" dup z15.d, #0                              \n\t" 
" dup z16.d, #0                              \n\t" 
" dup z17.d, #0                              \n\t" 
" dup z18.d, #0                              \n\t" 
" dup z19.d, #0                              \n\t" 
"                                            \n\t"
" fcmp d7,#0.0                               \n\t"
" beq .DBETAZEROCOLSTOREDS3                  \n\t" // Taking care of the beta==0 case.
"                                            \n\t"
" ld1d  z14.d, p0/z, [x23]                   \n\t" // Load column 4 of C
" ld1d  z15.d, p0/z, [x23, #1, MUL VL]       \n\t"
" ld1d  z16.d, p0/z, [x23, #2, MUL VL]       \n\t"
"                                            \n\t"
" ld1d  z17.d, p0/z, [x24]                   \n\t" // Load column 5 of C
" ld1d  z18.d, p0/z, [x24, #1, MUL VL]       \n\t"
" ld1d  z19.d, p0/z, [x24, #2, MUL VL]       \n\t"
"                                            \n\t"
" fmul z14.d, p0/m, z14.d, z7.d              \n\t" // Scale by beta
" fmul z15.d, p0/m, z15.d, z7.d              \n\t" // Scale by beta
" fmul z16.d, p0/m, z16.d, z7.d              \n\t" // Scale by beta
" fmul z17.d, p0/m, z17.d, z7.d              \n\t" // Scale by beta
" fmul z18.d, p0/m, z18.d, z7.d              \n\t" // Scale by beta
" fmul z19.d, p0/m, z19.d, z7.d              \n\t" // Scale by beta
"                                            \n\t"
" .DBETAZEROCOLSTOREDS3:                     \n\t"
"                                            \n\t"
" fmla z14.d, p0/m, z26.d, z6.d              \n\t" // Scale by alpha
" fmla z15.d, p0/m, z27.d, z6.d              \n\t" // Scale by alpha
" fmla z16.d, p0/m, z28.d, z6.d              \n\t" // Scale by alpha
" fmla z17.d, p0/m, z29.d, z6.d              \n\t" // Scale by alpha
" fmla z18.d, p0/m, z30.d, z6.d              \n\t" // Scale by alpha
" fmla z19.d, p0/m, z31.d, z6.d              \n\t" // Scale by alpha
"                                            \n\t"
" st1d  {z14.d}, p0, [x23]                   \n\t" // Store column 4 of C
" st1d  {z15.d}, p0, [x23, #1, MUL VL]       \n\t"
" st1d  {z16.d}, p0, [x23, #2, MUL VL]       \n\t"
"                                            \n\t"
" st1d  {z17.d}, p0, [x24]                   \n\t" // Store column 5 of C
" st1d  {z18.d}, p0, [x24, #1, MUL VL]       \n\t"
" st1d  {z19.d}, p0, [x24, #2, MUL VL]       \n\t"
"                                            \n\t"
" b .DEND                                    \n\t"
"                                            \n\t"
" .DGENSTORED:                               \n\t" // C is general-stride stored.
"                                            \n\t"
" index z8.d, xzr, x13                       \n\t" // Creating index for stride load&store access
" mul x15, x13, x11                          \n\t"
" index z9.d, x15, x13                       \n\t" // Creating index for stride load&store access
" mul x15, x13, x12                          \n\t"
" index z10.d, x15, x13                      \n\t" // Creating index for stride load&store access
"                                            \n\t"
" dup z0.d, #0                               \n\t" 
" dup z1.d, #0                               \n\t" 
" dup z2.d, #0                               \n\t" 
" dup z3.d, #0                               \n\t" 
" dup z4.d, #0                               \n\t" 
" dup z5.d, #0                               \n\t" 
"                                            \n\t"
" fcmp d7,#0.0                               \n\t"
" beq .DBETAZEROGENSTOREDS1                  \n\t" // Taking care of the beta==0 case.
"                                            \n\t"
" ld1d {z0.d}, p0/z, [x2, z8.d, LSL #3]      \n\t" // Load column 0 of C
" ld1d {z1.d}, p0/z, [x2, z9.d, LSL #3]      \n\t"
" ld1d {z2.d}, p0/z, [x2, z10.d, LSL #3]     \n\t"
"                                            \n\t"
" ld1d {z3.d}, p0/z, [x20, z8.d, LSL #3]     \n\t" // Load column 1 of C
" ld1d {z4.d}, p0/z, [x20, z9.d, LSL #3]     \n\t"
" ld1d {z5.d}, p0/z, [x20, z10.d, LSL #3]    \n\t"
"                                            \n\t"
" fmul z0.d, p0/m, z0.d, z7.d                \n\t" // Scale by beta
" fmul z1.d, p0/m, z1.d, z7.d                \n\t" // Scale by beta
" fmul z2.d, p0/m, z2.d, z7.d                \n\t" // Scale by beta
" fmul z3.d, p0/m, z3.d, z7.d                \n\t" // Scale by beta
" fmul z4.d, p0/m, z4.d, z7.d                \n\t" // Scale by beta
" fmul z5.d, p0/m, z5.d, z7.d                \n\t" // Scale by beta
"                                            \n\t"
" .DBETAZEROGENSTOREDS1:                     \n\t"
"                                            \n\t"
" fmla z0.d, p0/m, z14.d, z6.d               \n\t" // Scale by alpha
" fmla z1.d, p0/m, z15.d, z6.d               \n\t" // Scale by alpha
" fmla z2.d, p0/m, z16.d, z6.d               \n\t" // Scale by alpha
" fmla z3.d, p0/m, z17.d, z6.d               \n\t" // Scale by alpha
" fmla z4.d, p0/m, z18.d, z6.d               \n\t" // Scale by alpha
" fmla z5.d, p0/m, z19.d, z6.d               \n\t" // Scale by alpha
"                                            \n\t"
" st1d {z0.d}, p0, [x2, z8.d, LSL #3]        \n\t" // Store column 0 of C
" st1d {z1.d}, p0, [x2, z9.d, LSL #3]        \n\t"
" st1d {z2.d}, p0, [x2, z10.d, LSL #3]       \n\t"
"                                            \n\t"
" st1d {z3.d}, p0, [x20, z8.d, LSL #3]       \n\t" // Store column 1 of C
" st1d {z4.d}, p0, [x20, z9.d, LSL #3]       \n\t"
" st1d {z5.d}, p0, [x20, z10.d, LSL #3]      \n\t"
"                                            \n\t"
" dup z11.d, #0                              \n\t" 
" dup z12.d, #0                              \n\t" 
" dup z13.d, #0                              \n\t" 
" dup z14.d, #0                              \n\t" 
" dup z15.d, #0                              \n\t" 
" dup z16.d, #0                              \n\t" 
"                                            \n\t"
" fcmp d7,#0.0                               \n\t"
" beq .DBETAZEROGENSTOREDS2                  \n\t" // Taking care of the beta==0 case.
"                                            \n\t"
" ld1d {z11.d}, p0/z, [x21, z8.d, LSL #3]    \n\t" // Load column 2 of C
" ld1d {z12.d}, p0/z, [x21, z9.d, LSL #3]    \n\t"
" ld1d {z13.d}, p0/z, [x21, z10.d, LSL #3]   \n\t"
"                                            \n\t"
" ld1d {z14.d}, p0/z, [x22, z8.d, LSL #3]    \n\t" // Load column 3 of C
" ld1d {z15.d}, p0/z, [x22, z9.d, LSL #3]    \n\t"
" ld1d {z16.d}, p0/z, [x22, z10.d, LSL #3]   \n\t"
"                                            \n\t"
" fmul z11.d, p0/m, z11.d, z7.d              \n\t" // Scale by beta
" fmul z12.d, p0/m, z12.d, z7.d              \n\t" // Scale by beta
" fmul z13.d, p0/m, z13.d, z7.d              \n\t" // Scale by beta
" fmul z14.d, p0/m, z14.d, z7.d              \n\t" // Scale by beta
" fmul z15.d, p0/m, z15.d, z7.d              \n\t" // Scale by beta
" fmul z16.d, p0/m, z16.d, z7.d              \n\t" // Scale by beta
"                                            \n\t"
" .DBETAZEROGENSTOREDS2:                     \n\t"
"                                            \n\t"
" fmla z11.d, p0/m, z20.d, z6.d              \n\t" // Scale by alpha
" fmla z12.d, p0/m, z21.d, z6.d              \n\t" // Scale by alpha
" fmla z13.d, p0/m, z22.d, z6.d              \n\t" // Scale by alpha
" fmla z14.d, p0/m, z23.d, z6.d              \n\t" // Scale by alpha
" fmla z15.d, p0/m, z24.d, z6.d              \n\t" // Scale by alpha
" fmla z16.d, p0/m, z25.d, z6.d              \n\t" // Scale by alpha
"                                            \n\t"
" st1d {z11.d}, p0, [x21, z8.d, LSL #3]      \n\t" // Store column 2 of C
" st1d {z12.d}, p0, [x21, z9.d, LSL #3]      \n\t"
" st1d {z13.d}, p0, [x21, z10.d, LSL #3]     \n\t"
"                                            \n\t"
" st1d {z14.d}, p0, [x22, z8.d, LSL #3]      \n\t" // Store column 3 of C
" st1d {z15.d}, p0, [x22, z9.d, LSL #3]      \n\t"
" st1d {z16.d}, p0, [x22, z10.d, LSL #3]     \n\t"
"                                            \n\t"
" dup z0.d, #0                               \n\t" 
" dup z1.d, #0                               \n\t" 
" dup z2.d, #0                               \n\t" 
" dup z3.d, #0                               \n\t" 
" dup z4.d, #0                               \n\t" 
" dup z5.d, #0                               \n\t" 
"                                            \n\t"
" fcmp d7,#0.0                               \n\t"
" beq .DBETAZEROGENSTOREDS3                  \n\t" // Taking care of the beta==0 case.
"                                            \n\t"
" ld1d {z0.d}, p0/z, [x23, z8.d, LSL #3]     \n\t" // Load column 4 of C
" ld1d {z1.d}, p0/z, [x23, z9.d, LSL #3]     \n\t"
" ld1d {z2.d}, p0/z, [x23, z10.d, LSL #3]    \n\t"
"                                            \n\t"
" ld1d {z3.d}, p0/z, [x24, z8.d, LSL #3]     \n\t" // Load column 5 of C
" ld1d {z4.d}, p0/z, [x24, z9.d, LSL #3]     \n\t"
" ld1d {z5.d}, p0/z, [x24, z10.d, LSL #3]    \n\t"
"                                            \n\t"
" fmul z0.d, p0/m, z0.d, z7.d                \n\t" // Scale by beta
" fmul z1.d, p0/m, z1.d, z7.d                \n\t" // Scale by beta
" fmul z2.d, p0/m, z2.d, z7.d                \n\t" // Scale by beta
" fmul z3.d, p0/m, z3.d, z7.d                \n\t" // Scale by beta
" fmul z4.d, p0/m, z4.d, z7.d                \n\t" // Scale by beta
" fmul z5.d, p0/m, z5.d, z7.d                \n\t" // Scale by beta
"                                            \n\t"
" .DBETAZEROGENSTOREDS3:                     \n\t"
"                                            \n\t"
" fmla z0.d, p0/m, z26.d, z6.d               \n\t" // Scale by alpha
" fmla z1.d, p0/m, z27.d, z6.d               \n\t" // Scale by alpha
" fmla z2.d, p0/m, z28.d, z6.d               \n\t" // Scale by alpha
" fmla z3.d, p0/m, z29.d, z6.d               \n\t" // Scale by alpha
" fmla z4.d, p0/m, z30.d, z6.d               \n\t" // Scale by alpha
" fmla z5.d, p0/m, z31.d, z6.d               \n\t" // Scale by alpha
"                                            \n\t"
" st1d z0.d, p0, [x23, z8.d, LSL #3]         \n\t" // Store column 4 of C
" st1d z1.d, p0, [x23, z9.d, LSL #3]         \n\t"
" st1d z2.d, p0, [x23, z10.d, LSL #3]        \n\t"
"                                            \n\t"
" st1d z3.d, p0, [x24, z8.d, LSL #3]         \n\t" // Store column 5 of C
" st1d z4.d, p0, [x24, z9.d, LSL #3]         \n\t"
" st1d z5.d, p0, [x24, z10.d, LSL #3]        \n\t"
"                                            \n\t"
" .DEND:                                     \n\t" // Done!
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
 "x10","x11","x12","x13","x14","x16","x17",
 "x20","x21","x22","x23","x24","x25","x26",
 "x27",       
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

/*
   o 24x6 Double precision micro-kernel 
   o Runnable on ARMv8+SVE, compiled with aarch64 GCC.
   o Use it together with the armv8_sve BLIS configuration.
   o Tested on Juawei arm nodes with ARMIE emulator.
   o Due to the fact that mr and nr depend on size of the vector
     registers, this kernel only works when size=512!!

   February 2019.
   Bine Brank 

 * tests still need to be done to check performance
*/
void bli_dgemm_armv8a_sve512bits_asm_24x6
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
" ldr x7,%[alpha]                            \n\t" // Alpha address      
" ldr x8,%[beta]                             \n\t" // Beta address      
"                                            \n\t" 
" ldr x9,%[cs_c]                             \n\t" // Load cs_c
" lsl x10,x9,#3                              \n\t" // cs_c * sizeof(double)
"                                            \n\t"
" ldr x13,%[rs_c]                            \n\t" // Load rs_c.
" lsl x14,x13,#3                             \n\t" // rs_c * sizeof(double). 
"                                            \n\t"
" mov x11, #8                                \n\t"
" mov x12, #16                               \n\t"
"                                            \n\t"
" add x20,x2,x10                             \n\t" //Load address Column 1 of C
" add x21,x20,x10                            \n\t" //Load address Column 2 of C
" add x22,x21,x10                            \n\t" //Load address Column 3 of C
" add x23,x22,x10                            \n\t" //Load address Column 4 of C
" add x24,x23,x10                            \n\t" //Load address Column 5 of C
"                                            \n\t"
" prfm pldl1keep,[x2]                        \n\t" // Prefetch c.
" prfm pldl1keep,[x20]                       \n\t" // Prefetch c.
" prfm pldl1keep,[x21]                       \n\t" // Prefetch c.
" prfm pldl1keep,[x22]                       \n\t" // Prefetch c.
" prfm pldl1keep,[x23]                       \n\t" // Prefetch c.
" prfm pldl1keep,[x24]                       \n\t" // Prefetch c.
"                                            \n\t"
" whilelt p0.d, xzr, x11                     \n\t" // Creating all true predicate
"                                            \n\t"
" ld1d  z0.d, p0/z, [x0]                     \n\t" // Load a
" ld1d  z1.d, p0/z, [x0, #1, MUL VL]         \n\t"
" ld1d  z2.d, p0/z, [x0, #2, MUL VL]         \n\t"
"                                            \n\t"
" ld1rd  z3.d, p0/z, [x1]                    \n\t" // Load b
" ld1rd  z4.d, p0/z, [x1, #8]                \n\t"
" ld1rd  z5.d, p0/z, [x1, #16]               \n\t"
" ld1rd  z6.d, p0/z, [x1, #24]               \n\t"
" ld1rd  z7.d, p0/z, [x1, #32]               \n\t"
" ld1rd  z8.d, p0/z, [x1, #40]               \n\t"
"                                            \n\t"
"                                            \n\t"
" dup z14.d, #0                              \n\t" // Vector for accummulating column 0
" dup z15.d, #0                              \n\t" // Vector for accummulating column 0
" prfm PLDL1KEEP, [x1, #192]                 \n\t"
" dup z16.d, #0                              \n\t" // Vector for accummulating column 0
" prfm PLDL1KEEP, [x1, #256]                 \n\t"
"                                            \n\t"
" dup z17.d, #0                              \n\t" // Vector for accummulating column 1
" prfm PLDL1KEEP, [x1, #320]                 \n\t"
" dup z18.d, #0                              \n\t" // Vector for accummulating column 1
" dup z19.d, #0                              \n\t" // Vector for accummulating column 1
" prfm PLDL1KEEP, [x0, #768]                 \n\t"
"                                            \n\t"
" dup z20.d, #0                              \n\t" // Vector for accummulating column 2
" prfm PLDL1KEEP, [x0, #896]                 \n\t"
" dup z21.d, #0                              \n\t" // Vector for accummulating column 2
" prfm PLDL1KEEP, [x0, #1024]                \n\t"
" dup z22.d, #0                              \n\t" // Vector for accummulating column 2
"                                            \n\t"
" dup z23.d, #0                              \n\t" // Vector for accummulating column 3
" prfm PLDL1KEEP, [x0, #1152]                \n\t"
" dup z24.d, #0                              \n\t" // Vector for accummulating column 3
" prfm PLDL1KEEP, [x0, #1280]                \n\t"
" dup z25.d, #0                              \n\t" // Vector for accummulating column 3
" prfm PLDL1KEEP, [x0, #1408]                \n\t"
"                                            \n\t"
" dup z26.d, #0                              \n\t" // Vector for accummulating column 4
" dup z27.d, #0                              \n\t" // Vector for accummulating column 4
" dup z28.d, #0                              \n\t" // Vector for accummulating column 4
"                                            \n\t"
" dup z29.d, #0                              \n\t" // Vector for accummulating column 5
" dup z30.d, #0                              \n\t" // Vector for accummulating column 5
" dup z31.d, #0                              \n\t" // Vector for accummulating column 5
"                                            \n\t"
" cmp x5,#0                                  \n\t" // If k_iter == 0, jump to k_left.
" beq .D512CONSIDERKLEFT                     \n\t"
"                                            \n\t"
" add x0, x0, #192                           \n\t" //update address of A
" add x1, x1, #48                            \n\t" //update address of B
"                                            \n\t"
" cmp x5,1                                   \n\t" // If there is just one k_iter, jump to that one. 
" beq .D512LASTITER                          \n\t" // (as loop is do-while-like).
"                                            \n\t"
" D512LOOP:                                  \n\t" // Body
"                                            \n\t"
" fmla z14.d, p0/m, z0.d, z3.d               \n\t"
" prfm PLDL1KEEP, [x0, #1344]                \n\t" // 704 + 64 - 96 = 672
" fmla z15.d, p0/m, z1.d, z3.d               \n\t"
" prfm PLDL1KEEP, [x0, #1472]                \n\t" 
" fmla z16.d, p0/m, z2.d, z3.d               \n\t"
" prfm PLDL1KEEP, [x0, #1600]                \n\t" 
" ld1d   z9.d, p0/z, [x0]                    \n\t" 
" ld1rd  z3.d, p0/z, [x1]                    \n\t"
"                                            \n\t"
" fmla z17.d, p0/m, z0.d, z4.d               \n\t"
" prfm PLDL1KEEP, [x0, #1728]                \n\t" 
" fmla z18.d, p0/m, z1.d, z4.d               \n\t"
" prfm PLDL1KEEP, [x0, #1856]                \n\t" 
" fmla z19.d, p0/m, z2.d, z4.d               \n\t"
" prfm PLDL1KEEP, [x0, #1984]                \n\t" 
" ld1d   z10.d, p0/z, [x0, #1, MUL VL]       \n\t" 
" ld1rd  z4.d, p0/z, [x1, #8]                \n\t"
"                                            \n\t"
" fmla z20.d, p0/m, z0.d, z5.d               \n\t"
" fmla z21.d, p0/m, z1.d, z5.d               \n\t"
" fmla z22.d, p0/m, z2.d, z5.d               \n\t"
" ld1d   z11.d, p0/z, [x0, #2, MUL VL]       \n\t" 
" ld1rd  z5.d, p0/z, [x1, #16]               \n\t"
"                                            \n\t"
" fmla z23.d, p0/m, z0.d, z6.d               \n\t"
" fmla z24.d, p0/m, z1.d, z6.d               \n\t"
" fmla z25.d, p0/m, z2.d, z6.d               \n\t"
" ld1rd  z6.d, p0/z, [x1, #24]               \n\t"
"                                            \n\t"
" fmla z26.d, p0/m, z0.d, z7.d               \n\t"
" fmla z27.d, p0/m, z1.d, z7.d               \n\t"
" fmla z28.d, p0/m, z2.d, z7.d               \n\t"
" ld1rd  z7.d, p0/z, [x1, #32]               \n\t"
"                                            \n\t"
" fmla z29.d, p0/m, z0.d, z8.d               \n\t"
" fmla z30.d, p0/m, z1.d, z8.d               \n\t"
" fmla z31.d, p0/m, z2.d, z8.d               \n\t"
" ld1rd  z8.d, p0/z, [x1, #40]               \n\t"
"                                            \n\t"	//End it 1.
"                                            \n\t"
" fmla z14.d, p0/m, z9.d, z3.d               \n\t"
" fmla z15.d, p0/m, z10.d, z3.d              \n\t"
" fmla z16.d, p0/m, z11.d, z3.d              \n\t"
" ld1d   z0.d, p0/z, [x0, #3, MUL VL]        \n\t" 
" ld1rd  z3.d, p0/z, [x1, #48]               \n\t"
"                                            \n\t"
" fmla z17.d, p0/m, z9.d, z4.d               \n\t"
" fmla z18.d, p0/m, z10.d, z4.d              \n\t"
" fmla z19.d, p0/m, z11.d, z4.d              \n\t"
" ld1d   z1.d, p0/z, [x0, #4, MUL VL]        \n\t" 
" ld1rd  z4.d, p0/z, [x1, #56]               \n\t"
"                                            \n\t"
" fmla z20.d, p0/m, z9.d, z5.d               \n\t"
" fmla z21.d, p0/m, z10.d, z5.d              \n\t"
" fmla z22.d, p0/m, z11.d, z5.d              \n\t"
" prfm PLDL1KEEP, [x1, #336]                 \n\t"
" ld1d   z2.d, p0/z, [x0, #5, MUL VL]        \n\t" 
" prfm PLDL1KEEP, [x1, #400]                 \n\t"
" ld1rd  z5.d, p0/z, [x1, #64]               \n\t"
" prfm PLDL1KEEP, [x1, #464]                 \n\t"
"                                            \n\t"
" fmla z23.d, p0/m, z9.d, z6.d               \n\t"
" fmla z24.d, p0/m, z10.d, z6.d              \n\t"
" fmla z25.d, p0/m, z11.d, z6.d              \n\t"
" ld1rd  z6.d, p0/z, [x1, #72]               \n\t"
"                                            \n\t"
" fmla z26.d, p0/m, z9.d, z7.d               \n\t"
" fmla z27.d, p0/m, z10.d, z7.d              \n\t"
" fmla z28.d, p0/m, z11.d, z7.d              \n\t"
" ld1rd  z7.d, p0/z, [x1, #80]               \n\t"
"                                            \n\t"
" fmla z29.d, p0/m, z9.d, z8.d               \n\t"
" fmla z30.d, p0/m, z10.d, z8.d              \n\t"
" fmla z31.d, p0/m, z11.d, z8.d              \n\t"
" ld1rd  z8.d, p0/z, [x1, #88]               \n\t"
"                                            \n\t"	//End it 2.
"                                            \n\t"
" add x0, x0, #384                           \n\t" // Increasing x0 by 768/2
"                                            \n\t"
" fmla z14.d, p0/m, z0.d, z3.d               \n\t"
" fmla z15.d, p0/m, z1.d, z3.d               \n\t"
" fmla z16.d, p0/m, z2.d, z3.d               \n\t"
" ld1d   z9.d, p0/z, [x0]                    \n\t" 
" ld1rd  z3.d, p0/z, [x1, #96]               \n\t"
"                                            \n\t"
" fmla z17.d, p0/m, z0.d, z4.d               \n\t"
" fmla z18.d, p0/m, z1.d, z4.d               \n\t"
" fmla z19.d, p0/m, z2.d, z4.d               \n\t"
" ld1d   z10.d, p0/z, [x0, #1, MUL VL]       \n\t" 
" ld1rd  z4.d, p0/z, [x1, #104]              \n\t"
"                                            \n\t"
" fmla z20.d, p0/m, z0.d, z5.d               \n\t"
" fmla z21.d, p0/m, z1.d, z5.d               \n\t"
" fmla z22.d, p0/m, z2.d, z5.d               \n\t"
" ld1d   z11.d, p0/z, [x0, #2, MUL VL]       \n\t" 
" ld1rd  z5.d, p0/z, [x1, #112]              \n\t"
"                                            \n\t"
" fmla z23.d, p0/m, z0.d, z6.d               \n\t"
" fmla z24.d, p0/m, z1.d, z6.d               \n\t"
" fmla z25.d, p0/m, z2.d, z6.d               \n\t"
" ld1rd  z6.d, p0/z, [x1, #120]              \n\t"
"                                            \n\t"
" fmla z26.d, p0/m, z0.d, z7.d               \n\t"
" fmla z27.d, p0/m, z1.d, z7.d               \n\t"
" fmla z28.d, p0/m, z2.d, z7.d               \n\t"
" ld1rd  z7.d, p0/z, [x1, #128]              \n\t"
"                                            \n\t"
" fmla z29.d, p0/m, z0.d, z8.d               \n\t"
" fmla z30.d, p0/m, z1.d, z8.d               \n\t"
" fmla z31.d, p0/m, z2.d, z8.d               \n\t"
" ld1rd  z8.d, p0/z, [x1, #136]              \n\t"
"                                            \n\t"	//End it 3.
"                                            \n\t"
" fmla z14.d, p0/m, z9.d, z3.d               \n\t"
" fmla z15.d, p0/m, z10.d, z3.d              \n\t"
" fmla z16.d, p0/m, z11.d, z3.d              \n\t"
" ld1d   z0.d, p0/z, [x0, #3, MUL VL]        \n\t" 
" ld1rd  z3.d, p0/z, [x1, #144]              \n\t"
"                                            \n\t"
" fmla z17.d, p0/m, z9.d, z4.d               \n\t"
" fmla z18.d, p0/m, z10.d, z4.d              \n\t"
" fmla z19.d, p0/m, z11.d, z4.d              \n\t"
" ld1d   z1.d, p0/z, [x0, #4, MUL VL]        \n\t" 
" ld1rd  z4.d, p0/z, [x1, #152]              \n\t"
"                                            \n\t"
" fmla z20.d, p0/m, z9.d, z5.d               \n\t"
" fmla z21.d, p0/m, z10.d, z5.d              \n\t"
" fmla z22.d, p0/m, z11.d, z5.d              \n\t"
" ld1d   z2.d, p0/z, [x0, #5, MUL VL]        \n\t" 
" ld1rd  z5.d, p0/z, [x1, #160]              \n\t"
"                                            \n\t"
" fmla z23.d, p0/m, z9.d, z6.d               \n\t"
" fmla z24.d, p0/m, z10.d, z6.d              \n\t"
" fmla z25.d, p0/m, z11.d, z6.d              \n\t"
" ld1rd  z6.d, p0/z, [x1, #168]              \n\t"
"                                            \n\t"
" fmla z26.d, p0/m, z9.d, z7.d               \n\t"
" fmla z27.d, p0/m, z10.d, z7.d              \n\t"
" fmla z28.d, p0/m, z11.d, z7.d              \n\t"
" ld1rd  z7.d, p0/z, [x1, #176]              \n\t"
"                                            \n\t"
" fmla z29.d, p0/m, z9.d, z8.d               \n\t"
" fmla z30.d, p0/m, z10.d, z8.d              \n\t"
" fmla z31.d, p0/m, z11.d, z8.d              \n\t"
" ld1rd  z8.d, p0/z, [x1, #184]              \n\t"
"                                            \n\t"	//End it 4.
"                                            \n\t"
" add x0, x0, #384                           \n\t" // incremenenting by 768/2 
" add x1, x1, #192                           \n\t"
"                                            \n\t"
" sub x5,x5,1                                \n\t" // i-=1
" cmp x5,1                                   \n\t" // Iterate again if we are not in k_iter == 1.
" bne D512LOOP                               \n\t"
"                                            \n\t"
".D512LASTITER:                              \n\t"
"                                            \n\t"
" fmla z14.d, p0/m, z0.d, z3.d               \n\t"
" fmla z15.d, p0/m, z1.d, z3.d               \n\t"
" fmla z16.d, p0/m, z2.d, z3.d               \n\t"
" ld1d   z9.d, p0/z, [x0]                    \n\t" 
" ld1rd  z3.d, p0/z, [x1]                    \n\t"
"                                            \n\t"
" fmla z17.d, p0/m, z0.d, z4.d               \n\t"
" fmla z18.d, p0/m, z1.d, z4.d               \n\t"
" fmla z19.d, p0/m, z2.d, z4.d               \n\t"
" ld1d   z10.d, p0/z, [x0, #1, MUL VL]       \n\t" 
" ld1rd  z4.d, p0/z, [x1, #8]                \n\t"
"                                            \n\t"
" fmla z20.d, p0/m, z0.d, z5.d               \n\t"
" fmla z21.d, p0/m, z1.d, z5.d               \n\t"
" fmla z22.d, p0/m, z2.d, z5.d               \n\t"
" ld1d   z11.d, p0/z, [x0, #2, MUL VL]       \n\t" 
" ld1rd  z5.d, p0/z, [x1, #16]               \n\t"
"                                            \n\t"
" fmla z23.d, p0/m, z0.d, z6.d               \n\t"
" fmla z24.d, p0/m, z1.d, z6.d               \n\t"
" fmla z25.d, p0/m, z2.d, z6.d               \n\t"
" ld1rd  z6.d, p0/z, [x1, #24]               \n\t"
"                                            \n\t"
" fmla z26.d, p0/m, z0.d, z7.d               \n\t"
" fmla z27.d, p0/m, z1.d, z7.d               \n\t"
" fmla z28.d, p0/m, z2.d, z7.d               \n\t"
" ld1rd  z7.d, p0/z, [x1, #32]               \n\t"
"                                            \n\t"
" fmla z29.d, p0/m, z0.d, z8.d               \n\t"
" fmla z30.d, p0/m, z1.d, z8.d               \n\t"
" fmla z31.d, p0/m, z2.d, z8.d               \n\t"
" ld1rd  z8.d, p0/z, [x1, #40]               \n\t"
"                                            \n\t"	//End it 1.
"                                            \n\t"
" fmla z14.d, p0/m, z9.d, z3.d               \n\t"
" fmla z15.d, p0/m, z10.d, z3.d              \n\t"
" fmla z16.d, p0/m, z11.d, z3.d              \n\t"
" ld1d   z0.d, p0/z, [x0, #3, MUL VL]        \n\t" 
" ld1rd  z3.d, p0/z, [x1, #48]               \n\t"
"                                            \n\t"
" fmla z17.d, p0/m, z9.d, z4.d               \n\t"
" fmla z18.d, p0/m, z10.d, z4.d              \n\t"
" fmla z19.d, p0/m, z11.d, z4.d              \n\t"
" ld1d   z1.d, p0/z, [x0, #4, MUL VL]        \n\t" 
" ld1rd  z4.d, p0/z, [x1, #56]               \n\t"
"                                            \n\t"
" fmla z20.d, p0/m, z9.d, z5.d               \n\t"
" fmla z21.d, p0/m, z10.d, z5.d              \n\t"
" fmla z22.d, p0/m, z11.d, z5.d              \n\t"
" ld1d   z2.d, p0/z, [x0, #5, MUL VL]        \n\t" 
" ld1rd  z5.d, p0/z, [x1, #64]               \n\t"
"                                            \n\t"
" fmla z23.d, p0/m, z9.d, z6.d               \n\t"
" fmla z24.d, p0/m, z10.d, z6.d              \n\t"
" fmla z25.d, p0/m, z11.d, z6.d              \n\t"
" ld1rd  z6.d, p0/z, [x1, #72]               \n\t"
"                                            \n\t"
" fmla z26.d, p0/m, z9.d, z7.d               \n\t"
" fmla z27.d, p0/m, z10.d, z7.d              \n\t"
" fmla z28.d, p0/m, z11.d, z7.d              \n\t"
" ld1rd  z7.d, p0/z, [x1, #80]               \n\t"
"                                            \n\t"
" fmla z29.d, p0/m, z9.d, z8.d               \n\t"
" fmla z30.d, p0/m, z10.d, z8.d              \n\t"
" fmla z31.d, p0/m, z11.d, z8.d              \n\t"
" ld1rd  z8.d, p0/z, [x1, #88]               \n\t"
"                                            \n\t"	//End it 2.
" add x0, x0, #384                           \n\t"
"                                            \n\t"
" fmla z14.d, p0/m, z0.d, z3.d               \n\t"
" fmla z15.d, p0/m, z1.d, z3.d               \n\t"
" fmla z16.d, p0/m, z2.d, z3.d               \n\t"
" ld1d   z9.d, p0/z, [x0]                    \n\t" 
" ld1rd  z3.d, p0/z, [x1, #96]               \n\t"
"                                            \n\t"
" fmla z17.d, p0/m, z0.d, z4.d               \n\t"
" fmla z18.d, p0/m, z1.d, z4.d               \n\t"
" fmla z19.d, p0/m, z2.d, z4.d               \n\t"
" ld1d   z10.d, p0/z, [x0, #1, MUL VL]       \n\t" 
" ld1rd  z4.d, p0/z, [x1, #104]              \n\t"
"                                            \n\t"
" fmla z20.d, p0/m, z0.d, z5.d               \n\t"
" fmla z21.d, p0/m, z1.d, z5.d               \n\t"
" fmla z22.d, p0/m, z2.d, z5.d               \n\t"
" ld1d   z11.d, p0/z, [x0, #2, MUL VL]       \n\t" 
" ld1rd  z5.d, p0/z, [x1, #112]              \n\t"
"                                            \n\t"
" fmla z23.d, p0/m, z0.d, z6.d               \n\t"
" fmla z24.d, p0/m, z1.d, z6.d               \n\t"
" fmla z25.d, p0/m, z2.d, z6.d               \n\t"
" ld1rd  z6.d, p0/z, [x1, #120]              \n\t"
"                                            \n\t"
" fmla z26.d, p0/m, z0.d, z7.d               \n\t"
" fmla z27.d, p0/m, z1.d, z7.d               \n\t"
" fmla z28.d, p0/m, z2.d, z7.d               \n\t"
" ld1rd  z7.d, p0/z, [x1, #128]              \n\t"
"                                            \n\t"
" fmla z29.d, p0/m, z0.d, z8.d               \n\t"
" fmla z30.d, p0/m, z1.d, z8.d               \n\t"
" fmla z31.d, p0/m, z2.d, z8.d               \n\t"
" ld1rd  z8.d, p0/z, [x1, #136]              \n\t"
"                                            \n\t"	//End it 3.
"                                            \n\t"
" fmla z14.d, p0/m, z9.d, z3.d               \n\t"
" fmla z15.d, p0/m, z10.d, z3.d              \n\t"
" fmla z16.d, p0/m, z11.d, z3.d              \n\t"
"                                            \n\t"
" fmla z17.d, p0/m, z9.d, z4.d               \n\t"
" fmla z18.d, p0/m, z10.d, z4.d              \n\t"
" fmla z19.d, p0/m, z11.d, z4.d              \n\t"
"                                            \n\t"
" fmla z20.d, p0/m, z9.d, z5.d               \n\t"
" fmla z21.d, p0/m, z10.d, z5.d              \n\t"
" fmla z22.d, p0/m, z11.d, z5.d              \n\t"
"                                            \n\t"
" fmla z23.d, p0/m, z9.d, z6.d               \n\t"
" fmla z24.d, p0/m, z10.d, z6.d              \n\t"
" fmla z25.d, p0/m, z11.d, z6.d              \n\t"
"                                            \n\t"
" add x0, x0, #192                           \n\t"
"                                            \n\t"
" fmla z26.d, p0/m, z9.d, z7.d               \n\t"
" fmla z27.d, p0/m, z10.d, z7.d              \n\t"
" fmla z28.d, p0/m, z11.d, z7.d              \n\t"
"                                            \n\t"
" fmla z29.d, p0/m, z9.d, z8.d               \n\t"
" fmla z30.d, p0/m, z10.d, z8.d              \n\t"
" fmla z31.d, p0/m, z11.d, z8.d              \n\t"
"                                            \n\t"	//End it 4.
"                                            \n\t"
" add x1, x1, #144                           \n\t"
"                                            \n\t"
" .D512CONSIDERKLEFT:                        \n\t" 
" cmp x6,0                                   \n\t" // If k_left == 0, we are done.
" beq .D512POSTACCUM                         \n\t" // else, we enter the k_left loop.
"                                            \n\t"
".D512LOOPKLEFT:                             \n\t"
"                                            \n\t"
" ld1d  z0.d, p0/z, [x0]                     \n\t" // Load a
" ld1d  z1.d, p0/z, [x0, #1, MUL VL]         \n\t"
" ld1d  z2.d, p0/z, [x0, #2, MUL VL]         \n\t"
" add x0, x0, #192                           \n\t"
"                                            \n\t"
" ld1rd  z3.d, p0/z, [x1]                    \n\t" // Load b
" ld1rd  z4.d, p0/z, [x1, #8]                \n\t"
" ld1rd  z5.d, p0/z, [x1, #16]               \n\t"
" ld1rd  z6.d, p0/z, [x1, #24]               \n\t"
" ld1rd  z7.d, p0/z, [x1, #32]               \n\t"
" ld1rd  z8.d, p0/z, [x1, #40]               \n\t"
" add x1, x1, #48                            \n\t"
"                                            \n\t"
" sub x6,x6,1                                \n\t"
"                                            \n\t"
" fmla z14.d, p0/m, z0.d, z3.d               \n\t"
" fmla z15.d, p0/m, z1.d, z3.d               \n\t"
" fmla z16.d, p0/m, z2.d, z3.d               \n\t"
"                                            \n\t"
" fmla z17.d, p0/m, z0.d, z4.d               \n\t"
" fmla z18.d, p0/m, z1.d, z4.d               \n\t"
" fmla z19.d, p0/m, z2.d, z4.d               \n\t"
"                                            \n\t"
" fmla z20.d, p0/m, z0.d, z5.d               \n\t"
" fmla z21.d, p0/m, z1.d, z5.d               \n\t"
" fmla z22.d, p0/m, z2.d, z5.d               \n\t"
"                                            \n\t"
" fmla z23.d, p0/m, z0.d, z6.d               \n\t"
" fmla z24.d, p0/m, z1.d, z6.d               \n\t"
" fmla z25.d, p0/m, z2.d, z6.d               \n\t"
"                                            \n\t"
" fmla z26.d, p0/m, z0.d, z7.d               \n\t"
" fmla z27.d, p0/m, z1.d, z7.d               \n\t"
" fmla z28.d, p0/m, z2.d, z7.d               \n\t"
"                                            \n\t"
" fmla z29.d, p0/m, z0.d, z8.d               \n\t"
" fmla z30.d, p0/m, z1.d, z8.d               \n\t"
" fmla z31.d, p0/m, z2.d, z8.d               \n\t"
"                                            \n\t"
" cmp x6,0                                   \n\t" // Iterate again.
" bne .D512LOOPKLEFT                         \n\t" // if i!=0.
"                                            \n\t"
" .D512POSTACCUM:                            \n\t"
"                                            \n\t"
" ld1rd  z6.d, p0/z, [x7]                    \n\t" // Load alpha
" ld1rd  z7.d, p0/z, [x8]                    \n\t" // Load beta
"                                            \n\t"
" cmp x13,#1                                 \n\t" // If rs_c != 1 (column-major)
" bne .D512GENSTORED                         \n\t"
"                                            \n\t"
" .D512COLSTORED:                            \n\t" // C is column-major.
"                                            \n\t"
" dup z0.d, #0                               \n\t" 
" dup z1.d, #0                               \n\t" 
" dup z2.d, #0                               \n\t" 
" dup z3.d, #0                               \n\t" 
" dup z4.d, #0                               \n\t" 
" dup z5.d, #0                               \n\t" 
"                                            \n\t"
" fcmp d7,#0.0                               \n\t"
" beq .D512BETAZEROCOLSTOREDS1               \n\t" // Taking care of the beta==0 case.
"                                            \n\t"
" ld1d  {z0.d}, p0/z, [x2]                   \n\t" // Load column 0 of C
" ld1d  {z1.d}, p0/z, [x2, #1, MUL VL]       \n\t"
" ld1d  {z2.d}, p0/z, [x2, #2, MUL VL]       \n\t"
"                                            \n\t"
" ld1d  {z3.d}, p0/z, [x20]                  \n\t" // Load column 1 of C
" ld1d  {z4.d}, p0/z, [x20, #1, MUL VL]      \n\t"
" ld1d  {z5.d}, p0/z, [x20, #2, MUL VL]      \n\t"
"                                            \n\t"
" fmul z0.d, p0/m, z0.d, z7.d                \n\t" // Scale by beta
" fmul z1.d, p0/m, z1.d, z7.d                \n\t" // Scale by beta
" fmul z2.d, p0/m, z2.d, z7.d                \n\t" // Scale by beta
" fmul z3.d, p0/m, z3.d, z7.d                \n\t" // Scale by beta
" fmul z4.d, p0/m, z4.d, z7.d                \n\t" // Scale by beta
" fmul z5.d, p0/m, z5.d, z7.d                \n\t" // Scale by beta
"                                            \n\t"
" .D512BETAZEROCOLSTOREDS1:                  \n\t"
"                                            \n\t"
" fmla z0.d, p0/m, z14.d, z6.d               \n\t" // Scale by alpha
" fmla z1.d, p0/m, z15.d, z6.d               \n\t" // Scale by alpha
" fmla z2.d, p0/m, z16.d, z6.d               \n\t" // Scale by alpha
" fmla z3.d, p0/m, z17.d, z6.d               \n\t" // Scale by alpha
" fmla z4.d, p0/m, z18.d, z6.d               \n\t" // Scale by alpha
" fmla z5.d, p0/m, z19.d, z6.d               \n\t" // Scale by alpha
"                                            \n\t"
" st1d  {z0.d}, p0, [x2]                     \n\t" // Store column 0 of C
" st1d  {z1.d}, p0, [x2, #1, MUL VL]         \n\t"
" st1d  {z2.d}, p0, [x2, #2, MUL VL]         \n\t"
"                                            \n\t"
" st1d  {z3.d}, p0, [x20]                    \n\t" // Store column 1 of C
" st1d  {z4.d}, p0, [x20, #1, MUL VL]        \n\t"
" st1d  {z5.d}, p0, [x20, #2, MUL VL]        \n\t"
"                                            \n\t"
" dup z8.d, #0                               \n\t" 
" dup z9.d, #0                               \n\t" 
" dup z10.d, #0                              \n\t" 
" dup z11.d, #0                              \n\t" 
" dup z12.d, #0                              \n\t" 
" dup z13.d, #0                              \n\t" 
"                                            \n\t"
" fcmp d7,#0.0                               \n\t"
" beq .D512BETAZEROCOLSTOREDS2               \n\t" // Taking care of the beta==0 case.
"                                            \n\t"
" ld1d  {z8.d}, p0/z, [x21]                  \n\t" // Load column 2 of C
" ld1d  {z9.d}, p0/z, [x21, #1, MUL VL]      \n\t"
" ld1d  {z10.d}, p0/z, [x21, #2, MUL VL]     \n\t"
"                                            \n\t"
" ld1d  {z11.d}, p0/z, [x22]                 \n\t" // Load column 3 of C
" ld1d  {z12.d}, p0/z, [x22, #1, MUL VL]     \n\t"
" ld1d  {z13.d}, p0/z, [x22, #2, MUL VL]     \n\t"
"                                            \n\t"
" fmul z8.d, p0/m, z8.d, z7.d                \n\t" // Scale by beta
" fmul z9.d, p0/m, z9.d, z7.d                \n\t" // Scale by beta
" fmul z10.d, p0/m, z10.d, z7.d              \n\t" // Scale by beta
" fmul z11.d, p0/m, z11.d, z7.d              \n\t" // Scale by beta
" fmul z12.d, p0/m, z12.d, z7.d              \n\t" // Scale by beta
" fmul z13.d, p0/m, z13.d, z7.d              \n\t" // Scale by beta
"                                            \n\t"
" .D512BETAZEROCOLSTOREDS2:                  \n\t"
"                                            \n\t"
" fmla z8.d, p0/m, z20.d, z6.d               \n\t" // Scale by alpha
" fmla z9.d, p0/m, z21.d, z6.d               \n\t" // Scale by alpha
" fmla z10.d, p0/m, z22.d, z6.d              \n\t" // Scale by alpha
" fmla z11.d, p0/m, z23.d, z6.d              \n\t" // Scale by alpha
" fmla z12.d, p0/m, z24.d, z6.d              \n\t" // Scale by alpha
" fmla z13.d, p0/m, z25.d, z6.d              \n\t" // Scale by alpha
"                                            \n\t"
" st1d  {z8.d}, p0, [x21]                    \n\t" // Store column 2 of C
" st1d  {z9.d}, p0, [x21, #1, MUL VL]        \n\t"
" st1d  {z10.d}, p0, [x21, #2, MUL VL]       \n\t"
"                                            \n\t"
" st1d  {z11.d}, p0, [x22]                   \n\t" // Store column 3 of C
" st1d  {z12.d}, p0, [x22, #1, MUL VL]       \n\t"
" st1d  {z13.d}, p0, [x22, #2, MUL VL]       \n\t"
"                                            \n\t"
" dup z14.d, #0                              \n\t" 
" dup z15.d, #0                              \n\t" 
" dup z16.d, #0                              \n\t" 
" dup z17.d, #0                              \n\t" 
" dup z18.d, #0                              \n\t" 
" dup z19.d, #0                              \n\t" 
"                                            \n\t"
" fcmp d7,#0.0                               \n\t"
" beq .D512BETAZEROCOLSTOREDS3               \n\t" // Taking care of the beta==0 case.
"                                            \n\t"
" ld1d  z14.d, p0/z, [x23]                   \n\t" // Load column 4 of C
" ld1d  z15.d, p0/z, [x23, #1, MUL VL]       \n\t"
" ld1d  z16.d, p0/z, [x23, #2, MUL VL]       \n\t"
"                                            \n\t"
" ld1d  z17.d, p0/z, [x24]                   \n\t" // Load column 5 of C
" ld1d  z18.d, p0/z, [x24, #1, MUL VL]       \n\t"
" ld1d  z19.d, p0/z, [x24, #2, MUL VL]       \n\t"
"                                            \n\t"
" fmul z14.d, p0/m, z14.d, z7.d              \n\t" // Scale by beta
" fmul z15.d, p0/m, z15.d, z7.d              \n\t" // Scale by beta
" fmul z16.d, p0/m, z16.d, z7.d              \n\t" // Scale by beta
" fmul z17.d, p0/m, z17.d, z7.d              \n\t" // Scale by beta
" fmul z18.d, p0/m, z18.d, z7.d              \n\t" // Scale by beta
" fmul z19.d, p0/m, z19.d, z7.d              \n\t" // Scale by beta
"                                            \n\t"
" .D512BETAZEROCOLSTOREDS3:                  \n\t"
"                                            \n\t"
" fmla z14.d, p0/m, z26.d, z6.d              \n\t" // Scale by alpha
" fmla z15.d, p0/m, z27.d, z6.d              \n\t" // Scale by alpha
" fmla z16.d, p0/m, z28.d, z6.d              \n\t" // Scale by alpha
" fmla z17.d, p0/m, z29.d, z6.d              \n\t" // Scale by alpha
" fmla z18.d, p0/m, z30.d, z6.d              \n\t" // Scale by alpha
" fmla z19.d, p0/m, z31.d, z6.d              \n\t" // Scale by alpha
"                                            \n\t"
" st1d  {z14.d}, p0, [x23]                   \n\t" // Store column 4 of C
" st1d  {z15.d}, p0, [x23, #1, MUL VL]       \n\t"
" st1d  {z16.d}, p0, [x23, #2, MUL VL]       \n\t"
"                                            \n\t"
" st1d  {z17.d}, p0, [x24]                   \n\t" // Store column 5 of C
" st1d  {z18.d}, p0, [x24, #1, MUL VL]       \n\t"
" st1d  {z19.d}, p0, [x24, #2, MUL VL]       \n\t"
"                                            \n\t"
" b .D512END                                 \n\t"
"                                            \n\t"
" .D512GENSTORED:                            \n\t" // C is general-stride stored.
"                                            \n\t"
" index z8.d, xzr, x13                       \n\t" // Creating index for stride load&store access
" mul x15, x13, x11                          \n\t"
" index z9.d, x15, x13                       \n\t" // Creating index for stride load&store access
" mul x15, x13, x12                          \n\t"
" index z10.d, x15, x13                      \n\t" // Creating index for stride load&store access
"                                            \n\t"
" dup z0.d, #0                               \n\t" 
" dup z1.d, #0                               \n\t" 
" dup z2.d, #0                               \n\t" 
" dup z3.d, #0                               \n\t" 
" dup z4.d, #0                               \n\t" 
" dup z5.d, #0                               \n\t" 
"                                            \n\t"
" fcmp d7,#0.0                               \n\t"
" beq .D512BETAZEROGENSTOREDS1               \n\t" // Taking care of the beta==0 case.
"                                            \n\t"
" ld1d {z0.d}, p0/z, [x2, z8.d, LSL #3]      \n\t" // Load column 0 of C
" ld1d {z1.d}, p0/z, [x2, z9.d, LSL #3]      \n\t"
" ld1d {z2.d}, p0/z, [x2, z10.d, LSL #3]     \n\t"
"                                            \n\t"
" ld1d {z3.d}, p0/z, [x20, z8.d, LSL #3]     \n\t" // Load column 1 of C
" ld1d {z4.d}, p0/z, [x20, z9.d, LSL #3]     \n\t"
" ld1d {z5.d}, p0/z, [x20, z10.d, LSL #3]    \n\t"
"                                            \n\t"
" fmul z0.d, p0/m, z0.d, z7.d                \n\t" // Scale by beta
" fmul z1.d, p0/m, z1.d, z7.d                \n\t" // Scale by beta
" fmul z2.d, p0/m, z2.d, z7.d                \n\t" // Scale by beta
" fmul z3.d, p0/m, z3.d, z7.d                \n\t" // Scale by beta
" fmul z4.d, p0/m, z4.d, z7.d                \n\t" // Scale by beta
" fmul z5.d, p0/m, z5.d, z7.d                \n\t" // Scale by beta
"                                            \n\t"
" .D512BETAZEROGENSTOREDS1:                  \n\t"
"                                            \n\t"
" fmla z0.d, p0/m, z14.d, z6.d               \n\t" // Scale by alpha
" fmla z1.d, p0/m, z15.d, z6.d               \n\t" // Scale by alpha
" fmla z2.d, p0/m, z16.d, z6.d               \n\t" // Scale by alpha
" fmla z3.d, p0/m, z17.d, z6.d               \n\t" // Scale by alpha
" fmla z4.d, p0/m, z18.d, z6.d               \n\t" // Scale by alpha
" fmla z5.d, p0/m, z19.d, z6.d               \n\t" // Scale by alpha
"                                            \n\t"
" st1d {z0.d}, p0, [x2, z8.d, LSL #3]        \n\t" // Store column 0 of C
" st1d {z1.d}, p0, [x2, z9.d, LSL #3]        \n\t"
" st1d {z2.d}, p0, [x2, z10.d, LSL #3]       \n\t"
"                                            \n\t"
" st1d {z3.d}, p0, [x20, z8.d, LSL #3]       \n\t" // Store column 1 of C
" st1d {z4.d}, p0, [x20, z9.d, LSL #3]       \n\t"
" st1d {z5.d}, p0, [x20, z10.d, LSL #3]      \n\t"
"                                            \n\t"
" dup z11.d, #0                              \n\t" 
" dup z12.d, #0                              \n\t" 
" dup z13.d, #0                              \n\t" 
" dup z14.d, #0                              \n\t" 
" dup z15.d, #0                              \n\t" 
" dup z16.d, #0                              \n\t" 
"                                            \n\t"
" fcmp d7,#0.0                               \n\t"
" beq .D512BETAZEROGENSTOREDS2               \n\t" // Taking care of the beta==0 case.
"                                            \n\t"
" ld1d {z11.d}, p0/z, [x21, z8.d, LSL #3]    \n\t" // Load column 2 of C
" ld1d {z12.d}, p0/z, [x21, z9.d, LSL #3]    \n\t"
" ld1d {z13.d}, p0/z, [x21, z10.d, LSL #3]   \n\t"
"                                            \n\t"
" ld1d {z14.d}, p0/z, [x22, z8.d, LSL #3]    \n\t" // Load column 3 of C
" ld1d {z15.d}, p0/z, [x22, z9.d, LSL #3]    \n\t"
" ld1d {z16.d}, p0/z, [x22, z10.d, LSL #3]   \n\t"
"                                            \n\t"
" fmul z11.d, p0/m, z11.d, z7.d              \n\t" // Scale by beta
" fmul z12.d, p0/m, z12.d, z7.d              \n\t" // Scale by beta
" fmul z13.d, p0/m, z13.d, z7.d              \n\t" // Scale by beta
" fmul z14.d, p0/m, z14.d, z7.d              \n\t" // Scale by beta
" fmul z15.d, p0/m, z15.d, z7.d              \n\t" // Scale by beta
" fmul z16.d, p0/m, z16.d, z7.d              \n\t" // Scale by beta
"                                            \n\t"
" .D512BETAZEROGENSTOREDS2:                  \n\t"
"                                            \n\t"
" fmla z11.d, p0/m, z20.d, z6.d              \n\t" // Scale by alpha
" fmla z12.d, p0/m, z21.d, z6.d              \n\t" // Scale by alpha
" fmla z13.d, p0/m, z22.d, z6.d              \n\t" // Scale by alpha
" fmla z14.d, p0/m, z23.d, z6.d              \n\t" // Scale by alpha
" fmla z15.d, p0/m, z24.d, z6.d              \n\t" // Scale by alpha
" fmla z16.d, p0/m, z25.d, z6.d              \n\t" // Scale by alpha
"                                            \n\t"
" st1d {z11.d}, p0, [x21, z8.d, LSL #3]      \n\t" // Store column 2 of C
" st1d {z12.d}, p0, [x21, z9.d, LSL #3]      \n\t"
" st1d {z13.d}, p0, [x21, z10.d, LSL #3]     \n\t"
"                                            \n\t"
" st1d {z14.d}, p0, [x22, z8.d, LSL #3]      \n\t" // Store column 3 of C
" st1d {z15.d}, p0, [x22, z9.d, LSL #3]      \n\t"
" st1d {z16.d}, p0, [x22, z10.d, LSL #3]     \n\t"
"                                            \n\t"
" dup z0.d, #0                               \n\t" 
" dup z1.d, #0                               \n\t" 
" dup z2.d, #0                               \n\t" 
" dup z3.d, #0                               \n\t" 
" dup z4.d, #0                               \n\t" 
" dup z5.d, #0                               \n\t" 
"                                            \n\t"
" fcmp d7,#0.0                               \n\t"
" beq .D512BETAZEROGENSTOREDS3               \n\t" // Taking care of the beta==0 case.
"                                            \n\t"
" ld1d {z0.d}, p0/z, [x23, z8.d, LSL #3]     \n\t" // Load column 4 of C
" ld1d {z1.d}, p0/z, [x23, z9.d, LSL #3]     \n\t"
" ld1d {z2.d}, p0/z, [x23, z10.d, LSL #3]    \n\t"
"                                            \n\t"
" ld1d {z3.d}, p0/z, [x24, z8.d, LSL #3]     \n\t" // Load column 5 of C
" ld1d {z4.d}, p0/z, [x24, z9.d, LSL #3]     \n\t"
" ld1d {z5.d}, p0/z, [x24, z10.d, LSL #3]    \n\t"
"                                            \n\t"
" fmul z0.d, p0/m, z0.d, z7.d                \n\t" // Scale by beta
" fmul z1.d, p0/m, z1.d, z7.d                \n\t" // Scale by beta
" fmul z2.d, p0/m, z2.d, z7.d                \n\t" // Scale by beta
" fmul z3.d, p0/m, z3.d, z7.d                \n\t" // Scale by beta
" fmul z4.d, p0/m, z4.d, z7.d                \n\t" // Scale by beta
" fmul z5.d, p0/m, z5.d, z7.d                \n\t" // Scale by beta
"                                            \n\t"
" .D512BETAZEROGENSTOREDS3:                  \n\t"
"                                            \n\t"
" fmla z0.d, p0/m, z26.d, z6.d               \n\t" // Scale by alpha
" fmla z1.d, p0/m, z27.d, z6.d               \n\t" // Scale by alpha
" fmla z2.d, p0/m, z28.d, z6.d               \n\t" // Scale by alpha
" fmla z3.d, p0/m, z29.d, z6.d               \n\t" // Scale by alpha
" fmla z4.d, p0/m, z30.d, z6.d               \n\t" // Scale by alpha
" fmla z5.d, p0/m, z31.d, z6.d               \n\t" // Scale by alpha
"                                            \n\t"
" st1d z0.d, p0, [x23, z8.d, LSL #3]         \n\t" // Store column 4 of C
" st1d z1.d, p0, [x23, z9.d, LSL #3]         \n\t"
" st1d z2.d, p0, [x23, z10.d, LSL #3]        \n\t"
"                                            \n\t"
" st1d z3.d, p0, [x24, z8.d, LSL #3]         \n\t" // Store column 5 of C
" st1d z4.d, p0, [x24, z9.d, LSL #3]         \n\t"
" st1d z5.d, p0, [x24, z10.d, LSL #3]        \n\t"
"                                            \n\t"
" .D512END:                                  \n\t" // Done!
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
 "x10","x11","x12","x13","x14","x16","x17",
 "x20","x21","x22","x23","x24","x25","x26",
 "x27",       
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


/*
   o 48x6 Double precision micro-kernel 
   o Runnable on ARMv8+SVE, compiled with aarch64 GCC.
   o Use it together with the armv8_sve BLIS configuration.
   o Tested on Juawei arm nodes with ARMIE emulator.
   o Due to the fact that mr and nr depend on size of the vector
     registers, this kernel only works when size=1024!

   February 2019.
   Bine Brank 

 * tests still need to be done to check performance
*/
void bli_dgemm_armv8a_sve1024bits_asm_48x6
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
" ldr x7,%[alpha]                            \n\t" // Alpha address      
" ldr x8,%[beta]                             \n\t" // Beta address      
"                                            \n\t" 
" ldr x9,%[cs_c]                             \n\t" // Load cs_c
" lsl x10,x9,#3                              \n\t" // cs_c * sizeof(double)
"                                            \n\t"
" ldr x13,%[rs_c]                            \n\t" // Load rs_c.
" lsl x14,x13,#3                             \n\t" // rs_c * sizeof(double). 
"                                            \n\t"
" mov x11, #16                               \n\t"
" mov x12, #32                               \n\t"
"                                            \n\t"
" add x20,x2,x10                             \n\t" //Load address Column 1 of C
" add x21,x20,x10                            \n\t" //Load address Column 2 of C
" add x22,x21,x10                            \n\t" //Load address Column 3 of C
" add x23,x22,x10                            \n\t" //Load address Column 4 of C
" add x24,x23,x10                            \n\t" //Load address Column 5 of C
"                                            \n\t"
" prfm pldl1keep,[x2]                        \n\t" // Prefetch c.
" prfm pldl1keep,[x20]                       \n\t" // Prefetch c.
" prfm pldl1keep,[x21]                       \n\t" // Prefetch c.
" prfm pldl1keep,[x22]                       \n\t" // Prefetch c.
" prfm pldl1keep,[x23]                       \n\t" // Prefetch c.
" prfm pldl1keep,[x24]                       \n\t" // Prefetch c.
"                                            \n\t"
" whilelt p0.d, xzr, x11                     \n\t" // Creating all true predicate
"                                            \n\t"
" ld1d  z0.d, p0/z, [x0]                     \n\t" // Load a
" ld1d  z1.d, p0/z, [x0, #1, MUL VL]         \n\t"
" ld1d  z2.d, p0/z, [x0, #2, MUL VL]         \n\t"
"                                            \n\t"
" ld1rd  z3.d, p0/z, [x1]                    \n\t" // Load b
" ld1rd  z4.d, p0/z, [x1, #8]                \n\t"
" ld1rd  z5.d, p0/z, [x1, #16]               \n\t"
" ld1rd  z6.d, p0/z, [x1, #24]               \n\t"
" ld1rd  z7.d, p0/z, [x1, #32]               \n\t"
" ld1rd  z8.d, p0/z, [x1, #40]               \n\t"
"                                            \n\t"
"                                            \n\t"
" dup z14.d, #0                              \n\t" // Vector for accummulating column 0
" dup z15.d, #0                              \n\t" // Vector for accummulating column 0
" prfm PLDL1KEEP, [x1, #192]                 \n\t"
" dup z16.d, #0                              \n\t" // Vector for accummulating column 0
" prfm PLDL1KEEP, [x1, #256]                 \n\t"
"                                            \n\t"
" dup z17.d, #0                              \n\t" // Vector for accummulating column 1
" prfm PLDL1KEEP, [x1, #320]                 \n\t"
" dup z18.d, #0                              \n\t" // Vector for accummulating column 1
" dup z19.d, #0                              \n\t" // Vector for accummulating column 1
" prfm PLDL1KEEP, [x0, #1536]                \n\t"
"                                            \n\t"
" dup z20.d, #0                              \n\t" // Vector for accummulating column 2
" prfm PLDL1KEEP, [x0, #1792]                \n\t"
" dup z21.d, #0                              \n\t" // Vector for accummulating column 2
" prfm PLDL1KEEP, [x0, #2048]                \n\t"
" dup z22.d, #0                              \n\t" // Vector for accummulating column 2
"                                            \n\t"
" dup z23.d, #0                              \n\t" // Vector for accummulating column 3
" prfm PLDL1KEEP, [x0, #2304]                \n\t"
" dup z24.d, #0                              \n\t" // Vector for accummulating column 3
" prfm PLDL1KEEP, [x0, #1560]                \n\t"
" dup z25.d, #0                              \n\t" // Vector for accummulating column 3
" prfm PLDL1KEEP, [x0, #2816]                \n\t"
"                                            \n\t"
" dup z26.d, #0                              \n\t" // Vector for accummulating column 4
" dup z27.d, #0                              \n\t" // Vector for accummulating column 4
" dup z28.d, #0                              \n\t" // Vector for accummulating column 4
"                                            \n\t"
" dup z29.d, #0                              \n\t" // Vector for accummulating column 5
" dup z30.d, #0                              \n\t" // Vector for accummulating column 5
" dup z31.d, #0                              \n\t" // Vector for accummulating column 5
"                                            \n\t"
" cmp x5,#0                                  \n\t" // If k_iter == 0, jump to k_left.
" beq .D1024CONSIDERKLEFT                    \n\t"
"                                            \n\t"
" add x0, x0, #384                           \n\t" //update address of A
" add x1, x1, #48                            \n\t" //update address of B
"                                            \n\t"
" cmp x5,1                                   \n\t" // If there is just one k_iter, jump to that one. 
" beq .D1024LASTITER                         \n\t" // (as loop is do-while-like).
"                                            \n\t"
" D1024LOOP:                                 \n\t" // Body
"                                            \n\t"
" fmla z14.d, p0/m, z0.d, z3.d               \n\t"
" prfm PLDL1KEEP, [x0, #2688]                \n\t" // 704 + 64 - 96 = 672
" fmla z15.d, p0/m, z1.d, z3.d               \n\t"
" prfm PLDL1KEEP, [x0, #2944]                \n\t" 
" fmla z16.d, p0/m, z2.d, z3.d               \n\t"
" prfm PLDL1KEEP, [x0, #3200]                \n\t" 
" ld1d   z9.d, p0/z, [x0]                    \n\t" 
" ld1rd  z3.d, p0/z, [x1]                    \n\t"
"                                            \n\t"
" fmla z17.d, p0/m, z0.d, z4.d               \n\t"
" prfm PLDL1KEEP, [x0, #3456]                \n\t" 
" fmla z18.d, p0/m, z1.d, z4.d               \n\t"
" prfm PLDL1KEEP, [x0, #3712]                \n\t" 
" fmla z19.d, p0/m, z2.d, z4.d               \n\t"
" prfm PLDL1KEEP, [x0, #3968]                \n\t" 
" ld1d   z10.d, p0/z, [x0, #1, MUL VL]       \n\t" 
" ld1rd  z4.d, p0/z, [x1, #8]                \n\t"
"                                            \n\t"
" fmla z20.d, p0/m, z0.d, z5.d               \n\t"
" fmla z21.d, p0/m, z1.d, z5.d               \n\t"
" fmla z22.d, p0/m, z2.d, z5.d               \n\t"
" ld1d   z11.d, p0/z, [x0, #2, MUL VL]       \n\t" 
" ld1rd  z5.d, p0/z, [x1, #16]               \n\t"
"                                            \n\t"
" fmla z23.d, p0/m, z0.d, z6.d               \n\t"
" fmla z24.d, p0/m, z1.d, z6.d               \n\t"
" fmla z25.d, p0/m, z2.d, z6.d               \n\t"
" ld1rd  z6.d, p0/z, [x1, #24]               \n\t"
"                                            \n\t"
" fmla z26.d, p0/m, z0.d, z7.d               \n\t"
" fmla z27.d, p0/m, z1.d, z7.d               \n\t"
" fmla z28.d, p0/m, z2.d, z7.d               \n\t"
" ld1rd  z7.d, p0/z, [x1, #32]               \n\t"
"                                            \n\t"
" fmla z29.d, p0/m, z0.d, z8.d               \n\t"
" fmla z30.d, p0/m, z1.d, z8.d               \n\t"
" fmla z31.d, p0/m, z2.d, z8.d               \n\t"
" ld1rd  z8.d, p0/z, [x1, #40]               \n\t"
"                                            \n\t"	//End it 1.
"                                            \n\t"
" fmla z14.d, p0/m, z9.d, z3.d               \n\t"
" fmla z15.d, p0/m, z10.d, z3.d              \n\t"
" fmla z16.d, p0/m, z11.d, z3.d              \n\t"
" ld1d   z0.d, p0/z, [x0, #3, MUL VL]        \n\t" 
" ld1rd  z3.d, p0/z, [x1, #48]               \n\t"
"                                            \n\t"
" fmla z17.d, p0/m, z9.d, z4.d               \n\t"
" fmla z18.d, p0/m, z10.d, z4.d              \n\t"
" fmla z19.d, p0/m, z11.d, z4.d              \n\t"
" ld1d   z1.d, p0/z, [x0, #4, MUL VL]        \n\t" 
" ld1rd  z4.d, p0/z, [x1, #56]               \n\t"
"                                            \n\t"
" fmla z20.d, p0/m, z9.d, z5.d               \n\t"
" fmla z21.d, p0/m, z10.d, z5.d              \n\t"
" fmla z22.d, p0/m, z11.d, z5.d              \n\t"
" prfm PLDL1KEEP, [x1, #336]                 \n\t"
" ld1d   z2.d, p0/z, [x0, #5, MUL VL]        \n\t" 
" prfm PLDL1KEEP, [x1, #400]                 \n\t"
" ld1rd  z5.d, p0/z, [x1, #64]               \n\t"
" prfm PLDL1KEEP, [x1, #464]                 \n\t"
"                                            \n\t"
" fmla z23.d, p0/m, z9.d, z6.d               \n\t"
" fmla z24.d, p0/m, z10.d, z6.d              \n\t"
" fmla z25.d, p0/m, z11.d, z6.d              \n\t"
" ld1rd  z6.d, p0/z, [x1, #72]               \n\t"
"                                            \n\t"
" fmla z26.d, p0/m, z9.d, z7.d               \n\t"
" fmla z27.d, p0/m, z10.d, z7.d              \n\t"
" fmla z28.d, p0/m, z11.d, z7.d              \n\t"
" ld1rd  z7.d, p0/z, [x1, #80]               \n\t"
"                                            \n\t"
" fmla z29.d, p0/m, z9.d, z8.d               \n\t"
" fmla z30.d, p0/m, z10.d, z8.d              \n\t"
" fmla z31.d, p0/m, z11.d, z8.d              \n\t"
" ld1rd  z8.d, p0/z, [x1, #88]               \n\t"
"                                            \n\t"	//End it 2.
"                                            \n\t"
" add x0, x0, #768                           \n\t" // Increasing x0 by 768/2
"                                            \n\t"
" fmla z14.d, p0/m, z0.d, z3.d               \n\t"
" fmla z15.d, p0/m, z1.d, z3.d               \n\t"
" fmla z16.d, p0/m, z2.d, z3.d               \n\t"
" ld1d   z9.d, p0/z, [x0]                    \n\t" 
" ld1rd  z3.d, p0/z, [x1, #96]               \n\t"
"                                            \n\t"
" fmla z17.d, p0/m, z0.d, z4.d               \n\t"
" fmla z18.d, p0/m, z1.d, z4.d               \n\t"
" fmla z19.d, p0/m, z2.d, z4.d               \n\t"
" ld1d   z10.d, p0/z, [x0, #1, MUL VL]       \n\t" 
" ld1rd  z4.d, p0/z, [x1, #104]              \n\t"
"                                            \n\t"
" fmla z20.d, p0/m, z0.d, z5.d               \n\t"
" fmla z21.d, p0/m, z1.d, z5.d               \n\t"
" fmla z22.d, p0/m, z2.d, z5.d               \n\t"
" ld1d   z11.d, p0/z, [x0, #2, MUL VL]       \n\t" 
" ld1rd  z5.d, p0/z, [x1, #112]              \n\t"
"                                            \n\t"
" fmla z23.d, p0/m, z0.d, z6.d               \n\t"
" fmla z24.d, p0/m, z1.d, z6.d               \n\t"
" fmla z25.d, p0/m, z2.d, z6.d               \n\t"
" ld1rd  z6.d, p0/z, [x1, #120]              \n\t"
"                                            \n\t"
" fmla z26.d, p0/m, z0.d, z7.d               \n\t"
" fmla z27.d, p0/m, z1.d, z7.d               \n\t"
" fmla z28.d, p0/m, z2.d, z7.d               \n\t"
" ld1rd  z7.d, p0/z, [x1, #128]              \n\t"
"                                            \n\t"
" fmla z29.d, p0/m, z0.d, z8.d               \n\t"
" fmla z30.d, p0/m, z1.d, z8.d               \n\t"
" fmla z31.d, p0/m, z2.d, z8.d               \n\t"
" ld1rd  z8.d, p0/z, [x1, #136]              \n\t"
"                                            \n\t"	//End it 3.
"                                            \n\t"
" fmla z14.d, p0/m, z9.d, z3.d               \n\t"
" fmla z15.d, p0/m, z10.d, z3.d              \n\t"
" fmla z16.d, p0/m, z11.d, z3.d              \n\t"
" ld1d   z0.d, p0/z, [x0, #3, MUL VL]        \n\t" 
" ld1rd  z3.d, p0/z, [x1, #144]              \n\t"
"                                            \n\t"
" fmla z17.d, p0/m, z9.d, z4.d               \n\t"
" fmla z18.d, p0/m, z10.d, z4.d              \n\t"
" fmla z19.d, p0/m, z11.d, z4.d              \n\t"
" ld1d   z1.d, p0/z, [x0, #4, MUL VL]        \n\t" 
" ld1rd  z4.d, p0/z, [x1, #152]              \n\t"
"                                            \n\t"
" fmla z20.d, p0/m, z9.d, z5.d               \n\t"
" fmla z21.d, p0/m, z10.d, z5.d              \n\t"
" fmla z22.d, p0/m, z11.d, z5.d              \n\t"
" ld1d   z2.d, p0/z, [x0, #5, MUL VL]        \n\t" 
" ld1rd  z5.d, p0/z, [x1, #160]              \n\t"
"                                            \n\t"
" fmla z23.d, p0/m, z9.d, z6.d               \n\t"
" fmla z24.d, p0/m, z10.d, z6.d              \n\t"
" fmla z25.d, p0/m, z11.d, z6.d              \n\t"
" ld1rd  z6.d, p0/z, [x1, #168]              \n\t"
"                                            \n\t"
" fmla z26.d, p0/m, z9.d, z7.d               \n\t"
" fmla z27.d, p0/m, z10.d, z7.d              \n\t"
" fmla z28.d, p0/m, z11.d, z7.d              \n\t"
" ld1rd  z7.d, p0/z, [x1, #176]              \n\t"
"                                            \n\t"
" fmla z29.d, p0/m, z9.d, z8.d               \n\t"
" fmla z30.d, p0/m, z10.d, z8.d              \n\t"
" fmla z31.d, p0/m, z11.d, z8.d              \n\t"
" ld1rd  z8.d, p0/z, [x1, #184]              \n\t"
"                                            \n\t"	//End it 4.
"                                            \n\t"
" add x0, x0, #768                           \n\t" // incremenenting by 768/2 
" add x1, x1, #192                           \n\t"
"                                            \n\t"
" sub x5,x5,1                                \n\t" // i-=1
" cmp x5,1                                   \n\t" // Iterate again if we are not in k_iter == 1.
" bne D1024LOOP                              \n\t"
"                                            \n\t"
".D1024LASTITER:                             \n\t"
"                                            \n\t"
" fmla z14.d, p0/m, z0.d, z3.d               \n\t"
" fmla z15.d, p0/m, z1.d, z3.d               \n\t"
" fmla z16.d, p0/m, z2.d, z3.d               \n\t"
" ld1d   z9.d, p0/z, [x0]                    \n\t" 
" ld1rd  z3.d, p0/z, [x1]                    \n\t"
"                                            \n\t"
" fmla z17.d, p0/m, z0.d, z4.d               \n\t"
" fmla z18.d, p0/m, z1.d, z4.d               \n\t"
" fmla z19.d, p0/m, z2.d, z4.d               \n\t"
" ld1d   z10.d, p0/z, [x0, #1, MUL VL]       \n\t" 
" ld1rd  z4.d, p0/z, [x1, #8]                \n\t"
"                                            \n\t"
" fmla z20.d, p0/m, z0.d, z5.d               \n\t"
" fmla z21.d, p0/m, z1.d, z5.d               \n\t"
" fmla z22.d, p0/m, z2.d, z5.d               \n\t"
" ld1d   z11.d, p0/z, [x0, #2, MUL VL]       \n\t" 
" ld1rd  z5.d, p0/z, [x1, #16]               \n\t"
"                                            \n\t"
" fmla z23.d, p0/m, z0.d, z6.d               \n\t"
" fmla z24.d, p0/m, z1.d, z6.d               \n\t"
" fmla z25.d, p0/m, z2.d, z6.d               \n\t"
" ld1rd  z6.d, p0/z, [x1, #24]               \n\t"
"                                            \n\t"
" fmla z26.d, p0/m, z0.d, z7.d               \n\t"
" fmla z27.d, p0/m, z1.d, z7.d               \n\t"
" fmla z28.d, p0/m, z2.d, z7.d               \n\t"
" ld1rd  z7.d, p0/z, [x1, #32]               \n\t"
"                                            \n\t"
" fmla z29.d, p0/m, z0.d, z8.d               \n\t"
" fmla z30.d, p0/m, z1.d, z8.d               \n\t"
" fmla z31.d, p0/m, z2.d, z8.d               \n\t"
" ld1rd  z8.d, p0/z, [x1, #40]               \n\t"
"                                            \n\t"	//End it 1.
"                                            \n\t"
" fmla z14.d, p0/m, z9.d, z3.d               \n\t"
" fmla z15.d, p0/m, z10.d, z3.d              \n\t"
" fmla z16.d, p0/m, z11.d, z3.d              \n\t"
" ld1d   z0.d, p0/z, [x0, #3, MUL VL]        \n\t" 
" ld1rd  z3.d, p0/z, [x1, #48]               \n\t"
"                                            \n\t"
" fmla z17.d, p0/m, z9.d, z4.d               \n\t"
" fmla z18.d, p0/m, z10.d, z4.d              \n\t"
" fmla z19.d, p0/m, z11.d, z4.d              \n\t"
" ld1d   z1.d, p0/z, [x0, #4, MUL VL]        \n\t" 
" ld1rd  z4.d, p0/z, [x1, #56]               \n\t"
"                                            \n\t"
" fmla z20.d, p0/m, z9.d, z5.d               \n\t"
" fmla z21.d, p0/m, z10.d, z5.d              \n\t"
" fmla z22.d, p0/m, z11.d, z5.d              \n\t"
" ld1d   z2.d, p0/z, [x0, #5, MUL VL]        \n\t" 
" ld1rd  z5.d, p0/z, [x1, #64]               \n\t"
"                                            \n\t"
" fmla z23.d, p0/m, z9.d, z6.d               \n\t"
" fmla z24.d, p0/m, z10.d, z6.d              \n\t"
" fmla z25.d, p0/m, z11.d, z6.d              \n\t"
" ld1rd  z6.d, p0/z, [x1, #72]               \n\t"
"                                            \n\t"
" fmla z26.d, p0/m, z9.d, z7.d               \n\t"
" fmla z27.d, p0/m, z10.d, z7.d              \n\t"
" fmla z28.d, p0/m, z11.d, z7.d              \n\t"
" ld1rd  z7.d, p0/z, [x1, #80]               \n\t"
"                                            \n\t"
" fmla z29.d, p0/m, z9.d, z8.d               \n\t"
" fmla z30.d, p0/m, z10.d, z8.d              \n\t"
" fmla z31.d, p0/m, z11.d, z8.d              \n\t"
" ld1rd  z8.d, p0/z, [x1, #88]               \n\t"
"                                            \n\t"	//End it 2.
" add x0, x0, #768                           \n\t"
"                                            \n\t"
" fmla z14.d, p0/m, z0.d, z3.d               \n\t"
" fmla z15.d, p0/m, z1.d, z3.d               \n\t"
" fmla z16.d, p0/m, z2.d, z3.d               \n\t"
" ld1d   z9.d, p0/z, [x0]                    \n\t" 
" ld1rd  z3.d, p0/z, [x1, #96]               \n\t"
"                                            \n\t"
" fmla z17.d, p0/m, z0.d, z4.d               \n\t"
" fmla z18.d, p0/m, z1.d, z4.d               \n\t"
" fmla z19.d, p0/m, z2.d, z4.d               \n\t"
" ld1d   z10.d, p0/z, [x0, #1, MUL VL]       \n\t" 
" ld1rd  z4.d, p0/z, [x1, #104]              \n\t"
"                                            \n\t"
" fmla z20.d, p0/m, z0.d, z5.d               \n\t"
" fmla z21.d, p0/m, z1.d, z5.d               \n\t"
" fmla z22.d, p0/m, z2.d, z5.d               \n\t"
" ld1d   z11.d, p0/z, [x0, #2, MUL VL]       \n\t" 
" ld1rd  z5.d, p0/z, [x1, #112]              \n\t"
"                                            \n\t"
" fmla z23.d, p0/m, z0.d, z6.d               \n\t"
" fmla z24.d, p0/m, z1.d, z6.d               \n\t"
" fmla z25.d, p0/m, z2.d, z6.d               \n\t"
" ld1rd  z6.d, p0/z, [x1, #120]              \n\t"
"                                            \n\t"
" fmla z26.d, p0/m, z0.d, z7.d               \n\t"
" fmla z27.d, p0/m, z1.d, z7.d               \n\t"
" fmla z28.d, p0/m, z2.d, z7.d               \n\t"
" ld1rd  z7.d, p0/z, [x1, #128]              \n\t"
"                                            \n\t"
" fmla z29.d, p0/m, z0.d, z8.d               \n\t"
" fmla z30.d, p0/m, z1.d, z8.d               \n\t"
" fmla z31.d, p0/m, z2.d, z8.d               \n\t"
" ld1rd  z8.d, p0/z, [x1, #136]              \n\t"
"                                            \n\t"	//End it 3.
"                                            \n\t"
" fmla z14.d, p0/m, z9.d, z3.d               \n\t"
" fmla z15.d, p0/m, z10.d, z3.d              \n\t"
" fmla z16.d, p0/m, z11.d, z3.d              \n\t"
"                                            \n\t"
" fmla z17.d, p0/m, z9.d, z4.d               \n\t"
" fmla z18.d, p0/m, z10.d, z4.d              \n\t"
" fmla z19.d, p0/m, z11.d, z4.d              \n\t"
"                                            \n\t"
" fmla z20.d, p0/m, z9.d, z5.d               \n\t"
" fmla z21.d, p0/m, z10.d, z5.d              \n\t"
" fmla z22.d, p0/m, z11.d, z5.d              \n\t"
"                                            \n\t"
" fmla z23.d, p0/m, z9.d, z6.d               \n\t"
" fmla z24.d, p0/m, z10.d, z6.d              \n\t"
" fmla z25.d, p0/m, z11.d, z6.d              \n\t"
"                                            \n\t"
" add x0, x0, #384                           \n\t"
"                                            \n\t"
" fmla z26.d, p0/m, z9.d, z7.d               \n\t"
" fmla z27.d, p0/m, z10.d, z7.d              \n\t"
" fmla z28.d, p0/m, z11.d, z7.d              \n\t"
"                                            \n\t"
" fmla z29.d, p0/m, z9.d, z8.d               \n\t"
" fmla z30.d, p0/m, z10.d, z8.d              \n\t"
" fmla z31.d, p0/m, z11.d, z8.d              \n\t"
"                                            \n\t"	//End it 4.
"                                            \n\t"
" add x1, x1, #144                           \n\t"
"                                            \n\t"
" .D1024CONSIDERKLEFT:                       \n\t" 
" cmp x6,0                                   \n\t" // If k_left == 0, we are done.
" beq .D1024POSTACCUM                        \n\t" // else, we enter the k_left loop.
"                                            \n\t"
".D1024LOOPKLEFT:                            \n\t"
"                                            \n\t"
" ld1d  z0.d, p0/z, [x0]                     \n\t" // Load a
" ld1d  z1.d, p0/z, [x0, #1, MUL VL]         \n\t"
" ld1d  z2.d, p0/z, [x0, #2, MUL VL]         \n\t"
" add x0, x0, #384                           \n\t"
"                                            \n\t"
" ld1rd  z3.d, p0/z, [x1]                    \n\t" // Load b
" ld1rd  z4.d, p0/z, [x1, #8]                \n\t"
" ld1rd  z5.d, p0/z, [x1, #16]               \n\t"
" ld1rd  z6.d, p0/z, [x1, #24]               \n\t"
" ld1rd  z7.d, p0/z, [x1, #32]               \n\t"
" ld1rd  z8.d, p0/z, [x1, #40]               \n\t"
" add x1, x1, #48                            \n\t"
"                                            \n\t"
" sub x6,x6,1                                \n\t"
"                                            \n\t"
" fmla z14.d, p0/m, z0.d, z3.d               \n\t"
" fmla z15.d, p0/m, z1.d, z3.d               \n\t"
" fmla z16.d, p0/m, z2.d, z3.d               \n\t"
"                                            \n\t"
" fmla z17.d, p0/m, z0.d, z4.d               \n\t"
" fmla z18.d, p0/m, z1.d, z4.d               \n\t"
" fmla z19.d, p0/m, z2.d, z4.d               \n\t"
"                                            \n\t"
" fmla z20.d, p0/m, z0.d, z5.d               \n\t"
" fmla z21.d, p0/m, z1.d, z5.d               \n\t"
" fmla z22.d, p0/m, z2.d, z5.d               \n\t"
"                                            \n\t"
" fmla z23.d, p0/m, z0.d, z6.d               \n\t"
" fmla z24.d, p0/m, z1.d, z6.d               \n\t"
" fmla z25.d, p0/m, z2.d, z6.d               \n\t"
"                                            \n\t"
" fmla z26.d, p0/m, z0.d, z7.d               \n\t"
" fmla z27.d, p0/m, z1.d, z7.d               \n\t"
" fmla z28.d, p0/m, z2.d, z7.d               \n\t"
"                                            \n\t"
" fmla z29.d, p0/m, z0.d, z8.d               \n\t"
" fmla z30.d, p0/m, z1.d, z8.d               \n\t"
" fmla z31.d, p0/m, z2.d, z8.d               \n\t"
"                                            \n\t"
" cmp x6,0                                   \n\t" // Iterate again.
" bne .D1024LOOPKLEFT                        \n\t" // if i!=0.
"                                            \n\t"
" .D1024POSTACCUM:                           \n\t"
"                                            \n\t"
" ld1rd  z6.d, p0/z, [x7]                    \n\t" // Load alpha
" ld1rd  z7.d, p0/z, [x8]                    \n\t" // Load beta
"                                            \n\t"
" cmp x13,#1                                 \n\t" // If rs_c != 1 (column-major)
" bne .D1024GENSTORED                        \n\t"
"                                            \n\t"
" .D1024COLSTORED:                           \n\t" // C is column-major.
"                                            \n\t"
" dup z0.d, #0                               \n\t" 
" dup z1.d, #0                               \n\t" 
" dup z2.d, #0                               \n\t" 
" dup z3.d, #0                               \n\t" 
" dup z4.d, #0                               \n\t" 
" dup z5.d, #0                               \n\t" 
"                                            \n\t"
" fcmp d7,#0.0                               \n\t"
" beq .D1024BETAZEROCOLSTOREDS1              \n\t" // Taking care of the beta==0 case.
"                                            \n\t"
" ld1d  {z0.d}, p0/z, [x2]                   \n\t" // Load column 0 of C
" ld1d  {z1.d}, p0/z, [x2, #1, MUL VL]       \n\t"
" ld1d  {z2.d}, p0/z, [x2, #2, MUL VL]       \n\t"
"                                            \n\t"
" ld1d  {z3.d}, p0/z, [x20]                  \n\t" // Load column 1 of C
" ld1d  {z4.d}, p0/z, [x20, #1, MUL VL]      \n\t"
" ld1d  {z5.d}, p0/z, [x20, #2, MUL VL]      \n\t"
"                                            \n\t"
" fmul z0.d, p0/m, z0.d, z7.d                \n\t" // Scale by beta
" fmul z1.d, p0/m, z1.d, z7.d                \n\t" // Scale by beta
" fmul z2.d, p0/m, z2.d, z7.d                \n\t" // Scale by beta
" fmul z3.d, p0/m, z3.d, z7.d                \n\t" // Scale by beta
" fmul z4.d, p0/m, z4.d, z7.d                \n\t" // Scale by beta
" fmul z5.d, p0/m, z5.d, z7.d                \n\t" // Scale by beta
"                                            \n\t"
" .D1024BETAZEROCOLSTOREDS1:                 \n\t"
"                                            \n\t"
" fmla z0.d, p0/m, z14.d, z6.d               \n\t" // Scale by alpha
" fmla z1.d, p0/m, z15.d, z6.d               \n\t" // Scale by alpha
" fmla z2.d, p0/m, z16.d, z6.d               \n\t" // Scale by alpha
" fmla z3.d, p0/m, z17.d, z6.d               \n\t" // Scale by alpha
" fmla z4.d, p0/m, z18.d, z6.d               \n\t" // Scale by alpha
" fmla z5.d, p0/m, z19.d, z6.d               \n\t" // Scale by alpha
"                                            \n\t"
" st1d  {z0.d}, p0, [x2]                     \n\t" // Store column 0 of C
" st1d  {z1.d}, p0, [x2, #1, MUL VL]         \n\t"
" st1d  {z2.d}, p0, [x2, #2, MUL VL]         \n\t"
"                                            \n\t"
" st1d  {z3.d}, p0, [x20]                    \n\t" // Store column 1 of C
" st1d  {z4.d}, p0, [x20, #1, MUL VL]        \n\t"
" st1d  {z5.d}, p0, [x20, #2, MUL VL]        \n\t"
"                                            \n\t"
" dup z8.d, #0                               \n\t" 
" dup z9.d, #0                               \n\t" 
" dup z10.d, #0                              \n\t" 
" dup z11.d, #0                              \n\t" 
" dup z12.d, #0                              \n\t" 
" dup z13.d, #0                              \n\t" 
"                                            \n\t"
" fcmp d7,#0.0                               \n\t"
" beq .D1024BETAZEROCOLSTOREDS2              \n\t" // Taking care of the beta==0 case.
"                                            \n\t"
" ld1d  {z8.d}, p0/z, [x21]                  \n\t" // Load column 2 of C
" ld1d  {z9.d}, p0/z, [x21, #1, MUL VL]      \n\t"
" ld1d  {z10.d}, p0/z, [x21, #2, MUL VL]     \n\t"
"                                            \n\t"
" ld1d  {z11.d}, p0/z, [x22]                 \n\t" // Load column 3 of C
" ld1d  {z12.d}, p0/z, [x22, #1, MUL VL]     \n\t"
" ld1d  {z13.d}, p0/z, [x22, #2, MUL VL]     \n\t"
"                                            \n\t"
" fmul z8.d, p0/m, z8.d, z7.d                \n\t" // Scale by beta
" fmul z9.d, p0/m, z9.d, z7.d                \n\t" // Scale by beta
" fmul z10.d, p0/m, z10.d, z7.d              \n\t" // Scale by beta
" fmul z11.d, p0/m, z11.d, z7.d              \n\t" // Scale by beta
" fmul z12.d, p0/m, z12.d, z7.d              \n\t" // Scale by beta
" fmul z13.d, p0/m, z13.d, z7.d              \n\t" // Scale by beta
"                                            \n\t"
" .D1024BETAZEROCOLSTOREDS2:                 \n\t"
"                                            \n\t"
" fmla z8.d, p0/m, z20.d, z6.d               \n\t" // Scale by alpha
" fmla z9.d, p0/m, z21.d, z6.d               \n\t" // Scale by alpha
" fmla z10.d, p0/m, z22.d, z6.d              \n\t" // Scale by alpha
" fmla z11.d, p0/m, z23.d, z6.d              \n\t" // Scale by alpha
" fmla z12.d, p0/m, z24.d, z6.d              \n\t" // Scale by alpha
" fmla z13.d, p0/m, z25.d, z6.d              \n\t" // Scale by alpha
"                                            \n\t"
" st1d  {z8.d}, p0, [x21]                    \n\t" // Store column 2 of C
" st1d  {z9.d}, p0, [x21, #1, MUL VL]        \n\t"
" st1d  {z10.d}, p0, [x21, #2, MUL VL]       \n\t"
"                                            \n\t"
" st1d  {z11.d}, p0, [x22]                   \n\t" // Store column 3 of C
" st1d  {z12.d}, p0, [x22, #1, MUL VL]       \n\t"
" st1d  {z13.d}, p0, [x22, #2, MUL VL]       \n\t"
"                                            \n\t"
" dup z14.d, #0                              \n\t" 
" dup z15.d, #0                              \n\t" 
" dup z16.d, #0                              \n\t" 
" dup z17.d, #0                              \n\t" 
" dup z18.d, #0                              \n\t" 
" dup z19.d, #0                              \n\t" 
"                                            \n\t"
" fcmp d7,#0.0                               \n\t"
" beq .D1024BETAZEROCOLSTOREDS3              \n\t" // Taking care of the beta==0 case.
"                                            \n\t"
" ld1d  z14.d, p0/z, [x23]                   \n\t" // Load column 4 of C
" ld1d  z15.d, p0/z, [x23, #1, MUL VL]       \n\t"
" ld1d  z16.d, p0/z, [x23, #2, MUL VL]       \n\t"
"                                            \n\t"
" ld1d  z17.d, p0/z, [x24]                   \n\t" // Load column 5 of C
" ld1d  z18.d, p0/z, [x24, #1, MUL VL]       \n\t"
" ld1d  z19.d, p0/z, [x24, #2, MUL VL]       \n\t"
"                                            \n\t"
" fmul z14.d, p0/m, z14.d, z7.d              \n\t" // Scale by beta
" fmul z15.d, p0/m, z15.d, z7.d              \n\t" // Scale by beta
" fmul z16.d, p0/m, z16.d, z7.d              \n\t" // Scale by beta
" fmul z17.d, p0/m, z17.d, z7.d              \n\t" // Scale by beta
" fmul z18.d, p0/m, z18.d, z7.d              \n\t" // Scale by beta
" fmul z19.d, p0/m, z19.d, z7.d              \n\t" // Scale by beta
"                                            \n\t"
" .D1024BETAZEROCOLSTOREDS3:                 \n\t"
"                                            \n\t"
" fmla z14.d, p0/m, z26.d, z6.d              \n\t" // Scale by alpha
" fmla z15.d, p0/m, z27.d, z6.d              \n\t" // Scale by alpha
" fmla z16.d, p0/m, z28.d, z6.d              \n\t" // Scale by alpha
" fmla z17.d, p0/m, z29.d, z6.d              \n\t" // Scale by alpha
" fmla z18.d, p0/m, z30.d, z6.d              \n\t" // Scale by alpha
" fmla z19.d, p0/m, z31.d, z6.d              \n\t" // Scale by alpha
"                                            \n\t"
" st1d  {z14.d}, p0, [x23]                   \n\t" // Store column 4 of C
" st1d  {z15.d}, p0, [x23, #1, MUL VL]       \n\t"
" st1d  {z16.d}, p0, [x23, #2, MUL VL]       \n\t"
"                                            \n\t"
" st1d  {z17.d}, p0, [x24]                   \n\t" // Store column 5 of C
" st1d  {z18.d}, p0, [x24, #1, MUL VL]       \n\t"
" st1d  {z19.d}, p0, [x24, #2, MUL VL]       \n\t"
"                                            \n\t"
" b .D1024END                                \n\t"
"                                            \n\t"
" .D1024GENSTORED:                           \n\t" // C is general-stride stored.
"                                            \n\t"
" index z8.d, xzr, x13                       \n\t" // Creating index for stride load&store access
" mul x15, x13, x11                          \n\t"
" index z9.d, x15, x13                       \n\t" // Creating index for stride load&store access
" mul x15, x13, x12                          \n\t"
" index z10.d, x15, x13                      \n\t" // Creating index for stride load&store access
"                                            \n\t"
" dup z0.d, #0                               \n\t" 
" dup z1.d, #0                               \n\t" 
" dup z2.d, #0                               \n\t" 
" dup z3.d, #0                               \n\t" 
" dup z4.d, #0                               \n\t" 
" dup z5.d, #0                               \n\t" 
"                                            \n\t"
" fcmp d7,#0.0                               \n\t"
" beq .D1024BETAZEROGENSTOREDS1              \n\t" // Taking care of the beta==0 case.
"                                            \n\t"
" ld1d {z0.d}, p0/z, [x2, z8.d, LSL #3]      \n\t" // Load column 0 of C
" ld1d {z1.d}, p0/z, [x2, z9.d, LSL #3]      \n\t"
" ld1d {z2.d}, p0/z, [x2, z10.d, LSL #3]     \n\t"
"                                            \n\t"
" ld1d {z3.d}, p0/z, [x20, z8.d, LSL #3]     \n\t" // Load column 1 of C
" ld1d {z4.d}, p0/z, [x20, z9.d, LSL #3]     \n\t"
" ld1d {z5.d}, p0/z, [x20, z10.d, LSL #3]    \n\t"
"                                            \n\t"
" fmul z0.d, p0/m, z0.d, z7.d                \n\t" // Scale by beta
" fmul z1.d, p0/m, z1.d, z7.d                \n\t" // Scale by beta
" fmul z2.d, p0/m, z2.d, z7.d                \n\t" // Scale by beta
" fmul z3.d, p0/m, z3.d, z7.d                \n\t" // Scale by beta
" fmul z4.d, p0/m, z4.d, z7.d                \n\t" // Scale by beta
" fmul z5.d, p0/m, z5.d, z7.d                \n\t" // Scale by beta
"                                            \n\t"
" .D1024BETAZEROGENSTOREDS1:                 \n\t"
"                                            \n\t"
" fmla z0.d, p0/m, z14.d, z6.d               \n\t" // Scale by alpha
" fmla z1.d, p0/m, z15.d, z6.d               \n\t" // Scale by alpha
" fmla z2.d, p0/m, z16.d, z6.d               \n\t" // Scale by alpha
" fmla z3.d, p0/m, z17.d, z6.d               \n\t" // Scale by alpha
" fmla z4.d, p0/m, z18.d, z6.d               \n\t" // Scale by alpha
" fmla z5.d, p0/m, z19.d, z6.d               \n\t" // Scale by alpha
"                                            \n\t"
" st1d {z0.d}, p0, [x2, z8.d, LSL #3]        \n\t" // Store column 0 of C
" st1d {z1.d}, p0, [x2, z9.d, LSL #3]        \n\t"
" st1d {z2.d}, p0, [x2, z10.d, LSL #3]       \n\t"
"                                            \n\t"
" st1d {z3.d}, p0, [x20, z8.d, LSL #3]       \n\t" // Store column 1 of C
" st1d {z4.d}, p0, [x20, z9.d, LSL #3]       \n\t"
" st1d {z5.d}, p0, [x20, z10.d, LSL #3]      \n\t"
"                                            \n\t"
" dup z11.d, #0                              \n\t" 
" dup z12.d, #0                              \n\t" 
" dup z13.d, #0                              \n\t" 
" dup z14.d, #0                              \n\t" 
" dup z15.d, #0                              \n\t" 
" dup z16.d, #0                              \n\t" 
"                                            \n\t"
" fcmp d7,#0.0                               \n\t"
" beq .D1024BETAZEROGENSTOREDS2              \n\t" // Taking care of the beta==0 case.
"                                            \n\t"
" ld1d {z11.d}, p0/z, [x21, z8.d, LSL #3]    \n\t" // Load column 2 of C
" ld1d {z12.d}, p0/z, [x21, z9.d, LSL #3]    \n\t"
" ld1d {z13.d}, p0/z, [x21, z10.d, LSL #3]   \n\t"
"                                            \n\t"
" ld1d {z14.d}, p0/z, [x22, z8.d, LSL #3]    \n\t" // Load column 3 of C
" ld1d {z15.d}, p0/z, [x22, z9.d, LSL #3]    \n\t"
" ld1d {z16.d}, p0/z, [x22, z10.d, LSL #3]   \n\t"
"                                            \n\t"
" fmul z11.d, p0/m, z11.d, z7.d              \n\t" // Scale by beta
" fmul z12.d, p0/m, z12.d, z7.d              \n\t" // Scale by beta
" fmul z13.d, p0/m, z13.d, z7.d              \n\t" // Scale by beta
" fmul z14.d, p0/m, z14.d, z7.d              \n\t" // Scale by beta
" fmul z15.d, p0/m, z15.d, z7.d              \n\t" // Scale by beta
" fmul z16.d, p0/m, z16.d, z7.d              \n\t" // Scale by beta
"                                            \n\t"
" .D1024BETAZEROGENSTOREDS2:                 \n\t"
"                                            \n\t"
" fmla z11.d, p0/m, z20.d, z6.d              \n\t" // Scale by alpha
" fmla z12.d, p0/m, z21.d, z6.d              \n\t" // Scale by alpha
" fmla z13.d, p0/m, z22.d, z6.d              \n\t" // Scale by alpha
" fmla z14.d, p0/m, z23.d, z6.d              \n\t" // Scale by alpha
" fmla z15.d, p0/m, z24.d, z6.d              \n\t" // Scale by alpha
" fmla z16.d, p0/m, z25.d, z6.d              \n\t" // Scale by alpha
"                                            \n\t"
" st1d {z11.d}, p0, [x21, z8.d, LSL #3]      \n\t" // Store column 2 of C
" st1d {z12.d}, p0, [x21, z9.d, LSL #3]      \n\t"
" st1d {z13.d}, p0, [x21, z10.d, LSL #3]     \n\t"
"                                            \n\t"
" st1d {z14.d}, p0, [x22, z8.d, LSL #3]      \n\t" // Store column 3 of C
" st1d {z15.d}, p0, [x22, z9.d, LSL #3]      \n\t"
" st1d {z16.d}, p0, [x22, z10.d, LSL #3]     \n\t"
"                                            \n\t"
" dup z0.d, #0                               \n\t" 
" dup z1.d, #0                               \n\t" 
" dup z2.d, #0                               \n\t" 
" dup z3.d, #0                               \n\t" 
" dup z4.d, #0                               \n\t" 
" dup z5.d, #0                               \n\t" 
"                                            \n\t"
" fcmp d7,#0.0                               \n\t"
" beq .D1024BETAZEROGENSTOREDS3              \n\t" // Taking care of the beta==0 case.
"                                            \n\t"
" ld1d {z0.d}, p0/z, [x23, z8.d, LSL #3]     \n\t" // Load column 4 of C
" ld1d {z1.d}, p0/z, [x23, z9.d, LSL #3]     \n\t"
" ld1d {z2.d}, p0/z, [x23, z10.d, LSL #3]    \n\t"
"                                            \n\t"
" ld1d {z3.d}, p0/z, [x24, z8.d, LSL #3]     \n\t" // Load column 5 of C
" ld1d {z4.d}, p0/z, [x24, z9.d, LSL #3]     \n\t"
" ld1d {z5.d}, p0/z, [x24, z10.d, LSL #3]    \n\t"
"                                            \n\t"
" fmul z0.d, p0/m, z0.d, z7.d                \n\t" // Scale by beta
" fmul z1.d, p0/m, z1.d, z7.d                \n\t" // Scale by beta
" fmul z2.d, p0/m, z2.d, z7.d                \n\t" // Scale by beta
" fmul z3.d, p0/m, z3.d, z7.d                \n\t" // Scale by beta
" fmul z4.d, p0/m, z4.d, z7.d                \n\t" // Scale by beta
" fmul z5.d, p0/m, z5.d, z7.d                \n\t" // Scale by beta
"                                            \n\t"
" .D1024BETAZEROGENSTOREDS3:                 \n\t"
"                                            \n\t"
" fmla z0.d, p0/m, z26.d, z6.d               \n\t" // Scale by alpha
" fmla z1.d, p0/m, z27.d, z6.d               \n\t" // Scale by alpha
" fmla z2.d, p0/m, z28.d, z6.d               \n\t" // Scale by alpha
" fmla z3.d, p0/m, z29.d, z6.d               \n\t" // Scale by alpha
" fmla z4.d, p0/m, z30.d, z6.d               \n\t" // Scale by alpha
" fmla z5.d, p0/m, z31.d, z6.d               \n\t" // Scale by alpha
"                                            \n\t"
" st1d z0.d, p0, [x23, z8.d, LSL #3]         \n\t" // Store column 4 of C
" st1d z1.d, p0, [x23, z9.d, LSL #3]         \n\t"
" st1d z2.d, p0, [x23, z10.d, LSL #3]        \n\t"
"                                            \n\t"
" st1d z3.d, p0, [x24, z8.d, LSL #3]         \n\t" // Store column 5 of C
" st1d z4.d, p0, [x24, z9.d, LSL #3]         \n\t"
" st1d z5.d, p0, [x24, z10.d, LSL #3]        \n\t"
"                                            \n\t"
" .D1024END:                                 \n\t" // Done!
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
 "x10","x11","x12","x13","x14","x16","x17",
 "x20","x21","x22","x23","x24","x25","x26",
 "x27",       
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
#if 0
void bli_cgemm_armv8a_opt_4x4
     (
       dim_t               k,
       scomplex*  restrict alpha,
       scomplex*  restrict a,
       scomplex*  restrict b,
       scomplex*  restrict beta,
       scomplex*  restrict c, inc_t rs_c, inc_t cs_c,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
}

void bli_zgemm_armv8a_opt_4x4
     (
       dim_t               k,
       dcomplex*  restrict alpha,
       dcomplex*  restrict a,
       dcomplex*  restrict b,
       dcomplex*  restrict beta,
       dcomplex*  restrict c, inc_t rs_c, inc_t cs_c,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
}

#endif

