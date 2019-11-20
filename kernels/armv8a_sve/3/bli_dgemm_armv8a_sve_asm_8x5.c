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

/*int get_vector_length()
{
    int size = 0;

    __asm__ volatile(
    " mov  %[size], #0          \n\t"
    " incb %[size]             \n\t"
    : [size] "=r" (size)
    :
    :
    ); 
    return size * 8;
}*/

/*
   o 8x5 Double precision micro-kernel 
   o Runnable on ARMv8+SVE, compiled with aarch64 GCC.
   o Use it together with the armv8_sve BLIS configuration.
   o Tested on Juawei arm nodes with ARMIE emulator.
   o Due to the fact that mr and nr depend on size of the vector
     registers, this kernel only works when size=256!!

 * tests still need to be done to check performance
*/



void bli_dgemm_armv8a_sve_asm_8x5
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

    // safety check
    /*if (get_vector_length() != 256)
    {
        fprintf(stderr, "Wrong SVE vector length! You compiled for vector length of 256bits, but your length is %d.\n", get_vector_length());
        exit(EXIT_FAILURE);
    }*/



	void* a_next = bli_auxinfo_next_a( data );
	void* b_next = bli_auxinfo_next_b( data );
	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k0 / 8;
	uint64_t k_left = k0 % 8;
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
// Moved to where the values are loaded and reusing x0 and x1
//" ldr x7,%[alpha]                            \n\t" // Alpha address      
//" ldr x8,%[beta]                             \n\t" // Beta address      
"                                            \n\t" 
" ldr x10,%[cs_c]                             \n\t" // Load cs_c
" lsl x10,x10,#3                              \n\t" // cs_c * sizeof(double)
"                                            \n\t"
" ldr x13,%[rs_c]                            \n\t" // Load rs_c.
//" lsl x14,x13,#3                             \n\t" // rs_c * sizeof(double). 
"                                            \n\t"
" mov x11, #4                                \n\t"
"                                            \n\t"
" add x20,x2,x10                             \n\t" //Load address Column 1 of C
" add x21,x20,x10                            \n\t" //Load address Column 2 of C
" add x22,x21,x10                            \n\t" //Load address Column 3 of C
" add x23,x22,x10                            \n\t" //Load address Column 4 of C
"                                            \n\t"
" prfm pldl1keep,[x2]                        \n\t" // Prefetch c.
" prfm pldl1keep,[x20]                       \n\t" // Prefetch c.
" prfm pldl1keep,[x21]                       \n\t" // Prefetch c.
" prfm pldl1keep,[x22]                       \n\t" // Prefetch c.
" prfm pldl1keep,[x23]                       \n\t" // Prefetch c.
"                                            \n\t"
" ptrue p0.d                                 \n\t" // Creating all true predicate
"                                            \n\t"
" ld1d  z0.d, p0/z, [x0]                     \n\t" // Load a
" ld1d  z1.d, p0/z, [x0, #1, MUL VL]         \n\t"
"                                            \n\t"
" ld1rd  z2.d, p0/z, [x1]                    \n\t" // Load b
" ld1rd  z3.d, p0/z, [x1, #8]                \n\t"
" ld1rd  z4.d, p0/z, [x1, #16]               \n\t"
" ld1rd  z5.d, p0/z, [x1, #24]               \n\t"
" ld1rd  z6.d, p0/z, [x1, #32]               \n\t"
"                                            \n\t"
"                                            \n\t"
" dup z12.d, #0                              \n\t" // Vector for accummulating column 0
" prfm PLDL1KEEP, [x1, #320]                 \n\t"
" dup z13.d, #0                              \n\t" // Vector for accummulating column 0
" prfm PLDL1KEEP, [x1, #384]                 \n\t"
" dup z14.d, #0                              \n\t" // Vector for accummulating column 1
" prfm PLDL1KEEP, [x1, #448]                 \n\t"
" dup z15.d, #0                              \n\t" // Vector for accummulating column 1
" prfm PLDL1KEEP, [x1, #512]                 \n\t"
" dup z16.d, #0                              \n\t" // Vector for accummulating column 2
" prfm PLDL1KEEP, [x1, #576]                 \n\t"
"                                            \n\t"
" dup z17.d, #0                              \n\t" // Vector for accummulating column 2
" prfm PLDL1KEEP, [x0, #512]                 \n\t"
" dup z18.d, #0                              \n\t" // Vector for accummulating column 3
" dup z19.d, #0                              \n\t" // Vector for accummulating column 3
" prfm PLDL1KEEP, [x0, #576]                 \n\t"
" dup z20.d, #0                              \n\t" // Vector for accummulating column 4
" prfm PLDL1KEEP, [x0, #640]                 \n\t"
" dup z21.d, #0                              \n\t" // Vector for accummulating column 4
"                                            \n\t"
" prfm PLDL1KEEP, [x0, #704]                 \n\t"
"                                            \n\t"
" prfm PLDL1KEEP, [x0, #768]                 \n\t"
" prfm PLDL1KEEP, [x0, #832]                 \n\t"
" prfm PLDL1KEEP, [x0, #896]                 \n\t"
" prfm PLDL1KEEP, [x0, #960]                 \n\t"
"                                            \n\t"
" cmp x5,#0                                  \n\t" // If k_iter == 0, jump to k_left.
" beq .D256CONSIDERKLEFT                     \n\t"
"                                            \n\t"
" add x0, x0, #64                            \n\t" //update address of A
" add x1, x1, #40                            \n\t" //update address of B
"                                            \n\t"
" cmp x5,1                                   \n\t" // If there is just one k_iter, jump to that one. 
" beq .D256LASTITER                          \n\t" // (as loop is do-while-like).
"                                            \n\t"
" D256LOOP:                                  \n\t" // Body
"                                            \n\t"
" fmla z12.d, p0/m, z0.d, z2.d               \n\t"
" prfm PLDL1KEEP, [x1, #600]                 \n\t"
" fmla z13.d, p0/m, z1.d, z2.d               \n\t"
" prfm PLDL1KEEP, [x1, #664]                 \n\t"
" ld1rd  z2.d, p0/z, [x1]                    \n\t"
"                                            \n\t"
" fmla z14.d, p0/m, z0.d, z3.d               \n\t"
" prfm PLDL1KEEP, [x1, #728]                 \n\t"
" fmla z15.d, p0/m, z1.d, z3.d               \n\t"
" prfm PLDL1KEEP, [x1, #792]                 \n\t"
" ld1rd  z3.d, p0/z, [x1, #8]                \n\t"
"                                            \n\t"
" fmla z16.d, p0/m, z0.d, z4.d               \n\t"
" prfm PLDL1KEEP, [x1, #856]                 \n\t"
" fmla z17.d, p0/m, z1.d, z4.d               \n\t"
" ld1rd  z4.d, p0/z, [x1, #16]               \n\t"
"                                            \n\t"
" fmla z18.d, p0/m, z0.d, z5.d               \n\t"
" prfm PLDL1KEEP, [x0, #960]                \n\t"  // 448 + 64 -64 
" fmla z19.d, p0/m, z1.d, z5.d               \n\t"
" prfm PLDL1KEEP, [x0, #1024]                \n\t"  
" ld1rd  z5.d, p0/z, [x1, #24]               \n\t"
"                                            \n\t"
" fmla z20.d, p0/m, z0.d, z6.d               \n\t"
" prfm PLDL1KEEP, [x0, #1088]                \n\t"  
" fmla z21.d, p0/m, z1.d, z6.d               \n\t"
" prfm PLDL1KEEP, [x0, #1152]                \n\t"  
" ld1rd  z6.d, p0/z, [x1, #32]               \n\t"
"                                            \n\t"
"                                            \n\t"
" prfm PLDL1KEEP, [x0, #1216]                \n\t"
" prfm PLDL1KEEP, [x0, #1280]                \n\t"  
"                                            \n\t"
" prfm PLDL1KEEP, [x0, #1344]                \n\t"  
" prfm PLDL1KEEP, [x0, #1408]                \n\t"  
"                                            \n\t"
" ld1d   z0.d, p0/z, [x0]                    \n\t" 
" ld1d   z1.d, p0/z, [x0, #1, MUL VL]        \n\t" 
"                                            \n\t"	//End it 1.
"                                            \n\t"
" fmla z12.d, p0/m, z0.d, z2.d               \n\t"
" fmla z13.d, p0/m, z1.d, z2.d               \n\t"
" ld1rd  z2.d, p0/z, [x1, #40]               \n\t"
"                                            \n\t"
" fmla z14.d, p0/m, z0.d, z3.d               \n\t"
" fmla z15.d, p0/m, z1.d, z3.d               \n\t"
" ld1rd  z3.d, p0/z, [x1, #48]               \n\t"
"                                            \n\t"
" fmla z16.d, p0/m, z0.d, z4.d               \n\t"
" fmla z17.d, p0/m, z1.d, z4.d               \n\t"
" ld1rd  z4.d, p0/z, [x1, #56]               \n\t"
"                                            \n\t"
" fmla z18.d, p0/m, z0.d, z5.d               \n\t"
" fmla z19.d, p0/m, z1.d, z5.d               \n\t"
" ld1rd  z5.d, p0/z, [x1, #64]               \n\t"
"                                            \n\t"
" fmla z20.d, p0/m, z0.d, z6.d               \n\t"
" fmla z21.d, p0/m, z1.d, z6.d               \n\t"
" ld1rd  z6.d, p0/z, [x1, #72]               \n\t"
"                                            \n\t"
" ld1d   z0.d, p0/z, [x0, #2, MUL VL]        \n\t" 
" ld1d   z1.d, p0/z, [x0, #3, MUL VL]        \n\t" 
"                                            \n\t"	//End it 2.
"                                            \n\t"
" fmla z12.d, p0/m, z0.d, z2.d               \n\t"
" fmla z13.d, p0/m, z1.d, z2.d               \n\t"
" ld1rd  z2.d, p0/z, [x1, #80]               \n\t"
"                                            \n\t"
" fmla z14.d, p0/m, z0.d, z3.d               \n\t"
" fmla z15.d, p0/m, z1.d, z3.d               \n\t"
" ld1rd  z3.d, p0/z, [x1, #88]               \n\t"
"                                            \n\t"
" fmla z16.d, p0/m, z0.d, z4.d               \n\t"
" fmla z17.d, p0/m, z1.d, z4.d               \n\t"
" ld1rd  z4.d, p0/z, [x1, #96]               \n\t"
"                                            \n\t"
" fmla z18.d, p0/m, z0.d, z5.d               \n\t"
" fmla z19.d, p0/m, z1.d, z5.d               \n\t"
" ld1rd  z5.d, p0/z, [x1, #104]               \n\t"
"                                            \n\t"
" fmla z20.d, p0/m, z0.d, z6.d               \n\t"
" fmla z21.d, p0/m, z1.d, z6.d               \n\t"
" ld1rd  z6.d, p0/z, [x1, #112]               \n\t"
"                                            \n\t"
" ld1d   z0.d, p0/z, [x0, #4, MUL VL]        \n\t" 
" ld1d   z1.d, p0/z, [x0, #5, MUL VL]        \n\t" 
"                                            \n\t"	//End it 3.
"                                            \n\t"
" fmla z12.d, p0/m, z0.d, z2.d               \n\t"
" fmla z13.d, p0/m, z1.d, z2.d               \n\t"
" ld1rd  z2.d, p0/z, [x1, #120]               \n\t"
"                                            \n\t"
" fmla z14.d, p0/m, z0.d, z3.d               \n\t"
" fmla z15.d, p0/m, z1.d, z3.d               \n\t"
" ld1rd  z3.d, p0/z, [x1, #128]               \n\t"
"                                            \n\t"
" fmla z16.d, p0/m, z0.d, z4.d               \n\t"
" fmla z17.d, p0/m, z1.d, z4.d               \n\t"
" ld1rd  z4.d, p0/z, [x1, #136]               \n\t"
"                                            \n\t"
" fmla z18.d, p0/m, z0.d, z5.d               \n\t"
" fmla z19.d, p0/m, z1.d, z5.d               \n\t"
" ld1rd  z5.d, p0/z, [x1, #144]               \n\t"
"                                            \n\t"
" fmla z20.d, p0/m, z0.d, z6.d               \n\t"
" fmla z21.d, p0/m, z1.d, z6.d               \n\t"
" ld1rd  z6.d, p0/z, [x1, #152]               \n\t"
"                                            \n\t"
" ld1d   z0.d, p0/z, [x0, #6, MUL VL]        \n\t" 
" ld1d   z1.d, p0/z, [x0, #7, MUL VL]        \n\t" 
" add x0, x0, #256                            \n\t" //update address of A
"                                            \n\t"	//End it 4.
"                                            \n\t"
" fmla z12.d, p0/m, z0.d, z2.d               \n\t"
" fmla z13.d, p0/m, z1.d, z2.d               \n\t"
" ld1rd  z2.d, p0/z, [x1, #160]               \n\t"
"                                            \n\t"
" fmla z14.d, p0/m, z0.d, z3.d               \n\t"
" fmla z15.d, p0/m, z1.d, z3.d               \n\t"
" ld1rd  z3.d, p0/z, [x1, #168]               \n\t"
"                                            \n\t"
" fmla z16.d, p0/m, z0.d, z4.d               \n\t"
" fmla z17.d, p0/m, z1.d, z4.d               \n\t"
" ld1rd  z4.d, p0/z, [x1, #176]               \n\t"
"                                            \n\t"
" fmla z18.d, p0/m, z0.d, z5.d               \n\t"
" fmla z19.d, p0/m, z1.d, z5.d               \n\t"
" ld1rd  z5.d, p0/z, [x1, #184]               \n\t"
"                                            \n\t"
" fmla z20.d, p0/m, z0.d, z6.d               \n\t"
" fmla z21.d, p0/m, z1.d, z6.d               \n\t"
" ld1rd  z6.d, p0/z, [x1, #192]               \n\t"
"                                            \n\t"
" ld1d   z0.d, p0/z, [x0]                    \n\t" 
" ld1d   z1.d, p0/z, [x0, #1, MUL VL]        \n\t" 
"                                            \n\t"	//End it 5.
"                                            \n\t"
" fmla z12.d, p0/m, z0.d, z2.d               \n\t"
" fmla z13.d, p0/m, z1.d, z2.d               \n\t"
" ld1rd  z2.d, p0/z, [x1, #200]               \n\t"
"                                            \n\t"
" fmla z14.d, p0/m, z0.d, z3.d               \n\t"
" fmla z15.d, p0/m, z1.d, z3.d               \n\t"
" ld1rd  z3.d, p0/z, [x1, #208]               \n\t"
"                                            \n\t"
" fmla z16.d, p0/m, z0.d, z4.d               \n\t"
" fmla z17.d, p0/m, z1.d, z4.d               \n\t"
" ld1rd  z4.d, p0/z, [x1, #216]               \n\t"
"                                            \n\t"
" fmla z18.d, p0/m, z0.d, z5.d               \n\t"
" fmla z19.d, p0/m, z1.d, z5.d               \n\t"
" ld1rd  z5.d, p0/z, [x1, #224]               \n\t"
"                                            \n\t"
" fmla z20.d, p0/m, z0.d, z6.d               \n\t"
" fmla z21.d, p0/m, z1.d, z6.d               \n\t"
" ld1rd  z6.d, p0/z, [x1, #232]               \n\t"
"                                            \n\t"
" ld1d   z0.d, p0/z, [x0, #2, MUL VL]        \n\t" 
" ld1d   z1.d, p0/z, [x0, #3, MUL VL]        \n\t" 
"                                            \n\t"	//End it 6.
"                                            \n\t"
" fmla z12.d, p0/m, z0.d, z2.d               \n\t"
" fmla z13.d, p0/m, z1.d, z2.d               \n\t"
" ld1rd  z2.d, p0/z, [x1, #240]               \n\t"
"                                            \n\t"
" fmla z14.d, p0/m, z0.d, z3.d               \n\t"
" fmla z15.d, p0/m, z1.d, z3.d               \n\t"
" ld1rd  z3.d, p0/z, [x1, #248]               \n\t"
"                                            \n\t"
" fmla z16.d, p0/m, z0.d, z4.d               \n\t"
" fmla z17.d, p0/m, z1.d, z4.d               \n\t"
" ld1rd  z4.d, p0/z, [x1, #256]               \n\t"
"                                            \n\t"
" fmla z18.d, p0/m, z0.d, z5.d               \n\t"
" fmla z19.d, p0/m, z1.d, z5.d               \n\t"
" ld1rd  z5.d, p0/z, [x1, #264]               \n\t"
"                                            \n\t"
" fmla z20.d, p0/m, z0.d, z6.d               \n\t"
" fmla z21.d, p0/m, z1.d, z6.d               \n\t"
" ld1rd  z6.d, p0/z, [x1, #272]               \n\t"
"                                            \n\t"
" ld1d   z0.d, p0/z, [x0, #4, MUL VL]        \n\t" 
" ld1d   z1.d, p0/z, [x0, #5, MUL VL]        \n\t" 
"                                            \n\t"	//End it 7.
"                                            \n\t"
" fmla z12.d, p0/m, z0.d, z2.d               \n\t"
" fmla z13.d, p0/m, z1.d, z2.d               \n\t"
" ld1rd  z2.d, p0/z, [x1, #280]               \n\t"
"                                            \n\t"
" fmla z14.d, p0/m, z0.d, z3.d               \n\t"
" fmla z15.d, p0/m, z1.d, z3.d               \n\t"
" ld1rd  z3.d, p0/z, [x1, #288]               \n\t"
"                                            \n\t"
" fmla z16.d, p0/m, z0.d, z4.d               \n\t"
" fmla z17.d, p0/m, z1.d, z4.d               \n\t"
" ld1rd  z4.d, p0/z, [x1, #296]               \n\t"
"                                            \n\t"
" fmla z18.d, p0/m, z0.d, z5.d               \n\t"
" fmla z19.d, p0/m, z1.d, z5.d               \n\t"
" ld1rd  z5.d, p0/z, [x1, #304]               \n\t"
"                                            \n\t"
" fmla z20.d, p0/m, z0.d, z6.d               \n\t"
" fmla z21.d, p0/m, z1.d, z6.d               \n\t"
" ld1rd  z6.d, p0/z, [x1, #312]               \n\t"
"                                            \n\t"
" ld1d   z0.d, p0/z, [x0, #6, MUL VL]        \n\t" 
" ld1d   z1.d, p0/z, [x0, #7, MUL VL]        \n\t" 
"                                            \n\t"	//End it 8.
"                                            \n\t"
" add x0, x0, #256                           \n\t" // incremenenting by 256 
" add x1, x1, #320                           \n\t"
"                                            \n\t"
" sub x5,x5,1                                \n\t" // i-=1
" cmp x5,1                                   \n\t" // Iterate again if we are not in k_iter == 1.
" bne D256LOOP                               \n\t"
"                                            \n\t"
" .D256LASTITER:                             \n\t"
"                                            \n\t"
" fmla z12.d, p0/m, z0.d, z2.d               \n\t"
" fmla z13.d, p0/m, z1.d, z2.d               \n\t"
" ld1rd  z2.d, p0/z, [x1]                    \n\t"
"                                            \n\t"
" fmla z14.d, p0/m, z0.d, z3.d               \n\t"
" fmla z15.d, p0/m, z1.d, z3.d               \n\t"
" ld1rd  z3.d, p0/z, [x1, #8]                \n\t"
"                                            \n\t"
" fmla z16.d, p0/m, z0.d, z4.d               \n\t"
" fmla z17.d, p0/m, z1.d, z4.d               \n\t"
" ld1rd  z4.d, p0/z, [x1, #16]               \n\t"
"                                            \n\t"
" fmla z18.d, p0/m, z0.d, z5.d               \n\t"
" fmla z19.d, p0/m, z1.d, z5.d               \n\t"
" ld1rd  z5.d, p0/z, [x1, #24]               \n\t"
"                                            \n\t"
" fmla z20.d, p0/m, z0.d, z6.d               \n\t"
" fmla z21.d, p0/m, z1.d, z6.d               \n\t"
" ld1rd  z6.d, p0/z, [x1, #32]               \n\t"
"                                            \n\t"
" ld1d   z0.d, p0/z, [x0]                    \n\t" 
" ld1d   z1.d, p0/z, [x0, #1, MUL VL]        \n\t" 
"                                            \n\t"	//End it 1.
"                                            \n\t"
" fmla z12.d, p0/m, z0.d, z2.d               \n\t"
" fmla z13.d, p0/m, z1.d, z2.d               \n\t"
" ld1rd  z2.d, p0/z, [x1, #40]               \n\t"
"                                            \n\t"
" fmla z14.d, p0/m, z0.d, z3.d               \n\t"
" fmla z15.d, p0/m, z1.d, z3.d               \n\t"
" ld1rd  z3.d, p0/z, [x1, #48]               \n\t"
"                                            \n\t"
" fmla z16.d, p0/m, z0.d, z4.d               \n\t"
" fmla z17.d, p0/m, z1.d, z4.d               \n\t"
" ld1rd  z4.d, p0/z, [x1, #56]               \n\t"
"                                            \n\t"
" fmla z18.d, p0/m, z0.d, z5.d               \n\t"
" fmla z19.d, p0/m, z1.d, z5.d               \n\t"
" ld1rd  z5.d, p0/z, [x1, #64]               \n\t"
"                                            \n\t"
" fmla z20.d, p0/m, z0.d, z6.d               \n\t"
" fmla z21.d, p0/m, z1.d, z6.d               \n\t"
" ld1rd  z6.d, p0/z, [x1, #72]               \n\t"
"                                            \n\t"
" ld1d   z0.d, p0/z, [x0, #2, MUL VL]        \n\t" 
" ld1d   z1.d, p0/z, [x0, #3, MUL VL]        \n\t" 
"                                            \n\t"	//End it 2.
"                                            \n\t"
"                                            \n\t"
" fmla z12.d, p0/m, z0.d, z2.d               \n\t"
" fmla z13.d, p0/m, z1.d, z2.d               \n\t"
" ld1rd  z2.d, p0/z, [x1, #80]               \n\t"
"                                            \n\t"
" fmla z14.d, p0/m, z0.d, z3.d               \n\t"
" fmla z15.d, p0/m, z1.d, z3.d               \n\t"
" ld1rd  z3.d, p0/z, [x1, #88]               \n\t"
"                                            \n\t"
" fmla z16.d, p0/m, z0.d, z4.d               \n\t"
" fmla z17.d, p0/m, z1.d, z4.d               \n\t"
" ld1rd  z4.d, p0/z, [x1, #96]               \n\t"
"                                            \n\t"
" fmla z18.d, p0/m, z0.d, z5.d               \n\t"
" fmla z19.d, p0/m, z1.d, z5.d               \n\t"
" ld1rd  z5.d, p0/z, [x1, #104]               \n\t"
"                                            \n\t"
" fmla z20.d, p0/m, z0.d, z6.d               \n\t"
" fmla z21.d, p0/m, z1.d, z6.d               \n\t"
" ld1rd  z6.d, p0/z, [x1, #112]               \n\t"
"                                            \n\t"
" ld1d   z0.d, p0/z, [x0, #4, MUL VL]        \n\t" 
" ld1d   z1.d, p0/z, [x0, #5, MUL VL]        \n\t" 
"                                            \n\t"	//End it 3.
"                                            \n\t"
"                                            \n\t"
" fmla z12.d, p0/m, z0.d, z2.d               \n\t"
" fmla z13.d, p0/m, z1.d, z2.d               \n\t"
" ld1rd  z2.d, p0/z, [x1, #120]               \n\t"
"                                            \n\t"
" fmla z14.d, p0/m, z0.d, z3.d               \n\t"
" fmla z15.d, p0/m, z1.d, z3.d               \n\t"
" ld1rd  z3.d, p0/z, [x1, #128]               \n\t"
"                                            \n\t"
" fmla z16.d, p0/m, z0.d, z4.d               \n\t"
" fmla z17.d, p0/m, z1.d, z4.d               \n\t"
" ld1rd  z4.d, p0/z, [x1, #136]               \n\t"
"                                            \n\t"
" fmla z18.d, p0/m, z0.d, z5.d               \n\t"
" fmla z19.d, p0/m, z1.d, z5.d               \n\t"
" ld1rd  z5.d, p0/z, [x1, #144]               \n\t"
"                                            \n\t"
" fmla z20.d, p0/m, z0.d, z6.d               \n\t"
" fmla z21.d, p0/m, z1.d, z6.d               \n\t"
" ld1rd  z6.d, p0/z, [x1, #152]               \n\t"
"                                            \n\t"
" ld1d   z0.d, p0/z, [x0, #6, MUL VL]        \n\t" 
" ld1d   z1.d, p0/z, [x0, #7, MUL VL]        \n\t" 
" add x0, x0, #256                            \n\t" //update address of A
"                                            \n\t"	//End it 4.
"                                            \n\t"
"                                            \n\t"
" fmla z12.d, p0/m, z0.d, z2.d               \n\t"
" fmla z13.d, p0/m, z1.d, z2.d               \n\t"
" ld1rd  z2.d, p0/z, [x1, #160]               \n\t"
"                                            \n\t"
" fmla z14.d, p0/m, z0.d, z3.d               \n\t"
" fmla z15.d, p0/m, z1.d, z3.d               \n\t"
" ld1rd  z3.d, p0/z, [x1, #168]               \n\t"
"                                            \n\t"
" fmla z16.d, p0/m, z0.d, z4.d               \n\t"
" fmla z17.d, p0/m, z1.d, z4.d               \n\t"
" ld1rd  z4.d, p0/z, [x1, #176]               \n\t"
"                                            \n\t"
" fmla z18.d, p0/m, z0.d, z5.d               \n\t"
" fmla z19.d, p0/m, z1.d, z5.d               \n\t"
" ld1rd  z5.d, p0/z, [x1, #184]               \n\t"
"                                            \n\t"
" fmla z20.d, p0/m, z0.d, z6.d               \n\t"
" fmla z21.d, p0/m, z1.d, z6.d               \n\t"
" ld1rd  z6.d, p0/z, [x1, #192]               \n\t"
"                                            \n\t"
" ld1d   z0.d, p0/z, [x0]                    \n\t" 
" ld1d   z1.d, p0/z, [x0, #1, MUL VL]        \n\t" 
"                                            \n\t"	//End it 5.
"                                            \n\t"
"                                            \n\t"
" fmla z12.d, p0/m, z0.d, z2.d               \n\t"
" fmla z13.d, p0/m, z1.d, z2.d               \n\t"
" ld1rd  z2.d, p0/z, [x1, #200]               \n\t"
"                                            \n\t"
" fmla z14.d, p0/m, z0.d, z3.d               \n\t"
" fmla z15.d, p0/m, z1.d, z3.d               \n\t"
" ld1rd  z3.d, p0/z, [x1, #208]               \n\t"
"                                            \n\t"
" fmla z16.d, p0/m, z0.d, z4.d               \n\t"
" fmla z17.d, p0/m, z1.d, z4.d               \n\t"
" ld1rd  z4.d, p0/z, [x1, #216]               \n\t"
"                                            \n\t"
" fmla z18.d, p0/m, z0.d, z5.d               \n\t"
" fmla z19.d, p0/m, z1.d, z5.d               \n\t"
" ld1rd  z5.d, p0/z, [x1, #224]               \n\t"
"                                            \n\t"
" fmla z20.d, p0/m, z0.d, z6.d               \n\t"
" fmla z21.d, p0/m, z1.d, z6.d               \n\t"
" ld1rd  z6.d, p0/z, [x1, #232]               \n\t"
"                                            \n\t"
" ld1d   z0.d, p0/z, [x0, #2, MUL VL]        \n\t" 
" ld1d   z1.d, p0/z, [x0, #3, MUL VL]        \n\t" 
"                                            \n\t"	//End it 6.
"                                            \n\t"
"                                            \n\t"
" fmla z12.d, p0/m, z0.d, z2.d               \n\t"
" fmla z13.d, p0/m, z1.d, z2.d               \n\t"
" ld1rd  z2.d, p0/z, [x1, #240]               \n\t"
"                                            \n\t"
" fmla z14.d, p0/m, z0.d, z3.d               \n\t"
" fmla z15.d, p0/m, z1.d, z3.d               \n\t"
" ld1rd  z3.d, p0/z, [x1, #248]               \n\t"
"                                            \n\t"
" fmla z16.d, p0/m, z0.d, z4.d               \n\t"
" fmla z17.d, p0/m, z1.d, z4.d               \n\t"
" ld1rd  z4.d, p0/z, [x1, #256]               \n\t"
"                                            \n\t"
" fmla z18.d, p0/m, z0.d, z5.d               \n\t"
" fmla z19.d, p0/m, z1.d, z5.d               \n\t"
" ld1rd  z5.d, p0/z, [x1, #264]               \n\t"
"                                            \n\t"
" fmla z20.d, p0/m, z0.d, z6.d               \n\t"
" fmla z21.d, p0/m, z1.d, z6.d               \n\t"
" ld1rd  z6.d, p0/z, [x1, #272]               \n\t"
"                                            \n\t"
" ld1d   z0.d, p0/z, [x0, #4, MUL VL]        \n\t" 
" ld1d   z1.d, p0/z, [x0, #5, MUL VL]        \n\t" 
"                                            \n\t"	//End it 7.
"                                            \n\t"
" fmla z12.d, p0/m, z0.d, z2.d               \n\t"
" fmla z13.d, p0/m, z1.d, z2.d               \n\t"
"                                            \n\t"
" fmla z14.d, p0/m, z0.d, z3.d               \n\t"
" fmla z15.d, p0/m, z1.d, z3.d               \n\t"
"                                            \n\t"
" fmla z16.d, p0/m, z0.d, z4.d               \n\t"
" fmla z17.d, p0/m, z1.d, z4.d               \n\t"
"                                            \n\t"
" fmla z18.d, p0/m, z0.d, z5.d               \n\t"
" fmla z19.d, p0/m, z1.d, z5.d               \n\t"
"                                            \n\t"
" fmla z20.d, p0/m, z0.d, z6.d               \n\t"
" fmla z21.d, p0/m, z1.d, z6.d               \n\t"
"                                            \n\t"
"                                            \n\t"
"                                            \n\t"	//End it 8.
"                                            \n\t"
" add x0, x0, #192                           \n\t" // incremenenting by 256*3/4 
" add x1, x1, #280                           \n\t" // incremenenting by 320*7/8 
"                                            \n\t"
" .D256CONSIDERKLEFT:                        \n\t" 
" cmp x6,0                                   \n\t" // If k_left == 0, we are done.
" beq .D256POSTACCUM                         \n\t" // else, we enter the k_left loop.
"                                            \n\t"
".D256LOOPKLEFT:                             \n\t"
"                                            \n\t"
" ld1d  z0.d, p0/z, [x0]                     \n\t" // Load a
" ld1d  z1.d, p0/z, [x0, #1, MUL VL]         \n\t"
" add x0, x0, #64                            \n\t"
"                                            \n\t"
" ld1rd  z2.d, p0/z, [x1]                    \n\t" // Load b
" ld1rd  z3.d, p0/z, [x1, #8]                \n\t"
" ld1rd  z4.d, p0/z, [x1, #16]               \n\t"
" ld1rd  z5.d, p0/z, [x1, #24]               \n\t"
" ld1rd  z6.d, p0/z, [x1, #32]               \n\t"
" add x1, x1, #40                            \n\t"
"                                            \n\t"
" sub x6,x6,1                                \n\t"
"                                            \n\t"
" fmla z12.d, p0/m, z0.d, z2.d               \n\t"
" fmla z13.d, p0/m, z1.d, z2.d               \n\t"
"                                            \n\t"
" fmla z14.d, p0/m, z0.d, z3.d               \n\t"
" fmla z15.d, p0/m, z1.d, z3.d               \n\t"
"                                            \n\t"
" fmla z16.d, p0/m, z0.d, z4.d               \n\t"
" fmla z17.d, p0/m, z1.d, z4.d               \n\t"
"                                            \n\t"
" fmla z18.d, p0/m, z0.d, z5.d               \n\t"
" fmla z19.d, p0/m, z1.d, z5.d               \n\t"
"                                            \n\t"
" fmla z20.d, p0/m, z0.d, z6.d               \n\t"
" fmla z21.d, p0/m, z1.d, z6.d               \n\t"
"                                            \n\t"
" cmp x6,0                                   \n\t" // Iterate again.
" bne .D256LOOPKLEFT                         \n\t" // if i!=0.
"                                            \n\t"
" .D256POSTACCUM:                            \n\t"
"                                            \n\t"
" ldr x0,%[alpha]                            \n\t" // Alpha address      
" ldr x1,%[beta]                             \n\t" // Beta address      
" ld1rd  z10.d, p0/z, [x0]                    \n\t" // Load alpha
" ld1rd  z11.d, p0/z, [x1]                    \n\t" // Load beta
"                                            \n\t"
" cmp x13,#1                                 \n\t" // If rs_c != 1 (column-major)
" bne .D256GENSTORED                         \n\t"
"                                            \n\t"
" .D256COLSTORED:                            \n\t" // C is column-major.
"                                            \n\t"
" dup z0.d, #0                               \n\t" 
" dup z1.d, #0                               \n\t" 
" dup z2.d, #0                               \n\t" 
" dup z3.d, #0                               \n\t" 
" dup z4.d, #0                               \n\t" 
" dup z5.d, #0                               \n\t" 
" dup z6.d, #0                               \n\t" 
" dup z7.d, #0                               \n\t" 
" dup z8.d, #0                               \n\t" 
" dup z9.d, #0                               \n\t" 
"                                            \n\t"
" fcmp d11,#0.0                               \n\t"
" beq .D256BETAZEROCOLSTOREDS1               \n\t" // Taking care of the beta==0 case.
"                                            \n\t"
" ld1d  {z0.d}, p0/z, [x2]                   \n\t" // Load column 0 of C
" ld1d  {z1.d}, p0/z, [x2, #1, MUL VL]       \n\t"
"                                            \n\t"
" ld1d  {z2.d}, p0/z, [x20]                  \n\t" // Load column 1 of C
" ld1d  {z3.d}, p0/z, [x20, #1, MUL VL]      \n\t"
"                                            \n\t"
" ld1d  {z4.d}, p0/z, [x21]                  \n\t" // Load column 2 of C
" ld1d  {z5.d}, p0/z, [x21, #1, MUL VL]      \n\t"
"                                            \n\t"
" ld1d  {z6.d}, p0/z, [x22]                  \n\t" // Load column 3 of C
" ld1d  {z7.d}, p0/z, [x22, #1, MUL VL]      \n\t"
"                                            \n\t"
" ld1d  {z8.d}, p0/z, [x23]                  \n\t" // Load column 3 of C
" ld1d  {z9.d}, p0/z, [x23, #1, MUL VL]      \n\t"
"                                            \n\t"
" fmul z0.d, p0/m, z0.d, z11.d                \n\t" // Scale by beta
" fmul z1.d, p0/m, z1.d, z11.d                \n\t" // Scale by beta
" fmul z2.d, p0/m, z2.d, z11.d                \n\t" // Scale by beta
" fmul z3.d, p0/m, z3.d, z11.d                \n\t" // Scale by beta
" fmul z4.d, p0/m, z4.d, z11.d                \n\t" // Scale by beta
" fmul z5.d, p0/m, z5.d, z11.d                \n\t" // Scale by beta
" fmul z6.d, p0/m, z6.d, z11.d                \n\t" // Scale by beta
" fmul z7.d, p0/m, z7.d, z11.d                \n\t" // Scale by beta
" fmul z8.d, p0/m, z8.d, z11.d                \n\t" // Scale by beta
" fmul z9.d, p0/m, z9.d, z11.d                \n\t" // Scale by beta
"                                            \n\t"
" .D256BETAZEROCOLSTOREDS1:                  \n\t"
"                                            \n\t"
" fmla z0.d, p0/m, z12.d, z10.d               \n\t" // Scale by alpha
" fmla z1.d, p0/m, z13.d, z10.d               \n\t" // Scale by alpha
" fmla z2.d, p0/m, z14.d, z10.d               \n\t" // Scale by alpha
" fmla z3.d, p0/m, z15.d, z10.d               \n\t" // Scale by alpha
" fmla z4.d, p0/m, z16.d, z10.d               \n\t" // Scale by alpha
" fmla z5.d, p0/m, z17.d, z10.d               \n\t" // Scale by alpha
" fmla z6.d, p0/m, z18.d, z10.d               \n\t" // Scale by alpha
" fmla z7.d, p0/m, z19.d, z10.d               \n\t" // Scale by alpha
" fmla z8.d, p0/m, z20.d, z10.d               \n\t" // Scale by alpha
" fmla z9.d, p0/m, z21.d, z10.d               \n\t" // Scale by alpha
"                                            \n\t"
" st1d  {z0.d}, p0, [x2]                     \n\t" // Store column 0 of C
" st1d  {z1.d}, p0, [x2, #1, MUL VL]         \n\t"
"                                            \n\t"
" st1d  {z2.d}, p0, [x20]                    \n\t" // Store column 1 of C
" st1d  {z3.d}, p0, [x20, #1, MUL VL]        \n\t"
"                                            \n\t"
" st1d  {z4.d}, p0, [x21]                    \n\t" // Store column 2 of C
" st1d  {z5.d}, p0, [x21, #1, MUL VL]        \n\t"
"                                            \n\t"
" st1d  {z6.d}, p0, [x22]                    \n\t" // Store column 3 of C
" st1d  {z7.d}, p0, [x22, #1, MUL VL]        \n\t"
"                                            \n\t"
" st1d  {z8.d}, p0, [x23]                    \n\t" // Store column 4 of C
" st1d  {z9.d}, p0, [x23, #1, MUL VL]        \n\t"
"                                            \n\t"
" b .D256END                                 \n\t"
"                                            \n\t"
" .D256GENSTORED:                            \n\t" // C is general-stride stored.
"                                            \n\t"
" index z30.d, xzr, x13                       \n\t" // Creating index for stride load&store access
" mul x3, x13, x11                          \n\t"
" index z31.d, x3, x13                       \n\t" // Creating index for stride load&store access
"                                            \n\t"
" dup z0.d, #0                               \n\t" 
" dup z1.d, #0                               \n\t" 
" dup z2.d, #0                               \n\t" 
" dup z3.d, #0                               \n\t" 
" dup z4.d, #0                               \n\t" 
" dup z5.d, #0                               \n\t" 
" dup z6.d, #0                               \n\t" 
" dup z7.d, #0                               \n\t" 
" dup z8.d, #0                               \n\t" 
" dup z9.d, #0                               \n\t" 
"                                            \n\t"
" fcmp d11,#0.0                               \n\t"
" beq .D256BETAZEROGENSTOREDS1               \n\t" // Taking care of the beta==0 case.
"                                            \n\t"
" ld1d {z0.d}, p0/z, [x2, z30.d, LSL #3]     \n\t" // Load column 0 of C
" ld1d {z1.d}, p0/z, [x2, z31.d, LSL #3]     \n\t"
"                                            \n\t"
" ld1d {z2.d}, p0/z, [x20, z30.d, LSL #3]    \n\t" // Load column 1 of C
" ld1d {z3.d}, p0/z, [x20, z31.d, LSL #3]    \n\t"
"                                            \n\t"
" ld1d {z4.d}, p0/z, [x21, z30.d, LSL #3]    \n\t" // Load column 2 of C
" ld1d {z5.d}, p0/z, [x21, z31.d, LSL #3]    \n\t"
"                                            \n\t"
" ld1d {z6.d}, p0/z, [x22, z30.d, LSL #3]    \n\t" // Load column 3 of C
" ld1d {z7.d}, p0/z, [x22, z31.d, LSL #3]    \n\t"
"                                            \n\t"
" ld1d {z8.d}, p0/z, [x23, z30.d, LSL #3]    \n\t" // Load column 4 of C
" ld1d {z9.d}, p0/z, [x23, z31.d, LSL #3]    \n\t"
"                                            \n\t"
" fmul z0.d, p0/m, z0.d, z11.d                \n\t" // Scale by beta
" fmul z1.d, p0/m, z1.d, z11.d                \n\t" // Scale by beta
" fmul z2.d, p0/m, z2.d, z11.d                \n\t" // Scale by beta
" fmul z3.d, p0/m, z3.d, z11.d                \n\t" // Scale by beta
" fmul z4.d, p0/m, z4.d, z11.d                \n\t" // Scale by beta
" fmul z5.d, p0/m, z5.d, z11.d                \n\t" // Scale by beta
" fmul z6.d, p0/m, z6.d, z11.d                \n\t" // Scale by beta
" fmul z7.d, p0/m, z7.d, z11.d                \n\t" // Scale by beta
" fmul z8.d, p0/m, z8.d, z11.d                \n\t" // Scale by beta
" fmul z9.d, p0/m, z9.d, z11.d                \n\t" // Scale by beta
"                                            \n\t"
" .D256BETAZEROGENSTOREDS1:                  \n\t"
"                                            \n\t"
" fmla z0.d, p0/m, z12.d, z10.d               \n\t" // Scale by alpha
" fmla z1.d, p0/m, z13.d, z10.d               \n\t" // Scale by alpha
" fmla z2.d, p0/m, z14.d, z10.d               \n\t" // Scale by alpha
" fmla z3.d, p0/m, z15.d, z10.d               \n\t" // Scale by alpha
" fmla z4.d, p0/m, z16.d, z10.d               \n\t" // Scale by alpha
" fmla z5.d, p0/m, z17.d, z10.d               \n\t" // Scale by alpha
" fmla z6.d, p0/m, z18.d, z10.d               \n\t" // Scale by alpha
" fmla z7.d, p0/m, z19.d, z10.d               \n\t" // Scale by alpha
" fmla z8.d, p0/m, z20.d, z10.d               \n\t" // Scale by alpha
" fmla z9.d, p0/m, z21.d, z10.d               \n\t" // Scale by alpha
"                                            \n\t"
" st1d {z0.d}, p0, [x2, z30.d, LSL #3]        \n\t" // Store column 0 of C
" st1d {z1.d}, p0, [x2, z31.d, LSL #3]        \n\t"
"                                            \n\t"
" st1d {z2.d}, p0, [x20, z30.d, LSL #3]       \n\t" // Store column 1 of C
" st1d {z3.d}, p0, [x20, z31.d, LSL #3]       \n\t"
"                                            \n\t"
" st1d {z4.d}, p0, [x21, z30.d, LSL #3]        \n\t" // Store column 2 of C
" st1d {z5.d}, p0, [x21, z31.d, LSL #3]        \n\t"
"                                            \n\t"
" st1d {z6.d}, p0, [x22, z30.d, LSL #3]       \n\t" // Store column 3 of C
" st1d {z7.d}, p0, [x22, z31.d, LSL #3]       \n\t"
"                                            \n\t"
" st1d {z8.d}, p0, [x23, z30.d, LSL #3]       \n\t" // Store column 4 of C
" st1d {z9.d}, p0, [x23, z31.d, LSL #3]       \n\t"
"                                            \n\t"
"                                            \n\t"
" .D256END:                                  \n\t" // Done!
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



