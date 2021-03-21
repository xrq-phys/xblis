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

#include "bli_a64fx_sector_cache.h"

//#define inblock_pref(off) PREFSVE421(pldl1keep,p0,x0,off)

#define inblock_pref(off) PREFANY(pldl1keep,x0,256*off)

#define inblock_nopref(off)



#define BLOCK4_2VX6_Z16_Z27(bvec0,bvec1,bvec2,bvec3,bvec4,bvec5,bvec6,bvec7,bvec8,bvec9,bvec10,bvec11, avec0,avec1,avec2,avec3, prefmode)\
LOAD2VEC_VOFF_Z    (avec2,avec3, p0, x0, 0, 1) \
inblock_ ##prefmode(16)\
MLA1ROW_LB_Z       (z16,z17, avec0,avec1, bvec0,bvec1,   p0, x1,0,8) \
MLA1ROW_LB_Z       (z18,z19, avec0,avec1, bvec2,bvec3,   p0, x1,16,24) \
MLA1ROW_LB_Z       (z20,z21, avec0,avec1, bvec4,bvec5,   p0, x1,32,40) \
MLA1ROW_LB_Z       (z22,z23, avec0,avec1, bvec6,bvec7,   p0, x1,48,56) \
MLA1ROW_LB_Z       (z24,z25, avec0,avec1, bvec8,bvec9,   p0, x1,64,72) \
MLA1ROW_LB_Z       (z26,z27, avec0,avec1, bvec10,bvec11, p0, x1,80,88) \
LOAD2VEC_VOFF_Z    (avec0,avec1, p0, x0, 2, 3) \
MLA1ROW_LB_Z       (z16,z17, avec2,avec3, bvec0,bvec1,   p0, x1,96,104) \
MLA1ROW_LB_Z       (z18,z19, avec2,avec3, bvec2,bvec3,   p0, x1,112,120) \
MLA1ROW_LB_Z       (z20,z21, avec2,avec3, bvec4,bvec5,   p0, x1,128,136) \
MLA1ROW_LB_Z       (z22,z23, avec2,avec3, bvec6,bvec7,   p0, x1,144,152) \
MLA1ROW_LB_Z       (z24,z25, avec2,avec3, bvec8,bvec9,   p0, x1,160,168) \
MLA1ROW_LB_Z       (z26,z27, avec2,avec3, bvec10,bvec11, p0, x1,176,184) \
LOAD2VEC_VOFF_Z    (avec2,avec3, p0, x0, 4, 5) \
MLA1ROW_LB_Z       (z16,z17, avec0,avec1, bvec0,bvec1,   p0, x1,192,200) \
MLA1ROW_LB_Z       (z18,z19, avec0,avec1, bvec2,bvec3,   p0, x1,208,216) \
MLA1ROW_LB_Z       (z20,z21, avec0,avec1, bvec4,bvec5,   p0, x1,224,232) \
MLA1ROW_LB_Z       (z22,z23, avec0,avec1, bvec6,bvec7,   p0, x1,240,248) \
MLA1ROW_LB_Z       (z24,z25, avec0,avec1, bvec8,bvec9,   p0, x1,256,264) \
MLA1ROW_LB_Z       (z26,z27, avec0,avec1, bvec10,bvec11, p0, x1,272,280) \
LOAD2VEC_VOFF_Z    (avec0,avec1, p0, x0, 6, 7) \
MLA1ROW_LB_Z       (z16,z17, avec2,avec3, bvec0,bvec1,   p0, x1,288,296) \
MLA1ROW_LB_Z       (z18,z19, avec2,avec3, bvec2,bvec3,   p0, x1,304,312) \
MLA1ROW_LB_Z       (z20,z21, avec2,avec3, bvec4,bvec5,   p0, x1,320,328) \
MLA1ROW_LB_Z       (z22,z23, avec2,avec3, bvec6,bvec7,   p0, x1,336,344) \
MLA1ROW_LB_Z       (z24,z25, avec2,avec3, bvec8,bvec9,   p0, x1,352,360) \
MLA1ROW_LB_Z       (z26,z27, avec2,avec3, bvec10,bvec11, p0, x1,368,376) \
"                                             \n\t" \
" incb x0, ALL, MUL #8                        \n\t" \
" add x1, x1, #384                            \n\t"  

#define ENDBLOCK4_2VX6_Z16_Z27(bvec0,bvec1,bvec2,bvec3,bvec4,bvec5,bvec6,bvec7,bvec8,bvec9,bvec10,bvec11, avec0,avec1,avec2,avec3)\
LOAD2VEC_VOFF_Z    (avec2,avec3, p0, x0, 0, 1) \
MLA1ROW_LB_Z       (z16,z17, avec0,avec1, bvec0,bvec1,   p0, x1,0,8) \
MLA1ROW_LB_Z       (z18,z19, avec0,avec1, bvec2,bvec3,   p0, x1,16,24) \
MLA1ROW_LB_Z       (z20,z21, avec0,avec1, bvec4,bvec5,   p0, x1,32,40) \
MLA1ROW_LB_Z       (z22,z23, avec0,avec1, bvec6,bvec7,   p0, x1,48,56) \
MLA1ROW_LB_Z       (z24,z25, avec0,avec1, bvec8,bvec9,   p0, x1,64,72) \
MLA1ROW_LB_Z       (z26,z27, avec0,avec1, bvec10,bvec11, p0, x1,80,88) \
LOAD2VEC_VOFF_Z    (avec0,avec1, p0, x0, 2, 3) \
MLA1ROW_LB_Z       (z16,z17, avec2,avec3, bvec0,bvec1,   p0, x1,96,104) \
MLA1ROW_LB_Z       (z18,z19, avec2,avec3, bvec2,bvec3,   p0, x1,112,120) \
MLA1ROW_LB_Z       (z20,z21, avec2,avec3, bvec4,bvec5,   p0, x1,128,136) \
MLA1ROW_LB_Z       (z22,z23, avec2,avec3, bvec6,bvec7,   p0, x1,144,152) \
MLA1ROW_LB_Z       (z24,z25, avec2,avec3, bvec8,bvec9,   p0, x1,160,168) \
MLA1ROW_LB_Z       (z26,z27, avec2,avec3, bvec10,bvec11, p0, x1,176,184) \
LOAD2VEC_VOFF_Z    (avec2,avec3, p0, x0, 4, 5) \
MLA1ROW_LB_Z       (z16,z17, avec0,avec1, bvec0,bvec1,   p0, x1,192,200) \
MLA1ROW_LB_Z       (z18,z19, avec0,avec1, bvec2,bvec3,   p0, x1,208,216) \
MLA1ROW_LB_Z       (z20,z21, avec0,avec1, bvec4,bvec5,   p0, x1,224,232) \
MLA1ROW_LB_Z       (z22,z23, avec0,avec1, bvec6,bvec7,   p0, x1,240,248) \
MLA1ROW_LB_Z       (z24,z25, avec0,avec1, bvec8,bvec9,   p0, x1,256,264) \
MLA1ROW_LB_Z       (z26,z27, avec0,avec1, bvec10,bvec11, p0, x1,272,280) \
"                                             \n\t" \
MLA1ROW_Z       (z16,z17, avec2,avec3, bvec0,bvec1,   p0) \
MLA1ROW_Z       (z18,z19, avec2,avec3, bvec2,bvec3,   p0) \
MLA1ROW_Z       (z20,z21, avec2,avec3, bvec4,bvec5,   p0) \
MLA1ROW_Z       (z22,z23, avec2,avec3, bvec6,bvec7,   p0) \
MLA1ROW_Z       (z24,z25, avec2,avec3, bvec8,bvec9,   p0) \
MLA1ROW_Z       (z26,z27, avec2,avec3, bvec10,bvec11, p0)


/* 2 vectors in m_r, n_r = 6
*/
void bli_zgemm_armv8a_sve_asm_2vx6
     (
       dim_t               k0,
       dcomplex*  restrict alpha,
       dcomplex*  restrict a,
       dcomplex*  restrict b,
       dcomplex*  restrict beta,
       dcomplex*  restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	void* a_next = bli_auxinfo_next_a( data );
	void* b_next = bli_auxinfo_next_b( data );

    // 8 k iterations in unrolled loop
	uint64_t k_iter = k0 / 4;
    // rest is handled separately
    uint64_t k_left = k0 % 4;

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.

	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

    // see macro definition for documentation
    A64FX_SETUP_SECTOR_CACHE_SIZES(0b0001011000010000)

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
" ldr x10,%[cs_c]                             \n\t" // Load cs_c
" lsl x10,x10,#4                              \n\t" // cs_c * sizeof(complex double)
"                                            \n\t"
" ldr x13,%[rs_c]                            \n\t" // Load rs_c.
//" lsl x14,x13,#4                             \n\t" // rs_c * sizeof(double). 
"                                            \n\t"
" mov x11,#0                                 \n\t"
" incd x11                                   \n\t"
"                                            \n\t"
A64FX_SET_CACHE_SECTOR(x2, 0x3, x20) // A64FX: Use cache sector 3 for C_r microtile
A64FX_SET_CACHE_SECTOR(x1, 0x2, x20) // A64FX: Use cache sector 2 for B_r micropanel
A64FX_SET_CACHE_SECTOR(x0, 0x1, x20) // A64FX: Use cache sector 1 for A_r micropanel
"                                            \n\t"
" add x20,x2,x10                             \n\t" //Load address Column 1 of C
" add x21,x20,x10                            \n\t" //Load address Column 2 of C
" add x22,x21,x10                            \n\t" //Load address Column 3 of C
" add x23,x22,x10                            \n\t" //Load address Column 4 of C
" add x24,x23,x10                            \n\t" //Load address Column 4 of C
"                                            \n\t"
#if defined(PREFETCH64) || defined(PREFETCH256)
" prfm pstl1keep,[x2]                        \n\t" // Prefetch c.
" prfm pstl1keep,[x20]                       \n\t" // Prefetch c.
" prfm pstl1keep,[x21]                       \n\t" // Prefetch c.
" prfm pstl1keep,[x22]                       \n\t" // Prefetch c.
" prfm pstl1keep,[x23]                       \n\t" // Prefetch c.
" prfm pstl1keep,[x24]                       \n\t" // Prefetch c.
#endif
// Something isn't right with SVE prefetches (Or I'm misunderstanding/miscalculating something)
/*
PREFSVE421(pldl1keep, p0, x0, 0)
PREFSVE1  (pldl1keep, p0, x0, 1)
PREFSVE21 (pldl1keep, p0, x0, 2)
PREFSVE1  (pldl1keep, p0, x0, 3)
PREFSVE421(pldl1keep, p0, x0, 4)
PREFSVE1  (pldl1keep, p0, x0, 5)
PREFSVE21 (pldl1keep, p0, x0, 6)
PREFSVE1  (pldl1keep, p0, x0, 7)
PREFSVE421(pldl1keep, p0, x0, 8)
PREFSVE1  (pldl1keep, p0, x0, 9)
PREFSVE21 (pldl1keep, p0, x0, 10)
PREFSVE1  (pldl1keep, p0, x0, 11)
PREFSVE421(pldl1keep, p0, x0, 12)
PREFSVE1  (pldl1keep, p0, x0, 13)
PREFSVE21 (pldl1keep, p0, x0, 14)
PREFSVE1  (pldl1keep, p0, x0, 15)*/
// A64FX: k_c should be pretty big so let's prefetch 4kiB
PREF256(pldl1keep, x0, 256*0)
PREF256(pldl1keep, x0, 256*1)
PREF256(pldl1keep, x0, 256*2)
PREF256(pldl1keep, x0, 256*3)
PREF256(pldl1keep, x0, 256*4)
PREF256(pldl1keep, x0, 256*5)
PREF256(pldl1keep, x0, 256*6)
PREF256(pldl1keep, x0, 256*7)
PREF256(pldl1keep, x0, 256*8)
PREF256(pldl1keep, x0, 256*9)
PREF256(pldl1keep, x0, 256*10)
PREF256(pldl1keep, x0, 256*11)
PREF256(pldl1keep, x0, 256*12)
PREF256(pldl1keep, x0, 256*13)
PREF256(pldl1keep, x0, 256*14)
PREF256(pldl1keep, x0, 256*15)
// Prefetching a bit more than 4kiB seems to give better performance
PREF256(pldl1keep, x1, 256*0)
PREF256(pldl1keep, x1, 256*1)
PREF256(pldl1keep, x1, 256*2)
PREF256(pldl1keep, x1, 256*3)
PREF256(pldl1keep, x1, 256*4)
PREF256(pldl1keep, x1, 256*5)
PREF256(pldl1keep, x1, 256*6)
PREF256(pldl1keep, x1, 256*7)
PREF256(pldl1keep, x1, 256*8)
PREF256(pldl1keep, x1, 256*9)
PREF256(pldl1keep, x1, 256*10)
PREF256(pldl1keep, x1, 256*11)
PREF256(pldl1keep, x1, 256*12)
PREF256(pldl1keep, x1, 256*13)
PREF256(pldl1keep, x1, 256*14)
PREF256(pldl1keep, x1, 256*15)
PREF256(pldl1keep, x1, 256*16)
PREF256(pldl1keep, x1, 256*17)
"                                            \n\t"
" ptrue p0.d                                 \n\t" // Creating all true predicate
"                                            \n\t"
/*************************************************
 *
 * Store A vectors in z0,z1,z2,z3, alternating between (z0,z1) and (z2,z3)
 *
 * load B values in z4-z15, rotating
 * LD1 has 11 cycles latency
 *
 *************************************************/
LOAD2VEC_Z(z0,z1,p0,x0)
"                                            \n\t"
"                                            \n\t"
LOAD12VEC_DIST_Z(z4,z5,z6,z7,z8,z9,z10,z11,z12,z13,z14,z15, p0,x1)
"                                            \n\t"
ZERO4VEC_Z(z16,z17,z18,z19)                          // c column 0-1
ZERO4VEC_Z(z20,z21,z22,z23)                          // c column 2-3 
ZERO4VEC_Z(z24,z25,z26,z27)                          // c column 4-5

"                                            \n\t"
"                                            \n\t"
" cmp x5,#0                                  \n\t" // If k_iter == 0, jump to k_left.
" beq .Z2VX6CONSIDERKLEFT                    \n\t"
"                                            \n\t"
" incb x0, ALL, MUL #2                       \n\t" // A = A+vecsize*2
" add x1, x1, #96                            \n\t" // B = B+12*sizeof(double)
"                                            \n\t"
" cmp x5,1                                   \n\t" // If there is just one k_iter, jump to that one. 
" beq .Z2VX6LASTITER                         \n\t" // (as loop is do-while-like).
"                                            \n\t" 
"                                            \n\t"
" .Z2VX6LOOP:                                \n\t" // Body
BLOCK4_2VX6_Z16_Z27(z4,z5,z6,z7,z8,z9,z10,z11,z12,z13,z14,z15,    z0,z1,z2,z3, pref)
"                                            \n\t"
" sub x5,x5,1                                \n\t" // i-=1
" cmp x5,1                                   \n\t" // Iterate again if we are not in k_iter == 1.
" bne .Z2VX6LOOP                             \n\t"
" .Z2VX6LASTITER:                            \n\t" // Body
ENDBLOCK4_2VX6_Z16_Z27(z4,z5,z6,z7,z8,z9,z10,z11,z12,z13,z14,z15,    z0,z1,z2,z3)
"                                            \n\t"
" add x1, x1, #288                           \n\t"
" incb x0, ALL, MUL #6                       \n\t" // 6 more Vectors loaded
"                                            \n\t"
" .Z2VX6CONSIDERKLEFT:                       \n\t" 
" cmp x6,0                                   \n\t" // If k_left == 0, we are done.
" beq .Z2VX6POSTACCUM                        \n\t" // else, we enter the k_left loop.
"                                            \n\t"
".Z2VX6LOOPKLEFT:                            \n\t"
"                                            \n\t"
LOAD2VEC_Z(z0,z1,p0,x0)
" incb x0, ALL, MUL #2                       \n\t" // Advance a pointer by 2 vectors
"                                            \n\t"
LOAD12VEC_DIST_Z(z4,z5,z6,z7,z8,z9,z10,z11,z12,z13,z14,z15, p0,x1)
" add x1, x1, #96                            \n\t" // advance b pointer by 12 doubles
"                                            \n\t"
" sub x6,x6,1                                \n\t"
"                                            \n\t"
MLA1ROW_Z(z16,z17, z0,z1, z4,z5,   p0) 
MLA1ROW_Z(z18,z19, z0,z1, z6,z7,   p0) 
MLA1ROW_Z(z20,z21, z0,z1, z8,z9,   p0)
MLA1ROW_Z(z22,z23, z0,z1, z10,z11, p0)
MLA1ROW_Z(z24,z25, z0,z1, z12,z13, p0)
MLA1ROW_Z(z26,z27, z0,z1, z14,z15, p0)
"                                            \n\t"
" cmp x6,0                                   \n\t" // Iterate again.
" bne .Z2VX6LOOPKLEFT                        \n\t" // if i!=0.
"                                            \n\t" // x6 is free from here
" .Z2VX6POSTACCUM:                           \n\t"

#if defined(PREFETCH64) || defined(PREFETCH256)
" mov x6, 0x1                                \n\t" // A64FX: Use cache sector 1 for A
" lsl x6, x6, 56                             \n\t"
" orr x3, x3, x6                             \n\t"
" prfm PLDL2KEEP, [x3,256*0]                 \n\t"
" prfm PLDL2KEEP, [x3,256*1]                 \n\t"
" prfm PLDL2KEEP, [x3,256*2]                 \n\t"
" prfm PLDL2KEEP, [x3,256*3]                 \n\t"
" prfm PLDL2KEEP, [x3,256*4]                 \n\t"
" prfm PLDL2KEEP, [x3,256*5]                 \n\t"
" prfm PLDL2KEEP, [x3,256*6]                 \n\t"
" prfm PLDL2KEEP, [x3,256*7]                 \n\t"
" prfm PLDL2KEEP, [x3,256*8]                 \n\t"
" prfm PLDL2KEEP, [x3,256*9]                 \n\t"
" prfm PLDL2KEEP, [x3,256*10]                \n\t"
" prfm PLDL2KEEP, [x3,256*11]                \n\t"
" prfm PLDL2KEEP, [x3,256*12]                \n\t"
" prfm PLDL2KEEP, [x3,256*13]                \n\t"
" prfm PLDL2KEEP, [x3,256*14]                \n\t"
" prfm PLDL2KEEP, [x3,256*15]                \n\t"
" mov x6, 0x2                                \n\t" // A64FX: Use cache sector 2 for B
" lsl x6, x6, 56                             \n\t"
" orr x4, x4, x6                             \n\t"
" prfm PLDL2KEEP, [x4,256*0]                 \n\t"
" prfm PLDL2KEEP, [x4,256*1]                 \n\t"
" prfm PLDL2KEEP, [x4,256*2]                 \n\t"
" prfm PLDL2KEEP, [x4,256*3]                 \n\t"
" prfm PLDL2KEEP, [x4,256*4]                 \n\t"
" prfm PLDL2KEEP, [x4,256*5]                 \n\t"
" prfm PLDL2KEEP, [x4,256*6]                 \n\t"
" prfm PLDL2KEEP, [x4,256*7]                 \n\t"
" prfm PLDL2KEEP, [x4,256*8]                 \n\t"
" prfm PLDL2KEEP, [x4,256*9]                 \n\t"
#endif
"                                            \n\t"
#if defined(USE_SVE_CMLA_INSTRUCTION)
" ld1rqd  {z0.d}, p0/z, [x7]                \n\t" // Load alpha
" ld1rqd  {z1.d}, p0/z, [x8]                \n\t" // Load beta
#else
" ld1rd  z0.d, p0/z, [x7]                   \n\t" // Load alpha
" ld1rd  z30.d, p0/z, [x7,#8]               \n\t" // Load alpha
" ld1rd  z1.d, p0/z, [x8]                   \n\t" // Load beta
" ld1rd  z31.d, p0/z, [x8,#8]               \n\t" // Load beta
#endif

"                                            \n\t"
" cmp x13,#1                                 \n\t" // If rs_c != 1 (column-major)
" bne .Z2VX6GENSTORED                        \n\t"
"                                            \n\t"
" .Z2VX6COLSTORED:                           \n\t" // C is column-major.
"                                            \n\t"

// Accumulated results are stored in z16-z27
// alpha in z0 (and z30 without CMLA), beta in z1( and z31 without CMLA)
// z2-z15 and z28-z29 are free
"                                            \n\t"
ZERO4VEC_Z(z2,z3,z4,z5)
ZERO4VEC_Z(z6,z7,z8,z9)
ZERO4VEC_Z(z10,z11,z12,z13)
"                                            \n\t"
CMPCZB_D(z1, z31, ".Z2VX6BETAZEROCONTCOLSTOREDS")
"                                            \n\t"
LOAD2VEC_Z(z14,z15,p0,x2)                         // Load Column 0
LOAD2VEC_Z(z28,z29,p0,x20)                        // Load Column 1
"                                            \n\t"
MLA2ROW_Z(z2,z3,z4,z5,     z14,z15,z28,z29, z1,z31, p0)
LOAD2VEC_Z(z14,z15,p0,x21)                        // Load Column 2
"                                            \n\t"
MLA2ROW_Z(z2,z3,z4,z5,     z16,z17,z18,z19, z0,z30, p0)
LOAD2VEC_Z(z16,z17,p0,x22)                        // Load Column 3
"                                            \n\t"
MLA2ROW_Z(z6,z7,z8,z9,     z14,z15,z16,z17, z1,z31, p0)
LOAD2VEC_Z(z18,z19,p0,x23)                        // Load Column 4
"                                            \n\t"
MLA2ROW_Z(z6,z7,z8,z9,     z20,z21,z22,z23, z0,z30, p0)
LOAD2VEC_Z(z20,z21,p0,x24)                        // Load Column 5
"                                            \n\t"
MLA2ROW_Z(z10,z11,z12,z13, z18,z19,z20,z21, z1,z31, p0)
"                                            \n\t"
MLA2ROW_Z(z10,z11,z12,z13, z24,z25,z26,z27, z0,z30, p0)
"                                            \n\t"
STOR2VEC_Z(z2,z3,   p0, x2)
STOR2VEC_Z(z4,z5,   p0,x20)
STOR2VEC_Z(z6,z7,   p0,x21)
STOR2VEC_Z(z8,z9,   p0,x22)
STOR2VEC_Z(z10,z11, p0,x23)
STOR2VEC_Z(z12,z13, p0,x24)
"                                            \n\t"
" b .Z2VX6END                                \n\t"       // Duplicate code for stores required due to lack of registers
"                                            \n\t"
" .Z2VX6BETAZEROCONTCOLSTOREDS:              \n\t"
"                                            \n\t"
MLA2ROW_Z(z2,z3,z4,z5,     z16,z17,z18,z19,  z0,z30, p0)
MLA2ROW_Z(z6,z7,z8,z9,     z20,z21,z22,z23,  z0,z30, p0)
MLA2ROW_Z(z10,z11,z12,z13, z24,z25,z26,z27,  z0,z30, p0)
"                                            \n\t"
STOR2VEC_Z(z2,z3,   p0, x2)
STOR2VEC_Z(z4,z5,   p0,x20)
STOR2VEC_Z(z6,z7,   p0,x21)
STOR2VEC_Z(z8,z9,   p0,x22)
STOR2VEC_Z(z10,z11, p0,x23)
STOR2VEC_Z(z12,z13, p0,x24)
"                                            \n\t"

" b .Z2VX6END                                \n\t"       // Duplicate code for stores required due to lack of registers
"                                            \n\t"
" .Z2VX6GENSTORED:                           \n\t" // C is general-stride stored.
"                                            \n\t" // Creating index for stride load&store access
MKINDC_2VEC_D(z2,z3, x13,x11, x3)
ZERO4VEC_Z(z4,z5,z6,z7)
ZERO4VEC_Z(z8,z9,z10,z11)
"                                            \n\t"
CMPCZB_D(z1, z31, ".Z2VX6BETAZEROGENSTOREDS")
"                                            \n\t"
"                                            \n\t"
LOAD2VEC_GENI_Z (z12,z13,  p0, x2, z2,z3)                  // Load Column 0
LOAD2VEC_GENI_Z (z14,z15,  p0,x20, z2,z3)                  // Load Column 1

MLA2ROW_Z(z4,z5,z6,z7,     z12,z13,z14,z15, z1,z31, p0)
LOAD2VEC_GENI_Z (z12,z13,  p0,x21, z2,z3)                  // Load Column 2
MLA2ROW_Z(z4,z5,z6,z7,     z16,z17,z18,z19, z0,z30, p0)
LOAD2VEC_GENI_Z (z14,z15,  p0,x22, z2,z3)                  // Load Column 3
MLA2ROW_Z(z8,z9,z10,z11,   z12,z13,z14,z15, z1,z31, p0)
ZERO4VEC_Z(z12,z13,z14,z15)
LOAD2VEC_GENI_Z (z16,z17,  p0,x23, z2,z3)                  // Load Column 4
MLA2ROW_Z(z8,z9,z10,z11,   z20,z21,z22,z23, z0,z30, p0)
LOAD2VEC_GENI_Z (z18,z19,  p0,x24, z2,z3)                  // Load Column 5
MLA2ROW_Z(z12,z13,z14,z15, z16,z17,z18,z19, z1,z31, p0)
STOR2VEC_GENI_Z(z4,z5,    p0, x2, z2,z3)
MLA2ROW_Z(z12,z13,z14,z15, z24,z25,z26,z27, z0,z30, p0)
STOR2VEC_GENI_Z(z6,z7,    p0,x20, z2,z3)
STOR2VEC_GENI_Z(z8,z9,    p0,x21, z2,z3)
STOR2VEC_GENI_Z(z10,z11,  p0,x22, z2,z3)
STOR2VEC_GENI_Z(z12,z13,  p0,x23, z2,z3)
STOR2VEC_GENI_Z(z14,z15,  p0,x24, z2,z3)
"                                           \n\t"
" b .Z2VX6END                               \n\t"      // Duplicate code for stores required due to lack of registers
"                                           \n\t"
" .Z2VX6BETAZEROGENSTOREDS:                 \n\t"
ZERO4VEC_Z(z12,z13,z14,z15)
"                                           \n\t"
MLA2ROW_Z(z4,z5,z6,z7,     z16,z17,z18,z19, z0,z30, p0)
MLA2ROW_Z(z8,z9,z10,z11,   z20,z21,z22,z23, z0,z30, p0)
STOR2VEC_GENI_Z(z4,z5,    p0, x2, z2,z3)
MLA2ROW_Z(z12,z13,z14,z15, z24,z25,z26,z27, z0,z30, p0)
STOR2VEC_GENI_Z(z6,z7,    p0,x20, z2,z3)
STOR2VEC_GENI_Z(z8,z9,    p0,x21, z2,z3)
STOR2VEC_GENI_Z(z10,z11,  p0,x22, z2,z3)
STOR2VEC_GENI_Z(z12,z13,  p0,x23, z2,z3)
STOR2VEC_GENI_Z(z14,z15,  p0,x23, z2,z3)
"                                            \n\t"
" .Z2VX6END:                                 \n\t"     // Done!
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
 "x7","x8",
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
