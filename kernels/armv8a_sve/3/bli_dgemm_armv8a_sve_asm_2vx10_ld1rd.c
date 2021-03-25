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

//#define inblock_pref(off) \
PREFSVE421(pldl1keep,p0,x0,off)

#define inblock_pref(off) \
PREFANY(pldl1keep,x0,256*off)

#define inblock_nopref(off)


#define BLOCK_2VX10_Z12_Z31(bvec0,bvec1,bvec2,bvec3,bvec4,bvec5,bvec6,bvec7,bvec8,bvec9, cur_avec0,cur_avec1, next_avec0,next_avec1, aoff0,aoff1, prefmode)\
LOAD1VEC_VOFF_D(next_avec0, p0, x0, aoff0)\
inblock_ ##prefmode(10)\
MLA2ROW_D(z12,z13, cur_avec0,cur_avec1, bvec0, p0)\
LOAD1VEC_VOFF_D(next_avec1, p0, x0, aoff1)\
LOAD1VEC_DIST_OFF_D(bvec0, p0, x1, 0)\
MLA2ROW_D(z14,z15, cur_avec0,cur_avec1, bvec1, p0)\
LOAD1VEC_DIST_OFF_D(bvec1, p0, x1, 8)\
MLA2ROW_D(z16,z17, cur_avec0,cur_avec1, bvec2, p0)\
MLA2ROW_D(z18,z19, cur_avec0,cur_avec1, bvec3, p0)\
LOAD1VEC_DIST_OFF_D(bvec2, p0, x1, 16)\
LOAD1VEC_DIST_OFF_D(bvec3, p0, x1, 24)\
MLA2ROW_D(z20,z21, cur_avec0,cur_avec1, bvec4, p0)\
MLA2ROW_D(z22,z23, cur_avec0,cur_avec1, bvec5, p0)\
LOAD1VEC_DIST_OFF_D(bvec4, p0, x1, 32)\
LOAD1VEC_DIST_OFF_D(bvec5, p0, x1, 40)\
MLA2ROW_D(z24,z25, cur_avec0,cur_avec1, bvec6, p0)\
MLA2ROW_D(z26,z27, cur_avec0,cur_avec1, bvec7, p0)\
LOAD1VEC_DIST_OFF_D(bvec6, p0, x1, 48)\
LOAD1VEC_DIST_OFF_D(bvec7, p0, x1, 56)\
MLA2ROW_D(z28,z29, cur_avec0,cur_avec1, bvec8, p0)\
MLA2ROW_D(z30,z31, cur_avec0,cur_avec1, bvec9, p0)\
LOAD1VEC_DIST_OFF_D(bvec8, p0, x1, 64)\
LOAD1VEC_DIST_OFF_D(bvec9, p0, x1, 72)\
" add x1, x1, #80 \n\t"

#define ENDBLOCK_2VX10_Z12_Z31(bvec0,bvec1,bvec2,bvec3,bvec4,bvec5,bvec6,bvec7,bvec8,bvec9, cur_avec0,cur_avec1)\
MLA2ROW_D(z12,z13, cur_avec0,cur_avec1, bvec0, p0)\
MLA2ROW_D(z14,z15, cur_avec0,cur_avec1, bvec1, p0)\
LOAD1VEC_DIST_OFF_D(bvec0, p0, x1, 0)\
MLA2ROW_D(z16,z17, cur_avec0,cur_avec1, bvec2, p0)\
MLA2ROW_D(z18,z19, cur_avec0,cur_avec1, bvec3, p0)\
MLA2ROW_D(z20,z21, cur_avec0,cur_avec1, bvec4, p0)\
LOAD1VEC_DIST_OFF_D(bvec1, p0, x1, 8)\
" add x1, x1, #16 \n\t"\
MLA2ROW_D(z22,z23, cur_avec0,cur_avec1, bvec5, p0)\
MLA2ROW_D(z24,z25, cur_avec0,cur_avec1, bvec6, p0)\
MLA2ROW_D(z26,z27, cur_avec0,cur_avec1, bvec7, p0)\
MLA2ROW_D(z28,z29, cur_avec0,cur_avec1, bvec8, p0)\
MLA2ROW_D(z30,z31, cur_avec0,cur_avec1, bvec9, p0)


/* 2 vectors in m_r, n_r = 10
*/
void bli_dgemm_armv8a_sve_asm_2vx10_ld1rd
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
	uint64_t k_iter = k0 / 4;
    // rest is handled separately
    uint64_t k_left = k0 % 4;

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.

	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

    uint64_t byte_size = get_sve_byte_size();

    // see macro definition for documentation
    //A64FX_SETUP_SECTOR_CACHE_SIZES(0b0000000101110000)
    //A64FX_SETUP_SECTOR_CACHE_SIZES(A64FX_SCC(0,1,3,0))
    //A64FX_SETUP_SECTOR_CACHE_SIZES_L2(A64FX_SCC_L2(1,31))
    //A64FX_SETUP_SECTOR_CACHE_SIZES(0b0001010100010001)
    
    //uint64_t sector_config = 0;
    //A64FX_READ_SECTOR_CACHE_SIZES(sector_config);
    //print_sector_cache_configuration(sector_config);

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
A64FX_SET_CACHE_SECTOR(x2, 0x3, x20) // A64FX: Use cache sector 3 for C_r microtile
A64FX_SET_CACHE_SECTOR(x1, 0x2, x20) // A64FX: Use cache sector 2 for B_r micropanel
A64FX_SET_CACHE_SECTOR(x0, 0x1, x20) // A64FX: Use cache sector 1 for A_r micropanel
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
" add x20,x24,x10 \n\t"
" prfm pstl1keep,[x21]                       \n\t" // Prefetch c.
" add x21,x20,x10 \n\t"
" prfm pstl1keep,[x22]                       \n\t" // Prefetch c.
" add x22,x21,x10 \n\t"
" prfm pstl1keep,[x23]                       \n\t" // Prefetch c.
" add x23,x22,x10 \n\t"
" prfm pstl1keep,[x24]                       \n\t" // Prefetch c.
" prfm pstl1keep,[x20]                       \n\t" // Prefetch c.
" add x20,x2,x10 \n\t"                             // RESET is here
" prfm pstl1keep,[x21]                       \n\t" // Prefetch c.
" add x21,x20,x10 \n\t"
" prfm pstl1keep,[x22]                       \n\t" // Prefetch c.
" add x22,x21,x10 \n\t"
" prfm pstl1keep,[x23]                       \n\t" // Prefetch c.
" add x23,x22,x10 \n\t"
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
//PREF256(pldl1keep, x0, 256*10)
//PREF256(pldl1keep, x0, 256*11)
//PREF256(pldl1keep, x0, 256*12)
//PREF256(pldl1keep, x0, 256*13)
//PREF256(pldl1keep, x0, 256*14)
//PREF256(pldl1keep, x0, 256*15)
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
 * load B values in z4-z11, rotating
 * LD1 has 11 cycles latency
 *
 *************************************************/
LOAD2VEC_D(z0,z1,p0,x0)
"                                            \n\t"
"                                            \n\t"
LOAD8VEC_DIST_D(z4,z5,z6,z7,z8,z9,z10,z11, p0,x1)
"                                            \n\t"
ZERO4VEC_D(z12,z13,z14,z15)                          // c columns 0-1
ZERO4VEC_D(z16,z17,z18,z19)                          // c columns 2-3
ZERO4VEC_D(z20,z21,z22,z23)                          // c columns 4-5
ZERO4VEC_D(z24,z25,z26,z27)                          // c columns 6-7
ZERO4VEC_D(z28,z29,z30,z31)                          // c columns 8-9

"                                            \n\t"
"                                            \n\t"
" cmp x5,#0                                  \n\t" // If k_iter == 0, jump to k_left.
" beq .D2VX10CONSIDERKLEFT                   \n\t"
"                                            \n\t"
" incb x0, ALL, MUL #2                       \n\t" // A = A+vecsize*2
" add x1, x1, #64                            \n\t" // B = B+8*sizeof(double)
"                                            \n\t"
" cmp x5,1                                   \n\t" // If there is just one k_iter, jump to that one. 
" beq .D2VX10LASTITER                        \n\t" // (as loop is do-while-like).
"                                            \n\t" 
"                                            \n\t"
" .D2VX10LOOP:                               \n\t" // Body
BLOCK_2VX10_Z12_Z31(z4,z5,z6,z7,z8,z9,z10,z11,z4,z5,    z0,z1, z2,z3, 0,1,nopref)
"                                            \n\t"
BLOCK_2VX10_Z12_Z31(z6,z7,z8,z9,z10,z11,z4,z5,z6,z7,    z2,z3, z0,z1, 2,3,pref)
" incb x0, ALL, MUL #4                       \n\t" // Next 4 A vectors
BLOCK_2VX10_Z12_Z31(z8,z9,z10,z11,z4,z5,z6,z7,z8,z9,    z0,z1, z2,z3, 0,1,nopref)
"                                            \n\t"
BLOCK_2VX10_Z12_Z31(z10,z11,z4,z5,z6,z7,z8,z9,z10,z11,  z2,z3, z0,z1, 2,3,pref)
" incb x0, ALL, MUL #4                       \n\t" // Next 4 A vectors
"                                            \n\t"
" sub x5,x5,1                                \n\t" // i-=1
" cmp x5,1                                   \n\t" // Iterate again if we are not in k_iter == 1.
" bne .D2VX10LOOP                             \n\t"
" .D2VX10LASTITER:                            \n\t" // Body
BLOCK_2VX10_Z12_Z31(z4,z5,z6,z7,z8,z9,z10,z11,z4,z5,    z0,z1, z2,z3, 0,1,nopref)
"                                            \n\t"
BLOCK_2VX10_Z12_Z31(z6,z7,z8,z9,z10,z11,z4,z5,z6,z7,    z2,z3, z0,z1, 2,3,nopref)
"                                            \n\t"
BLOCK_2VX10_Z12_Z31(z8,z9,z10,z11,z4,z5,z6,z7,z8,z9,    z0,z1, z2,z3, 4,5,nopref)
"                                            \n\t"
ENDBLOCK_2VX10_Z12_Z31(z10,z11,z4,z5,z6,z7,z8,z9,z10,z11,  z2,z3)
"                                            \n\t"
" incb x0, ALL, MUL #6                       \n\t" // 6 more Vectors loaded
"                                            \n\t"
" .D2VX10CONSIDERKLEFT:                       \n\t" 
" cmp x6,0                                   \n\t" // If k_left == 0, we are done.
" beq .D2VX10POSTACCUM                        \n\t" // else, we enter the k_left loop.
"                                            \n\t"
".D2VX10LOOPKLEFT:                            \n\t"
"                                            \n\t"
LOAD2VEC_D(z0,z1,p0,x0)
" incb x0, ALL, MUL #2                       \n\t" // Advance a pointer by 2 vectors
"                                            \n\t"
LOAD1VEC_DIST_OFF_D(z2,  p0, x1,  0)
LOAD1VEC_DIST_OFF_D(z3,  p0, x1,  8)
LOAD1VEC_DIST_OFF_D(z4,  p0, x1, 16)
LOAD1VEC_DIST_OFF_D(z5,  p0, x1, 24)
LOAD1VEC_DIST_OFF_D(z6,  p0, x1, 32)
LOAD1VEC_DIST_OFF_D(z7,  p0, x1, 40)
LOAD1VEC_DIST_OFF_D(z8,  p0, x1, 48)
LOAD1VEC_DIST_OFF_D(z9,  p0, x1, 56)
LOAD1VEC_DIST_OFF_D(z10, p0, x1, 64)
LOAD1VEC_DIST_OFF_D(z11, p0, x1, 72)
" add x1, x1, #80                            \n\t" // advance b pointer by 10 doubles
"                                            \n\t"
" sub x6,x6,1                                \n\t"
"                                            \n\t"
MLA2ROW_D(z12,z13,  z0,z1,  z2, p0)
MLA2ROW_D(z14,z15,  z0,z1,  z3, p0)
                          
MLA2ROW_D(z16,z17,  z0,z1,  z4, p0)
MLA2ROW_D(z18,z19,  z0,z1,  z5, p0)
                          
MLA2ROW_D(z20,z21,  z0,z1,  z6, p0)
MLA2ROW_D(z22,z23,  z0,z1,  z7, p0)
                          
MLA2ROW_D(z24,z25,  z0,z1,  z8, p0)
MLA2ROW_D(z26,z27,  z0,z1,  z9, p0)

MLA2ROW_D(z28,z29,  z0,z1, z10, p0)
MLA2ROW_D(z30,z31,  z0,z1, z11, p0)
"                                            \n\t"
" cmp x6,0                                   \n\t" // Iterate again.
" bne .D2VX10LOOPKLEFT                       \n\t" // if i!=0.
"                                            \n\t" // x6 is free from here
" .D2VX10POSTACCUM:                          \n\t"

A64FX_SET_CACHE_SECTOR(x3,0x1,x6)
PREF256(pldl2keep,x3,256*0)
//PREF256(pldl2keep,x3,256*1)
//PREF256(pldl2keep,x3,256*2)
//PREF256(pldl2keep,x3,256*3)
//PREF256(pldl2keep,x3,256*4)
//PREF256(pldl2keep,x3,256*5)
//PREF256(pldl2keep,x3,256*6)
//PREF256(pldl2keep,x3,256*7)
//PREF256(pldl2keep,x3,256*8)
//PREF256(pldl2keep,x3,256*9)
//PREF256(pldl2keep,x3,256*10)
//PREF256(pldl2keep,x3,256*11)
//PREF256(pldl2keep,x3,256*12)
//PREF256(pldl2keep,x3,256*13)
//PREF256(pldl2keep,x3,256*14)
//PREF256(pldl2keep,x3,256*15)
A64FX_SET_CACHE_SECTOR(x4,0x2,x6)
PREF256(pldl2keep,x4,256*0)
//PREF256(pldl2keep,x4,256*1)
//PREF256(pldl2keep,x4,256*2)
//PREF256(pldl2keep,x4,256*3)
//PREF256(pldl2keep,x4,256*4)
//PREF256(pldl2keep,x4,256*5)
//PREF256(pldl2keep,x4,256*6)
//PREF256(pldl2keep,x4,256*7)
//PREF256(pldl2keep,x4,256*8)
//PREF256(pldl2keep,x4,256*9)
//PREF256(pldl2keep,x4,256*10)
//PREF256(pldl2keep,x4,256*11)
//PREF256(pldl2keep,x4,256*12)
//PREF256(pldl2keep,x4,256*13)
//PREF256(pldl2keep,x4,256*14)
//PREF256(pldl2keep,x4,256*15)
//PREF256(pldl2keep,x4,256*16)
//PREF256(pldl2keep,x4,256*17)
"                                            \n\t"
" ld1rd  z0.d, p0/z, [x7]                    \n\t" // Load alpha
" ld1rd  z1.d, p0/z, [x8]                   \n\t" // Load beta

"                                            \n\t"
" cmp x13,#1                                 \n\t" // If rs_c != 1 (column-major)
" bne .D2VX10GENSTORED                       \n\t"
"                                            \n\t"
" .D2VX10COLSTORED:                          \n\t" // C is column-major.
"                                            \n\t"

// Accumulated results are stored in z12-z31
// alpha in z0, beta in z1 - put alpha is in z0.d[0], beta in z1.d[1]
// z2-z11 are free
"                                        \n\t"

" fcmp d1,#0.0                           \n\t"
" beq .D2VX10BETAZEROCONTCOLSTOREDS      \n\t" // multiply with beta if beta isn't zero
"                                        \n\t"
// interleaved saving

MUL2ROW_D(z12,z13, z12,z13, z0, p0)
LOAD2VEC_D(z2,z3,p0,x2)                         // Load Column 0
MUL2ROW_D(z14,z15, z14,z15, z0, p0)
LOAD2VEC_D(z4,z5,p0,x20)                        // Load Column 1
MUL2ROW_D(z16,z17, z16,z17, z0, p0)
LOAD2VEC_D(z6,z7,p0,x21)                        // Load Column 2
MUL2ROW_D(z18,z19, z18,z19, z0, p0)
LOAD2VEC_D(z8,z9,p0,x22)                        // Load Column 3
"                                            \n\t"

//  0   1   2   3   4   5   6   7   8   9
//  ^   ^   ^   ^   ^   ^
// x2 x20 x21 x22 x23 x24

MUL2ROW_D(z20,z21, z20,z21, z0, p0)
LOAD2VEC_D(z10,z11, p0, x23)                           // Load Column 4
MLA2ROW_D(z12,z13, z2,z3,     z1, p0)
STOR2VEC_D(z12,z13,  p0,  x2)                          // Store Column 0
" add x2,x24,x10                            \n\t"      // Load address Column 6 of C
MUL2ROW_D(z22,z23, z22,z23, z0, p0)
LOAD2VEC_D(z12,z13, p0, x24)                           // Load Column 5
MLA2ROW_D(z14,z15, z4,z5,     z1, p0)
STOR2VEC_D(z14,z15,  p0, x20)                          // Store Column 1
" add x20,x2,x10                            \n\t"      // Load address Column 7 of C

//  0   1   2   3   4   5   6   7   8   9
//          ^   ^   ^   ^   ^   ^
//        x21 x22 x23 x24  x2 x20

MUL2ROW_D(z24,z25, z24,z25, z0, p0)
LOAD2VEC_D(z14,z15, p0,x2)                             // Load Column 6
MLA2ROW_D(z16,z17, z6,z7,     z1, p0)
STOR2VEC_D(z16,z17, p0, x21)                           // Store Column 2
" add x21,x20,x10                           \n\t"      // Load address Column 8 of C
MUL2ROW_D(z26,z27, z26,z27, z0, p0)
LOAD2VEC_D(z16,z17, p0,x20)                            // Load Column 7
MLA2ROW_D(z18,z19, z8,z9,     z1, p0)
STOR2VEC_D(z18,z19, p0, x22)                           // Store Column 3
" add x22,x21,x10                           \n\t"      // Load address Column 9 of C

//  0   1   2   3   4   5   6   7   8   9
//                  ^   ^   ^   ^   ^   ^
//                x23 x24  x2 x20 x21 x22
MUL2ROW_D(z28,z29, z28,z29, z0, p0)
LOAD2VEC_D(z18,z19,p0,x21)                              // Load Column 8
MLA2ROW_D(z20,z21, z10,z11, z1, p0)
STOR2VEC_D(z20,z21, p0, x23)                            // Store Column 4

MUL2ROW_D(z30,z31, z30,z31, z0, p0)
LOAD2VEC_D(z20,z21,p0,x22)                              // Load Column 9
MLA2ROW_D(z22,z23, z12,z13, z1, p0)
STOR2VEC_D(z22,z23, p0, x24)                            // Store Column 5

//  0   1   2   3   4   5   6   7   8   9
//                  ^   ^   ^   ^   ^   ^
//                x23 x24  x2 x20 x21 x22
MLA2ROW_D(z24,z25, z14,z15, z1, p0)
STOR2VEC_D(z24,z25, p0, x2)                             // Store Column 6
MLA2ROW_D(z26,z27, z16,z17, z1, p0)
STOR2VEC_D(z26,z27, p0, x20)                            // Store Column 7
MLA2ROW_D(z28,z29, z18,z19, z1, p0)
STOR2VEC_D(z28,z29, p0, x21)                            // Store Column 8
MLA2ROW_D(z30,z31, z20,z21, z1, p0)
STOR2VEC_D(z30,z31, p0, x22)                            // Store Column 9






// blocked saving
/*
LOAD2VEC_D(z2,z3,p0,x2)                         // Load Column 0
LOAD2VEC_D(z4,z5,p0,x20)                        // Load Column 1
LOAD2VEC_D(z6,z7,p0,x21)                        // Load Column 2
LOAD2VEC_D(z8,z9,p0,x22)                        // Load Column 3
"                                            \n\t"
MUL4ROW_D(z12,z13,z14,z15, z12,z13,z14,z15, z0, p0)
MUL4ROW_D(z16,z17,z18,z19, z16,z17,z18,z19, z0, p0)

//  0   1   2   3   4   5   6   7   8   9
//  ^   ^   ^   ^
// x2 x20 x21 x22

MUL4ROW_D(z20,z21,z22,z23, z20,z21,z22,z23, z0, p0)
MLA4ROW_D(z12,z13,z14,z15, z2,z3,z4,z5,     z1, p0)
STOR2VEC_D(z12,z13,  p0,  x2)                          // Store Column 0
" add x2,x24,x10                            \n\t"      // Load address Column 6 of C
STOR2VEC_D(z14,z15,  p0, x20)                          // Store Column 1
" add x20,x2,x10                            \n\t"      // Load address Column 7 of C
LOAD2VEC_D(z10,z11, p0, x23)                           // Load Column 4
LOAD2VEC_D(z12,z13, p0, x24)                           // Load Column 5

//  0   1   2   3   4   5   6   7   8   9
//          ^   ^   ^   ^
//        x21 x22  x2 x20

MUL4ROW_D(z24,z25,z26,z27, z24,z25,z26,z27, z0, p0)
MLA4ROW_D(z16,z17,z18,z19, z6,z7,z8,z9,     z1, p0)
STOR2VEC_D(z16,z17, p0, x21)                           // Store Column 2
" add x21,x20,x10                           \n\t"      // Load address Column 8 of C
STOR2VEC_D(z18,z19, p0, x22)                           // Store Column 3
" add x22,x21,x10                           \n\t"      // Load address Column 9 of C
LOAD2VEC_D(z14,z15, p0,x2)                             // Load Column 6
LOAD2VEC_D(z16,z17, p0,x20)                            // Load Column 7

//  0   1   2   3   4   5   6   7   8   9
//                  ^   ^   ^   ^
//                 x2 x20 x21 x22
MUL4ROW_D(z28,z29,z30,z31, z28,z29,z30,z31, z0, p0)
MLA4ROW_D(z20,z21,z22,z23, z10,z11,z12,z13, z1, p0)
STOR2VEC_D(z20,z21, p0, x23)                            // Store Column 4
STOR2VEC_D(z22,z23, p0, x24)                            // Store Column 5
LOAD2VEC_D(z18,z19,p0,x21)                              // Load Column 8
LOAD2VEC_D(z20,z21,p0,x22)                              // Load Column 9

//  0   1   2   3   4   5   6   7   8   9
//                          ^   ^   ^   ^
//                        x21 x22  x2 x20
MLA4ROW_D(z24,z25,z26,z27, z14,z15,z16,z17, z1, p0)
MLA4ROW_D(z28,z29,z30,z31, z18,z19,z20,z21, z1, p0)
STOR2VEC_D(z24,z25, p0, x2)                             // Store Column 6
STOR2VEC_D(z26,z27, p0, x20)                            // Store Column 7
STOR2VEC_D(z28,z29, p0, x21)                            // Store Column 8
STOR2VEC_D(z30,z31, p0, x22)                            // Store Column 9
*/

" b .D2VX10END                              \n\t"       // Duplicate code for stores required due to lack of registers
"                                           \n\t"
" .D2VX10BETAZEROCONTCOLSTOREDS:            \n\t"
// No need to zero anything as we are storing the scaled accumulated A*B values
MUL4ROW_D(z12,z13,z14,z15, z12,z13,z14,z15, z0, p0)
MUL4ROW_D(z16,z17,z18,z19, z16,z17,z18,z19, z0, p0)
MUL4ROW_D(z20,z21,z22,z23, z20,z21,z22,z23, z0, p0)
MUL4ROW_D(z24,z25,z26,z27, z24,z25,z26,z27, z0, p0)
MUL4ROW_D(z28,z29,z30,z31, z28,z29,z30,z31, z0, p0)
STOR2VEC_D(z12,z13,p0,x2)                               // Store Column 0
" add x2,x24,x10                            \n\t"       // Load address Column 6 of C
STOR2VEC_D(z14,z15,p0,x20)                              // Store Column 1
" add x20,x2,x10                            \n\t"       // Load address Column 7 of C
STOR2VEC_D(z16,z17,p0,x21)                              // Store Column 2
" add x21,x20,x10                           \n\t"       // Load address Column 8 of C
STOR2VEC_D(z18,z19,p0,x22)                              // Store Column 3
" add x22,x21,x10                           \n\t"       // Load address Column 9 of C
STOR2VEC_D(z20,z21,p0,x23)                              // Store Column 4
STOR2VEC_D(z22,z23,p0,x24)                              // Store Column 5
STOR2VEC_D(z24,z25,p0,x2)                               // Store Column 6
STOR2VEC_D(z26,z27,p0,x20)                              // Store Column 7
STOR2VEC_D(z28,z29,p0,x21)                              // Store Column 8
STOR2VEC_D(z30,z31,p0,x22)                              // Store Column 9

"                                           \n\t"
" b .D2VX10END                              \n\t"
"                                           \n\t"
" .D2VX10GENSTORED:                         \n\t" // C is general-stride stored.

"                                           \n\t" // Creating index for stride load&store access
" index z2.d, xzr, x13                      \n\t" // 0, stride*double, 2*stride*double, ...
" mul x3, x13, x11                          \n\t" // x3 <- stride*double*vecsize
" index z3.d, x3, x13                       \n\t" // stride*double*vecsize, (vecsize+1)*stride*double, (vecsize+2)*stride*double, ...
"                                           \n\t"
" fcmp d1,#0.0                              \n\t"
" beq .D2VX10BETAZEROGENSTOREDS             \n\t" // multiply with beta if beta isn't zero
"                                           \n\t"
"                                           \n\t"
LOAD2VEC_GENI_D (z4,z5,  p0, x2, z2,z3)                  // Load Column 0
LOAD2VEC_GENI_D (z6,z7,  p0,x20, z2,z3)                  // Load Column 1
LOAD2VEC_GENI_D (z8,z9,  p0,x21, z2,z3)                  // Load Column 2
LOAD2VEC_GENI_D (z10,z11,p0,x22, z2,z3)                  // Load Column 3
"                                           \n\t"

// Let's do Accum=Accum*alpha, then Accum=Accum+C*beta and add some interleaving
MUL4ROW_D(z12,z13,z14,z15, z12,z13,z14,z15, z0, p0)
MUL4ROW_D(z16,z17,z18,z19, z16,z17,z18,z19, z0, p0)
//  0   1   2   3   4   5   6   7   8   9
//  ^   ^   ^   ^
// x2 x20 x21 x22

MUL4ROW_D(z20,z21,z22,z23, z20,z21,z22,z23, z0, p0)
MLA4ROW_D(z12,z13,z14,z15, z4,z5,z6,z7,     z1, p0)
STOR2VEC_GENI_D(z12,z13,  p0,  x2, z2,z3)              // Store Column 0
" add x2,x24,x10                            \n\t"      // Load address Column 6 of C
STOR2VEC_GENI_D(z14,z15,  p0, x20, z2,z3)              // Store Column 1
" add x20,x2,x10                            \n\t"      // Load address Column 7 of C
LOAD2VEC_GENI_D (z12,z13, p0, x23, z2,z3)              // Load Column 4
LOAD2VEC_GENI_D (z14,z15, p0, x24, z2,z3)              // Load Column 5

//  0   1   2   3   4   5   6   7   8   9
//          ^   ^   ^   ^
//        x21 x22  x2 x20

MUL4ROW_D(z24,z25,z26,z27, z24,z25,z26,z27, z0, p0)
MLA4ROW_D(z16,z17,z18,z19, z8,z9,z10,z11,   z1, p0)
STOR2VEC_GENI_D(z16,z17, p0, x21, z2,z3)               // Store Column 2
" add x21,x20,x10                           \n\t"      // Load address Column 8 of C
STOR2VEC_GENI_D(z18,z19, p0, x22, z2,z3)               // Store Column 3
" add x22,x21,x10                           \n\t"      // Load address Column 9 of C
LOAD2VEC_GENI_D(z16,z17, p0, x2, z2,z3)                // Load Column 6
LOAD2VEC_GENI_D(z18,z19, p0, x20, z2,z3)               // Load Column 7

//  0   1   2   3   4   5   6   7   8   9
//                  ^   ^   ^   ^
//                 x2 x20 x21 x22
MUL4ROW_D(z28,z29,z30,z31, z28,z29,z30,z31, z0, p0)
MLA4ROW_D(z20,z21,z22,z23, z12,z13,z14,z15, z1, p0)
STOR2VEC_GENI_D(z20,z21, p0, x23, z2,z3)               // Store Column 4
STOR2VEC_GENI_D(z22,z23, p0, x24, z2,z3)               // Store Column 5
LOAD2VEC_GENI_D(z20,z21, p0, x21, z2,z3)               // Load Column 8
LOAD2VEC_GENI_D(z22,z23, p0, x22, z2,z3)               // Load Column 9

//  0   1   2   3   4   5   6   7   8   9
//                          ^   ^   ^   ^
//                        x21 x22  x2 x20
MLA4ROW_D(z24,z25,z26,z27, z16,z17,z18,z19, z1, p0)
MLA4ROW_D(z28,z29,z30,z31, z20,z21,z22,z23, z1, p0)
STOR2VEC_GENI_D(z24,z25, p0,  x2, z2,z3)               // Store Column 6
STOR2VEC_GENI_D(z26,z27, p0, x20, z2,z3)               // Store Column 7
STOR2VEC_GENI_D(z28,z29, p0, x21, z2,z3)               // Store Column 8
STOR2VEC_GENI_D(z30,z31, p0, x22, z2,z3)               // Store Column 9

" b .D2VX10END                              \n\t"      // Duplicate code for stores required due to lack of registers
"                                           \n\t"
" .D2VX10BETAZEROGENSTOREDS:                \n\t"
// No need to zero anything as we are storing the scaled accumulated A*B values
MUL4ROW_D(z12,z13,z14,z15, z12,z13,z14,z15, z0, p0)
MUL4ROW_D(z16,z17,z18,z19, z16,z17,z18,z19, z0, p0)
MUL4ROW_D(z20,z21,z22,z23, z20,z21,z22,z23, z0, p0)
MUL4ROW_D(z24,z25,z26,z27, z24,z25,z26,z27, z0, p0)
MUL4ROW_D(z28,z29,z30,z31, z28,z29,z30,z31, z0, p0)
STOR2VEC_GENI_D(z12,z13, p0,  x2, z2,z3)               // Store Column 0
" add x2,x24,x10                            \n\t"      // Load address Column 6 of C
STOR2VEC_GENI_D(z14,z15, p0, x20, z2,z3)               // Store Column 1
" add x20,x2,x10                            \n\t"      // Load address Column 7 of C
STOR2VEC_GENI_D(z16,z17, p0, x21, z2,z3)               // Store Column 2
" add x21,x20,x10                           \n\t"      // Load address Column 8 of C
STOR2VEC_GENI_D(z18,z19, p0, x22, z2,z3)               // Store Column 3
" add x22,x21,x10                           \n\t"      // Load address Column 9 of C
STOR2VEC_GENI_D(z20,z21, p0, x23, z2,z3)               // Store Column 4
STOR2VEC_GENI_D(z22,z23, p0, x24, z2,z3)               // Store Column 5
STOR2VEC_GENI_D(z24,z25, p0,  x2, z2,z3)               // Store Column 6
STOR2VEC_GENI_D(z26,z27, p0, x20, z2,z3)               // Store Column 7
STOR2VEC_GENI_D(z28,z29, p0, x21, z2,z3)               // Store Column 8
STOR2VEC_GENI_D(z30,z31, p0, x22, z2,z3)               // Store Column 9
"                                            \n\t"
" .D2VX10END:                                \n\t"     // Done!
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
