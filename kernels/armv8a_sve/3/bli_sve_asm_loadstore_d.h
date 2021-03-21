/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2019-2020, Forschunszentrum Juelich

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

/****************************************************************
 * ============================================================ *
 *                    real loads/stores                         *
 * ============================================================ *
 ****************************************************************/

/****************************************************************
 *            LOAD n vectors starting from address              *
 ****************************************************************/

#define LOAD1VEC_D(vec,preg,areg)\
    " ld1d  " #vec ".d, " #preg "/z, [" #areg "]           \n\t"

#define LOAD1VEC_VOFF_D(vec, preg, areg, off)\
    " ld1d   " #vec ".d, " #preg "/z, [" #areg ", #" #off", MUL VL]\n\t"

#define LOAD2VEC_D(vec1,vec2,preg,areg)\
    " ld1d  " #vec1 ".d, " #preg "/z, [" #areg "]           \n\t"\
    " ld1d  " #vec2 ".d, " #preg "/z, [" #areg ",#1, MUL VL]\n\t"

#define LOAD4VEC_D(vec1, vec2, vec3, vec4, preg,areg)\
    " ld1d  " #vec1 ".d, " #preg "/z, [" #areg "]           \n\t"\
    " ld1d  " #vec2 ".d, " #preg "/z, [" #areg ",#1, MUL VL]\n\t"\
    " ld1d  " #vec3 ".d, " #preg "/z, [" #areg ",#2, MUL VL]\n\t"\
    " ld1d  " #vec4 ".d, " #preg "/z, [" #areg ",#3, MUL VL]\n\t"

#define LOAD1VEC_CONT_D(vec,preg,areg,avec) LOAD1VEC_D(vec,preg,areg)

#define LOAD2VEC_CONT_D(vec1,vec2,preg,areg,avec1,avec2) LOAD2VEC_D(vec1,vec2,preg,areg)

/****************************************************************
 *          GATHER n vectors starting from address              *
 ****************************************************************/

#define LOAD1VEC_GENI_D(vec, preg, areg, avec)\
    " ld1d  " #vec ".d, " #preg "/z, [" #areg "," #avec ".d, LSL #3]\n\t"

#define LOAD2VEC_GENI_D(vec1,vec2, preg, areg, avec1,avec2)\
    " ld1d  " #vec1 ".d, " #preg "/z, [" #areg "," #avec1 ".d, LSL #3]\n\t"\
    " ld1d  " #vec2 ".d, " #preg "/z, [" #areg "," #avec2 ".d, LSL #3]\n\t"

#define LOAD4VEC_GENI_D(vec1,vec2,vec3,vec4, preg,areg, avec1,avec2,avec3,avec4)\
    " ld1d  " #vec1 ".d, " #preg "/z, [" #areg "," #avec1 ".d, LSL #3]\n\t"\
    " ld1d  " #vec2 ".d, " #preg "/z, [" #areg "," #avec2 ".d, LSL #3]\n\t"\
    " ld1d  " #vec3 ".d, " #preg "/z, [" #areg "," #avec3 ".d, LSL #3]\n\t"\
    " ld1d  " #vec4 ".d, " #preg "/z, [" #areg "," #avec4 ".d, LSL #3]\n\t"

/****************************************************************
 *       LOAD n doubles from address and replicate each         *
 *                       across a vector                        *
 ****************************************************************/

#define LDR_NOADDR_D(vec1,preg)\
    " ld1rd  " #vec1 ".d, " #preg "/z"
#define OA_D(areg,offset)\
    ",[" #areg ", #" #offset"]"

#define LOAD1VEC_DIST_OFF_D(vec1,preg,areg,off)\
    LDR_NOADDR_D(vec1,preg)OA_D(areg,off)"\n\t"

#define LOAD1VEC_DIST_D(vec1,preg,areg)\
    LDR_NOADDR_D(vec1,preg)OA_D(areg,0)"\n\t"

#define LOAD2VEC_DIST_D(vec1,vec2,preg,areg)\
    LDR_NOADDR_D(vec1,preg)OA_D(areg,0)"\n\t"\
    LDR_NOADDR_D(vec2,preg)OA_D(areg,8)"\n\t"

#define LOAD4VEC_DIST_D(vec1,vec2,vec3,vec4,preg,areg)\
    LDR_NOADDR_D(vec1,preg)OA_D(areg,0)"\n\t"\
    LDR_NOADDR_D(vec2,preg)OA_D(areg,8)"\n\t"\
    LDR_NOADDR_D(vec3,preg)OA_D(areg,16)"\n\t"\
    LDR_NOADDR_D(vec4,preg)OA_D(areg,24)"\n\t"

#define LOAD5VEC_DIST_D(vec1,vec2,vec3,vec4,vec5, preg,areg)\
    LDR_NOADDR_D(vec1,preg)OA_D(areg,0)"\n\t"\
    LDR_NOADDR_D(vec2,preg)OA_D(areg,8)"\n\t"\
    LDR_NOADDR_D(vec3,preg)OA_D(areg,16)"\n\t"\
    LDR_NOADDR_D(vec4,preg)OA_D(areg,24)"\n\t"\
    LDR_NOADDR_D(vec5,preg)OA_D(areg,32)"\n\t"

#define LOAD8VEC_DIST_D(vec1,vec2,vec3,vec4,vec5,vec6,vec7,vec8,preg,areg)\
    LDR_NOADDR_D(vec1,preg)OA_D(areg,0)"\n\t"\
    LDR_NOADDR_D(vec2,preg)OA_D(areg,8)"\n\t"\
    LDR_NOADDR_D(vec3,preg)OA_D(areg,16)"\n\t"\
    LDR_NOADDR_D(vec4,preg)OA_D(areg,24)"\n\t"\
    LDR_NOADDR_D(vec5,preg)OA_D(areg,32)"\n\t"\
    LDR_NOADDR_D(vec6,preg)OA_D(areg,40)"\n\t"\
    LDR_NOADDR_D(vec7,preg)OA_D(areg,48)"\n\t"\
    LDR_NOADDR_D(vec8,preg)OA_D(areg,56)"\n\t"

#define LOAD10VEC_DIST_D(vec1,vec2,vec3,vec4,vec5,vec6,vec7,vec8,vec9,vec10,preg,areg)\
    LDR_NOADDR_D(vec1,preg)OA_D(areg,0)"\n\t"\
    LDR_NOADDR_D(vec2,preg)OA_D(areg,8)"\n\t"\
    LDR_NOADDR_D(vec3,preg)OA_D(areg,16)"\n\t"\
    LDR_NOADDR_D(vec4,preg)OA_D(areg,24)"\n\t"\
    LDR_NOADDR_D(vec5,preg)OA_D(areg,32)"\n\t"\
    LDR_NOADDR_D(vec6,preg)OA_D(areg,40)"\n\t"\
    LDR_NOADDR_D(vec7,preg)OA_D(areg,48)"\n\t"\
    LDR_NOADDR_D(vec8,preg)OA_D(areg,56)"\n\t"\
    LDR_NOADDR_D(vec9,preg)OA_D(areg,64)"\n\t"\
    LDR_NOADDR_D(vec10,preg)OA_D(areg,72)"\n\t"

#define LOAD12VEC_DIST_D(vec1,vec2,vec3,vec4,vec5,vec6,vec7,vec8,vec9,vec10,vec11,vec12,preg,areg)\
    LDR_NOADDR_D(vec1,preg)OA_D(areg,0)"\n\t"\
    LDR_NOADDR_D(vec2,preg)OA_D(areg,8)"\n\t"\
    LDR_NOADDR_D(vec3,preg)OA_D(areg,16)"\n\t"\
    LDR_NOADDR_D(vec4,preg)OA_D(areg,24)"\n\t"\
    LDR_NOADDR_D(vec5,preg)OA_D(areg,32)"\n\t"\
    LDR_NOADDR_D(vec6,preg)OA_D(areg,40)"\n\t"\
    LDR_NOADDR_D(vec7,preg)OA_D(areg,48)"\n\t"\
    LDR_NOADDR_D(vec8,preg)OA_D(areg,56)"\n\t"\
    LDR_NOADDR_D(vec9,preg)OA_D(areg,64)"\n\t"\
    LDR_NOADDR_D(vec10,preg)OA_D(areg,72)"\n\t"\
    LDR_NOADDR_D(vec11,preg)OA_D(areg,80)"\n\t"\
    LDR_NOADDR_D(vec12,preg)OA_D(areg,88)"\n\t"

/****************************************************************
 *     LOAD n*2 doubles from address and replicate each pair    *
 *                    across a vector                           *
 ****************************************************************/

#define LDRQ_NOADDR_D(vec1,preg)\
    " ld1rqd  " #vec1 ".d, " #preg "/z"
#define OA_D(areg,offset)\
    ",[" #areg ", #" #offset"]"

#define LOADVEC_QDIST_OFF_D(vec1,preg,areg,off)\
    LDRQ_NOADDR_D(vec1,preg)OA_D(areg,off)"\n\t"

#define LOADVEC_QDIST_D(vec1,preg,areg)\
    LOADVEC_QDIST_OFF_D(vec1,preg,areg,0)"\n\t"

#define LOAD2VEC_QDIST_D(vec1,vec2,preg,areg)\
    LOADVEC_QDIST_OFF_D(vec1,preg,areg,0)"\n\t"\
    LOADVEC_QDIST_OFF_D(vec2,preg,areg,16)"\n\t"

#define LOAD4VEC_QDIST_D(vec1,vec2,vec3,vec4,preg,areg)\
    LOADVEC_QDIST_OFF_D(vec1,preg,areg,0)"\n\t"\
    LOADVEC_QDIST_OFF_D(vec2,preg,areg,16)"\n\t"\
    LOADVEC_QDIST_OFF_D(vec3,preg,areg,32)"\n\t"\
    LOADVEC_QDIST_OFF_D(vec4,preg,areg,48)"\n\t"

#define LOAD6VEC_QDIST_D(vec1,vec2,vec3,vec4,vec5,vec6,preg,areg)\
    LOADVEC_QDIST_OFF_D(vec1,preg,areg,0)"\n\t"\
    LOADVEC_QDIST_OFF_D(vec2,preg,areg,16)"\n\t"\
    LOADVEC_QDIST_OFF_D(vec3,preg,areg,32)"\n\t"\
    LOADVEC_QDIST_OFF_D(vec4,preg,areg,48)"\n\t"\
    LOADVEC_QDIST_OFF_D(vec5,preg,areg,64)"\n\t"\
    LOADVEC_QDIST_OFF_D(vec6,preg,areg,80)"\n\t"

/****************************************************************
 * ============================================================ *
 *                  complex loads/stores                        *
 * ============================================================ *
 ****************************************************************/

#if defined(USE_SVE_CMLA_INSTRUCTION)
// When using the fused complex mupliply-accumulate
// load {real, imag} into every 128 bit part of the vector

/****************************************************************
 *            LOAD n vectors starting from address              *
 ****************************************************************/
#define LOAD2VEC_Z(vec1, vec2, preg, areg) LOAD2VEC_D(vec1, vec2, preg, areg)

#define LOAD2VEC_VOFF_Z(vec1, vec2, preg, areg, off1, off2)\
    " ld1d  " #vec1 ".d, " #preg "/z, [" #areg ",#"#off1",   MUL VL]           \n\t"\
    " ld1d  " #vec2 ".d, " #preg "/z, [" #areg ",#"#off2", MUL VL]\n\t"

/****************************************************************
 *   LOAD n complex doubles from address and replicate each     *
 *                       across a vector                        *
 ****************************************************************/

#define LDC2_DIST_NOADDR_D(vec1, vec2, preg)\
    " ld1rqd {" #vec1 ".d}, " #preg "/z"

#define LOAD2VEC_DIST_OFF_Z(vec1, vec2, preg, areg, off1, off2)\
    LDC2_DIST_NOADDR_D(vec1, vec2, preg)OA_D(areg,off1)"\n\t"

// Load 2 vectors and distribute the first complex number to all 128 parts
#define LOAD2VEC_DIST_Z(vec1, vec2, preg, areg) LOAD2VEC_DIST_OFF_Z(vec1, vec2, preg, areg, 0)

#define LOAD4VEC_DIST_OFF_Z(vec1, vec2, vec3, vec4, preg, areg, off1, off2, off3, off4)\
    LOAD2VEC_DIST_OFF_Z(vec1, vec2, preg, areg, off1, off2)\
    LOAD2VEC_DIST_OFF_Z(vec3, vec4, preg, areg, off3, off4)

#define LOAD4VEC_DIST_Z(vec1, vec2, vec3, vec4, preg, areg)\
    LOAD4VEC_DIST_OFF_Z(vec1, vec2, vec3, vec4, preg, areg, 0, 8, 16, 24)

#define LOAD8VEC_DIST_Z(vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8, preg, areg)\
    LOAD4VEC_DIST_OFF_Z(vec1, vec2, vec3, vec4, preg, areg, 0, 8, 16, 24)\
    LOAD4VEC_DIST_OFF_Z(vec5, vec6, vec7, vec8, preg, areg, 32, 40, 48, 56)

#define LOAD10VEC_DIST_Z(vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8, vec9, vec10, preg, areg)\
    LOAD4VEC_DIST_OFF_Z(vec1, vec2, vec3, vec4, preg, areg, 0, 8, 16, 24)\
    LOAD4VEC_DIST_OFF_Z(vec5, vec6, vec7, vec8, preg, areg, 32, 40, 48, 56)\
    LOAD2VEC_DIST_OFF_Z(vec9, vec10, preg, areg, 64, 72)

#define LOAD12VEC_DIST_Z(vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8, vec9, vec10, vec11, vec12, preg, areg)\
    LOAD4VEC_DIST_OFF_Z(vec1, vec2, vec3, vec4, preg, areg, 0, 8, 16, 24)\
    LOAD4VEC_DIST_OFF_Z(vec5, vec6, vec7, vec8, preg, areg, 32, 40, 48, 56)\
    LOAD4VEC_DIST_OFF_Z(vec9, vec10, vec11, vec12, preg, areg, 64, 72, 80, 88)

/****************************************************************
 *         Create gather-load/scatter-store index               *
 ****************************************************************/
// Make an index for gather-load/scatter-store
// Here it has to be
// stride = estride*sizeof(double complex)
// ivec1 = {0,8,stride, stride+8, ... }
// ivec2 = {nelem*stride, nelem*stride+8, (nelem+1)*stride, (nelem+1)*stride+8, ... }
#define MKINDC_2VEC_D(ivec1, ivec2, stridereg, nelemreg, sparereg)\
    " lsl " #sparereg", "#stridereg", #1\n\t"\
    " index "#ivec1".d,xzr," #sparereg "\n\t"\
    " mov "  #ivec2".d," #ivec1".d\n\t"\
    " add "  #ivec2".d," #ivec2".d,#1\n\t"\
    " zip1 " #ivec1".d," #ivec1".d," #ivec2".d\n\t"\
    " mul "  #sparereg", "#sparereg", "#nelemreg"\n\t"\
    " lsr " #sparereg", " #sparereg", #1\n\t"\
    " dup "  #ivec2".d," #sparereg"\n\t"\
    " add "  #ivec2".d," #ivec2 ".d," #ivec1".d\n\t"

/****************************************************************
 *            STORE n vectors starting from address             *
 ****************************************************************/
#define STOR2VEC_Z(vec1, vec2, preg, areg) STOR2VEC_D(vec1, vec2, preg, areg)

#else
// Otherwise
// load {real} {imag} into separate vectors

/****************************************************************
 *            LOAD n vectors starting from address             *
 ****************************************************************/
#define LOAD2VEC_Z(vec1, vec2, preg, areg)\
    " ld2d {" #vec1 ".d," #vec2 ".d}, " #preg "/z, [" #areg "]\n\t"

#define LOAD2VEC_VOFF_Z(vec1, vec2, preg, areg, off1, off2)\
    " ld2d {" #vec1 ".d," #vec2 ".d}, " #preg "/z, [" #areg ",#"#off1", MUL VL]\n\t"

#define LOAD2VEC_DIST_OFF_Z(vec_r, vec_i, preg,areg,off1,off2)\
    LDR_NOADDR_D(vec_r,preg)OA_D(areg,off1)"\n\t"\
    LDR_NOADDR_D(vec_i,preg)OA_D(areg,off2)"\n\t"

#define LOAD2VEC_DIST_Z(vec_r, vec_i, preg, areg) LOAD2VEC_DIST_D(vec_r, vec_i, preg, areg)

#define LOAD4VEC_DIST_Z(vec1_r, vec1_i, vec2_r, vec2_i, preg, areg)\
    LOAD4VEC_DIST_D(vec1_r, vec1_i, vec2_r, vec2_i, preg, areg)

#define LOAD8VEC_DIST_Z(vec1_r, vec1_i, vec2_r, vec2_i, vec3_r, vec3_i, vec4_r, vec4_i, preg, areg)\
    LOAD8VEC_DIST_D(vec1_r, vec1_i, vec2_r, vec2_i, vec3_r, vec3_i, vec4_r, vec4_i, preg, areg)

#define LOAD10VEC_DIST_Z(vec1_r, vec1_i, vec2_r, vec2_i, vec3_r, vec3_i, vec4_r, vec4_i, vec5_r, vec5_i, preg, areg)\
    LOAD10VEC_DIST_D(vec1_r, vec1_i, vec2_r, vec2_i, vec3_r, vec3_i, vec4_r, vec4_i, vec5_r, vec5_i, preg, areg)

#define LOAD12VEC_DIST_Z(vec1_r,vec1_i, vec2_r,vec2_i, vec3_r,vec3_i, vec4_r,vec4_i, vec5_r,vec5_i, vec6_r,vec6_i, preg, areg)\
    LOAD12VEC_DIST_D(vec1_r,vec1_i, vec2_r,vec2_i, vec3_r,vec3_i, vec4_r,vec4_i, vec5_r,vec5_i, vec6_r,vec6_i, preg, areg)

// Make an index for gather-load/scatter-store
// Here it has to be
// stride = estride*sizeof(double complex)
// ivec1 = {0,stride,  2*stride,    ... }
// ivec2 = {8,stride+8,2*stride+8,  ... }
#define MKINDC_2VEC_D(ivec1, ivec2, stridereg, nelemreg, sparereg)\
    " lsl " #sparereg", "#stridereg", #1\n\t"\
    " index "#ivec1".d,xzr," #sparereg "\n\t"\
    " mov "#ivec2".d, "#ivec1".d\n\t"\
    " add "#ivec2".d, "#ivec2".d,#1\n\t"

#define STOR2VEC_Z(vec1, vec2, preg, areg)\
    " st2d {" #vec1 ".d," #vec2 ".d}, " #preg ", [" #areg "]\n\t"

#endif

#define LOAD2VEC_CONT_Z(vec1,vec2,preg,areg,avec1,avec2) LOAD2VEC_Z(vec1,vec2,preg,areg)

#define LOAD2VEC_GENI_Z(vec1,vec2,preg,areg,avec1,avec2)\
    " ld1d  " #vec1 ".d, " #preg "/z, [" #areg "," #avec1 ".d, LSL #3]\n\t"\
    " ld1d  " #vec2 ".d, " #preg "/z, [" #areg "," #avec2 ".d, LSL #3]\n\t"

// Store 1 vector to contiguous memory
#define STOR1VEC_D(vec,preg,areg)\
    " st1d  {" #vec ".d}, " #preg ", [" #areg "]           \n\t"

#define STOR1VEC_CONT_D(vec,preg,areg,avec) STOR1VEC_D(vec,preg,areg)

// Store 1 vector with generic indexing (scatter-store)
#define STOR1VEC_GENI_D(vec,preg,areg,avec)\
    " st1d  {" #vec ".d}, " #preg ", [" #areg "," #avec ".d, LSL #3]\n\t"

// Store 2 vectors to contiguous memory
#define STOR2VEC_D(vec1,vec2,preg,areg)\
    " st1d  {" #vec1 ".d}, " #preg ", [" #areg "]           \n\t"\
    " st1d  {" #vec2 ".d}, " #preg ", [" #areg ",#1, MUL VL]\n\t"

#define STOR4VEC_D(vec1,vec2,vec3,vec4,preg,areg)\
    " st1d  {" #vec1 ".d}, " #preg ", [" #areg "]           \n\t"\
    " st1d  {" #vec2 ".d}, " #preg ", [" #areg ",#1, MUL VL]\n\t"\
    " st1d  {" #vec3 ".d}, " #preg ", [" #areg ",#2, MUL VL]\n\t"\
    " st1d  {" #vec4 ".d}, " #preg ", [" #areg ",#3, MUL VL]\n\t"

#define STOR2VEC_CONT_D(vec1,vec2,preg,areg,avec1,avec2) STOR2VEC_D(vec1,vec2,preg,areg)

#define STOR4VEC_CONT_D(vec1,vec2,vec3,vec4,preg,areg,avec1,avec2) STOR4VEC_D(vec1,vec2,vec3,vec4,preg,areg)

// Store 2 vectors with generic indexing (scatter-store)
#define STOR2VEC_GENI_D(vec1,vec2,preg,areg,avec1,avec2)\
    " st1d  {" #vec1 ".d}, " #preg ", [" #areg "," #avec1 ".d, LSL #3]\n\t"\
    " st1d  {" #vec2 ".d}, " #preg ", [" #areg "," #avec2 ".d, LSL #3]\n\t"

#define STOR4VEC_GENI_D(vec1,vec2,vec3,vec4, preg,areg,avec1,avec2,avec3,avec4)\
    " st1d  {" #vec1 ".d}, " #preg ", [" #areg "," #avec1 ".d, LSL #3]\n\t"\
    " st1d  {" #vec2 ".d}, " #preg ", [" #areg "," #avec2 ".d, LSL #3]\n\t"\
    " st1d  {" #vec3 ".d}, " #preg ", [" #areg "," #avec3 ".d, LSL #3]\n\t"\
    " st1d  {" #vec4 ".d}, " #preg ", [" #areg "," #avec4 ".d, LSL #3]\n\t"

#define STOR2VEC_CONT_Z(vec1,vec2,preg,areg,avec1,avec2) STOR2VEC_Z(vec1,vec2,preg,areg)

// Store 2 vectors with generic indexing (scatter-store)
#define STOR2VEC_GENI_Z(vec1,vec2,preg,areg,avec1,avec2)\
    " st1d  {" #vec1 ".d}, " #preg ", [" #areg "," #avec1 ".d, LSL #3]\n\t"\
    " st1d  {" #vec2 ".d}, " #preg ", [" #areg "," #avec2 ".d, LSL #3]\n\t"


// Some macros used for fixed size kernels

#if defined(USE_SVE_CMLA_INSTRUCTION)
#define LDR_BVEC_D(vec1, vec2, preg, reg, offset1, offset2)\
    " ld1rqd  " #vec1 ".d, " #preg "/z, [" #reg ", #" #offset1 "]   \n\t"
#else
#define LDR_BVEC_D(vec1, vec2, preg, reg, offset1, offset2)\
    " ld1rd  " #vec1 ".d, " #preg "/z, [" #reg ", #" #offset1 "]   \n\t"\
    " ld1rd  " #vec2 ".d, " #preg "/z, [" #reg ", #" #offset2 "]   \n\t"
#endif

#if defined(USE_SVE_CMLA_INSTRUCTION)
#define LD_AVEC_D(vec1, vec2, preg, reg)\
    " ld1d  " #vec1 ".d, " #preg "/z, [" #reg "]           \n\t"\
    " ld1d  " #vec2 ".d, " #preg "/z, [" #reg ",#1, MUL VL]\n\t"
#else
#define LD_AVEC_D(vec1, vec2, preg, reg)\
    " ld2d {" #vec1 ".d," #vec2 ".d}, " #preg "/z, [" #reg "]\n\t"
#endif

#if defined(USE_SVE_CMLA_INSTRUCTION)
#define ST_AVEC_D(vec1, vec2, preg, reg)\
    " st1d  " #vec1 ".d, " #preg ", [" #reg "]           \n\t"\
    " st1d  " #vec2 ".d, " #preg ", [" #reg ",#1, MUL VL]\n\t"
#else
#define ST_AVEC_D(vec1, vec2, preg, reg)\
    " st2d {" #vec1 ".d," #vec2 ".d}, " #preg ", [" #reg "]\n\t"
#endif




/****************************************************************
 * ============================================================ *
 *                         prefetches                           *
 * ============================================================ *
 ****************************************************************/

#if defined(PREFETCH64)
    #define PREF64(type,areg,off)\
" prfm " #type ",[" #areg ",#" #off "]\n\t"
#else
    #define PREF64(type,areg,off)
#endif

#if defined(PREFETCH256)
    #define PREF256(type,areg,off)\
" prfm " #type ",[" #areg ",#" #off "]\n\t"
#else
    #define PREF256(type,areg,off)
#endif

#if defined(PREFETCH64) || defined(PREFETCH256)
    #define PREFANY(type,areg,off)\
" prfm " #type ",[" #areg ",#" #off "]\n\t"
#else
    #define PREFANY(type,areg,off)
#endif


// One cache line is one vector
#if defined(PREFETCHSVE1)
    #define PREFSVE1(type,preg,areg,voff)\
" prfd " #type "," #preg ", [" #areg ", #" #voff ", MUL VL]        \n\t"
#else
    #define PREFSVE1(type,preg,areg,voff)
#endif


// One cache line is two vectors
#if defined(PREFETCHSVE2)
    #define PREFSVE2(type,preg,areg,voff)\
" prfd " #type "," #preg ", [" #areg ", #" #voff ", MUL VL]        \n\t"
#else
    #define PREFSVE2(type,preg,areg,voff)
#endif


// One cache line is four vectors
#if defined(PREFETCHSVE4)
    #define PREFSVE4(type,preg,areg,voff)\
" prfd " #type "," #preg ", [" #areg ", #" #voff ", MUL VL]        \n\t"
#else
    #define PREFSVE4(type,preg,areg,voff)
#endif

// One cache line is 2 or 1 vector
#if defined(PREFETCHSVE1) || defined(PREFETCHSVE2)
    #define PREFSVE21(type,preg,areg,voff)\
" prfd " #type "," #preg ", [" #areg ", #" #voff ", MUL VL]        \n\t"
#else
    #define PREFSVE21(type,preg,areg,voff)
#endif

// One cache line is four, two or one vector
#if defined(PREFETCHSVE1) || defined(PREFETCHSVE2) || defined(PREFETCHSVE4)
    #define PREFSVE421(type,preg,areg,voff)\
" prfd " #type "," #preg ", [" #areg ", #" #voff ", MUL VL]        \n\t"
#else
    #define PREFSVE421(type,preg,areg,voff)
#endif
