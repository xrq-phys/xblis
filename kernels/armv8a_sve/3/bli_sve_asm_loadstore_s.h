/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2019, Forschunszentrum Juelich

   Author(s): Stepan Nassyr, s.nassyr@fz-juelich.se

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

#define LOAD1VEC_S(vec,preg,areg)\
    " ld1w  " #vec ".s, " #preg "/z, [" #areg "]           \n\t"

#define LOAD1VEC_VOFF_S(vec, preg, areg, off)\
    " ld1w   " #vec ".s, " #preg "/z, [" #areg ", #" #off", MUL VL]\n\t"

#define LOAD1VEC_CONT_S(vec,preg,areg,avec) LOAD1VEC_S(vec,preg,areg)

#define LOAD1VEC_GENI_S(vec,preg,areg,avec)\
    " ld1w  " #vec ".s, " #preg "/z, [" #areg "," #avec ".s, UXTW #2]\n\t"

#define LOAD2VEC_S(vec1,vec2,preg,areg)\
    " ld1w  " #vec1 ".s, " #preg "/z, [" #areg "]           \n\t"\
    " ld1w  " #vec2 ".s, " #preg "/z, [" #areg ",#1, MUL VL]\n\t"

#define LOAD2VEC_CONT_S(vec1,vec2,preg,areg,avec1,avec2) LOAD2VEC_S(vec1,vec2,preg,areg)

#define LOAD2VEC_GENI_S(vec1,vec2,preg,areg,avec1,avec2)\
    " ld1w  " #vec1 ".s, " #preg "/z, [" #areg "," #avec1 ".s, UXTW #2]\n\t"\
    " ld1w  " #vec2 ".s, " #preg "/z, [" #areg "," #avec2 ".s, UXTW #2]\n\t"

#define LDR_NOADDR_S(vec1,preg)\
    " ld1rw  " #vec1 ".s, " #preg "/z"
#define OA_S(areg,offset)\
    ",[" #areg ", #" #offset"]"

#define LOAD1VEC_DIST_OFF_S(vec1,preg,areg,off)\
    LDR_NOADDR_S(vec1,preg)OA_S(areg,off)"\n\t"

#define LOADVEC_DIST_S(vec1,preg,areg)\
    LDR_NOADDR_S(vec1,preg)OA_S(areg,0)"\n\t"

#define LOAD2VEC_DIST_S(vec1,vec2,preg,areg)\
    LDR_NOADDR_S(vec1,preg)OA_S(areg,0)"\n\t"\
    LDR_NOADDR_S(vec2,preg)OA_S(areg,4)"\n\t"

#define LOAD4VEC_DIST_S(vec1,vec2,vec3,vec4,preg,areg)\
    LDR_NOADDR_S(vec1,preg)OA_S(areg,0)"\n\t"\
    LDR_NOADDR_S(vec2,preg)OA_S(areg,4)"\n\t"\
    LDR_NOADDR_S(vec3,preg)OA_S(areg,8)"\n\t"\
    LDR_NOADDR_S(vec4,preg)OA_S(areg,12)"\n\t"

#define LOAD8VEC_DIST_S(vec1,vec2,vec3,vec4,vec5,vec6,vec7,vec8,preg,areg)\
    LDR_NOADDR_S(vec1,preg)OA_S(areg,0)"\n\t"\
    LDR_NOADDR_S(vec2,preg)OA_S(areg,4)"\n\t"\
    LDR_NOADDR_S(vec3,preg)OA_S(areg,8)"\n\t"\
    LDR_NOADDR_S(vec4,preg)OA_S(areg,12)"\n\t"\
    LDR_NOADDR_S(vec5,preg)OA_S(areg,16)"\n\t"\
    LDR_NOADDR_S(vec6,preg)OA_S(areg,20)"\n\t"\
    LDR_NOADDR_S(vec7,preg)OA_S(areg,24)"\n\t"\
    LDR_NOADDR_S(vec8,preg)OA_S(areg,28)"\n\t"


#if defined(USE_SVE_CMLA_INSTRUCTION)
// When using the fused complex mupliply-accumulate
// load {real, imag} into every 128 bit part of the vector

// Load 2 vectors worth of complex numbers
#define LOAD2VEC_C(vec1, vec2, preg, areg) LOAD2VEC_S(vec1, vec2, preg, areg)

#define LOAD2VEC_VOFF_C(vec1, vec2, preg, areg, off1, off2)\
    " ld1w  " #vec1 ".s, " #preg "/z, [" #areg ",#"#off1",   MUL VL]           \n\t"\
    " ld1w  " #vec2 ".s, " #preg "/z, [" #areg ",#"#off2", MUL VL]\n\t"

#define LDC2_DIST_NOADDR_S(vec1, vec2, preg)\
    " ld1rd {" #vec1 ".d}, " #preg "/z"
#define LOAD2VEC_DIST_OFF_C(vec1, vec2, preg, areg, off1, off2)\
    LDC2_DIST_NOADDR_S(vec1, vec2, preg)OA_S(areg,off1)"\n\t"

// Load 2 vectors and distribute the first complex number to all 128 parts
#define LOAD2VEC_DIST_C(vec1, vec2, preg, areg) LOAD2VEC_DIST_OFF_C(vec1, vec2, preg, areg, 0)

#define LOAD4VEC_DIST_OFF_C(vec1, vec2, vec3, vec4, preg, areg, off1, off2, off3, off4)\
    LOAD2VEC_DIST_OFF_C(vec1, vec2, preg, areg, off1, off2)\
    LOAD2VEC_DIST_OFF_C(vec3, vec4, preg, areg, off3, off4)

#define LOAD4VEC_DIST_C(vec1, vec2, vec3, vec4, preg, areg)\
    LOAD4VEC_DIST_OFF_C(vec1, vec2, vec3, vec4, preg, areg, 0, 4, 8, 12)

#define LOAD8VEC_DIST_C(vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8, preg, areg)\
    LOAD4VEC_DIST_OFF_C(vec1, vec2, vec3, vec4, preg, areg, 0, 4, 8, 12)\
    LOAD4VEC_DIST_OFF_C(vec5, vec6, vec7, vec8, preg, areg, 16, 20, 24, 28)

// Make an index for gather-load/scatter-store
// Here it has to be
// stride = estride*sizeof(double complex)
// ivec1 = {0,8,stride, stride+8, ... }
// ivec2 = {nelem*stride, nelem*stride+8, (nelem+1)*stride, (nelem+1)*stride+8, ... }
#define MKINDC_2VEC_S(ivec1, ivec2, stridereg, nelemreg, sparereg)\
    " lsl " #sparereg", "#stridereg", #1\n\t"\
    " index "#ivec1".s,wzr," #sparereg "\n\t"\
    " mov "  #ivec2".d," #ivec1".d\n\t"\
    " add "  #ivec2".s," #ivec2".s,#1\n\t"\
    " zip1 " #ivec1".s," #ivec1".s," #ivec2".s\n\t"\
    " mul "  #sparereg", "#sparereg", "#nelemreg"\n\t"\
    " lsr " #sparereg", " #sparereg", #1\n\t"\
    " dup "  #ivec2".s," #sparereg"\n\t"\
    " add "  #ivec2".s," #ivec2 ".s," #ivec1".s\n\t"

// Load 2 vectors worth of complex numbers
#define STOR2VEC_C(vec1, vec2, preg, areg) STOR2VEC_S(vec1, vec2, preg, areg)

#else
// Otherwise
// load {real} {imag} into separate vectors

#define LOAD2VEC_C(vec1, vec2, preg, areg)\
    " ld2w {" #vec1 ".s," #vec2 ".s}, " #preg "/z, [" #areg "]\n\t"

#define LOAD2VEC_VOFF_C(vec1, vec2, preg, areg, off1, off2)\
    " ld2w {" #vec1 ".s," #vec2 ".s}, " #preg "/z, [" #areg ",#"#off1", MUL VL]\n\t"

#define LOAD2VEC_DIST_OFF_C(vec_r, vec_i, preg,areg,off1,off2)\
    LDR_NOADDR_S(vec_r,preg)OA_S(areg,off1)"\n\t"\
    LDR_NOADDR_S(vec_i,preg)OA_S(areg,off2)"\n\t"

#define LOAD2VEC_DIST_C(vec_r, vec_i, preg, areg) LOAD2VEC_DIST_S(vec_r, vec_i, preg, areg)
#define LOAD4VEC_DIST_C(vec1_r, vec1_i, vec2_r, vec2_i, preg, areg)\
    LOAD4VEC_DIST_S(vec1_r, vec1_i, vec2_r, vec2_i, preg, areg)
#define LOAD8VEC_DIST_C(vec1_r, vec1_i, vec2_r, vec2_i, vec3_r, vec3_i, vec4_r, vec4_i, preg, areg)\
    LOAD8VEC_DIST_S(vec1_r, vec1_i, vec2_r, vec2_i, vec3_r, vec3_i, vec4_r, vec4_i, preg, areg)

// Make an index for gather-load/scatter-store
// Here it has to be
// stride = estride*sizeof(double complex)
// ivec1 = {0,stride,  2*stride,    ... }
// ivec2 = {8,stride+8,2*stride+8,  ... }
#define MKINDC_2VEC_S(ivec1, ivec2, stridereg, nelemreg, sparereg)\
    " lsl " #sparereg", "#stridereg", #1\n\t"\
    " index "#ivec1".s,wzr," #sparereg "\n\t"\
    " mov "#ivec2".d, "#ivec1".d\n\t"\
    " add "#ivec2".s, "#ivec2".s,#1\n\t"

#define STOR2VEC_C(vec1, vec2, preg, areg)\
    " st2w {" #vec1 ".s," #vec2 ".s}, " #preg ", [" #areg "]\n\t"

#endif

#define LOAD2VEC_CONT_C(vec1,vec2,preg,areg,avec1,avec2) LOAD2VEC_C(vec1,vec2,preg,areg)

#define LOAD2VEC_GENI_C(vec1,vec2,preg,areg,avec1,avec2)\
    " ld1w  " #vec1 ".s, " #preg "/z, [" #areg "," #avec1 ".s, UXTW #2]\n\t"\
    " ld1w  " #vec2 ".s, " #preg "/z, [" #areg "," #avec2 ".s, UXTW #2]\n\t"

// Store 1 vector to contiguous memory
#define STOR1VEC_S(vec,preg,areg)\
    " st1w  {" #vec ".s}, " #preg ", [" #areg "]           \n\t"

#define STOR1VEC_CONT_S(vec,preg,areg,avec) STOR1VEC_S(vec,preg,areg)

// Store 1 vector with generic indexing (scatter-store)
#define STOR1VEC_GENI_S(vec,preg,areg,avec)\
    " st1w  {" #vec ".s}, " #preg ", [" #areg "," #avec ".s, UXTW #2]\n\t"

// Store 2 vectors to contiguous memory
#define STOR2VEC_S(vec1,vec2,preg,areg)\
    " st1w  {" #vec1 ".s}, " #preg ", [" #areg "]           \n\t"\
    " st1w  {" #vec2 ".s}, " #preg ", [" #areg ",#1, MUL VL]\n\t"

#define STOR2VEC_CONT_S(vec1,vec2,preg,areg,avec1,avec2) STOR2VEC_S(vec1,vec2,preg,areg)

// Store 2 vectors with generic indexing (scatter-store)
#define STOR2VEC_GENI_S(vec1,vec2,preg,areg,avec1,avec2)\
    " st1w  {" #vec1 ".s}, " #preg ", [" #areg "," #avec1 ".s, UXTW #2]\n\t"\
    " st1w  {" #vec2 ".s}, " #preg ", [" #areg "," #avec2 ".s, UXTW #2]\n\t"

#define STOR2VEC_CONT_C(vec1,vec2,preg,areg,avec1,avec2) STORC2VEC_S(vec1,vec2,preg,areg)

// Store 2 vectors with generic indexing (scatter-store)
#define STOR2VEC_GENI_C(vec1,vec2,preg,areg,avec1,avec2)\
    " st1w  {" #vec1 ".s}, " #preg ", [" #areg "," #avec1 ".s, UXTW #2]\n\t"\
    " st1w  {" #vec2 ".s}, " #preg ", [" #areg "," #avec2 ".s, UXTW #2]\n\t"


// Some macros used for fixed size kernels

#if defined(USE_SVE_CMLA_INSTRUCTION)
#define LDR_BVEC_S(vec1, vec2, preg, reg, offset1, offset2)\
    " ld1rd  " #vec1 ".s, " #preg "/z, [" #reg ", #" #offset1 "]   \n\t"
#else
#define LDR_BVEC_S(vec1, vec2, preg, reg, offset1, offset2)\
    " ld1rw  " #vec1 ".s, " #preg "/z, [" #reg ", #" #offset1 "]   \n\t"\
    " ld1rw  " #vec2 ".s, " #preg "/z, [" #reg ", #" #offset2 "]   \n\t"
#endif

#if defined(USE_SVE_CMLA_INSTRUCTION)
#define LD_AVEC_S(vec1, vec2, preg, reg)\
    " ld1w  " #vec1 ".s, " #preg "/z, [" #reg "]           \n\t"\
    " ld1w  " #vec2 ".s, " #preg "/z, [" #reg ",#1, MUL VL]\n\t"
#else
#define LD_AVEC_S(vec1, vec2, preg, reg)\
    " ld2w {" #vec1 ".s," #vec2 ".s}, " #preg "/z, [" #reg "]\n\t"
#endif

#if defined(USE_SVE_CMLA_INSTRUCTION)
#define ST_AVEC_S(vec1, vec2, preg, reg)\
    " st1w  " #vec1 ".s, " #preg ", [" #reg "]           \n\t"\
    " st1w  " #vec2 ".s, " #preg ", [" #reg ",#1, MUL VL]\n\t"
#else
#define ST_AVEC_S(vec1, vec2, preg, reg)\
    " st2w {" #vec1 ".s," #vec2 ".s}, " #preg ", [" #reg "]\n\t"
#endif


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
