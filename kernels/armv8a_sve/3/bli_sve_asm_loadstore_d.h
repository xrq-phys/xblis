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

#define LOAD1VEC(vec,preg,areg)\
    " ld1d  " #vec ".d, " #preg "/z, [" #areg "]           \n\t"

#define LOAD1VEC_CONT(vec,preg,areg,avec) LOAD1VEC(vec,preg,areg)

#define LOAD1VEC_GENI(vec,preg,areg,avec)\
    " ld1d  " #vec ".d, " #preg "/z, [" #areg "," #avec ".d, LSL #3]\n\t"

#define LOAD2VEC(vec1,vec2,preg,areg)\
    " ld1d  " #vec1 ".d, " #preg "/z, [" #areg "]           \n\t"\
    " ld1d  " #vec2 ".d, " #preg "/z, [" #areg ",#1, MUL VL]\n\t"

#define LOAD4VEC(vec1, vec2, vec3, vec4, preg,areg)\
    " ld1d  " #vec1 ".d, " #preg "/z, [" #areg "]           \n\t"\
    " ld1d  " #vec2 ".d, " #preg "/z, [" #areg ",#1, MUL VL]\n\t"\
    " ld1d  " #vec3 ".d, " #preg "/z, [" #areg ",#2, MUL VL]\n\t"\
    " ld1d  " #vec4 ".d, " #preg "/z, [" #areg ",#3, MUL VL]\n\t"

#define LOAD2VEC_CONT(vec1,vec2,preg,areg,avec1,avec2) LOAD2VEC(vec1,vec2,preg,areg)

#define LOAD2VEC_GENI(vec1,vec2,preg,areg,avec1,avec2)\
    " ld1d  " #vec1 ".d, " #preg "/z, [" #areg "," #avec1 ".d, LSL #3]\n\t"\
    " ld1d  " #vec2 ".d, " #preg "/z, [" #areg "," #avec2 ".d, LSL #3]\n\t"

#define LOAD4VEC_GENI(vec1,vec2,vec3,vec4, preg,areg, avec1,avec2,avec3,avec4)\
    " ld1d  " #vec1 ".d, " #preg "/z, [" #areg "," #avec1 ".d, LSL #3]\n\t"\
    " ld1d  " #vec2 ".d, " #preg "/z, [" #areg "," #avec2 ".d, LSL #3]\n\t"\
    " ld1d  " #vec3 ".d, " #preg "/z, [" #areg "," #avec3 ".d, LSL #3]\n\t"\
    " ld1d  " #vec4 ".d, " #preg "/z, [" #areg "," #avec4 ".d, LSL #3]\n\t"

#define LDR_NOADDR(vec1,preg)\
    " ld1rd  " #vec1 ".d, " #preg "/z"
#define OA(areg,offset)\
    ",[" #areg ", #" #offset"]"

#define LOADVEC_DIST(vec1,preg,areg)\
    LDR_NOADDR(vec1,preg)OA(areg,0)"\n\t"

#define LOAD2VEC_DIST(vec1,vec2,preg,areg)\
    LDR_NOADDR(vec1,preg)OA(areg,0)"\n\t"\
    LDR_NOADDR(vec2,preg)OA(areg,8)"\n\t"

#define LOAD4VEC_DIST(vec1,vec2,vec3,vec4,preg,areg)\
    LDR_NOADDR(vec1,preg)OA(areg,0)"\n\t"\
    LDR_NOADDR(vec2,preg)OA(areg,8)"\n\t"\
    LDR_NOADDR(vec3,preg)OA(areg,16)"\n\t"\
    LDR_NOADDR(vec4,preg)OA(areg,24)"\n\t"

#define LOAD8VEC_DIST(vec1,vec2,vec3,vec4,vec5,vec6,vec7,vec8,preg,areg)\
    LDR_NOADDR(vec1,preg)OA(areg,0)"\n\t"\
    LDR_NOADDR(vec2,preg)OA(areg,8)"\n\t"\
    LDR_NOADDR(vec3,preg)OA(areg,16)"\n\t"\
    LDR_NOADDR(vec4,preg)OA(areg,24)"\n\t"\
    LDR_NOADDR(vec5,preg)OA(areg,32)"\n\t"\
    LDR_NOADDR(vec6,preg)OA(areg,40)"\n\t"\
    LDR_NOADDR(vec7,preg)OA(areg,48)"\n\t"\
    LDR_NOADDR(vec8,preg)OA(areg,56)"\n\t"


#if defined(USE_SVE_CMLA_INSTRUCTION)
// When using the fused complex mupliply-accumulate
// load {real, imag} into every 128 bit part of the vector

// Load 2 vectors worth of complex numbers
#define LOADC2VEC(vec1, vec2, preg, areg) LOAD2VEC(vec1, vec2, preg, areg)

#define LOADC2VEC_VOFF(vec1, vec2, preg, areg, off1, off2)\
    " ld1d  " #vec1 ".d, " #preg "/z, [" #areg ",#"#off1",   MUL VL]           \n\t"\
    " ld1d  " #vec2 ".d, " #preg "/z, [" #areg ",#"#off2", MUL VL]\n\t"

#define LDC2_DIST_NOADDR(vec1, vec2, preg)\
    " ld1rqd {" #vec1 ".d}, " #preg "/z"
#define LOADC2VEC_DIST_OFF(vec1, vec2, preg, areg, off1, off2)\
    LDC2_DIST_NOADDR(vec1, vec2, preg)OA(areg,off1)"\n\t"

// Load 2 vectors and distribute the first complex number to all 128 parts
#define LOADC2VEC_DIST(vec1, vec2, preg, areg) LOADC2VEC_DIST_OFF(vec1, vec2, preg, areg, 0)

#define LOADC4VEC_DIST_OFF(vec1, vec2, vec3, vec4, preg, areg, off1, off2, off3, off4)\
    LOADC2VEC_DIST_OFF(vec1, vec2, preg, areg, off1, off2)\
    LOADC2VEC_DIST_OFF(vec3, vec4, preg, areg, off3, off4)

#define LOADC4VEC_DIST(vec1, vec2, vec3, vec4, preg, areg)\
    LOADC4VEC_DIST_OFF(vec1, vec2, vec3, vec4, preg, areg, 0, 8, 16, 24)

#define LOADC8VEC_DIST(vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8, preg, areg)\
    LOADC4VEC_DIST_OFF(vec1, vec2, vec3, vec4, preg, areg, 0, 8, 16, 24)\
    LOADC4VEC_DIST_OFF(vec5, vec6, vec7, vec8, preg, areg, 32, 40, 48, 56)

// Make an index for gather-load/scatter-store
// Here it has to be
// stride = estride*sizeof(double complex)
// ivec1 = {0,8,stride, stride+8, ... }
// ivec2 = {nelem*stride, nelem*stride+8, (nelem+1)*stride, (nelem+1)*stride+8, ... }
#define MKINDC_2VEC(ivec1, ivec2, stridereg, nelemreg, sparereg)\
    " lsl " #sparereg", "#stridereg", #1\n\t"\
    " index "#ivec1".d,xzr," #sparereg "\n\t"\
    " mov "  #ivec2".d," #ivec1".d\n\t"\
    " add "  #ivec2".d," #ivec2".d,#1\n\t"\
    " zip1 " #ivec1".d," #ivec1".d," #ivec2".d\n\t"\
    " mul "  #sparereg", "#sparereg", "#nelemreg"\n\t"\
    " lsr " #sparereg", " #sparereg", #1\n\t"\
    " dup "  #ivec2".d," #sparereg"\n\t"\
    " add "  #ivec2".d," #ivec2 ".d," #ivec1".d\n\t"

// Load 2 vectors worth of complex numbers
#define STORC2VEC(vec1, vec2, preg, areg) STOR2VEC(vec1, vec2, preg, areg)

#else
// Otherwise
// load {real} {imag} into separate vectors

#define LOADC2VEC(vec1, vec2, preg, areg)\
    " ld2d {" #vec1 ".d," #vec2 ".d}, " #preg "/z, [" #areg "]\n\t"

#define LOADC2VEC_VOFF(vec1, vec2, preg, areg, off1, off2)\
    " ld2d {" #vec1 ".d," #vec2 ".d}, " #preg "/z, [" #areg ",#"#off1", MUL VL]\n\t"

#define LOADC2VEC_DIST_OFF(vec_r, vec_i, preg,areg,off1,off2)\
    LDR_NOADDR(vec_r,preg)OA(areg,off1)"\n\t"\
    LDR_NOADDR(vec_i,preg)OA(areg,off2)"\n\t"

#define LOADC2VEC_DIST(vec_r, vec_i, preg, areg) LOAD2VEC_DIST(vec_r, vec_i, preg, areg)
#define LOADC4VEC_DIST(vec1_r, vec1_i, vec2_r, vec2_i, preg, areg)\
    LOAD4VEC_DIST(vec1_r, vec1_i, vec2_r, vec2_i, preg, areg)
#define LOADC8VEC_DIST(vec1_r, vec1_i, vec2_r, vec2_i, vec3_r, vec3_i, vec4_r, vec4_i, preg, areg)\
    LOAD8VEC_DIST(vec1_r, vec1_i, vec2_r, vec2_i, vec3_r, vec3_i, vec4_r, vec4_i, preg, areg)

// Make an index for gather-load/scatter-store
// Here it has to be
// stride = estride*sizeof(double complex)
// ivec1 = {0,stride,  2*stride,    ... }
// ivec2 = {8,stride+8,2*stride+8,  ... }
#define MKINDC_2VEC(ivec1, ivec2, stridereg, nelemreg, sparereg)\
    " lsl " #sparereg", "#stridereg", #1\n\t"\
    " index "#ivec1".d,xzr," #sparereg "\n\t"\
    " mov "#ivec2".d, "#ivec1".d\n\t"\
    " add "#ivec2".d, "#ivec2".d,#1\n\t"

#define STORC2VEC(vec1, vec2, preg, areg)\
    " st2d {" #vec1 ".d," #vec2 ".d}, " #preg ", [" #areg "]\n\t"

#endif

#define LOADC2VEC_CONT(vec1,vec2,preg,areg,avec1,avec2) LOADC2VEC(vec1,vec2,preg,areg)

#define LOADC2VEC_GENI(vec1,vec2,preg,areg,avec1,avec2)\
    " ld1d  " #vec1 ".d, " #preg "/z, [" #areg "," #avec1 ".d, LSL #3]\n\t"\
    " ld1d  " #vec2 ".d, " #preg "/z, [" #areg "," #avec2 ".d, LSL #3]\n\t"

// Store 1 vector to contiguous memory
#define STOR1VEC(vec,preg,areg)\
    " st1d  {" #vec ".d}, " #preg ", [" #areg "]           \n\t"

#define STOR1VEC_CONT(vec,preg,areg,avec) STOR1VEC(vec,preg,areg)

// Store 1 vector with generic indexing (scatter-store)
#define STOR1VEC_GENI(vec,preg,areg,avec)\
    " st1d  {" #vec ".d}, " #preg ", [" #areg "," #avec ".d, LSL #3]\n\t"

// Store 2 vectors to contiguous memory
#define STOR2VEC(vec1,vec2,preg,areg)\
    " st1d  {" #vec1 ".d}, " #preg ", [" #areg "]           \n\t"\
    " st1d  {" #vec2 ".d}, " #preg ", [" #areg ",#1, MUL VL]\n\t"

#define STOR4VEC(vec1,vec2,vec3,vec4,preg,areg)\
    " st1d  {" #vec1 ".d}, " #preg ", [" #areg "]           \n\t"\
    " st1d  {" #vec2 ".d}, " #preg ", [" #areg ",#1, MUL VL]\n\t"\
    " st1d  {" #vec3 ".d}, " #preg ", [" #areg ",#2, MUL VL]\n\t"\
    " st1d  {" #vec4 ".d}, " #preg ", [" #areg ",#3, MUL VL]\n\t"

#define STOR2VEC_CONT(vec1,vec2,preg,areg,avec1,avec2) STOR2VEC(vec1,vec2,preg,areg)

#define STOR4VEC_CONT(vec1,vec2,vec3,vec4,preg,areg,avec1,avec2) STOR4VEC(vec1,vec2,vec3,vec4,preg,areg)

// Store 2 vectors with generic indexing (scatter-store)
#define STOR2VEC_GENI(vec1,vec2,preg,areg,avec1,avec2)\
    " st1d  {" #vec1 ".d}, " #preg ", [" #areg "," #avec1 ".d, LSL #3]\n\t"\
    " st1d  {" #vec2 ".d}, " #preg ", [" #areg "," #avec2 ".d, LSL #3]\n\t"

#define STOR4VEC_GENI(vec1,vec2,vec3,vec4, preg,areg,avec1,avec2,avec3,avec4)\
    " st1d  {" #vec1 ".d}, " #preg ", [" #areg "," #avec1 ".d, LSL #3]\n\t"\
    " st1d  {" #vec2 ".d}, " #preg ", [" #areg "," #avec2 ".d, LSL #3]\n\t"\
    " st1d  {" #vec3 ".d}, " #preg ", [" #areg "," #avec3 ".d, LSL #3]\n\t"\
    " st1d  {" #vec4 ".d}, " #preg ", [" #areg "," #avec4 ".d, LSL #3]\n\t"

#define STORC2VEC_CONT(vec1,vec2,preg,areg,avec1,avec2) STORC2VEC(vec1,vec2,preg,areg)

// Store 2 vectors with generic indexing (scatter-store)
#define STORC2VEC_GENI(vec1,vec2,preg,areg,avec1,avec2)\
    " st1d  {" #vec1 ".d}, " #preg ", [" #areg "," #avec1 ".d, LSL #3]\n\t"\
    " st1d  {" #vec2 ".d}, " #preg ", [" #areg "," #avec2 ".d, LSL #3]\n\t"


// Some macros used for fixed size kernels

#if defined(USE_SVE_CMLA_INSTRUCTION)
#define LDR_BVEC(vec1, vec2, preg, reg, offset1, offset2)\
    " ld1rqd  " #vec1 ".d, " #preg "/z, [" #reg ", #" #offset1 "]   \n\t"
#else
#define LDR_BVEC(vec1, vec2, preg, reg, offset1, offset2)\
    " ld1rd  " #vec1 ".d, " #preg "/z, [" #reg ", #" #offset1 "]   \n\t"\
    " ld1rd  " #vec2 ".d, " #preg "/z, [" #reg ", #" #offset2 "]   \n\t"
#endif

#if defined(USE_SVE_CMLA_INSTRUCTION)
#define LD_AVEC(vec1, vec2, preg, reg)\
    " ld1d  " #vec1 ".d, " #preg "/z, [" #reg "]           \n\t"\
    " ld1d  " #vec2 ".d, " #preg "/z, [" #reg ",#1, MUL VL]\n\t"
#else
#define LD_AVEC(vec1, vec2, preg, reg)\
    " ld2d {" #vec1 ".d," #vec2 ".d}, " #preg "/z, [" #reg "]\n\t"
#endif

#if defined(USE_SVE_CMLA_INSTRUCTION)
#define ST_AVEC(vec1, vec2, preg, reg)\
    " st1d  " #vec1 ".d, " #preg ", [" #reg "]           \n\t"\
    " st1d  " #vec2 ".d, " #preg ", [" #reg ",#1, MUL VL]\n\t"
#else
#define ST_AVEC(vec1, vec2, preg, reg)\
    " st2d {" #vec1 ".d," #vec2 ".d}, " #preg ", [" #reg "]\n\t"
#endif
