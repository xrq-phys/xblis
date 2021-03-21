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

//#define USE_SVE_CMLA_INSTRUCTION


#if defined(DEBUG)
#include "bli_sve_asm_debug.h"
#endif
#include "bli_sve_asm_loadstore_s.h"

#define COMBINE2(a,b) a ## _ ## b

#define ZEROVEC(vec1)\
    " dup " #vec1 ".s, #0\n\t"

#define ZERO2VEC_S(vec1,vec2)\
    ZEROVEC(vec1)\
    ZEROVEC(vec2)

#define ZERO4VEC_S(vec1,vec2,vec3,vec4)\
    ZERO2VEC_S(vec1,vec2)\
    ZERO2VEC_S(vec3,vec4)

#define ZERO8VEC_S(vec1,vec2,vec3,vec4,vec5,vec6,vec7,vec8)\
    ZERO4VEC_S(vec1,vec2,vec3,vec4)\
    ZERO4VEC_S(vec5,vec6,vec7,vec8)

#define MLA1ROW_S(cvec, avec, bvec, preg)\
    " fmla " #cvec ".s, " #preg "/m, " #avec ".s, " #bvec ".s\n\t"

#define MLA2ROW_S(c1, c2 , a1, a2, bvec, preg)\
    MLA1ROW_S(c1,a1,bvec,preg)\
    MLA1ROW_S(c2,a2,bvec,preg)

#define MLA4ROW_S(c1, c2, c3, c4, a1, a2, a3, a4, bvec, preg)\
    MLA2ROW_S(c1,c2,a1,a2,bvec,preg)\
    MLA2ROW_S(c3,c4,a3,a4,bvec,preg)


#define MUL1ROW_S(c1, a1, bvec, preg)\
    " fmul " #c1 ".s, " #preg "/m, " #a1 ".s, " #bvec ".s\n\t"

#define MUL2ROW_S(c1, c2 , a1, a2, bvec, preg)\
    MUL1ROW_S(c1,a1,bvec,preg)\
    MUL1ROW_S(c2,a2,bvec,preg)

#define MUL4ROW_S(c1, c2, c3, c4, a1, a2, a3, a4, bvec, preg)\
    MUL2ROW_S(c1,c2,a1,a2,bvec,preg)\
    MUL2ROW_S(c3,c4,a3,a4,bvec,preg)

#define MLA2X2ROW_S(c11, c12, c21, c22, a1, a2, bvec1,bvec2, preg)\
    MLA2ROW_S(c11, c12, a1, a2, bvec1, preg)\
    MLA2ROW_S(c21, c22, a1, a2, bvec2, preg)

#define MLA1ROW_LA_LB_S(cvec, avec, bvec, preg, nextavec, aareg, avoff, bareg, bboff)\
    MLA1ROW_S(cvec, avec, bvec, preg)\
    " ld1w   " #nextavec ".s, " #preg "/z, [" #aareg ", #" #avoff", MUL VL]\n\t"\
    " ld1rw  " #bvec ".s, "#preg"/z, [" #bareg",#" #bboff "]\n\t"

#define MLA2ROW_LA_LB_S(c1, c2 , a1, a2, bvec, preg, nextavec, aareg, avoff, bareg, bboff)\
    MLA2ROW_S(c1, c2, a1, a2, bvec, preg)\
    " ld1w   " #nextavec ".s, " #preg "/z, [" #aareg ", #" #avoff", MUL VL]\n\t"\
    " ld1rw  " #bvec ".s, "#preg"/z, [" #bareg",#" #bboff "]\n\t"

#define MLA2X2ROW_LA_LB_S(c11,c12,c21,c22, a1, a2, bvec1,bvec2, preg, nextavec, aareg, avoff, bareg, bboff1,bboff2)\
    MLA2ROW_S(c11, c12, a1, a2, bvec1, preg)\
    MLA2ROW_S(c21, c22, a1, a2, bvec2, preg)\
    " ld1w   " #nextavec ".s, " #preg "/z, [" #aareg ", #" #avoff", MUL VL]\n\t"\
    " ld1rw  " #bvec1 ".s, "#preg"/z, [" #bareg",#" #bboff1 "]\n\t"\
    " ld1rw  " #bvec2 ".s, "#preg"/z, [" #bareg",#" #bboff2 "]\n\t"

#define MLA4ROW_LA_LB_S(c1, c2, c3, c4, a1, a2, a3, a4, bvec, preg, nextavec, aareg, avoff, bareg, bboff)\
    MLA4ROW_S(c1, c2, c3, c4, a1, a2, a3, a4, bvec, preg)\
    " ld1w   " #nextavec ".s, " #preg "/z, [" #aareg ", #" #avoff", MUL VL]\n\t"\
    " ld1rw  " #bvec ".s, "#preg"/z, [" #bareg",#" #bboff "]\n\t"


#define MLA2X2ROW_LB_S(c11, c12, c21, c22, a1, a2, bvec1,bvec2, preg, bareg, bboff1,bboff2)\
    MLA2ROW_S(c11, c12, a1, a2, bvec1, preg)\
    MLA2ROW_S(c21, c22, a1, a2, bvec2, preg)\
    " ld1rw  " #bvec1 ".s, "#preg"/z, [" #bareg",#" #bboff1 "]\n\t"\
    " ld1rw  " #bvec2 ".s, "#preg"/z, [" #bareg",#" #bboff2 "]\n\t"

#define MLA1ROW_LB_S(cvec, avec, bvec, preg,  bareg, bboff)\
    MLA1ROW_S(cvec, avec, bvec, preg)\
    " ld1rw  " #bvec ".s, "#preg"/z, [" #bareg",#" #bboff "]\n\t"

#define MLA2ROW_LB_S(c1, c2 , a1, a2, bvec, preg,  bareg, bboff)\
    MLA2ROW_S(c1, c2, a1, a2, bvec, preg)\
    " ld1rw  " #bvec ".s, "#preg"/z, [" #bareg",#" #bboff "]\n\t"

#define MLA4ROW_LB_S(c1, c2, c3, c4, a1, a2, a3, a4, bvec, preg,  bareg, bboff)\
    MLA4ROW_S(c1, c2, c3, c4, a1, a2, a3, a4, bvec, preg)\
    " ld1rw  " #bvec ".s, "#preg"/z, [" #bareg",#" #bboff "]\n\t"

#define ZEROVEC_C(vec1) ZEROVEC_S(vec1)

#define ZERO2VEC_C(vec1,vec2) ZERO2VEC_S(vec1,vec2)

#define ZERO4VEC_C(vec1,vec2,vec3,vec4) ZERO4VEC_S(vec1,vec2,vec3,vec4)

#define ZERO8VEC_C(vec1,vec2,vec3,vec4,vec5,vec6,vec7,vec8)\
    ZERO8VEC_S(vec1,vec2,vec3,vec4,vec5,vec6,vec7,vec8)

#if defined(USE_SVE_CMLA_INSTRUCTION)
    #define MLA1ROW_ILV_C(cvec1, cvec2, avec1, avec2, bvec1, bvec2, preg, ilv1, ilv2, ilv3, ilv4)\
        " fcmla " #cvec1 ".s, " #preg "/m, " #avec1 ".s, " #bvec1 ".s, #0\n\t"\
        ilv1\
        " fcmla " #cvec2 ".s, " #preg "/m, " #avec2 ".s, " #bvec1 ".s, #0\n\t"\
        ilv2\
        " fcmla " #cvec1 ".s, " #preg "/m, " #avec1 ".s, " #bvec1 ".s, #90\n\t"\
        ilv3\
        " fcmla " #cvec2 ".s, " #preg "/m, " #avec2 ".s, " #bvec1 ".s, #90\n\t"\
        ilv4\
        "\n\t"
#else
    #define MLA1ROW_ILV_C(cvec_r, cvec_i, avec_r, avec_i, bvec_r, bvec_i, preg, ilv1, ilv2, ilv3, ilv4)\
        " fmla " #cvec_r ".s, " #preg "/m, " #avec_r ".s, " #bvec_r ".s\n\t"\
        ilv1\
        " fmla " #cvec_i ".s, " #preg "/m, " #avec_r ".s, " #bvec_i ".s\n\t"\
        ilv2\
        " fmls " #cvec_r ".s, " #preg "/m, " #avec_i ".s, " #bvec_i ".s\n\t"\
        ilv3\
        " fmla " #cvec_i ".s, " #preg "/m, " #avec_i ".s, " #bvec_r ".s\n\t"\
        ilv4\
        "\n\t"
#endif

#define MLA1ROW_C(cvec1, cvec2, avec1, avec2, bvec1, bvec2, preg)\
        MLA1ROW_ILV_C(cvec1, cvec2, avec1, avec2, bvec1, bvec2, preg,"","","","")

#define MLA1ROW_ILV_LA_LB_C(c1, c2, a1, a2, bvec1, bvec2, preg, ilv1, ilv2, ilv3, ilv4, nextavec1, nextavec2, aareg, avoff1, avoff2, bareg, bboff1, bboff2)\
    MLA1ROW_ILV_C(c1, c2, a1, a2, bvec1, bvec2, preg, ilv1, ilv2, ilv3, ilv4 )\
    LOAD2VEC_VOFF_C(nextavec1, nextavec2, preg, aareg, avoff1, avoff2)\
    LOAD2VEC_DIST_OFF_C(bvec1, bvec2, preg, bareg, bboff1,bboff2)

#define MLA1ROW_ILV_LB_C(c1, c2, a1, a2, bvec1, bvec2, preg, ilv1, ilv2, ilv3, ilv4, bareg, bboff1, bboff2)\
    MLA1ROW_ILV_C(c1, c2, a1, a2, bvec1, bvec2, preg, ilv1, ilv2, ilv3, ilv4)\
    LOAD2VEC_DIST_OFF_C(bvec1, bvec2, preg, bareg, bboff1, bboff2)

#define MLA1ROW_LA_LB_C(c1, c2, a1, a2, bvec1, bvec2, preg, nextavec1, nextavec2, aareg, avoff1, avoff2, bareg, bboff1, bboff2)\
    MLA1ROW_ILV_LA_LB_C(c1, c2, a1, a2, bvec1, bvec2, preg, "", "", "", "", nextavec1, nextavec2, aareg, avoff1, avoff2, bareg, bboff1, bboff2)

#define MLA1ROW_LB_C(c1, c2, a1, a2, bvec1, bvec2, preg, bareg, bboff1, bboff2)\
    MLA1ROW_ILV_LB_C(c1, c2, a1, a2, bvec1, bvec2, preg, "", "", "", "", bareg, bboff1, bboff2)

#define MLA2ROW_C(cvec1, cvec2, cvec3, cvec4, avec1, avec2, avec3, avec4, bvec1, bvec2, preg)\
    MLA1ROW_C(cvec1, cvec2, avec1, avec2, bvec1, bvec2, preg)\
    MLA1ROW_C(cvec3, cvec4, avec3, avec4, bvec1, bvec2, preg)

#define PFL1(areg,preg,offset) " prfd pldl1keep, "#preg", [" #areg ", #" #offset ", MUL VL]\n\t"


// Check if a complex number is 0
// TODO: Only first complex number needs to be checked, 
#if defined(USE_SVE_CMLA_INSTRUCTION)
#define CMPCZB_S(vec1,vec2,label)\
" fcmeq p1.s, p0/z, " #vec1 ".s, #0.0\n\t"\
" nots p1.b, p0/z, p1.b\n\t"\
" b.none " label "\n\t"
#else
#define CMPCZB_S(vec1,vec2,label)\
" fcmeq p1.s, p0/z, " #vec1 ".s, #0.0\n\t"\
" fcmeq p2.s, p0/z, " #vec2 ".s, #0.0\n\t"\
" ands p1.b, p0/z, p1.b, p2.b\n\t"\
" b.any " label "\n\t"
#endif
