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

//#define USE_SVE_CMLA_INSTRUCTION


#if defined(DEBUG)
#include "bli_sve_asm_debug.h"
#endif
#include "bli_sve_asm_loadstore_d.h"


#define ZEROVEC_D(vec1)\
    " dup " #vec1 ".d, #0\n\t"

#define ZERO2VEC_D(vec1,vec2)\
    ZEROVEC_D(vec1)\
    ZEROVEC_D(vec2)

#define ZERO4VEC_D(vec1,vec2,vec3,vec4)\
    ZERO2VEC_D(vec1,vec2)\
    ZERO2VEC_D(vec3,vec4)

#define ZERO8VEC_D(vec1,vec2,vec3,vec4,vec5,vec6,vec7,vec8)\
    ZERO4VEC_D(vec1,vec2,vec3,vec4)\
    ZERO4VEC_D(vec5,vec6,vec7,vec8)

#define MLA1COL_D(cvec, avec, bvec, preg)\
    " fmla " #cvec ".d, " #preg "/m, " #avec ".d, " #bvec ".d\n\t"

#define MLA2COL_D(cvec1,cvec2, avec, bvec1,bvec2, preg)\
    MLA1COL_D(cvec1, avec, bvec1, preg)\
    MLA1COL_D(cvec2, avec, bvec2, preg)

#define MLA4COL_D(cvec1,cvec2,cvec3,cvec4, avec, bvec1,bvec2,bvec3,bvec4, preg)\
    MLA2COL_D(cvec1,cvec2, avec, bvec1,bvec2, preg)\
    MLA2COL_D(cvec3,cvec4, avec, bvec3,bvec4, preg)

#define MLA5COL_D(cvec1,cvec2,cvec3,cvec4,cvec5, avec, bvec1,bvec2,bvec3,bvec4,bvec5, preg)\
    MLA4COL_D(cvec1,cvec2,cvec3,cvec4, avec, bvec1,bvec2,bvec3,bvec4, preg)\
    MLA1COL_D(cvec5, avec, bvec5, preg)


#define MLA1ROW_D(cvec, avec, bvec, preg)\
    " fmla " #cvec ".d, " #preg "/m, " #avec ".d, " #bvec ".d\n\t"

#define MLA2ROW_D(c1, c2 , a1, a2, bvec, preg)\
    MLA1ROW_D(c1,a1,bvec,preg)\
    MLA1ROW_D(c2,a2,bvec,preg)

#define MLA4ROW_D(c1, c2, c3, c4, a1, a2, a3, a4, bvec, preg)\
    MLA2ROW_D(c1,c2,a1,a2,bvec,preg)\
    MLA2ROW_D(c3,c4,a3,a4,bvec,preg)

#define MLA1ROW_I_D(cvec, avec, bvec, ind, preg)\
    " fmla " #cvec ".d, " #avec ".d, " #bvec ".d[" #ind "]\n\t"

#define MLA2ROW_I_D(c1, c2 , a1, a2, bvec, ind, preg)\
    MLA1ROW_I_D(c1,a1,bvec,ind,preg)\
    MLA1ROW_I_D(c2,a2,bvec,ind,preg)

#define MLA4ROW_I_D(c1,c2,c3,c4, a1,a2,a3,a4, bvec, ind, preg)\
    MLA2ROW_I_D(c1,c2, a1,a2, bvec,ind,preg)\
    MLA2ROW_I_D(c3,c4, a3,a4, bvec,ind,preg)


#define MUL1ROW_D(c1, a1, bvec, preg)\
    " fmul " #c1 ".d, " #preg "/m, " #a1 ".d, " #bvec ".d\n\t"

#define MUL2ROW_D(c1, c2 , a1, a2, bvec, preg)\
    MUL1ROW_D(c1,a1,bvec,preg)\
    MUL1ROW_D(c2,a2,bvec,preg)

#define MUL4ROW_D(c1, c2, c3, c4, a1, a2, a3, a4, bvec, preg)\
    MUL2ROW_D(c1,c2,a1,a2,bvec,preg)\
    MUL2ROW_D(c3,c4,a3,a4,bvec,preg)

#define MUL1ROW_I_D(c1, a1, bvec, ind, preg)\
    " fmul " #c1 ".d, " #a1 ".d, " #bvec ".d["#ind"]\n\t"

#define MUL2ROW_I_D(c1, c2 , a1, a2, bvec, ind, preg)\
    MUL1ROW_I_D(c1,a1,bvec,ind,preg)\
    MUL1ROW_I_D(c2,a2,bvec,ind,preg)

#define MUL4ROW_I_D(c1,c2,c3,c4, a1,a2,a3,a4, bvec, ind, preg)\
    MUL2ROW_I_D(c1,c2, a1,a2, bvec,ind,preg)\
    MUL2ROW_I_D(c3,c4, a3,a4, bvec,ind,preg)

#define MLA2X2ROW_D(c11, c12, c21, c22, a1, a2, bvec1,bvec2, preg)\
    MLA2ROW_D(c11, c12, a1, a2, bvec1, preg)\
    MLA2ROW_D(c21, c22, a1, a2, bvec2, preg)

#define MLA2ROW_I_LA_D(c1, c2 , a1, a2, bvec, ind, preg, nextavec, aareg, avoff)\
    MLA2ROW_I_D(c1, c2, a1, a2, bvec, ind, preg)\
    " ld1d   " #nextavec ".d, " #preg "/z, [" #aareg ", #" #avoff", MUL VL]\n\t"

#define MLA2ROW_I_LA_LB_D(c1, c2 , a1, a2, bvec, ind, preg, nextavec, aareg, avoff, bareg, bboff)\
    MLA2ROW_I_D(c1, c2, a1, a2, bvec, ind, preg)\
    " ld1d   " #nextavec ".d, " #preg "/z, [" #aareg ", #" #avoff", MUL VL]\n\t"\
    LOADVEC_QDIST_D(bvec,preg,bareg)

#define MLA1ROW_LA_LB_D(cvec, avec, bvec, preg, nextavec, aareg, avoff, bareg, bboff)\
    MLA1ROW_D(cvec, avec, bvec, preg)\
    " ld1d   " #nextavec ".d, " #preg "/z, [" #aareg ", #" #avoff", MUL VL]\n\t"\
    " ld1rd  " #bvec ".d, "#preg"/z, [" #bareg",#" #bboff "]\n\t"

#define MLA2ROW_LA_LB_D(c1, c2 , a1, a2, bvec, preg, nextavec, aareg, avoff, bareg, bboff)\
    MLA2ROW_D(c1, c2, a1, a2, bvec, preg)\
    " ld1d   " #nextavec ".d, " #preg "/z, [" #aareg ", #" #avoff", MUL VL]\n\t"\
    " ld1rd  " #bvec ".d, "#preg"/z, [" #bareg",#" #bboff "]\n\t"


#define MLA2X2ROW_LA_LB_D(c11,c12,c21,c22, a1, a2, bvec1,bvec2, preg, nextavec, aareg, avoff, bareg, bboff1,bboff2)\
    MLA2ROW_D(c11, c12, a1, a2, bvec1, preg)\
    MLA2ROW_D(c21, c22, a1, a2, bvec2, preg)\
    " ld1d   " #nextavec ".d, " #preg "/z, [" #aareg ", #" #avoff", MUL VL]\n\t"\
    " ld1rd  " #bvec1 ".d, "#preg"/z, [" #bareg",#" #bboff1 "]\n\t"\
    " ld1rd  " #bvec2 ".d, "#preg"/z, [" #bareg",#" #bboff2 "]\n\t"

#define MLA4ROW_LA_LB_D(c1, c2, c3, c4, a1, a2, a3, a4, bvec, preg, nextavec, aareg, avoff, bareg, bboff)\
    MLA4ROW_D(c1, c2, c3, c4, a1, a2, a3, a4, bvec, preg)\
    " ld1d   " #nextavec ".d, " #preg "/z, [" #aareg ", #" #avoff", MUL VL]\n\t"\
    " ld1rd  " #bvec ".d, "#preg"/z, [" #bareg",#" #bboff "]\n\t"


#define MLA2X2ROW_LB_D(c11, c12, c21, c22, a1, a2, bvec1,bvec2, preg, bareg, bboff1,bboff2)\
    MLA2ROW_D(c11, c12, a1, a2, bvec1, preg)\
    MLA2ROW_D(c21, c22, a1, a2, bvec2, preg)\
    " ld1rd  " #bvec1 ".d, "#preg"/z, [" #bareg",#" #bboff1 "]\n\t"\
    " ld1rd  " #bvec2 ".d, "#preg"/z, [" #bareg",#" #bboff2 "]\n\t"

#define MLA1ROW_LB_D(cvec, avec, bvec, preg,  bareg, bboff)\
    MLA1ROW_D(cvec, avec, bvec, preg)\
    " ld1rd  " #bvec ".d, "#preg"/z, [" #bareg",#" #bboff "]\n\t"

#define MLA2ROW_LB_D(c1, c2 , a1, a2, bvec, preg,  bareg, bboff)\
    MLA2ROW_D(c1, c2, a1, a2, bvec, preg)\
    " ld1rd  " #bvec ".d, "#preg"/z, [" #bareg",#" #bboff "]\n\t"

#define MLA4ROW_LB_D(c1, c2, c3, c4, a1, a2, a3, a4, bvec, preg,  bareg, bboff)\
    MLA4ROW_D(c1, c2, c3, c4, a1, a2, a3, a4, bvec, preg)\
    " ld1rd  " #bvec ".d, "#preg"/z, [" #bareg",#" #bboff "]\n\t"



#define ZEROVEC_Z(vec1) ZEROVEC_D(vec1)

#define ZERO2VEC_Z(vec1,vec2) ZERO2VEC_D(vec1,vec2)

#define ZERO4VEC_Z(vec1,vec2,vec3,vec4) ZERO4VEC_D(vec1,vec2,vec3,vec4)

#define ZERO8VEC_Z(vec1,vec2,vec3,vec4,vec5,vec6,vec7,vec8)\
    ZERO8VEC_D(vec1,vec2,vec3,vec4,vec5,vec6,vec7,vec8)


#if defined(USE_SVE_CMLA_INSTRUCTION)
    #define MLA1ROW_ILV_Z(cvec1, cvec2, avec1, avec2, bvec1, bvec2, preg, ilv1, ilv2, ilv3, ilv4)\
        " fcmla " #cvec1 ".d, " #preg "/m, " #avec1 ".d, " #bvec1 ".d, #0\n\t"\
        ilv1\
        " fcmla " #cvec2 ".d, " #preg "/m, " #avec2 ".d, " #bvec1 ".d, #0\n\t"\
        ilv2\
        " fcmla " #cvec1 ".d, " #preg "/m, " #avec1 ".d, " #bvec1 ".d, #90\n\t"\
        ilv3\
        " fcmla " #cvec2 ".d, " #preg "/m, " #avec2 ".d, " #bvec1 ".d, #90\n\t"\
        ilv4\
        "\n\t"
#else
    #define MLA1ROW_ILV_Z(cvec_r, cvec_i, avec_r, avec_i, bvec_r, bvec_i, preg, ilv1, ilv2, ilv3, ilv4)\
        " fmla " #cvec_r ".d, " #preg "/m, " #avec_r ".d, " #bvec_r ".d\n\t"\
        ilv1\
        " fmla " #cvec_i ".d, " #preg "/m, " #avec_r ".d, " #bvec_i ".d\n\t"\
        ilv2\
        " fmls " #cvec_r ".d, " #preg "/m, " #avec_i ".d, " #bvec_i ".d\n\t"\
        ilv3\
        " fmla " #cvec_i ".d, " #preg "/m, " #avec_i ".d, " #bvec_r ".d\n\t"\
        ilv4\
        "\n\t"
#endif

#define MLA1ROW_Z(cvec1, cvec2, avec1, avec2, bvec1, bvec2, preg)\
        MLA1ROW_ILV_Z(cvec1, cvec2, avec1, avec2, bvec1, bvec2, preg,"","","","")

#define MLA1ROW_ILV_LA_LB_Z(c1, c2, a1, a2, bvec1, bvec2, preg, ilv1, ilv2, ilv3, ilv4, nextavec1, nextavec2, aareg, avoff1, avoff2, bareg, bboff1, bboff2)\
    MLA1ROW_ILV_Z(c1, c2, a1, a2, bvec1, bvec2, preg, ilv1, ilv2, ilv3, ilv4 )\
    LOAD2VEC_VOFF_Z(nextavec1, nextavec2, preg, aareg, avoff1, avoff2)\
    LOAD2VEC_DIST_OFF_Z(bvec1, bvec2, preg, bareg, bboff1,bboff2)

#define MLA1ROW_ILV_LB_Z(c1, c2, a1, a2, bvec1, bvec2, preg, ilv1, ilv2, ilv3, ilv4, bareg, bboff1, bboff2)\
    MLA1ROW_ILV_Z(c1, c2, a1, a2, bvec1, bvec2, preg, ilv1, ilv2, ilv3, ilv4)\
    LOAD2VEC_DIST_OFF_Z(bvec1, bvec2, preg, bareg, bboff1, bboff2)

#define MLA1ROW_LA_LB_Z(c1, c2, a1, a2, bvec1, bvec2, preg, nextavec1, nextavec2, aareg, avoff1, avoff2, bareg, bboff1, bboff2)\
    MLA1ROW_ILV_LA_LB_Z(c1, c2, a1, a2, bvec1, bvec2, preg, "", "", "", "", nextavec1, nextavec2, aareg, avoff1, avoff2, bareg, bboff1, bboff2)

//#if defined(USE_SVE_CMLA_INSTRUCTION)
#define MLA1ROW_LB_Z(c1, c2, a1, a2, bvec1, bvec2, preg, bareg, bboff1, bboff2)\
    MLA1ROW_ILV_LB_Z(c1, c2, a1, a2, bvec1, bvec2, preg, "", "", "", "", bareg, bboff1, bboff2)
// not faster
/*#else
#define MLA1ROW_LB_Z(c1, c2, a1, a2, bvec1, bvec2, preg, bareg, bboff1, bboff2)\
        " fmla " #c1 ".d, " #preg "/m, " #a1 ".d, " #bvec1 ".d\n\t"\
        " fmla " #c2 ".d, " #preg "/m, " #a2 ".d, " #bvec1 ".d\n\t"\
        LDR_NOADDR_D(bvec1,preg)OA_D(bareg,bboff1)"\n\t"\
        " fmls " #c1 ".d, " #preg "/m, " #a2 ".d, " #bvec2 ".d\n\t"\
        " fmla " #c2 ".d, " #preg "/m, " #a1 ".d, " #bvec2 ".d\n\t"\
        LDR_NOADDR_D(bvec2,preg)OA_D(bareg,bboff2)"\n\t"
#endif*/

#define MLA2ROW_Z(cvec1, cvec2, cvec3, cvec4, avec1, avec2, avec3, avec4, bvec1, bvec2, preg)\
    MLA1ROW_Z(cvec1, cvec2, avec1, avec2, bvec1, bvec2, preg)\
    MLA1ROW_Z(cvec3, cvec4, avec3, avec4, bvec1, bvec2, preg)

#define PFL1(areg,preg,offset) " prfd pldl1keep, "#preg", [" #areg ", #" #offset ", MUL VL]\n\t"


// Check if a complex number is 0
// TODO: Only first complex number needs to be checked, 
#if defined(USE_SVE_CMLA_INSTRUCTION)
#define CMPCZB_D(vec1,vec2,label)\
" fcmeq p1.d, p0/z, " #vec1 ".d, #0.0\n\t"\
" nots p1.b, p0/z, p1.b\n\t"\
" b.none " label "\n\t"
#else
#define CMPCZB_D(vec1,vec2,label)\
" fcmeq p1.d, p0/z, " #vec1 ".d, #0.0\n\t"\
" fcmeq p2.d, p0/z, " #vec2 ".d, #0.0\n\t"\
" ands p1.b, p0/z, p1.b, p2.b\n\t"\
" b.any " label "\n\t"
#endif
