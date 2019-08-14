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
#include "bli_sve_asm_loadstore.h"

#define COMBINE2(a,b) a ## _ ## b

#define ZEROVEC(vec1)\
    " dup " #vec1 ".d, #0\n\t"

#define ZERO2VEC(vec1,vec2)\
    ZEROVEC(vec1)\
    ZEROVEC(vec2)

#define ZERO4VEC(vec1,vec2,vec3,vec4)\
    ZERO2VEC(vec1,vec2)\
    ZERO2VEC(vec3,vec4)

#define ZERO8VEC(vec1,vec2,vec3,vec4,vec5,vec6,vec7,vec8)\
    ZERO4VEC(vec1,vec2,vec3,vec4)\
    ZERO4VEC(vec5,vec6,vec7,vec8)

#define MLA1ROW(cvec, avec, bvec, preg)\
    " fmla " #cvec ".d, " #preg "/m, " #avec ".d, " #bvec ".d\n\t"

#define MLA2ROW(c1, c2 , a1, a2, bvec, preg)\
    MLA1ROW(c1,a1,bvec,preg)\
    MLA1ROW(c2,a2,bvec,preg)

#define MLA4ROW(c1, c2, c3, c4, a1, a2, a3, a4, bvec, preg)\
    MLA2ROW(c1,c2,a1,a2,bvec,preg)\
    MLA2ROW(c3,c4,a3,a4,bvec,preg)


#define MUL1ROW(c1, a1, bvec, preg)\
    " fmul " #c1 ".d, " #preg "/m, " #a1 ".d, " #bvec ".d\n\t"

#define MUL2ROW(c1, c2 , a1, a2, bvec, preg)\
    MUL1ROW(c1,a1,bvec,preg)\
    MUL1ROW(c2,a2,bvec,preg)

#define MUL4ROW(c1, c2, c3, c4, a1, a2, a3, a4, bvec, preg)\
    MUL2ROW(c1,c2,a1,a2,bvec,preg)\
    MUL2ROW(c3,c4,a3,a4,bvec,preg)

#define MLA2X2ROW(c11, c12, c21, c22, a1, a2, bvec1,bvec2, preg)\
    MLA2ROW(c11, c12, a1, a2, bvec1, preg)\
    MLA2ROW(c21, c22, a1, a2, bvec2, preg)

#define MLA1ROW_LA_LB(cvec, avec, bvec, preg, nextavec, aareg, avoff, bareg, bboff)\
    MLA1ROW(cvec, avec, bvec, preg)\
    " ld1d   " #nextavec ".d, " #preg "/z, [" #aareg ", #" #avoff", MUL VL]\n\t"\
    " ld1rd  " #bvec ".d, "#preg"/z, [" #bareg",#" #bboff "]\n\t"

#define MLA2ROW_LA_LB(c1, c2 , a1, a2, bvec, preg, nextavec, aareg, avoff, bareg, bboff)\
    MLA2ROW(c1, c2, a1, a2, bvec, preg)\
    " ld1d   " #nextavec ".d, " #preg "/z, [" #aareg ", #" #avoff", MUL VL]\n\t"\
    " ld1rd  " #bvec ".d, "#preg"/z, [" #bareg",#" #bboff "]\n\t"

#define MLA2X2ROW_LA_LB(c11,c12,c21,c22, a1, a2, bvec1,bvec2, preg, nextavec, aareg, avoff, bareg, bboff1,bboff2)\
    MLA2ROW(c11, c12, a1, a2, bvec1, preg)\
    MLA2ROW(c21, c22, a1, a2, bvec2, preg)\
    " ld1d   " #nextavec ".d, " #preg "/z, [" #aareg ", #" #avoff", MUL VL]\n\t"\
    " ld1rd  " #bvec1 ".d, "#preg"/z, [" #bareg",#" #bboff1 "]\n\t"\
    " ld1rd  " #bvec2 ".d, "#preg"/z, [" #bareg",#" #bboff2 "]\n\t"

#define MLA4ROW_LA_LB(c1, c2, c3, c4, a1, a2, a3, a4, bvec, preg, nextavec, aareg, avoff, bareg, bboff)\
    MLA4ROW(c1, c2, c3, c4, a1, a2, a3, a4, bvec, preg)\
    " ld1d   " #nextavec ".d, " #preg "/z, [" #aareg ", #" #avoff", MUL VL]\n\t"\
    " ld1rd  " #bvec ".d, "#preg"/z, [" #bareg",#" #bboff "]\n\t"


#define MLA2X2ROW_LB(c11, c12, c21, c22, a1, a2, bvec1,bvec2, preg, bareg, bboff1,bboff2)\
    MLA2ROW(c11, c12, a1, a2, bvec1, preg)\
    MLA2ROW(c21, c22, a1, a2, bvec2, preg)\
    " ld1rd  " #bvec1 ".d, "#preg"/z, [" #bareg",#" #bboff1 "]\n\t"\
    " ld1rd  " #bvec2 ".d, "#preg"/z, [" #bareg",#" #bboff2 "]\n\t"

#define MLA1ROW_LB(cvec, avec, bvec, preg,  bareg, bboff)\
    MLA1ROW(cvec, avec, bvec, preg)\
    " ld1rd  " #bvec ".d, "#preg"/z, [" #bareg",#" #bboff "]\n\t"

#define MLA2ROW_LB(c1, c2 , a1, a2, bvec, preg,  bareg, bboff)\
    MLA2ROW(c1, c2, a1, a2, bvec, preg)\
    " ld1rd  " #bvec ".d, "#preg"/z, [" #bareg",#" #bboff "]\n\t"

#define MLA4ROW_LB(c1, c2, c3, c4, a1, a2, a3, a4, bvec, preg,  bareg, bboff)\
    MLA4ROW(c1, c2, c3, c4, a1, a2, a3, a4, bvec, preg)\
    " ld1rd  " #bvec ".d, "#preg"/z, [" #bareg",#" #bboff "]\n\t"

#if defined(USE_SVE_CMLA_INSTRUCTION)
    #define CMLA1ROW_ILV(cvec1, cvec2, avec1, avec2, bvec1, bvec2, preg, ilv1, ilv2, ilv3, ilv4)\
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
    #define CMLA1ROW_ILV(cvec_r, cvec_i, avec_r, avec_i, bvec_r, bvec_i, preg, ilv1, ilv2, ilv3, ilv4)\
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

#define CMLA1ROW(cvec1, cvec2, avec1, avec2, bvec1, bvec2, preg)\
        CMLA1ROW_ILV(cvec1, cvec2, avec1, avec2, bvec1, bvec2, preg,"","","","")

#define CMLA1ROW_ILV_LA_LB(c1, c2, a1, a2, bvec1, bvec2, preg, ilv1, ilv2, ilv3, ilv4, nextavec1, nextavec2, aareg, avoff1, avoff2, bareg, bboff1, bboff2)\
    CMLA1ROW_ILV(c1, c2, a1, a2, bvec1, bvec2, preg, ilv1, ilv2, ilv3, ilv4 )\
    LOADC2VEC_VOFF(nextavec1, nextavec2, preg, aareg, avoff1, avoff2)\
    LOADC2VEC_DIST_OFF(bvec1, bvec2, preg, bareg, bboff1,bboff2)

#define CMLA1ROW_ILV_LB(c1, c2, a1, a2, bvec1, bvec2, preg, ilv1, ilv2, ilv3, ilv4, bareg, bboff1, bboff2)\
    CMLA1ROW_ILV(c1, c2, a1, a2, bvec1, bvec2, preg, ilv1, ilv2, ilv3, ilv4)\
    LOADC2VEC_DIST_OFF(bvec1, bvec2, preg, bareg, bboff1, bboff2)

#define CMLA1ROW_LA_LB(c1, c2, a1, a2, bvec1, bvec2, preg, nextavec1, nextavec2, aareg, avoff1, avoff2, bareg, bboff1, bboff2)\
    CMLA1ROW_ILV_LA_LB(c1, c2, a1, a2, bvec1, bvec2, preg, "", "", "", "", nextavec1, nextavec2, aareg, avoff1, avoff2, bareg, bboff1, bboff2)

#define CMLA1ROW_LB(c1, c2, a1, a2, bvec1, bvec2, preg, bareg, bboff1, bboff2)\
    CMLA1ROW_ILV_LB(c1, c2, a1, a2, bvec1, bvec2, preg, "", "", "", "", bareg, bboff1, bboff2)

#define CMLA2ROW(cvec1, cvec2, cvec3, cvec4, avec1, avec2, avec3, avec4, bvec1, bvec2, preg)\
    CMLA1ROW(cvec1, cvec2, avec1, avec2, bvec1, bvec2, preg)\
    CMLA1ROW(cvec3, cvec4, avec3, avec4, bvec1, bvec2, preg)

#define PFL1(areg,preg,offset) " prfd pldl1keep, "#preg", [" #areg ", #" #offset ", MUL VL]\n\t"

// Zero 2 columns of C,
// Load 2 columns of C and multiply by beta if beta !=0
// Add accumulated A*B values multiplied by alpha
// Store 2 columns of C
// Contiguous memory (CONT) or generic index (GENI) specified by addressing
// labelnr for the beta case jumps
#define FINC_2COL(fsuf, addressing,c0,c1,c2,c3,ca0,ca1, avec0,avec1, alpha, beta, acc0,acc1,acc2,acc3,labelnr)\
ZERO4VEC(c0,c1,c2,c3)\
"                                            \n\t"\
" fcmp d" #beta ",#0.0                       \n\t"\
" beq .D" #fsuf "BETAZERO" #addressing "COLSTOREDS" #labelnr "       \n\t"\
"                                            \n\t"\
COMBINE2(LOAD2VEC,addressing) (c0,c1,p0,ca0,avec0,avec1)\
COMBINE2(LOAD2VEC,addressing) (c2,c3,p0,ca1,avec0,avec1)\
"                                            \n\t"\
MUL4ROW(c0,c1,c2,c3,c0,c1,c2,c3,z ##beta,p0)\
"                                            \n\t"\
" .D" #fsuf "BETAZERO" #addressing "COLSTOREDS" #labelnr ":          \n\t"\
"                                            \n\t"\
MLA4ROW(c0,c1,c2,c3,acc0,acc1,acc2,acc3,z ##alpha,p0)\
"                                            \n\t"\
COMBINE2(STOR2VEC,addressing) (c0,c1,p0,ca0,avec0,avec1)\
COMBINE2(STOR2VEC,addressing) (c2,c3,p0,ca1,avec0,avec1)         

// 1x vector variant
// Zero 4 columns of C,
// Load 4 columns of C and multiply by beta if beta !=0
// Add accumulated A*B values multiplied by alpha
// Store 4 columns of C
// Contiguous memory (CONT) or generic index (GENI) specified by addressing
// labelnr for the beta case jumps
#define FINC_4COL_1X(fsuf, addressing,c0,c1,c2,c3, ca0,ca1,ca2,ca3, avec, alpha, beta, acc0,acc1,acc2,acc3, labelnr)\
ZERO4VEC(c0,c1,c2,c3)\
"                                            \n\t"\
" fcmp d" #beta ",#0.0                       \n\t"\
" beq .D" #fsuf "BETAZERO" #addressing "COLSTOREDS" #labelnr "       \n\t"\
"                                            \n\t"\
COMBINE2(LOAD1VEC,addressing) (c0,p0,ca0,avec)\
COMBINE2(LOAD1VEC,addressing) (c1,p0,ca1,avec)\
COMBINE2(LOAD1VEC,addressing) (c2,p0,ca2,avec)\
COMBINE2(LOAD1VEC,addressing) (c3,p0,ca3,avec)\
"                                            \n\t"\
MUL4ROW(c0,c1,c2,c3,c0,c1,c2,c3,z ##beta,p0)\
"                                            \n\t"\
" .D" #fsuf "BETAZERO" #addressing "COLSTOREDS" #labelnr ":          \n\t"\
"                                            \n\t"\
MLA4ROW(c0,c1,c2,c3,acc0,acc1,acc2,acc3,z ##alpha,p0)\
"                                            \n\t"\
COMBINE2(STOR1VEC,addressing) (c0,p0,ca0,avec)\
COMBINE2(STOR1VEC,addressing) (c1,p0,ca1,avec)\
COMBINE2(STOR1VEC,addressing) (c2,p0,ca2,avec)\
COMBINE2(STOR1VEC,addressing) (c3,p0,ca3,avec)

// 2x vector variant
// Zero 4 columns of C,
// Load 4 columns of C and multiply by beta if beta !=0
// Add accumulated A*B values multiplied by alpha
// Store 4 columns of C
// Contiguous memory (CONT) or generic index (GENI) specified by addressing
// labelnr for the beta case jumps
#define FINC_4COL(fsuf, addressing,c0,c1,c2,c3,c4,c5,c6,c7, ca0,ca1,ca2,ca3, avec0,avec1, alpha, beta, acc0,acc1,acc2,acc3,acc4,acc5,acc6,acc7, labelnr)\
ZERO8VEC(c0,c1,c2,c3,c4,c5,c6,c7)\
"                                            \n\t"\
" fcmp d" #beta ",#0.0                       \n\t"\
" beq .D" #fsuf "BETAZERO" #addressing "COLSTOREDS" #labelnr "       \n\t"\
"                                            \n\t"\
COMBINE2(LOAD2VEC,addressing) (c0,c1,p0,ca0,avec0,avec1)\
COMBINE2(LOAD2VEC,addressing) (c2,c3,p0,ca1,avec0,avec1)\
COMBINE2(LOAD2VEC,addressing) (c4,c5,p0,ca2,avec0,avec1)\
COMBINE2(LOAD2VEC,addressing) (c6,c7,p0,ca3,avec0,avec1)\
"                                            \n\t"\
MUL4ROW(c0,c1,c2,c3,c0,c1,c2,c3,z ##beta,p0)\
MUL4ROW(c4,c5,c6,c7,c4,c5,c6,c7,z ##beta,p0)\
"                                            \n\t"\
" .D" #fsuf "BETAZERO" #addressing "COLSTOREDS" #labelnr ":          \n\t"\
"                                            \n\t"\
MLA4ROW(c0,c1,c2,c3,acc0,acc1,acc2,acc3,z ##alpha,p0)\
MLA4ROW(c4,c5,c6,c7,acc4,acc5,acc6,acc7,z ##alpha,p0)\
"                                            \n\t"\
COMBINE2(STOR2VEC,addressing) (c0,c1,p0,ca0,avec0,avec1)\
COMBINE2(STOR2VEC,addressing) (c2,c3,p0,ca1,avec0,avec1)\
COMBINE2(STOR2VEC,addressing) (c4,c5,p0,ca2,avec0,avec1)\
COMBINE2(STOR2VEC,addressing) (c6,c7,p0,ca3,avec0,avec1)

// Check if a complex number is 0
// TODO: Only first complex number needs to be checked, 
#if defined(USE_SVE_CMLA_INSTRUCTION)
#define CMPCZB(vec1,vec2,label)\
" fcmeq p1.d, p0/z, " #vec1 ".d, #0.0\n\t"\
" nots p1.b, p0/z, p1.b\n\t"\
" b.none " label "\n\t"
#else
#define CMPCZB(vec1,vec2,label)\
" fcmeq p1.d, p0/z, " #vec1 ".d, #0.0\n\t"\
" fcmeq p2.d, p0/z, " #vec2 ".d, #0.0\n\t"\
" ands p1.b, p0/z, p1.b, p2.b\n\t"\
" b.any " label "\n\t"
#endif

// complex variant, 1row = 2 vectors worth of double complex
// Zero 4 columns of C,
// Load 4 columns of C and multiply by beta if beta !=0
// Add accumulated A*B values multiplied by alpha
// Store 4 columns of C
// Contiguous memory (CONT) or generic index (GENI) specified by addressing
// labelnr for the beta case jumps
// TODO: csX are needed because there is no fcmul instruction - find a better way without using
//       8 additional vector registers
#define CFINC_4COL(fsuf, addressing, c0,c1,c2,c3,c4,c5,c6,c7, cs0,cs1,cs2,cs3,cs4,cs5,cs6,cs7, ca0,ca1,ca2,ca3, avec1,avec2, alpha1,alpha2,beta1,beta2, acc0,acc1,acc2,acc3,acc4,acc5,acc6,acc7, labelnr)\
ZERO8VEC(c0,c1,c2,c3,c4,c5,c6,c7)\
"                                            \n\t"\
CMPCZB(z ##beta1, z ##beta2, ".Z" #fsuf "BETAZERO" #addressing "COLSTOREDS" #labelnr)\
"                                            \n\t"\
COMBINE2(LOADC2VEC,addressing) (cs0,cs1,p0,ca0,avec1,avec2)\
COMBINE2(LOADC2VEC,addressing) (cs2,cs3,p0,ca1,avec1,avec2)\
COMBINE2(LOADC2VEC,addressing) (cs4,cs5,p0,ca2,avec1,avec2)\
COMBINE2(LOADC2VEC,addressing) (cs6,cs7,p0,ca3,avec1,avec2)\
"                                            \n\t"\
CMLA2ROW(c0,c1,c2,c3,cs0,cs1,cs2,cs3,z ##beta1,z ##beta2,p0)\
CMLA2ROW(c4,c5,c6,c7,cs4,cs5,cs6,cs7,z ##beta1,z ##beta2,p0)\
"                                            \n\t"\
" .Z" #fsuf "BETAZERO" #addressing "COLSTOREDS" #labelnr ":          \n\t"\
"                                            \n\t"\
CMLA2ROW(c0,c1,c2,c3,acc0,acc1,acc2,acc3,z ##alpha1 , z ##alpha2 ,p0)\
CMLA2ROW(c4,c5,c6,c7,acc4,acc5,acc6,acc7,z ##alpha1 , z ##alpha2 ,p0)\
"                                            \n\t"\
COMBINE2(STORC2VEC,addressing) (c0,c1,p0,ca0,avec1,avec2)\
COMBINE2(STORC2VEC,addressing) (c2,c3,p0,ca1,avec1,avec2)\
COMBINE2(STORC2VEC,addressing) (c4,c5,p0,ca2,avec1,avec2)\
COMBINE2(STORC2VEC,addressing) (c6,c7,p0,ca3,avec1,avec2)

