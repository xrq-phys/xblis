#include "bli_sve_asm_mla_d.h"

#define COMBINE2(a,b) a ## _ ## b

// Zero 2 columns of C,
// Load 2 columns of C and multiply by beta if beta !=0
// Add accumulated A*B values multiplied by alpha
// Store 2 columns of C
// Contiguous memory (CONT) or generic index (GENI) specified by addressing
// labelnr for the beta case jumps
#define FINC_2COL(fsuf, addressing,c0,c1,c2,c3,ca0,ca1, avec0,avec1, alpha, beta, acc0,acc1,acc2,acc3,labelnr)\
ZERO4VEC_D(c0,c1,c2,c3)\
"                                            \n\t"\
" fcmp d" #beta ",#0.0                       \n\t"\
" beq .D" #fsuf "BETAZERO" #addressing "COLSTOREDS" #labelnr "       \n\t"\
"                                            \n\t"\
COMBINE2(LOAD2VEC,addressing) (c0,c1,p0,ca0,avec0,avec1)\
COMBINE2(LOAD2VEC,addressing) (c2,c3,p0,ca1,avec0,avec1)\
"                                            \n\t"\
MUL4ROW_D(c0,c1,c2,c3,c0,c1,c2,c3,z ##beta,p0)\
"                                            \n\t"\
" .D" #fsuf "BETAZERO" #addressing "COLSTOREDS" #labelnr ":          \n\t"\
"                                            \n\t"\
MLA4ROW_D(c0,c1,c2,c3,acc0,acc1,acc2,acc3,z ##alpha,p0)\
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
ZERO4VEC_D(c0,c1,c2,c3)\
"                                            \n\t"\
" fcmp d" #beta ",#0.0                       \n\t"\
" beq .D" #fsuf "BETAZERO" #addressing "COLSTOREDS" #labelnr "       \n\t"\
"                                            \n\t"\
COMBINE2(LOAD1VEC,addressing) (c0,p0,ca0,avec)\
COMBINE2(LOAD1VEC,addressing) (c1,p0,ca1,avec)\
COMBINE2(LOAD1VEC,addressing) (c2,p0,ca2,avec)\
COMBINE2(LOAD1VEC,addressing) (c3,p0,ca3,avec)\
"                                            \n\t"\
MUL4ROW_D(c0,c1,c2,c3,c0,c1,c2,c3,z ##beta,p0)\
"                                            \n\t"\
" .D" #fsuf "BETAZERO" #addressing "COLSTOREDS" #labelnr ":          \n\t"\
"                                            \n\t"\
MLA4ROW_D(c0,c1,c2,c3,acc0,acc1,acc2,acc3,z ##alpha,p0)\
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
ZERO8VEC_D(c0,c1,c2,c3,c4,c5,c6,c7)\
"                                            \n\t"\
" fcmp d" #beta ",#0.0                       \n\t"\
" beq .D" #fsuf "BETAZERO" #addressing "COLSTOREDS" #labelnr "       \n\t"\
"                                            \n\t"\
COMBINE2(LOAD2VEC,addressing) (c0,c1,p0,ca0,avec0,avec1)\
COMBINE2(LOAD2VEC,addressing) (c2,c3,p0,ca1,avec0,avec1)\
COMBINE2(LOAD2VEC,addressing) (c4,c5,p0,ca2,avec0,avec1)\
COMBINE2(LOAD2VEC,addressing) (c6,c7,p0,ca3,avec0,avec1)\
"                                            \n\t"\
MUL4ROW_D(c0,c1,c2,c3,c0,c1,c2,c3,z ##beta,p0)\
MUL4ROW_D(c4,c5,c6,c7,c4,c5,c6,c7,z ##beta,p0)\
"                                            \n\t"\
" .D" #fsuf "BETAZERO" #addressing "COLSTOREDS" #labelnr ":          \n\t"\
"                                            \n\t"\
MLA4ROW_D(c0,c1,c2,c3,acc0,acc1,acc2,acc3,z ##alpha,p0)\
MLA4ROW_D(c4,c5,c6,c7,acc4,acc5,acc6,acc7,z ##alpha,p0)\
"                                            \n\t"\
COMBINE2(STOR2VEC,addressing) (c0,c1,p0,ca0,avec0,avec1)\
COMBINE2(STOR2VEC,addressing) (c2,c3,p0,ca1,avec0,avec1)\
COMBINE2(STOR2VEC,addressing) (c4,c5,p0,ca2,avec0,avec1)\
COMBINE2(STOR2VEC,addressing) (c6,c7,p0,ca3,avec0,avec1)


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
ZERO8VEC_D(c0,c1,c2,c3,c4,c5,c6,c7)\
"                                            \n\t"\
CMPCZB_D(z ##beta1, z ##beta2, ".Z" #fsuf "BETAZERO" #addressing "COLSTOREDS" #labelnr)\
"                                            \n\t"\
COMBINE2(LOAD2VEC,addressing) (cs0,cs1,p0,ca0,avec1,avec2)\
COMBINE2(LOAD2VEC,addressing) (cs2,cs3,p0,ca1,avec1,avec2)\
COMBINE2(LOAD2VEC,addressing) (cs4,cs5,p0,ca2,avec1,avec2)\
COMBINE2(LOAD2VEC,addressing) (cs6,cs7,p0,ca3,avec1,avec2)\
"                                            \n\t"\
MLA2ROW_Z(c0,c1,c2,c3,cs0,cs1,cs2,cs3,z ##beta1,z ##beta2,p0)\
MLA2ROW_Z(c4,c5,c6,c7,cs4,cs5,cs6,cs7,z ##beta1,z ##beta2,p0)\
"                                            \n\t"\
" .Z" #fsuf "BETAZERO" #addressing "COLSTOREDS" #labelnr ":          \n\t"\
"                                            \n\t"\
MLA2ROW_Z(c0,c1,c2,c3,acc0,acc1,acc2,acc3,z ##alpha1 , z ##alpha2 ,p0)\
MLA2ROW_Z(c4,c5,c6,c7,acc4,acc5,acc6,acc7,z ##alpha1 , z ##alpha2 ,p0)\
"                                            \n\t"\
COMBINE2(STOR2VEC,addressing) (c0,c1,p0,ca0,avec1,avec2)\
COMBINE2(STOR2VEC,addressing) (c2,c3,p0,ca1,avec1,avec2)\
COMBINE2(STOR2VEC,addressing) (c4,c5,p0,ca2,avec1,avec2)\
COMBINE2(STOR2VEC,addressing) (c6,c7,p0,ca3,avec1,avec2)
