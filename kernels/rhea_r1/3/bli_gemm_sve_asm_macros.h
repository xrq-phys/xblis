#define COMBINE2(a,b) a ## _ ## b

#define LOAD2VEC(vec1,vec2,preg,areg)\
    " ld1d  " #vec1 ".d, " #preg "/z, [" #areg "]           \n\t"\
    " ld1d  " #vec2 ".d, " #preg "/z, [" #areg ",#1, MUL VL]\n\t"

#define LOAD2VEC_CONT(vec1,vec2,preg,areg,avec1,avec2) LOAD2VEC(vec1,vec2,preg,areg)

#define LOAD2VEC_GENI(vec1,vec2,preg,areg,avec1,avec2)\
    " ld1d  " #vec1 ".d, " #preg "/z, [" #areg "," #avec1 ".d, LSL #3]\n\t"\
    " ld1d  " #vec2 ".d, " #preg "/z, [" #areg "," #avec2 ".d, LSL #3]\n\t"

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

#define MLA1ROW(c1, a1, bvec, preg)\
    " fmla " #c1 ".d, " #preg "/m, " #a1 ".d, " #bvec ".d\n\t"

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

#define MLA2ROW_LB(c1, c2 , a1, a2, bvec, preg,  bareg, bboff)\
    MLA2ROW(c1, c2, a1, a2, bvec, preg)\
    " ld1rd  " #bvec ".d, "#preg"/z, [" #bareg",#" #bboff "]\n\t"

#define MLA4ROW_LB(c1, c2, c3, c4, a1, a2, a3, a4, bvec, preg,  bareg, bboff)\
    MLA4ROW(c1, c2, c3, c4, a1, a2, a3, a4, bvec, preg)\
    " ld1rd  " #bvec ".d, "#preg"/z, [" #bareg",#" #bboff "]\n\t"


// Load 2 vectors from contiguous memory
#define STOR2VEC(vec1,vec2,preg,areg)\
    " st1d  {" #vec1 ".d}, " #preg ", [" #areg "]           \n\t"\
    " st1d  {" #vec2 ".d}, " #preg ", [" #areg ",#1, MUL VL]\n\t"

#define STOR2VEC_CONT(vec1,vec2,preg,areg,avec1,avec2) STOR2VEC(vec1,vec2,preg,areg)

// Load 2 vectors with generic indexing (scatter-store)
#define STOR2VEC_GENI(vec1,vec2,preg,areg,avec1,avec2)\
    " st1d  {" #vec1 ".d}, " #preg ", [" #areg "," #avec1 ".d, LSL #3]\n\t"\
    " st1d  {" #vec2 ".d}, " #preg ", [" #areg "," #avec2 ".d, LSL #3]\n\t"

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


static void print_marker(uint64_t val)
{
    printf("Marker %lu\n",val);
}

static void print_pointer(void* p)
{
    printf("Pointer: 0x%p\n", p);
}

#define MAX_COUNTERS 32
static int initialized = 0;
static uint64_t counters[MAX_COUNTERS];
static void print_counter(uint64_t counter, uint64_t inc)
{
    if(!initialized)
    {
        for(int i = 0; i < MAX_COUNTERS; i++)
        {
            counters[i] = 0;
        }
        initialized = 1;
    }
    printf("Counter %lu: %lu\n", counter, counters[counter]);
    counters[counter] += inc;
}

#define SAVEALLREGS\
    " stp x0, x1,   [sp, #-16]!\n\t"\
    " stp x2, x3,   [sp, #-16]!\n\t"\
    " stp x4, x5,   [sp, #-16]!\n\t"\
    " stp x6, x7,   [sp, #-16]!\n\t"\
    " stp x8, x9,   [sp, #-16]!\n\t"\
    " stp x10, x11, [sp, #-16]!\n\t"\
    " stp x12, x13, [sp, #-16]!\n\t"\
    " stp x14, x15, [sp, #-16]!\n\t"\
    " stp x16, x17, [sp, #-16]!\n\t"\
    " stp x18, x19, [sp, #-16]!\n\t"\
    " stp x20, x21, [sp, #-16]!\n\t"\
    " stp x22, x23, [sp, #-16]!\n\t"\
    " stp x24, x25, [sp, #-16]!\n\t"\
    " stp x26, x27, [sp, #-16]!\n\t"\
    " stp x28, x29, [sp, #-16]!\n\t"\
    " stp x30, x0,  [sp, #-16]!\n\t"\
    " sub sp,sp,#16\n\t"

#define RESTOREALLREGS\
    " add sp,sp,#16\n\t"\
    " ldp x30, x0,  [sp], #16\n\t"\
    " ldp x28, x29, [sp], #16\n\t"\
    " ldp x26, x27, [sp], #16\n\t"\
    " ldp x24, x25, [sp], #16\n\t"\
    " ldp x22, x23, [sp], #16\n\t"\
    " ldp x20, x21, [sp], #16\n\t"\
    " ldp x18, x19, [sp], #16\n\t"\
    " ldp x16, x17, [sp], #16\n\t"\
    " ldp x14, x15, [sp], #16\n\t"\
    " ldp x12, x13, [sp], #16\n\t"\
    " ldp x10, x11, [sp], #16\n\t"\
    " ldp x8, x9,   [sp], #16\n\t"\
    " ldp x6, x7,   [sp], #16\n\t"\
    " ldp x4, x5,   [sp], #16\n\t"\
    " ldp x2, x3,   [sp], #16\n\t"\
    " ldp x0, x1,   [sp], #16\n\t"

#define PMARKER(x)\
    SAVEALLREGS\
    " mov x0, #"#x"\n\t"\
    " bl print_marker\n\t"\
    RESTOREALLREGS

#define PREGVAL(x)\
    SAVEALLREGS\
    " mov x0, "#x"\n\t"\
    " bl print_marker\n\t"\
    RESTOREALLREGS

#define PPOINTER(x)\
    SAVEALLREGS\
    " mov x0, "#x"\n\t"\
    " bl print_pointer\n\t"\
    RESTOREALLREGS

#define PCOUNTER(x,y)\
    SAVEALLREGS\
    " mov x0, "#x"\n\t"\
    " mov x1, "#y"\n\t"\
    " bl print_counter\n\t"\
    RESTOREALLREGS
 
