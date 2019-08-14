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


#include <stdint.h>

void print_marker(uint64_t val);

void print_pointer(void* p);

#define MAX_COUNTERS 32
void print_counter(uint64_t counter, uint64_t inc);

void print_dvector(double* ptr, uint64_t nelem);
void print_slvector(int64_t* ptr, uint64_t nelem);

#define SAVEALLZREGS\
    " mov x0,sp\n\t"\
    " decb x0\n\t"\
    " st1d z0.d,p0,[x0]\n\t"\
    " decb x0\n\t"\
    " st1d z1.d,p0,[x0]\n\t"\
    " decb x0\n\t"\
    " st1d z2.d,p0,[x0]\n\t"\
    " decb x0\n\t"\
    " st1d z3.d,p0,[x0]\n\t"\
    " decb x0\n\t"\
    " st1d z4.d,p0,[x0]\n\t"\
    " decb x0\n\t"\
    " st1d z5.d,p0,[x0]\n\t"\
    " decb x0\n\t"\
    " st1d z6.d,p0,[x0]\n\t"\
    " decb x0\n\t"\
    " st1d z7.d,p0,[x0]\n\t"\
    " decb x0\n\t"\
    " st1d z8.d,p0,[x0]\n\t"\
    " decb x0\n\t"\
    " st1d z9.d,p0,[x0]\n\t"\
    " decb x0\n\t"\
    " st1d z10.d,p0,[x0]\n\t"\
    " decb x0\n\t"\
    " st1d z11.d,p0,[x0]\n\t"\
    " decb x0\n\t"\
    " st1d z12.d,p0,[x0]\n\t"\
    " decb x0\n\t"\
    " st1d z13.d,p0,[x0]\n\t"\
    " decb x0\n\t"\
    " st1d z14.d,p0,[x0]\n\t"\
    " decb x0\n\t"\
    " st1d z15.d,p0,[x0]\n\t"\
    " decb x0\n\t"\
    " st1d z16.d,p0,[x0]\n\t"\
    " decb x0\n\t"\
    " st1d z17.d,p0,[x0]\n\t"\
    " decb x0\n\t"\
    " st1d z18.d,p0,[x0]\n\t"\
    " decb x0\n\t"\
    " st1d z19.d,p0,[x0]\n\t"\
    " decb x0\n\t"\
    " st1d z20.d,p0,[x0]\n\t"\
    " decb x0\n\t"\
    " st1d z21.d,p0,[x0]\n\t"\
    " decb x0\n\t"\
    " st1d z22.d,p0,[x0]\n\t"\
    " decb x0\n\t"\
    " st1d z23.d,p0,[x0]\n\t"\
    " decb x0\n\t"\
    " st1d z24.d,p0,[x0]\n\t"\
    " decb x0\n\t"\
    " st1d z25.d,p0,[x0]\n\t"\
    " decb x0\n\t"\
    " st1d z26.d,p0,[x0]\n\t"\
    " decb x0\n\t"\
    " st1d z27.d,p0,[x0]\n\t"\
    " decb x0\n\t"\
    " st1d z28.d,p0,[x0]\n\t"\
    " decb x0\n\t"\
    " st1d z29.d,p0,[x0]\n\t"\
    " decb x0\n\t"\
    " st1d z30.d,p0,[x0]\n\t"\
    " decb x0\n\t"\
    " st1d z31.d,p0,[x0]\n\t"\
    " decb x0\n\t"\
    " mov sp, x0\n\t"

#define RESTOREALLZREGS\
    " mov x0, sp\n\t"\
    " incb x0\n\t"\
    " ld1d z31.d,p0/z,[x0]\n\t"\
    " incb x0\n\t"\
    " ld1d z30.d,p0/z,[x0]\n\t"\
    " incb x0\n\t"\
    " ld1d z29.d,p0/z,[x0]\n\t"\
    " incb x0\n\t"\
    " ld1d z28.d,p0/z,[x0]\n\t"\
    " incb x0\n\t"\
    " ld1d z27.d,p0/z,[x0]\n\t"\
    " incb x0\n\t"\
    " ld1d z26.d,p0/z,[x0]\n\t"\
    " incb x0\n\t"\
    " ld1d z25.d,p0/z,[x0]\n\t"\
    " incb x0\n\t"\
    " ld1d z24.d,p0/z,[x0]\n\t"\
    " incb x0\n\t"\
    " ld1d z23.d,p0/z,[x0]\n\t"\
    " incb x0\n\t"\
    " ld1d z22.d,p0/z,[x0]\n\t"\
    " incb x0\n\t"\
    " ld1d z21.d,p0/z,[x0]\n\t"\
    " incb x0\n\t"\
    " ld1d z20.d,p0/z,[x0]\n\t"\
    " incb x0\n\t"\
    " ld1d z19.d,p0/z,[x0]\n\t"\
    " incb x0\n\t"\
    " ld1d z18.d,p0/z,[x0]\n\t"\
    " incb x0\n\t"\
    " ld1d z17.d,p0/z,[x0]\n\t"\
    " incb x0\n\t"\
    " ld1d z16.d,p0/z,[x0]\n\t"\
    " incb x0\n\t"\
    " ld1d z15.d,p0/z,[x0]\n\t"\
    " incb x0\n\t"\
    " ld1d z14.d,p0/z,[x0]\n\t"\
    " incb x0\n\t"\
    " ld1d z13.d,p0/z,[x0]\n\t"\
    " incb x0\n\t"\
    " ld1d z12.d,p0/z,[x0]\n\t"\
    " incb x0\n\t"\
    " ld1d z11.d,p0/z,[x0]\n\t"\
    " incb x0\n\t"\
    " ld1d z10.d,p0/z,[x0]\n\t"\
    " incb x0\n\t"\
    " ld1d z9.d,p0/z,[x0]\n\t"\
    " incb x0\n\t"\
    " ld1d z8.d,p0/z,[x0]\n\t"\
    " incb x0\n\t"\
    " ld1d z7.d,p0/z,[x0]\n\t"\
    " incb x0\n\t"\
    " ld1d z6.d,p0/z,[x0]\n\t"\
    " incb x0\n\t"\
    " ld1d z5.d,p0/z,[x0]\n\t"\
    " incb x0\n\t"\
    " ld1d z4.d,p0/z,[x0]\n\t"\
    " incb x0\n\t"\
    " ld1d z3.d,p0/z,[x0]\n\t"\
    " incb x0\n\t"\
    " ld1d z2.d,p0/z,[x0]\n\t"\
    " incb x0\n\t"\
    " ld1d z1.d,p0/z,[x0]\n\t"\
    " incb x0\n\t"\
    " ld1d z0.d,p0/z,[x0]\n\t"\
    " incb x0\n\t"\
    " mov sp,x0\n\t"

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
    SAVEALLZREGS\
    " sub sp,sp,#16\n\t"

#define RESTOREALLREGS\
    " add sp,sp,#16\n\t"\
    RESTOREALLZREGS\
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

#define PDVEC(x,ptr,nelem)\
    SAVEALLREGS\
    " ldr x0, %[" #ptr "]\n\t"\
    " mov x1, "#nelem"\n\t"\
    " st1d {"#x".d},p0,[x0]\n\t"\
    " bl print_dvector\n\t"\
    RESTOREALLREGS

#define PSLVEC(x,ptr,nelem)\
    SAVEALLREGS\
    " ldr x0, %[" #ptr "]\n\t"\
    " mov x1, "#nelem"\n\t"\
    " st1d {"#x".d},p0,[x0]\n\t"\
    " bl print_slvector\n\t"\
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

#define INASM_START_TRACE ".inst 0x2520e020\n\t"
#define INASM_STOP_TRACE ".inst 0x2520e040\n\t"
