/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2019, Forschungszentrum Juelich, Germany

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
#include "sve_architecture.h"

uint64_t get_sve_byte_size()
{
    uint64_t byte_size = 0;
    __asm__ volatile(
            " mov %[byte_size],#0\n\t"
            " incb %[byte_size]\n\t"
            : [byte_size] "=r" (byte_size)
            :
            :
            ); 
    return byte_size;
}

void  adjust_sve_mr_nr_d(int* m_r, int* n_r)
{
#if SVE_VECSIZE == SVE_VECSIZE_VLA

    int onevec = (get_sve_byte_size())/8;

#warning Testing 2vx10
    *m_r = 2*onevec;
    *n_r = 10;
    return;

    if(*m_r > 2*onevec)
    {
        *m_r = 4*onevec;
        *n_r = 5;
    }
    else if (*m_r == 2*onevec)
    {
        *m_r = 2*onevec;
        if(*n_r != 9)
        {
            *n_r = 8;
        }
    }
    else
    {
        *m_r = onevec;
        *n_r = 8;
    }
#elif SVE_VECSIZE == SVE_VECSIZE_256
    *m_r = 8;
    *n_r = 10;
#elif SVE_VECSIZE == SVE_VECSIZE_512
    *m_r = 16;
    *n_r = 10;
#elif SVE_VECSIZE == SVE_VECSIZE_1024
    *m_r = 32;
    *n_r = 10;
#endif
}

void  adjust_sve_mr_nr_s(int* m_r, int* n_r)
{
    //if not implemented, set to -1
#if SVE_VECSIZE == SVE_VECSIZE_VLA
    int onevec = (get_sve_byte_size())/4;
    *m_r = 2*onevec;
    *n_r = 8;
#elif SVE_VECSIZE == SVE_VECSIZE_256
    *m_r = 16;
    *n_r = 8;
#elif SVE_VECSIZE == SVE_VECSIZE_512
    *m_r = -1;
    *n_r = -1;
#elif SVE_VECSIZE == SVE_VECSIZE_1024
    *m_r = -1;
    *n_r = -1;
#endif
}

void  adjust_sve_mr_nr_z(int* m_r, int* n_r)
{
#if SVE_VECSIZE == SVE_VECSIZE_VLA
    *m_r = (2*get_sve_byte_size())/16;
    *n_r = 4;
#elif SVE_VECSIZE == SVE_VECSIZE_256
    *m_r = 4;
    *n_r = 4;
#elif SVE_VECSIZE == SVE_VECSIZE_512
    *m_r = 8;
    *n_r = 4;
#elif SVE_VECSIZE == SVE_VECSIZE_1024
    *m_r = 16;
    *n_r = 4;
#endif
}
