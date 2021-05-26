/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

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

//#ifndef BLIS_FAMILY_H
//#define BLIS_FAMILY_H


// -- MEMORY ALLOCATION --------------------------------------------------------

#define BLIS_SIMD_ALIGN_SIZE    256
#define BLIS_SIMD_NUM_REGISTERS 32

#ifdef 1 // BLIS_ENABLE_MULTITHREADING
#define BLIS_GEMM_DYNAMIC_BLOCK_SIZE_UPDATE(cntx, rntm,  c) {           \
                                                                        \
    const dim_t ic = rntm->thrloop[ BLIS_MC ];                          \
    const dim_t jr = rntm->thrloop[ BLIS_NR ];                          \
    const dim_t m = bli_obj_length(&c);                                 \
    const dim_t n = bli_obj_width(&c);                                  \
                                                                        \
    blksz_t blkszs[ BLIS_NUM_BLKSZS ];                                  \
    if (ic >= 2 && m < 2400) {                                          \
        bli_blksz_init_easy(&blkszs[BLIS_MC],   128,    64, -1, -1 );   \
        bli_blksz_init_easy(&blkszs[BLIS_KC],  3072,  3072, -1, -1 );   \
        bli_blksz_init_easy(&blkszs[BLIS_NC], 23040, 26880, -1, -1 );   \
                                                                        \
        bli_cntx_set_blkszs(                                            \
            BLIS_NAT, 3,                                                \
            BLIS_NC, &blkszs[BLIS_NC], BLIS_NR,                         \
            BLIS_KC, &blkszs[BLIS_KC], BLIS_KR,                         \
            BLIS_MC, &blkszs[BLIS_MC], BLIS_MR,                         \
            cntx);                                                      \
    } else {                                                            \
      /* Do not touch block size otherwise.
       * TODO: Process also small matrices. */                          \
    }                                                                   \
}
#else
#define BLIS_GEMM_DYNAMIC_BLOCK_SIZE_UPDATE(cntx, rntm, c) {}
#endif


//#endif

