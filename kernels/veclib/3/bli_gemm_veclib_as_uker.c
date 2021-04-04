/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2021, The University of Tokyo
   Copyright (C) 2020, Apple Inc. (?)

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

#include "blis.h"
#include <assert.h>

//
// Prototype BLAS-to-BLIS interfaces.
// Required to call Accelerate.
//
#undef  GENTPROT
#define GENTPROT( ftype, ch, blasname ) \
\
BLIS_EXPORT_BLAS void PASTEF77(ch,blasname) \
     ( \
       const f77_char* transa, \
       const f77_char* transb, \
       const f77_int*  m, \
       const f77_int*  n, \
       const f77_int*  k, \
       const ftype*    alpha, \
       const ftype*    a, const f77_int* lda, \
       const ftype*    b, const f77_int* ldb, \
       const ftype*    beta, \
             ftype*    c, const f77_int* ldc  \
     );
INSERT_GENTPROT_BLAS( gemm )

#define F77_NAME( cchar, opr ) cchar ##opr ##_
#define UKR_NAME( cchar ) bli_## cchar ##gemm_veclib_as_uker

#define EXPAND_VECLIB_BRIDGE( ctype, cchar, env_m0, m0_default, env_n0, n0_default ) \
void UKR_NAME( cchar ) \
     ( \
       dim_t               k0,    \
       ctype*     restrict alpha, \
       ctype*     restrict a,     \
       ctype*     restrict b,     \
       ctype*     restrict beta,  \
       ctype*     restrict c, inc_t rs_c0, inc_t cs_c0, \
       auxinfo_t* restrict data,  \
       cntx_t*    restrict cntx   \
     ) \
{ \
    dim_t m0 = bli_env_get_var(#env_m0, m0_default); \
    dim_t n0 = bli_env_get_var(#env_n0, n0_default); \
\
    /* TODO: Support G target. */ \
    assert ( rs_c0 == 1 ); \
    { \
        f77_int m = m0;      \
        f77_int n = n0;      \
        f77_int k = k0;      \
        f77_int lda = m0;    \
        f77_int ldb = n0;    \
        f77_int ldc = cs_c0; \
\
        char fcc_transa = 'n'; \
        char fcc_transb = 't'; \
\
        F77_NAME( cchar, gemm ) \
          ( &fcc_transa, \
            &fcc_transb, \
            &m,          \
            &n,          \
            &k,          \
            alpha,       \
            a, &lda,     \
            b, &ldb,     \
            beta,        \
            c, &ldc );   \
    } \
}
EXPAND_VECLIB_BRIDGE( float,    s, BLIS_SVE_VECLIB_MR_S, 320, BLIS_SVE_VECLIB_NR_S, 16 )
EXPAND_VECLIB_BRIDGE( double,   d, BLIS_SVE_VECLIB_MR_D, 160, BLIS_SVE_VECLIB_NR_D, 16 )
EXPAND_VECLIB_BRIDGE( scomplex, c, BLIS_SVE_VECLIB_MR_C, 160, BLIS_SVE_VECLIB_NR_C, 16 )
EXPAND_VECLIB_BRIDGE( dcomplex, z, BLIS_SVE_VECLIB_MR_Z, 160, BLIS_SVE_VECLIB_NR_Z, 16 )

