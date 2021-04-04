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

#include "blis.h"

void bli_cntx_init_veclib( cntx_t* cntx )
{
	blksz_t blkszs[ BLIS_NUM_BLKSZS ];

	// Set default kernel blocksizes and functions.
	bli_cntx_init_veclib_ref( cntx );

	// -------------------------------------------------------------------------

	// Update the context with optimized native gemm micro-kernels and
	// their storage preferences.
	bli_cntx_set_l3_nat_ukrs
	(
	  4,
	  BLIS_GEMM_UKR, BLIS_FLOAT,    bli_sgemm_veclib_as_uker, FALSE,
	  BLIS_GEMM_UKR, BLIS_DOUBLE,   bli_dgemm_veclib_as_uker, FALSE,
	  BLIS_GEMM_UKR, BLIS_SCOMPLEX, bli_cgemm_veclib_as_uker, FALSE,
	  BLIS_GEMM_UKR, BLIS_DCOMPLEX, bli_zgemm_veclib_as_uker, FALSE,
	  cntx
	);

	dim_t m0_s = bli_env_get_var("BLIS_SVE_VECLIB_MR_S", 320);
	dim_t n0_s = bli_env_get_var("BLIS_SVE_VECLIB_NR_S",  16);
	dim_t k0_s = bli_env_get_var("BLIS_SVE_VECLIB_KC_S", 100);
	dim_t m0_d = bli_env_get_var("BLIS_SVE_VECLIB_MR_D", 160);
	dim_t n0_d = bli_env_get_var("BLIS_SVE_VECLIB_NR_D",  16);
	dim_t k0_d = bli_env_get_var("BLIS_SVE_VECLIB_KC_D", 100);
	dim_t m0_c = bli_env_get_var("BLIS_SVE_VECLIB_MR_C", 160);
	dim_t n0_c = bli_env_get_var("BLIS_SVE_VECLIB_NR_C",  16);
	dim_t k0_c = bli_env_get_var("BLIS_SVE_VECLIB_KC_C", 100);
	dim_t m0_z = bli_env_get_var("BLIS_SVE_VECLIB_MR_Z", 160);
	dim_t n0_z = bli_env_get_var("BLIS_SVE_VECLIB_NR_Z",  16);
	dim_t k0_z = bli_env_get_var("BLIS_SVE_VECLIB_KC_Z", 100);

	// Initialize level-3 blocksize objects with architecture-specific values.
	//                                           s      d      c      z
	bli_blksz_init_easy( &blkszs[ BLIS_MR ],  m0_s,  m0_d,  m0_c,  m0_z );
	bli_blksz_init_easy( &blkszs[ BLIS_NR ],  n0_s,  n0_d,  n0_c,  n0_z );
	bli_blksz_init_easy( &blkszs[ BLIS_MC ], 10240, 10240, 10240, 10240 );
	bli_blksz_init_easy( &blkszs[ BLIS_KC ],  k0_s,  k0_d,  k0_c,  k0_z );
	bli_blksz_init_easy( &blkszs[ BLIS_NC ], 10240, 10240, 10240, 10240 );

	// Update the context with the current architecture's register and cache
	// blocksizes (and multiples) for native execution.
	bli_cntx_set_blkszs
	(
	  BLIS_NAT, 5,
	  BLIS_NC, &blkszs[ BLIS_NC ], BLIS_NR,
	  BLIS_KC, &blkszs[ BLIS_KC ], BLIS_KR,
	  BLIS_MC, &blkszs[ BLIS_MC ], BLIS_MR,
	  BLIS_NR, &blkszs[ BLIS_NR ], BLIS_NR,
	  BLIS_MR, &blkszs[ BLIS_MR ], BLIS_MR,
	  cntx
	);
}

