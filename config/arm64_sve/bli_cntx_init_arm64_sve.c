/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
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

#include "blis.h"

#include "sve_architecture.h"
#include "sve_helpers.h"

void* get_sve_dgemm_bli_kernel(int m_r, int n_r)
{
    void* kptr = NULL;
#if SVE_VECSIZE == SVE_VECSIZE_VLA
    // TODO: More VLA kernels + selection depending on m_r/n_r
    kptr = (void*) bli_dgemm_arm64_sve_asm_2vx8;
#elif SVE_VECSIZE == SVE_VECSIZE_256
    kptr = (void*) bli_dgemm_arm64_sve_8x10;
#elif SVE_VECSIZE == SVE_VECSIZE_512
    kptr = (void*) bli_dgemm_arm64_sve_16x10;
#elif SVE_VECSIZE == SVE_VECSIZE_1024
    kptr = (void*) bli_dgemm_arm64_sve_32x10;
#else
#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)
#error "Chosen SVE vector size (" STR(SVE_VECSIZE) ") invalid or not implemented!"
#undef STR
#undef STR_HELPER
#endif

#if SVE_VECSIZE != SVE_VECSIZE_VLA
    int sve_bit_size = get_sve_byte_size()*8;
    if(SVE_VECSIZE != sve_bit_size)
    {
        fprintf(STDERR,"Error: Runtime vector size (%d) does not match compile-time size (%d)!\n", sve_bit_size, SVE_VECSIZE);
        kptr = NULL;
    }
#endif

    return kptr;

    (void) m_r;
    (void) n_r;
}

void bli_cntx_init_arm64_sve( cntx_t* cntx )
{


    int S_Data   = 8;
    int simd_size = get_sve_byte_size()/S_Data;

    int m_r_d = (int) ceil(sqrt((double)simd_size*L_VFMA*N_VFMA)/simd_size)*simd_size;
    int n_r_d = (int) ceil(((double)simd_size*L_VFMA*N_VFMA)/m_r_d);

    adjust_sve_mr_nr_d(&m_r_d,&n_r_d);

    int k_c_d = (int) (floor(((double)W_L1-1.0)/(1.0+((double)n_r_d)/m_r_d)) * N_L1*C_L1)/(m_r_d*S_Data);

    int C_Ac = W_L2 - 1 - ceil(((double)k_c_d*m_r_d*S_Data)/(C_L2*N_L2));
    int m_c_d = C_Ac * (N_L2 * C_L2)/(k_c_d*S_Data);
    m_c_d -= (m_c_d%m_r_d);

    int C_Bc = W_L3 - 1 - ceil(((double)k_c_d*m_c_d*S_Data)/(C_L3*N_L3));
    int n_c_d = C_Bc * (N_L3 * C_L3)/(k_c_d*S_Data);
    n_c_d -= (n_c_d%n_r_d);
    

    S_Data   = 4;
    simd_size = get_sve_byte_size()/S_Data;

    int m_r_s = (int) ceil(sqrt((double)simd_size*L_VFMA*N_VFMA)/simd_size)*simd_size;
    int n_r_s = (int) ceil(((double)simd_size*L_VFMA*N_VFMA)/m_r_s);

    adjust_sve_mr_nr_s(&m_r_s,&n_r_s);

    int k_c_s = (int) (floor(((double)W_L1-1.0)/(1.0+((double)n_r_s)/m_r_s)) * N_L1*C_L1)/(m_r_s*S_Data);

    C_Ac = W_L2 - 1 - ceil(((double)k_c_s*m_r_s*S_Data)/(C_L2*N_L2));
    int m_c_s = C_Ac * (N_L2 * C_L2)/(k_c_s*S_Data);
    m_c_s -= (m_c_s%m_r_s);

    C_Bc = W_L3 - 1 - ceil(((double)k_c_s*m_c_s*S_Data)/(C_L3*N_L3));
    int n_c_s = C_Bc * (N_L3 * C_L3)/(k_c_s*S_Data);
    n_c_s -= (n_c_s%n_r_s);

	blksz_t blkszs[ BLIS_NUM_BLKSZS ];

	// Set default kernel blocksizes and functions.
	bli_cntx_init_arm64_sve_ref( cntx );

	// -------------------------------------------------------------------------

	// Update the context with optimized native gemm micro-kernels and
	// their storage preferences.
	bli_cntx_set_l3_nat_ukrs
	(
	  1,
      //BLIS_GEMM_UKR, BLIS_FLOAT,    get_sve_sgemm_bli_kernel(m_r_s,n_r_s), FALSE,
	  BLIS_GEMM_UKR, BLIS_DOUBLE,   get_sve_dgemm_bli_kernel(m_r_d,n_r_d), FALSE,
	  cntx
	);

	// Initialize level-3 blocksize objects with architecture-specific values.
	//                                           s      d      c      z
	bli_blksz_init_easy( &blkszs[ BLIS_MR ], m_r_s,   m_r_d,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_NR ], n_r_s,   n_r_d,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_MC ], m_c_s,   m_c_d,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_KC ], k_c_s,   k_c_d,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_NC ], n_c_s,   n_c_d,    -1,    -1 );

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
