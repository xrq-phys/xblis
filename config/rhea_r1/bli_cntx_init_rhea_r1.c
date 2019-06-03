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

#include "sve_helpers.h"

void* get_sve_dgemm_bli_kernel(int m_r, int n_r)
{
    return (void*) bli_dgemm_rhea_r1_asm_2vx8;
    (void) m_r;
    (void) n_r;
}

void bli_cntx_init_rhea_r1( cntx_t* cntx )
{

    // Technical information:
    // 64kb 4-way associative L1 cache
    // 256kb 8-way L2
    // 1MB 16 way L3
    // 2x 256bit(?) SVE engines
    // Assumption: fmla latency: 5 cycles (Cortex-A72)
    //
    // L_vfma = 5
    // N_vfma = 2
    //
    // m_r = ceil(sqrt(N_vec*L_vfma*N_vfma)/N_vec)*N_vec
    // n_r = ceil(N_vec*L_vfma*N_vfma/m_r)
    //
    // Assumption: Cacheline size = 64 bytes
    //
    // N_L1 = 256
    // C_L1 = 64
    // S_Data = sve_vector_size/2
    //
    // k_c = C_Ar*N_L1*C_L1/(m_r*S_Data)
    //
    // C_Ar <= floor((W_L1 - 1)/(1+n_r/m_r))
    //
    // 256bit:
    //
    // N_vec = 4
    //
    // m_r = 8
    // n_r = 5
    // C_Ar = 1
    // k_c = 256
    //
    // 512bit:
    //
    // N_vec = 8
    //
    // m_r = 16
    // n_r = 5
    // C_Ar = 2
    // k_c = 256
    //
    // 1024bit:
    //
    // N_vec = 16
    //
    // m_r = 16
    // n_r = 10
    // C_Ar = 1
    // k_c = 128
 
    // Number of cache lines in a set
    #define N_L1 256
    // L1 associativity
    #define W_L1 4
    // Cacheline size
    #define C_L1 64
    // FMA latency (chained)
    #define L_VFMA 5
    // Number of SVE engines
    #define N_VFMA 2   

    #define N_L2 512
    #define W_L2 8
    #define C_L2 64

    #define N_L3 16384
    #define W_L3 16
    #define C_L3 64

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
    #undef N_L1
    #undef W_L1
    #undef C_L1
    #undef N_L2
    #undef W_L2
    #undef C_L2
    #undef N_L3
    #undef W_L3
    #undef C_L3
    #undef L_VFMA
    #undef N_VFMA

	blksz_t blkszs[ BLIS_NUM_BLKSZS ];

	// Set default kernel blocksizes and functions.
	bli_cntx_init_rhea_r1_ref( cntx );

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

