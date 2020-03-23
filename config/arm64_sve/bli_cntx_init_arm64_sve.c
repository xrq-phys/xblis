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

void* get_sve_sgemm_bli_kernel(int m_r, int n_r)
{
    void* kptr = NULL;
#if SVE_VECSIZE == SVE_VECSIZE_VLA
    kptr = (void*) bli_sgemm_armv8a_sve_asm_2vx8;
#elif SVE_VECSIZE == SVE_VECSIZE_256
    kptr = (void*) bli_sgemm_armv8a_sve_asm_16x8;
#else
#endif

    return kptr;
}


void* get_sve_dgemm_bli_kernel(int m_r, int n_r)
{
    void* kptr = NULL;
    int sve_bit_size = get_sve_byte_size()*8;
#if SVE_VECSIZE == SVE_VECSIZE_VLA
    // TODO: More VLA kernels + selection depending on m_r/n_r
    if(sve_bit_size == m_r*64)
    {
        kptr = (void*) bli_dgemm_armv8a_sve_asm_1vx8;
    }
    else if(4*sve_bit_size == m_r*64)
    {
        kptr = (void*) bli_dgemm_armv8a_sve_asm_4vx5;
    }
    else
    {
        if(9 == n_r)
        {
            kptr = (void*) bli_dgemm_armv8a_sve_asm_2vx9;
        }
        else
        {
            kptr = (void*) bli_dgemm_armv8a_sve_asm_2vx8;
        }
    }
#elif SVE_VECSIZE == SVE_VECSIZE_256
    kptr = (void*) bli_dgemm_armv8a_sve_asm_8x10;
#elif SVE_VECSIZE == SVE_VECSIZE_512
    kptr = (void*) bli_dgemm_armv8a_sve_asm_16x10;
#elif SVE_VECSIZE == SVE_VECSIZE_1024
    kptr = (void*) bli_dgemm_armv8a_sve_asm_32x10;
#else
#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)
#pragma message "Vector size: (" STR(SVE_VECSIZE) ")"
#error "Chosen SVE vector size  invalid or not implemented!"
#undef STR
#undef STR_HELPER
#endif

#if SVE_VECSIZE != SVE_VECSIZE_VLA
    if(SVE_VECSIZE != sve_bit_size)
    {
        fprintf(stderr,"Error: Runtime vector size (%d) does not match compile-time size (%d)!\n", sve_bit_size, SVE_VECSIZE);
        kptr = NULL;
        exit(EXIT_FAILURE);
    }
#endif

    return kptr;

    (void) n_r;
}

void* get_sve_zgemm_bli_kernel(int m_r, int n_r)
{
    void* kptr = NULL;

#if SVE_VECSIZE == SVE_VECSIZE_VLA
    kptr = (void*) bli_zgemm_armv8a_sve_asm_2vx4;
#elif SVE_VECSIZE == SVE_VECSIZE_256
    kptr = (void*) bli_zgemm_armv8a_sve_asm_4x4;
#elif SVE_VECSIZE == SVE_VECSIZE_512
    kptr = (void*) bli_zgemm_armv8a_sve_asm_8x4;
#elif SVE_VECSIZE == SVE_VECSIZE_1024
    kptr = (void*) bli_zgemm_armv8a_sve_asm_16x4;
#else
#endif

    return kptr;
}

void bli_cntx_init_arm64_sve( cntx_t* cntx )
{

    dim_t n_vfma = bli_env_get_var("BLIS_SVE_N_VFMA",N_VFMA);
    dim_t l_vfma = bli_env_get_var("BLIS_SVE_L_VFMA",L_VFMA);
    dim_t w_l1   = bli_env_get_var("BLIS_SVE_W_L1",W_L1);
    dim_t n_l1   = bli_env_get_var("BLIS_SVE_N_L1",N_L1);
    dim_t c_l1   = bli_env_get_var("BLIS_SVE_C_L1",C_L1);
    dim_t w_l2   = bli_env_get_var("BLIS_SVE_W_L2",W_L2);
    dim_t n_l2   = bli_env_get_var("BLIS_SVE_N_L2",N_L2);
    dim_t c_l2   = bli_env_get_var("BLIS_SVE_C_L2",C_L2);
    dim_t w_l3   = bli_env_get_var("BLIS_SVE_W_L3",W_L3);
    dim_t n_l3   = bli_env_get_var("BLIS_SVE_N_L3",N_L3);
    dim_t c_l3   = bli_env_get_var("BLIS_SVE_C_L3",C_L3);

    // double
    int S_Data   = 8;
    int simd_size = get_sve_byte_size()/S_Data;

    int m_r_d = (int) ceil(sqrt((double)simd_size*l_vfma*n_vfma)/simd_size)*simd_size;
    int n_r_d = (int) ceil(((double)simd_size*l_vfma*n_vfma)/m_r_d);

    adjust_sve_mr_nr_d(&m_r_d,&n_r_d);

    int k_c_d = (int) (floor(((double)w_l1-1.0)/(1.0+((double)n_r_d)/m_r_d)) * n_l1*c_l1)/(m_r_d*S_Data);

    int C_Ac = w_l2 - 1 - ceil(((double)k_c_d*m_r_d*S_Data)/(c_l2*n_l2));
    int m_c_d = C_Ac * (n_l2 * c_l2)/(k_c_d*S_Data);
    m_c_d -= (m_c_d%m_r_d);

    int C_Bc = w_l3 - 1 - ceil(((double)k_c_d*m_c_d*S_Data)/(c_l3*n_l3));
    int n_c_d = C_Bc * (n_l3 * c_l3)/(k_c_d*S_Data);
    n_c_d -= (n_c_d%n_r_d);
    

    // float
    S_Data   = 4;
    simd_size = get_sve_byte_size()/S_Data;

    int m_r_s = (int) ceil(sqrt((double)simd_size*l_vfma*n_vfma)/simd_size)*simd_size;
    int n_r_s = (int) ceil(((double)simd_size*l_vfma*n_vfma)/m_r_s);

    adjust_sve_mr_nr_s(&m_r_s,&n_r_s);

    int k_c_s = (int) (floor(((double)w_l1-1.0)/(1.0+((double)n_r_s)/m_r_s)) * n_l1*c_l1)/(m_r_s*S_Data);

    C_Ac = w_l2 - 1 - ceil(((double)k_c_s*m_r_s*S_Data)/(c_l2*n_l2));
    int m_c_s = C_Ac * (n_l2 * c_l2)/(k_c_s*S_Data);
    m_c_s -= (m_c_s%m_r_s);

    C_Bc = w_l3 - 1 - ceil(((double)k_c_s*m_c_s*S_Data)/(c_l3*n_l3));
    int n_c_s = C_Bc * (n_l3 * c_l3)/(k_c_s*S_Data);
    n_c_s -= (n_c_s%n_r_s);

    // if kernel not implemented, set everything to -1 -> back to default
    if (m_r_s == -1 || n_r_s == -1)
        k_c_s = m_c_s = n_c_s = -1;

    // double complex
    S_Data   = 16;
    simd_size = get_sve_byte_size()/S_Data;

    int m_r_z = (int) ceil(sqrt((double)simd_size*l_vfma*n_vfma)/simd_size)*simd_size;
    int n_r_z = (int) ceil(((double)simd_size*l_vfma*n_vfma)/m_r_z);

    adjust_sve_mr_nr_z(&m_r_z,&n_r_z);

    int k_c_z = (int) (floor(((double)w_l1-1.0)/(1.0+((double)n_r_z)/m_r_z)) * n_l1*c_l1)/(m_r_z*S_Data);

    C_Ac = w_l2 - 1 - ceil(((double)k_c_z*m_r_z*S_Data)/(c_l2*n_l2));
    int m_c_z = C_Ac * (n_l2 * c_l2)/(k_c_z*S_Data);
    m_c_z -= (m_c_z%m_r_z);

    C_Bc = w_l3 - 1 - ceil(((double)k_c_z*m_c_z*S_Data)/(c_l3*n_l3));
    int n_c_z = C_Bc * (n_l3 * c_l3)/(k_c_z*S_Data);
    n_c_z -= (n_c_z%n_r_z);

	blksz_t blkszs[ BLIS_NUM_BLKSZS ];

	// Set default kernel blocksizes and functions.
	bli_cntx_init_arm64_sve_ref( cntx );

	// -------------------------------------------------------------------------

	// Update the context with optimized native gemm micro-kernels and
	// their storage preferences.
	bli_cntx_set_l3_nat_ukrs
	(
#if SVE_VECSIZE == SVE_VECSIZE_VLA
	  3,
      BLIS_GEMM_UKR, BLIS_FLOAT,    get_sve_sgemm_bli_kernel(m_r_s, n_r_s), FALSE,
	  BLIS_GEMM_UKR, BLIS_DOUBLE,   get_sve_dgemm_bli_kernel(m_r_d, n_r_d), FALSE,
      BLIS_GEMM_UKR, BLIS_DCOMPLEX, get_sve_zgemm_bli_kernel(m_r_z, n_r_z), FALSE,
#elif SVE_VECSIZE == SVE_VECSIZE_256
	  3,
      BLIS_GEMM_UKR, BLIS_FLOAT,    get_sve_sgemm_bli_kernel(m_r_s, n_r_s), FALSE,
	  BLIS_GEMM_UKR, BLIS_DOUBLE,   get_sve_dgemm_bli_kernel(m_r_d, n_r_d), FALSE,
      BLIS_GEMM_UKR, BLIS_DCOMPLEX, get_sve_zgemm_bli_kernel(m_r_z, n_r_z), FALSE,
#else
	  2,
      //BLIS_GEMM_UKR, BLIS_FLOAT,    get_sve_sgemm_bli_kernel(m_r_s,n_r_s), FALSE,
	  BLIS_GEMM_UKR, BLIS_DOUBLE,   get_sve_dgemm_bli_kernel(m_r_d, n_r_d), FALSE,
      BLIS_GEMM_UKR, BLIS_DCOMPLEX, get_sve_zgemm_bli_kernel(m_r_z, n_r_z), FALSE,
#endif
	  cntx
	);

    // TODO: clean this up
//#if SVE_VECSIZE != SVE_VECSIZE_VLA
 //   m_r_z = n_r_z = m_c_z = k_c_z = n_c_z = -1;
//#endif


	// Initialize level-3 blocksize objects with architecture-specific values.
	//                                                  s        d      c       z
	bli_blksz_init_easy( &blkszs[ BLIS_MR ],        m_r_s,   m_r_d,    -1,  m_r_z);
	bli_blksz_init_easy( &blkszs[ BLIS_NR ],        n_r_s,   n_r_d,    -1,  n_r_z);
	bli_blksz_init_easy( &blkszs[ BLIS_MC ],        m_c_s,   m_c_d,    -1,  m_c_z);
	bli_blksz_init_easy( &blkszs[ BLIS_KC ],        k_c_s,   k_c_d,    -1,  k_c_z);
	bli_blksz_init_easy( &blkszs[ BLIS_NC ],        n_c_s,   n_c_d,    -1,  n_c_z);

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
