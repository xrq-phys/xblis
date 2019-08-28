/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2019, Forschunszentrum Juelich

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

GEMM_UKR_PROT( double,   d, gemm_armv8a_sve_asm_12x6 )
GEMM_UKR_PROT( double,   d, gemm_armv8a_sve_asm_8x10 )
GEMM_UKR_PROT( double,   d, gemm_armv8a_sve_asm_16x10 )
GEMM_UKR_PROT( double,   d, gemm_armv8a_sve_asm_32x10 )

/* Vary number of vectors for m_r
 * Vary n_r
 */

GEMM_UKR_PROT(double, d, gemm_armv8a_sve_asm_1vx8)
GEMM_UKR_PROT(double, d, gemm_armv8a_sve_asm_2vx4)
GEMM_UKR_PROT(double, d, gemm_armv8a_sve_asm_2vx8)
//GEMM_UKR_PROT(double, d, gemm_armv8a_sve_asm_4vx4)
//GEMM_UKR_PROT(double, d, gemm_armv8a_sve_asm_4vx8)
//GEMM_UKR_PROT(double, d, gemm_armv8a_sve_asm_4vx16)

GEMM_UKR_PROT(dcomplex, z, gemm_armv8a_sve_asm_4x4)
GEMM_UKR_PROT(dcomplex, z, gemm_armv8a_sve_asm_8x4)
GEMM_UKR_PROT(dcomplex, z, gemm_armv8a_sve_asm_16x4)
GEMM_UKR_PROT(dcomplex, z, gemm_armv8a_sve_asm_2vx4)

void* get_sve_dgemm_bli_kernel(int m_r, int n_r);
