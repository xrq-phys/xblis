/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2020, Linaro Limited

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
#include <stdio.h>

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#else
#error "No Arm SVE intrinsics support in compiler"
#endif // __ARM_FEATURE_SVE

// assumption:
//   SVE vector length = 512 bits.

void bli_dpackm_armsve512_asm_10xk
     (
       conj_t           conja,
       pack_t           schema,
       dim_t            cdim_,
       dim_t            n_,
       dim_t            n_max_,
       void*   restrict kappa_,
       void*   restrict a_, inc_t inca_, inc_t lda_,
       void*   restrict p_,              inc_t ldp_,
       cntx_t* restrict cntx
     )
{
    double*       a     = ( double* )a_;
    double*       p     = ( double* )p_;
    double*       kappa = ( double* )kappa_;
    const int64_t cdim  = cdim_;
    const int64_t mnr   = 10;
    const int64_t n     = n_;
    const int64_t n_max = n_max_;
    const int64_t inca  = inca_;
    const int64_t lda   = lda_;
    const int64_t ldp   = ldp_;

    double* restrict alpha1     = a;
    double* restrict alpha1_8   = alpha1 + 8 * inca;
    double* restrict alpha1_p2  = alpha1 + 2 * inca;
    double* restrict alpha1_p4  = alpha1 + 4 * inca;
    double* restrict alpha1_p6  = alpha1 + 6 * inca;
    double* restrict alpha1_m2  = alpha1 - 2 * inca;
    double* restrict alpha1_m4  = alpha1 - 4 * inca;
    double* restrict alpha1_m6  = alpha1 - 6 * inca;
    double* restrict pi1        = p;
    const   svbool_t all_active  = svptrue_b64();
    const   svbool_t active_0to2 = svwhilelt_b64(0, 2);
    const   svbool_t active_0to4 = svwhilelt_b64(0, 4);
    const   svbool_t active_0to6 = svwhilelt_b64(0, 6);
    const   svbool_t active_2to8 = svnot_z(all_active, active_0to2);
    const   svbool_t active_4to8 = svnot_z(all_active, active_0to4);
    const   svbool_t active_6to8 = svnot_z(all_active, active_0to6);
    svfloat64_t      z_a0;
    svfloat64_t      z_a8_fh2;
    svfloat64_t      z_a8_lh6;
    svfloat64_t      z_a16_fh4;
    svfloat64_t      z_a16_lh4;
    svfloat64_t      z_a24_fh6;
    svfloat64_t      z_a24_lh2;
    svfloat64_t      z_a32;
    svuint64_t       z_index;

    // creating index for gather/scatter
    //   with each element as: 0, 1*inca, 2*inca, 3*inca
    z_index = svindex_u64( 0, inca * sizeof( double ) );

    if ( cdim == mnr )
    {
        if ( bli_deq1( *kappa ) )
        {
            if ( inca == 1 )  // continous memory. packA style
            {
                dim_t k = n;
                // 2 pack into 3 case.
                if ( ldp == mnr )
                {
                    for ( ; k >= 4; k -= 4 )
                    {
                        // load 10 continuous elments from *a
                        z_a0      = svld1_f64( all_active, alpha1 );
                        z_a8_fh2  = svld1_vnum_f64( active_0to2, alpha1, 1 );

                        // forward address - 0 to 1
                        alpha1   += lda;
                        alpha1_m2 = alpha1 - 2 * inca;
                        alpha1_p6 = alpha1 + 6 * inca;

                        // load 10 continuous elments from *a, filling last half of z8
                        z_a8_lh6  = svld1_f64( active_2to8, alpha1_m2 );
                        z_a8_fh2  = svadd_f64_z( all_active, z_a8_fh2, z_a8_lh6 );
                        z_a16_fh4 = svld1_f64( active_0to4, alpha1_p6 );

                        // forward address - 1 to 2
                        alpha1   += lda;
                        alpha1_m4 = alpha1 - 4 * inca;
                        alpha1_p4 = alpha1 + 4 * inca;

                        // load 10 continuous elments from *a, filling last half of z16
                        z_a16_lh4 = svld1_f64( active_4to8, alpha1_m4 );
                        z_a16_fh4 = svadd_f64_z( all_active, z_a16_fh4, z_a16_lh4 );
                        z_a24_fh6 = svld1_f64( active_0to6, alpha1_p4 );

                        // forward address - 2 to 3
                        alpha1   += lda;
                        alpha1_m6 = alpha1 - 6 * inca;
                        alpha1_p2 = alpha1 + 2 * inca;

                        // load 10 continuous elments from *a, filling last half of z24
                        z_a24_lh2 = svld1_f64( active_6to8, alpha1_m6 );
                        z_a24_fh6 = svadd_f64_z( all_active, z_a24_fh6, z_a24_lh2 );
                        z_a32     = svld1_f64( all_active, alpha1_p2 );

                        // stored packed data into *p
                        svst1_f64( all_active, pi1, z_a0 );
                        svst1_vnum_f64( all_active, pi1, 1, z_a8_fh2 );
                        svst1_vnum_f64( all_active, pi1, 2, z_a16_fh4 );
                        svst1_vnum_f64( all_active, pi1, 3, z_a24_fh6 );
                        svst1_vnum_f64( all_active, pi1, 4, z_a32 );

                        // forward packing target address
                        alpha1   += lda;
                        alpha1_8  = alpha1 + 8 * inca;
                        pi1      += 4 * ldp;
                    }
                }
                // line-by-line packing case.
                for ( ; k != 0; --k )
                {
                    // load 12 continuous elments from *a
                    z_a0      = svld1_f64( all_active, alpha1 );
                    z_a8_fh2  = svld1_vnum_f64( active_0to2, alpha1, 1 );

                    // store them into *p
                    svst1_f64( all_active, pi1, z_a0 );
                    svst1_vnum_f64( active_0to2, pi1, 1, z_a8_fh2 );

                    alpha1   += lda;
                    alpha1_8  = alpha1 + 8 * inca;
                    pi1      += ldp;
                }
            }
            else  // gather/scatter load/store. packB style
            {
                dim_t k = n;
                if ( ldp == mnr )
                {
                    for ( ; k >= 4; k -= 4 )
                    {
                        // gather load from *a
                        z_a0      = svld1_gather_u64offset_f64( all_active, alpha1, z_index );
                        z_a8_fh2  = svld1_gather_u64offset_f64( active_0to2, alpha1_8, z_index );

                        // forward address - 0 to 1
                        alpha1   += lda;
                        alpha1_m2 = alpha1 - 2 * inca;
                        alpha1_p6 = alpha1 + 6 * inca;

                        // gather load from *a, filling last half of z8
                        z_a8_lh6  = svld1_gather_u64offset_f64( active_2to8, alpha1_m2, z_index );
                        z_a8_fh2  = svadd_f64_z( all_active, z_a8_fh2, z_a8_lh6 );
                        z_a16_fh4 = svld1_gather_u64offset_f64( active_0to4, alpha1_p6, z_index );

                        // forward address - 1 to 2
                        alpha1   += lda;
                        alpha1_m4 = alpha1 - 4 * inca;
                        alpha1_p4 = alpha1 + 4 * inca;

                        // gather load from *a, filling last half of z16
                        z_a16_lh4 = svld1_gather_u64offset_f64( active_4to8, alpha1_m4, z_index );
                        z_a16_fh4 = svadd_f64_z( all_active, z_a16_fh4, z_a16_lh4 );
                        z_a24_fh6 = svld1_gather_u64offset_f64( active_0to6, alpha1_p4, z_index );

                        // forward address - 2 to 3
                        alpha1   += lda;
                        alpha1_m6 = alpha1 - 6 * inca;
                        alpha1_p2 = alpha1 + 2 * inca;

                        // gather load from *a, filling last half of z24
                        z_a24_lh2 = svld1_gather_u64offset_f64( active_6to8, alpha1_m6, z_index );
                        z_a24_fh6 = svadd_f64_z( all_active, z_a24_fh6, z_a24_lh2 );
                        z_a32     = svld1_gather_u64offset_f64( all_active, alpha1_p2, z_index );

                        // stored packed data into *p
                        svst1_f64( all_active, pi1, z_a0 );
                        svst1_vnum_f64( all_active, pi1, 1, z_a8_fh2 );
                        svst1_vnum_f64( all_active, pi1, 2, z_a16_fh4 );
                        svst1_vnum_f64( all_active, pi1, 3, z_a24_fh6 );
                        svst1_vnum_f64( all_active, pi1, 4, z_a32 );

                        // forward packing target address
                        alpha1   += lda;
                        alpha1_8  = alpha1 + 8 * inca;
                        pi1      += 4 * ldp;
                    }
                }
                for ( ; k != 0; --k )
                {
                    // gather load from *a
                    z_a0      = svld1_gather_u64offset_f64( all_active, alpha1, z_index );
                    z_a8_fh2  = svld1_gather_u64offset_f64( active_0to2, alpha1_8, z_index );

                    // scatter store into *p
                    svst1_f64( all_active, pi1, z_a0 );
                    svst1_vnum_f64( active_0to2, pi1, 1, z_a8_fh2 );

                    alpha1   += lda;
                    alpha1_8  = alpha1 + 8 * inca;
                    pi1      += ldp;
                }
            }
        }
        else  // *kappa != 1.0
        {
            // load kappa into vector
            svfloat64_t z_kappa;

            z_kappa = svdup_f64( *kappa );

            if ( inca == 1 )  // continous memory. packA style
            {
                dim_t k = n;
                if ( ldp == mnr )
                {
                    for ( ; k >= 4; k -= 4 )
                    {
                        // load 10 continuous elments from *a
                        z_a0      = svld1_f64( all_active, alpha1 );
                        z_a8_fh2  = svld1_vnum_f64( active_0to2, alpha1, 1 );

                        // forward address - 0 to 1
                        alpha1   += lda;
                        alpha1_m2 = alpha1 - 2 * inca;
                        alpha1_p6 = alpha1 + 6 * inca;

                        // load 10 continuous elments from *a, filling last half of z8
                        z_a8_lh6  = svld1_f64( active_2to8, alpha1_m2 );
                        z_a8_fh2  = svadd_f64_z( all_active, z_a8_fh2, z_a8_lh6 );
                        z_a16_fh4 = svld1_f64( active_0to4, alpha1_p6 );

                        // forward address - 1 to 2
                        alpha1   += lda;
                        alpha1_m4 = alpha1 - 4 * inca;
                        alpha1_p4 = alpha1 + 4 * inca;

                        // load 10 continuous elments from *a, filling last half of z16
                        z_a16_lh4 = svld1_f64( active_4to8, alpha1_m4 );
                        z_a16_fh4 = svadd_f64_z( all_active, z_a16_fh4, z_a16_lh4 );
                        z_a24_fh6 = svld1_f64( active_0to6, alpha1_p4 );

                        // forward address - 2 to 3
                        alpha1   += lda;
                        alpha1_m6 = alpha1 - 6 * inca;
                        alpha1_p2 = alpha1 + 2 * inca;

                        // load 10 continuous elments from *a, filling last half of z24
                        z_a24_lh2 = svld1_f64( active_6to8, alpha1_m6 );
                        z_a24_fh6 = svadd_f64_z( all_active, z_a24_fh6, z_a24_lh2 );
                        z_a32     = svld1_f64( all_active, alpha1_p2 );

                        // multiply by *kappa
                        z_a0      = svmul_lane_f64( z_a0, z_kappa, 0 );
                        z_a8_fh2  = svmul_lane_f64( z_a8_fh2, z_kappa, 0 );
                        z_a16_fh4 = svmul_lane_f64( z_a16_fh4, z_kappa, 0 );
                        z_a24_fh6 = svmul_lane_f64( z_a24_fh6, z_kappa, 0 );
                        z_a32     = svmul_lane_f64( z_a32, z_kappa, 0 );

                        // stored packed data into *p
                        svst1_f64( all_active, pi1, z_a0 );
                        svst1_vnum_f64( all_active, pi1, 1, z_a8_fh2 );
                        svst1_vnum_f64( all_active, pi1, 2, z_a16_fh4 );
                        svst1_vnum_f64( all_active, pi1, 3, z_a24_fh6 );
                        svst1_vnum_f64( all_active, pi1, 4, z_a32 );

                        // forward packing target address
                        alpha1   += lda;
                        alpha1_8  = alpha1 + 8 * inca;
                        pi1      += 4 * ldp;
                    }
                }
                for ( ; k != 0; --k )
                {
                    // load 12 continuous elments from *a
                    z_a0      = svld1_f64( all_active, alpha1 );
                    z_a8_fh2  = svld1_vnum_f64( active_0to2, alpha1, 1 );

                    // multiply by *kappa
                    z_a0      = svmul_lane_f64( z_a0, z_kappa, 0 );
                    z_a8_fh2  = svmul_lane_f64( z_a8_fh2, z_kappa, 0 );

                    // store them into *p
                    svst1_f64( all_active, pi1, z_a0 );
                    svst1_vnum_f64( active_0to2, pi1, 1, z_a8_fh2 );

                    alpha1   += lda;
                    alpha1_8  = alpha1 + 8 * inca;
                    pi1      += ldp;
                }
            }
            else  // gather/scatter load/store. packB style
            {
                dim_t k = n;
                if ( ldp == mnr )
                {
                    for ( ; k >= 4; k -= 4 )
                    {
                        // gather load from *a
                        z_a0      = svld1_gather_u64offset_f64( all_active, alpha1, z_index );
                        z_a8_fh2  = svld1_gather_u64offset_f64( active_0to2, alpha1_8, z_index );

                        // forward address - 0 to 1
                        alpha1   += lda;
                        alpha1_m2 = alpha1 - 2 * inca;
                        alpha1_p6 = alpha1 + 6 * inca;

                        // gather load from *a, filling last half of z8
                        z_a8_lh6  = svld1_gather_u64offset_f64( active_2to8, alpha1_m2, z_index );
                        z_a8_fh2  = svadd_f64_z( all_active, z_a8_fh2, z_a8_lh6 );
                        z_a16_fh4 = svld1_gather_u64offset_f64( active_0to4, alpha1_p6, z_index );

                        // forward address - 1 to 2
                        alpha1   += lda;
                        alpha1_m4 = alpha1 - 4 * inca;
                        alpha1_p4 = alpha1 + 4 * inca;

                        // gather load from *a, filling last half of z16
                        z_a16_lh4 = svld1_gather_u64offset_f64( active_4to8, alpha1_m4, z_index );
                        z_a16_fh4 = svadd_f64_z( all_active, z_a16_fh4, z_a16_lh4 );
                        z_a24_fh6 = svld1_gather_u64offset_f64( active_0to6, alpha1_p4, z_index );

                        // forward address - 2 to 3
                        alpha1   += lda;
                        alpha1_m6 = alpha1 - 6 * inca;
                        alpha1_p2 = alpha1 + 2 * inca;

                        // gather load from *a, filling last half of z24
                        z_a24_lh2 = svld1_gather_u64offset_f64( active_6to8, alpha1_m6, z_index );
                        z_a24_fh6 = svadd_f64_z( all_active, z_a24_fh6, z_a24_lh2 );
                        z_a32     = svld1_gather_u64offset_f64( all_active, alpha1_p2, z_index );

                        // multiply by *kappa
                        z_a0      = svmul_lane_f64( z_a0, z_kappa, 0 );
                        z_a8_fh2  = svmul_lane_f64( z_a8_fh2, z_kappa, 0 );
                        z_a16_fh4 = svmul_lane_f64( z_a16_fh4, z_kappa, 0 );
                        z_a24_fh6 = svmul_lane_f64( z_a24_fh6, z_kappa, 0 );
                        z_a32     = svmul_lane_f64( z_a32, z_kappa, 0 );

                        // stored packed data into *p
                        svst1_f64( all_active, pi1, z_a0 );
                        svst1_vnum_f64( all_active, pi1, 1, z_a8_fh2 );
                        svst1_vnum_f64( all_active, pi1, 2, z_a16_fh4 );
                        svst1_vnum_f64( all_active, pi1, 3, z_a24_fh6 );
                        svst1_vnum_f64( all_active, pi1, 4, z_a32 );

                        // forward packing target address
                        alpha1   += lda;
                        alpha1_8  = alpha1 + 8 * inca;
                        pi1      += 4 * ldp;
                    }
                }
                for ( ; k != 0; --k )
                {
                    // gather load from *a
                    z_a0      = svld1_gather_u64offset_f64( all_active, alpha1, z_index );
                    z_a8_fh2  = svld1_gather_u64offset_f64( active_0to2, alpha1_8, z_index );

                    // multiply by *kappa
                    z_a0      = svmul_lane_f64( z_a0, z_kappa, 0 );
                    z_a8_fh2  = svmul_lane_f64( z_a8_fh2, z_kappa, 0 );

                    // scatter store into *p
                    svst1_f64( all_active, pi1, z_a0 );
                    svst1_vnum_f64( active_0to2, pi1, 1, z_a8_fh2 );

                    alpha1   += lda;
                    alpha1_8  = alpha1 + 8 * inca;
                    pi1      += ldp;
                }
            }
        } // end of if ( *kappa == 1.0 )
    }
    else // if ( cdim < mnr )
    {
        bli_dscal2m_ex
        (
          0,
          BLIS_NONUNIT_DIAG,
          BLIS_DENSE,
          ( trans_t )conja,
          cdim,
          n,
          kappa,
          a, inca, lda,
          p, 1,    ldp,
          cntx,
          NULL
        );

        // if ( cdim < mnr )
        {
            const dim_t      i      = cdim;
            const dim_t      m_edge = mnr - i;
            const dim_t      n_edge = n_max;
            double* restrict p_edge = p + (i  )*1;

            bli_dset0s_mxn
            (
              m_edge,
              n_edge,
              p_edge, 1, ldp
            );
        }
    }

    if ( n < n_max )
    {
        const dim_t      j      = n;
        const dim_t      m_edge = mnr;
        const dim_t      n_edge = n_max - j;
        double* restrict p_edge = p + (j  )*ldp;

        bli_dset0s_mxn
        (
          m_edge,
          n_edge,
          p_edge, 1, ldp
        );
    }
}
