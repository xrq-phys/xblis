"""

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2019, Forschungszentrjum Juelich, Germany

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

"""
#!/usr/bin/env python3

import sys
import argparse
from math import *

def init_parser():

    parser = argparse.ArgumentParser(description="Calculate BLIS block sizes from architectural values")

    parser.add_argument("-V","--simdsize", metavar="simd_size", type=int,
            required=True, help="Size of the vector register in bytes")
    parser.add_argument("-N","--simdengines", metavar="count", type=int,
            required=True, help="Number of SIMD engines")
    parser.add_argument("-l","--fmalatency", metavar="cycles", type=int,
            required=True, help="Latency of an FMA instruction (when chained) in cpu cycles")
    parser.add_argument("-C","--cachelinesize", metavar="cacheline_size", type=int,
            required=True, help="Cache line size (assumed same for all cache levels)")
    parser.add_argument("-L","--l1dsize", metavar="l1_size", type=int,
            required=True, help="L1 data cache size")
    parser.add_argument("-W","--l1dassoc", metavar="l1_associativity", type=int,
            required=True, help="L1 data cache associativity")
    parser.add_argument("-K","--l2size", metavar="l2_size", type=int,
            required=True, help="L2 cache size")
    parser.add_argument("-Q","--l2assoc", metavar="l2_associativity", type=int,
            required=True, help="L2 cache associativity")
    parser.add_argument("-M","--l3size", metavar="l3_size", type=int,
            required=True, help="L3 cache size")
    parser.add_argument("-R","--l3assoc", metavar="l3_associativity", type=int,
            required=True, help="L3 cache associativity")

    return parser

def main():

    parser = init_parser()
    if(len(sys.argv) == 1):
        parser.print_usage()
        return
    args = parser.parse_args()

    # double
    S_data = 8
    L_vfma = args.fmalatency
    N_vfma = args.simdengines
    N_vec  = args.simdsize/S_data
    W_L1   = args.l1dassoc
    C_L1   = args.cachelinesize
    N_L1   = args.l1dsize/W_L1/C_L1

    W_L2   = args.l2assoc
    C_L2   = args.cachelinesize
    N_L2   = args.l2size/W_L2/C_L2

    W_L3   = args.l3assoc
    C_L3   = args.cachelinesize
    N_L3   = args.l3size/W_L3/C_L3

    print(f"Entered parameters:")
    print(f"N_vec:  {N_vec}")
    print(f"L_vfma: {L_vfma}")
    print(f"N_vfma: {N_vfma}")

    m_r =ceil(sqrt(N_vec*L_vfma*N_vfma)/N_vec)*N_vec;
    n_r =ceil((N_vec*L_vfma*N_vfma)/m_r);
    # Reuse: B_r
    # Keep in L1: 1xB_r micropanel, 2xAr 2xCr
    # evict A_r:
    # A_r size: k_c*m_r
    # k_c*m_r*S_data = C_Ar * N_L1*C_L1
    # C_Ar = floor((W_L1-1.0)/(1.0+n_r/m_r)) (see paper)
    k_c =floor((floor((W_L1-1.0)/(1.0+n_r/m_r)) * N_L1*C_L1)/(m_r*S_data));

    # Reuse: A_C
    # Keep in L2: 1x A_C, 2x m_c/m_r Cr microtiles, 2x B_r micropanel
    # Evict B_r microtiles
    # size: k_c*n_r
    # size of A_C: k_c*m_c
    # 
    # 1 CL for C_r, calc how many for B_r, rest for A_C
    C_Ac = W_L2-1 - ceil((k_c*m_r*S_data)/(C_L2*N_L2))
    m_c = C_Ac *(N_L2*C_L2)/(k_c*S_data)


    # Reuse: B_C
    # Keep in L3: 1x B_C, 2x m_c/m_r Cr microtiles, 2x A_C 
    # Evict A_C
    # size B_C: k_c*n_c
    #
    # 1 CL for C_r, calc how many for A_C, rest for B_C
    C_Bc = W_L3-1 - ceil((k_c*m_c*S_data)/(C_L3*N_L3))
    n_c = C_Bc*(N_L3*C_L3)/(k_c*S_data)

    print("Double precision: ")
    print(f"m_r = {m_r}")
    print(f"n_r = {n_r}")
    print(f"k_c = {k_c}")
    print(f"m_c = {m_c}")
    print(f"n_c = {n_c}")


    # single
    S_data = 4
    L_vfma = args.fmalatency
    N_vfma = args.simdengines
    N_vec  = args.simdsize/S_data
    W_L1   = args.l1dassoc
    C_L1   = args.cachelinesize
    N_L1   = args.l1dsize/W_L1/C_L1

    W_L2   = args.l2assoc
    C_L2   = args.cachelinesize
    N_L2   = args.l2size/W_L2/C_L2

    W_L3   = args.l3assoc
    C_L3   = args.cachelinesize
    N_L3   = args.l3size/W_L3/C_L3

    m_r =ceil(sqrt(N_vec*L_vfma*N_vfma)/N_vec)*N_vec;
    n_r =ceil((N_vec*L_vfma*N_vfma)/m_r);
    k_c =(floor((W_L1-1.0)/(1.0+n_r/m_r)) * N_L1*C_L1)/(m_r*S_data);
    m_c =floor(N_L2*C_L2*(W_L2-1)/(k_c*S_data))-n_r;
    n_c =floor(N_L3*C_L3*(W_L3-1)/(k_c*S_data))-m_c;

    print("Single precision: ")
    print(f"m_r = {m_r}")
    print(f"n_r = {n_r}")
    print(f"k_c = {k_c}")
    print(f"m_c = {m_c}")
    print(f"n_c = {n_c}")

if __name__ == "__main__":
    main()
