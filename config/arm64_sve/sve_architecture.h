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

#ifndef SVE_ARCHITECTURE_H
#define SVE_ARCHITECTURE_H

// Use vector length agnostic kernels
#define SVE_VECSIZE_VLA 0

// Use fixed-size kernels
#define SVE_VECSIZE_128 128
#define SVE_VECSIZE_256 256
#define SVE_VECSIZE_384 384
#define SVE_VECSIZE_512 512
#define SVE_VECSIZE_640 640
#define SVE_VECSIZE_768 768
#define SVE_VECSIZE_896 896
#define SVE_VECSIZE_1024 1024
#define SVE_VECSIZE_1152 1152
#define SVE_VECSIZE_1280 1280
#define SVE_VECSIZE_1408 1408
#define SVE_VECSIZE_1536 1536
#define SVE_VECSIZE_1664 1664
#define SVE_VECSIZE_1792 1792
#define SVE_VECSIZE_1920 1920
#define SVE_VECSIZE_2048 2048

#define SVE_VECSIZE SVE_VECSIZE_VLA

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

#endif
