// include after blis.h so that the kernels are known


// Write down the index directly so it's easier to look up
enum kernel_indices
{
    ukr_2vx8                =  1,
    ukr_2vx8_ld1rd          =  2,
    ukr_2vx8_ld1rqd         =  3,
    ukr_2vx9                =  4,
    ukr_2vx10_ld1rd         =  5,
    ukr_2vx10_ld1rd_colwise =  6,
    ukr_2vx10_ld1rqd        =  7,
    ukr_2vx12               =  8,
    ukr_2vx12_dup           =  9,
    ukr_2vx12_ld1rqd        = 10,
    ukr_4vx5                = 11,
    ukr_4vx5_ld1rd_colwise  = 12
};

void* sve_get_override_kernel_d(int kernel_idx);

void sve_override_mr_nr_d(int kernel_idx, int vec_size, int* m_r, int* n_r);
