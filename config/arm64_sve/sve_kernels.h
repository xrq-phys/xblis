// include after blis.h so that the kernels are known

void* sve_get_override_kernel_s(int kernel_idx);

// Write down the index directly so it's easier to look up
enum kernel_indices_d
{
    dukr_2vx8                =  1,
    dukr_2vx8_ld1rd          =  2,
    dukr_2vx8_ld1rqd         =  3,
    dukr_2vx9                =  4,
    dukr_2vx10_ld1rd         =  5,
    dukr_2vx10_ld1rd_colwise =  6,
    dukr_2vx10_ld1rqd        =  7,
    dukr_2vx12               =  8,
    dukr_2vx12_dup           =  9,
    dukr_2vx12_ld1rd         = 10,
    dukr_2vx12_ld1rqd        = 11,
    dukr_4vx5                = 12,
    dukr_4vx5_ld1rd_colwise  = 13,
    dukr_2vx10_unindexed     = 14
};

void* sve_get_override_kernel_d(int kernel_idx);

void sve_override_mr_nr_d(int kernel_idx, int vec_size, int* m_r, int* n_r);

enum kernel_indices_z
{
    zukr_2vx4                =  1,
    zukr_2vx5                =  2,
    zukr_2vx6                =  3,
    zukr_2vx8                =  4,
    zukr_2vx10               =  5,
    zukr_2vx12               =  6,
};

void* sve_get_override_kernel_z(int kernel_idx);

void sve_override_mr_nr_z(int kernel_idx, int vec_size, int* m_r, int* n_r);

enum kernel_indices_c
{
    cukr_2vx10               =  1,
};

void* sve_get_override_kernel_c(int kernel_idx);

void sve_override_mr_nr_c(int kernel_idx, int vec_size, int* m_r, int* n_r);
