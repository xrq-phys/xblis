function [ r_val ] = load_data( ...
                                filetemp, ...
                                dirpath, ...
                                thr_str, ...
                                opsupname, ...
                                vartemp, ...
                                opname, ...
                                impl_str ...
                              )

filepath = sprintf( filetemp, dirpath, thr_str, opsupname );
run( filepath )
varname = sprintf( vartemp, thr_str, opname, impl_str );
data = eval( varname );  % e.g. data_st_dgemm_blissup( :, : );

r_val = data;

