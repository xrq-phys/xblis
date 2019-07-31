This branch of BLIS library provides micro kernels for Arm SVE extension.

There are currently 4 available configurations:

rhea_r1:
    This configuration uses a vector-length-agnostic micro kernel, which works
    for all SVE vector lengths (128, 256, ..., 2048)
    
    Configure with configuration name 'rhea_r1'. For more information on build system,
    refer to README.md.

    Note: 
    This microkernel uses vector prefetch instructions, which (to this day) are
    not yet supported in gem5. This results in bad performance. If running in gem5,
    one should use other 3 kernels.


cortexa76_sve256:
    This configuration uses a micro kernel, which is specifically written for
    vector length of 256. Running this configuration with different vector 
    length results in error.

    Configure with configuration name 'cortexa76_sve256'. For more information on build system,
    refer to README.md.


cortexa76_sve512:
    Same as above, but for vector length of 512.


cortexa76_sve1024:
    Same as above, but for vector length of 1024.
