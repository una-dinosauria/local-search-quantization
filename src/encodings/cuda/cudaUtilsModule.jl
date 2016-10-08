module CudaUtilsModule

using CUDArt

const ptxdict = Dict()
const mdlist = Array(CuModule, 0)

function mdinit(devlist)
  global ptxdict
  global mdlist
  isempty(mdlist) || error("mdlist is not empty")
  for dev in devlist
    device(dev)
    md = CuModule("src/encodings/cuda/cudautils.ptx", false) # false means it will not be automatically finalized

    # Utils functions
    ptxdict[(dev, "setup_kernel")] = CuFunction(md, "setup_kernel")
    ptxdict[(dev, "perturb")]      = CuFunction(md, "perturb")
    ptxdict[(dev, "veccost")]      = CuFunction(md, "veccost")
    ptxdict[(dev, "veccost2")]     = CuFunction(md, "veccost2")
    ptxdict[(dev, "vec_add")]      = CuFunction(md, "vec_add")

    # ICM functions
    ptxdict[(dev, "condition_icm")]  = CuFunction(md, "condition_icm");
    ptxdict[(dev, "condition_icm2")] = CuFunction(md, "condition_icm2");
    ptxdict[(dev, "condition_icm3")] = CuFunction(md, "condition_icm3");

    push!(mdlist, md)
  end
end

mdclose() = (for md in mdlist; unload(md); end; empty!(mdlist); empty!(ptxdict))

function finit( )
  mdclose()
end

function init( devlist )
  mdinit( devlist )
end

# Accumulates binaries to unaries for icm conditioning
function condition_icm(
  nblocks::Integer, nthreads::Integer,
  d_ub::CudaArray{Cfloat, 2},     # in/out. h-by-n. where we accumulate
  d_bb::CudaArray{Cfloat, 2},     # in. binaries
  d_codek::CudaArray{Cuchar, 1},  # in. n-long. indices into d_bb
  n::Cint,                        # in. size( d_ub, 2 )
  h::Cint)                        # in. size( d_ub, 1 )

  fun = ptxdict[(device(), "condition_icm")];

  CUDArt.launch( fun, nblocks, nthreads,
    (d_ub, d_bb, d_codek, n, h));

  return nothing
end

function condition_icm2(
  nblocks::Integer, nthreads::CUDArt.CudaDims,
  d_ub::CudaArray{Cfloat, 2},     # in/out. h-by-n. where we accumulate
  d_bb::CudaArray{Cfloat, 2},     # in. binaries
  d_codek::CudaArray{Cuchar, 1},  # in. n-long. indices into d_bb
  n::Cint,                        # in. size( d_ub, 2 )
  h::Cint)                        # in. size( d_ub, 1 )

  fun = ptxdict[(device(), "condition_icm2")];

  CUDArt.launch( fun, nblocks, nthreads,
    (d_ub, d_bb, d_codek, n, h));

  return nothing
end

function condition_icm3(
  nblocks::Integer, nthreads::CUDArt.CudaDims,
  d_ub::CudaArray{Cfloat, 2},     # in/out. h-by-n. where we accumulate
  d_bbs::CudaArray{Cfloat, 2},     # in. binaries
  d_codek::CudaArray{Cuchar, 2},  # in. n-long. indices into d_bb
  conditioning::Cint,
  m::Cint,
  n::Cint)                        # in. size( d_ub, 2 )

  fun = ptxdict[(device(), "condition_icm3")];

  CUDArt.launch( fun, nblocks, nthreads,
    (d_ub,
    d_bbs,
    d_codek,
    conditioning,
    m,
    n));

  return nothing
end

function vec_add(
  nblocks::CUDArt.CudaDims, nthreads::CUDArt.CudaDims,
  d_matrix::CudaArray{Cfloat},
  d_vec::CudaArray{Cfloat},
  n::Cint,
  h::Cint)

  fun = ptxdict[( device(), "vec_add" )];
  CUDArt.launch( fun, nblocks, nthreads,
    ( d_matrix, d_vec, n, h ));
end

function veccost(
  nblocks::CUDArt.CudaDims, nthreads::CUDArt.CudaDims,
  d_rx::CudaArray{Cfloat},
  d_codebooks::CudaArray{Cfloat},
  d_codes::CudaArray{Cuchar},
  d_veccost::CudaArray{Cfloat}, # out.
  m::Cint,
  n::Cint)

  fun = ptxdict[( device(), "veccost" )];
  CUDArt.launch( fun, nblocks, nthreads,
    ( d_rx, d_codebooks, d_codes, d_veccost, m, n ));
end

function veccost2(
  nblocks::CUDArt.CudaDims, nthreads::CUDArt.CudaDims,
  d_rx::CudaArray{Cfloat},
  d_codebooks::CudaArray{Cfloat},
  d_codes::CudaArray{Cuchar},
  d_veccost::CudaArray{Cfloat}, # out.
  d::Cint,
  m::Cint,
  n::Cint)

  fun = ptxdict[( device(), "veccost2" )];
  CUDArt.launch( fun, nblocks, nthreads,
    ( d_rx, d_codebooks, d_codes, d_veccost, d, m, n ),
    shmem_bytes=Int( d*sizeof(Cfloat)) );
end

function perturb(
  nblocks::CUDArt.CudaDims, nthreads::CUDArt.CudaDims,
  state::CudaPtr,
  codes::CudaArray{Cuchar},
  n::Cint,
  m::Cint,
  k::Cint)

  fun = ptxdict[( device(), "perturb" )];
  CUDArt.launch( fun, nblocks, nthreads,
    ( state, codes, n, m, k ));
end

function setup_kernel(
  nblocks::CUDArt.CudaDims, nthreads::CUDArt.CudaDims,
  n::Cint,
  state::CudaPtr )

  fun = ptxdict[( device(), "setup_kernel" )];
  CUDArt.launch( fun, nblocks, nthreads,
    ( n, state ));
end

end #cudaUtilsModule
