# Local-search Quantization

This is the code for the papers

* Martinez, J., Clement, J., Hoos H. H. and Little, J. J.:
[*Revisiting additive quantization*](https://www.cs.ubc.ca/~julm/papers/eccv16.pdf),
from ECCV 2016, and
* Martinez, J., Hoos H. H. and Little, J. J.:
[*Solving multi-codebook quantization in the GPU*](https://www.cs.ubc.ca/~julm/papers/eccvw16.pdf),
from VSM (ECCV workshops) 2016.

The code in this repository was mostly written by [Julieta Martinez](http://www.cs.ubc.ca/~julm/) and Joris Clement.

## Dependencies

Our code is mostly written in [Julia](http://julialang.org/), and should run
under version 0.5.2 or later. To get Julia, go to the
[Julia downloads page](http://julialang.org/downloads/) and install the latest
stable release.

We use a number of dependencies that you have to install using
`Pkg.install( "package_name" )`, where `package_name` is

* [HDF5](https://github.com/JuliaIO/HDF5.jl) -- for reading/writing data
* [Distributions](https://github.com/JuliaStats/Distributions.jl) -- for random inits
* [Distances](https://github.com/JuliaStats/Distances.jl) -- for quick distance computation
* [DistributedArrays](https://github.com/JuliaParallel/DistributedArrays.jl) -- for parallelizing in the CPU
* [IterativeSolvers](https://github.com/JuliaLang/IterativeSolvers.jl) -- for [LSQR](https://github.com/JuliaMath/IterativeSolvers.jl/blob/master/src/lsqr.jl); while #1 is open, you will have to also run `Pkg.checkout("IterativeSolvers","master")` to get the bug-free latest version
* [Clustering](https://github.com/JuliaStats/Clustering.jl) -- for k-means

To run encoding in a GPU, you will also need

* [CUDArt](https://github.com/JuliaGPU/CUDArt.jl) -- the CUDA runtime environment
* [CUBLAS](https://github.com/JuliaGPU/CUBLAS.jl) -- for fast matrix multiplication in the GPU
* A CUDA-enabled GPU with compute capability 3.5 or higher. We have tested our code on K40 and Titan X GPUs

Finally, to run the sparse encoding demo you will need Matlab to run the
[SPGL1](https://github.com/mpf/spgl1) solver by van den Berg and Friedlander, as
well as the [MATLAB.jl](https://github.com/JuliaInterop/MATLAB.jl) package to
call Matlab functions from Julia.

## Demos

First, clone this repository and download the [SIFT1M](http://corpus-texmex.irisa.fr/)
dataset. To do so run the following commands:

```bash
git clone git@github.com:jltmtz/local-search-quantization.git
cd local-search-quantization
mkdir data
cd data
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xvzf sift.tar.gz
rm sift.tar.gz
cd ..
```

Also, compile the auxiliary search cpp code:
```bash
cd src/linscan/cpp/
./compile.sh
cd ../../../
```

For expedience, the following demos train on the first 10K vectors of the
SIFT1M dataset. To reproduce the paper results you will have to use the full
training set with 100K vectors.

There are 3 main functionalities showcased in this code:

### 1) Baselines and LSQ demo with encoding in the CPU
Simply run
```bash
julia demos/demo_pq.jl
julia demos/demo_opq.jl
julia demos/demo_lsq.jl
```
This will train PQ, OPQ, and LSQ on a subset of SIFT1M, encode the base set and
compute a recall@N curve. To get better speed in LSQ, you can also run the code
on parallel in multiple cores using
```bash
julia -p n demos/demo_lsq.jl
```
Where `n` is the number of CPU cores on your machine.

### 2) LSQ demo with encoding in the GPU
If you have a CUDA-enabled GPU, you might want to try out encoding in the GPU.

First, compile the CUDA code:

```bash
cd src/encodings/cuda
./compile.sh
cd ../../../
```
and then run
```bash
julia demos/demo_lsq_gpu.jl
```

or

```bash
julia -p n demos/demo_lsq_gpu.jl
```
Where `n` is the number of CPU cores on your machine.

### 3) LSQ demo with sparse encoding

This is very similar to demo #1, but the learned codebooks will be sparse.

First of all, you have to download the [SPGL1](https://github.com/mpf/spgl1)
solver by van den Berg and Friedlander, and add the function that implements
Expression 8 to the package

```bash
cd matlab
git clone git@github.com:mpf/spgl1.git
mv sparse_lsq_fun.m spgl1/
mv splitarray.m spgl1/
cd ..
```

Now you should be able to run the demo

```bash
julia -p n demos/demo_lsq_sparse.jl
```
Where `n` is the number of CPU cores on your machine.

Note that you need MATLAB installed on your computer to run this demo, as well
as well as the [MATLAB.jl](https://github.com/JuliaInterop/MATLAB.jl) package to
call Matlab functions from Julia. Granted, getting all this to work can be a bit
of a pain -- if at this point you (like me) love Julia more than any other
language, please consider porting [SPGL1](https://github.com/mpf/spgl1) to Julia.

## Citing

Thank for your interest in our research! If you find this code useful, please
consider citing our paper

```
Julieta Martinez, Joris Clement, Holger H. Hoos, James J. Little. "Revisiting
additive quantization", ECCV 2016.
```

If you use our GPU implementation please consider citing

```
Julieta Martinez, Holger H. Hoos, James J. Little. "Solving multi-codebook
quantization in the GPU", 4th Workshop on Web-scale Vision and Social Media
(VSM), at ECCV 2016.
```

## FAQ

* **Q:** *What is ChainQ?*

  **A:** ChainQ is a quantization method inspired by [optimized tree quantization (OTQ)](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Babenko_Tree_Quantization_for_2015_CVPR_paper.pdf). Instead of learning the
  dimension splitting and sharing among codebooks (which OTQ finds using Gurobi),
  we simply take the natural splitting and sharing given by contiguous dimensions.
  Therefore, our codebooks form a chain, not a general tree. This means we can
  solve encoding optimally using the Viterbi algorithm.

* **Q:** *LSQ is very slow...?*

  **A:** Compared to PQ and OPQ yes, but (a) it gives much better compression rates,
  and (b) it is much better in quality and speed compared to
  [additive quantization (AQ)](http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Babenko_Additive_Quantization_for_2014_CVPR_paper.pdf) (our most similar baseline). The authors have made the [AQ code available](https://github.com/arbabenko/Quantizations), so you can compare yourself :)


* **Q:** *The code does not reproduce the results of the paper...?*

  **A:** The demos train on 10K vectors and for 10 iterations. To reproduce the
  results of the paper, train with the whole 100K vectors and do it for 100
  iterations. You can also control the number of ILS iterations to use
  for database encoding in the LSQ demos; which corresponds to LSQ-16 and LSQ-32
  in the paper.

* **Q:** *Why do I see all those warnings when I run your code?*

  **A:** Julia 0.5 issues a warning when a method is redefined more than once in
  the Main scope. This is annoying for many people and will disappear in Julia
  0.6 (see https://github.com/JuliaLang/julia/issues/18725)

## Acknowledgments

Some of our evaluation code and our OPQ implementation has been adapted from
[Cartesian k-means](https://github.com/norouzi/ckmeans) by [Mohamad Norouzi](https://github.com/norouzi)
and [optimized product quantization](http://kaiminghe.com/cvpr13/index.html) by [Kaiming He](http://kaiminghe.com/).

## License
MIT
