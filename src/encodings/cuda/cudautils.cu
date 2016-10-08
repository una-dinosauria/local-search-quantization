#include <cuda_runtime.h>
#include <math.h> // for truncf
#include <stdio.h>
#include <curand_kernel.h>

/*******************************************
** Helper functions for CUDA LSQ encoding **
** Written by Julieta Martinez, 2016      **
** jltmtzc@gmail.com                      **
** https://www.cs.ubc.ca/~julm/           **
*******************************************/

// Create a global cuda random state. Used for perturbations.
__device__ void _setup_kernel(
  int n,                // number of codes
  curandState *state) { // memory where the curand state will be initialized

  int x = threadIdx.x + blockIdx.x * blockDim.x;
  if (x < n) {
    // Each thread gets same seed, a different sequence number, no offset
    curand_init((unsigned long long)clock(), x, 0, &state[x]);
  }

}

// Perturb codes using reservoir sampling.
__device__ void _perturb(
  curandState *state,   // random state
  unsigned char *codes, // codes to perturb
  int n,                // number of codes
  int m,                // number of codebooks
  int n_to_perturb ) {  // how many codes we want to perturb

  int x = blockIdx.x;
  int y = threadIdx.y;

  // Copy the codes to local memory
  __shared__ unsigned char local_codes [16];
  local_codes[y] = codes[(x*m) + y];
  __syncthreads();

  if (y == 0) {

    // Get the rand generator state from global memory
    curandState localState = state[x];

    // reservoir sampling loop
    float n_needed = n_to_perturb;
    float n_left   = m;

    float a_rand;

    for (int i=0; i<m; i++) {
      a_rand = curand_uniform( &localState );
      if (a_rand < (n_needed / n_left)) {

        // FIXME hard-coding 256 entries in each codebook
        a_rand = curand_uniform( &localState )*256;
        local_codes[i] = (unsigned char) truncf(a_rand);

        n_needed -= 1.0f;
        if (n_needed <= 0.0f) {
          // Then we have all the numbers we want
          break;
        }
      }

      // Otherwise decrease the number of codes we can change
      n_left -= 1.0f;
    }

    // Save the state back
    state[x] = localState;
  }
  __syncthreads();

  // Save the codes back
  codes[(x*m)+y] = local_codes[y];

}

// Compute cost of a certain quantization. Optimized according to
// http://developer.download.nvidia.com/assets/cuda/files/reduction.pdf. This is
// the method described in the eccvw paper.
__device__ void _veccost(
  float *d_rx,            // data to use (X)
  float *d_codebooks,     // codebooks (C)
  unsigned char *d_codes, // the codes (B)
  float *d_veccost,       // where to save the cost
  int m,                  // number of codebooks
  int n) {                // number of vectors in X

  // FIXME hard-coding 256 entries in each codebook, and 128 dimensions
  const int H = 256; // size of each codebook
  const int D = 128; // dimensionality of each vector

  int x = threadIdx.x + blockIdx.x * blockDim.x; // 1-to-n
  int y = threadIdx.y; // 1-to-128

  if ( x >= n ) {return; }

  /** Copy rx to shared memory **/
  __shared__ float local_rx[ D ];
  local_rx[ y ] = d_rx[ x*D + y ];
  __syncthreads();

  // Loop through each codebook
  for (int i=0; i<m; i++) {
    local_rx[ y ] -= d_codebooks[ H*D*i + d_codes[ n*i + x ]*D + y ];
  }
  // Square all the input
  local_rx[ y ] = local_rx[ y ]*local_rx[ y ];
  __syncthreads();

  // Now reduce by summing along D
  if ( y >= 64 ) { return; }
  local_rx[y] += local_rx[ y+64 ];
  __syncthreads();

  if ( y >= 32 ) { return; }
  local_rx[y] += local_rx[ y+32 ];

  if ( y >= 16 ) { return; }
  local_rx[y] += local_rx[ y+16 ];

  if ( y >= 8 ) { return; }
   local_rx[y] += local_rx[ y+8 ];

  if ( y >= 4 ) { return; }
  local_rx[y] += local_rx[ y+4 ];

  if ( y >= 2 ) { return; }
  local_rx[y] += local_rx[ y+2 ];

  if ( y >= 1 ) { return; }
  local_rx[y] += local_rx[ y+1 ];

  d_veccost[ x ] = local_rx[y];
}


// Compute cost of a certain quantization -- maximize thread workload
// This implementation is preferred to _veccost1 (above) as it is almost as fast
// but does not hard-code the vector dimensionality.
__device__ void _veccost2(
  float *d_rx,            // data to use (X)
  float *d_codebooks,     // codebooks (C)
  unsigned char *d_codes, // the codes (B)
  float *d_veccost,       // where to save the cost
  int d,                  // dimensionality of the data
  int m,                  // number of codebooks
  int n) {                // number of vectors in X

  // FIXME hard-coding 256 entries in each codebook
  const int H = 256; // size of each codebook

  int x = threadIdx.x + blockIdx.x * blockDim.x; // 1-to-n
  int y = threadIdx.y; // 1-to-d

  if ( x >= n ) {return; }

  /** Copy rx to shared memory **/
  extern __shared__ float local_rx[];
  local_rx[ y ] = d_rx[ x*d + y ];
  __syncthreads();

  // Loop through each codebook
  for (int i=0; i<m; i++) {
    local_rx[ y ] -= d_codebooks[ H*d*i + d_codes[ n*i + x ]*d + y ];
  }
  // Square all the inputs
  local_rx[ y ] = local_rx[ y ]*local_rx[ y ];
  __syncthreads();

  // Leave only one thread
  if( y > 0 ) {return; }

  for ( int i=1; i<d; i++ ) {
    local_rx[0] += local_rx[i];
  }

  d_veccost[ x ] = local_rx[0];
}

// Adds a vector to all the columns of a matrix
__device__ void _vec_add( float *matrix, float *vec, int n, int h) {

  int x = threadIdx.x + blockIdx.x * blockDim.x; // 1-to-n
  int y = threadIdx.y; // 1-to-256

  if (x < n) {
    matrix[ x*h + y ] += vec[ y ];
  }
}

/****************************
** Encoding ICM functions **
*****************************/

/** Per-column version. Very non-coalesced **/
__device__ void _condition_icm(
  float *d_ub,
  float *d_bb,
  unsigned char  *d_codek,
  int n, int h) {

  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if ( idx < n ) {
    unsigned char d_codek_idx = d_codek[idx];

    for( int i=0; i<h; i++) {
      d_ub[ idx*h + i ] += d_bb[ d_codek_idx*h + i ];
    }
  }
}

/** per-element version. More coalesced **/
__device__ void _condition_icm2(
  float *d_ub,
  float *d_bb,
  unsigned char  *d_codek,
  int n, int h) {

  int i_idx = threadIdx.x + blockIdx.x * blockDim.x;
  int y_idx = threadIdx.y;

  if ( i_idx < n ) {
    unsigned char d_codek_idx = d_codek[ i_idx ];

    d_ub[ i_idx*h + y_idx ] += d_bb[ d_codek_idx*h + y_idx ];
  }
}

// Version used in final code. Does conditioning and minimization.
__device__ void _condition_icm3(
  float *d_ub,             // unary terms
  float *d_bb,             // binary terms
  unsigned char *d_codek,  // codes
  int conditioning,        // which codebook we are minimizing in ICM
  int m,                   // number of codebooks
  int n) {                 // number of vectors

  // FIXME hard-coding 256 entries in each codebook
  const int H = 256;

  int i_idx = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y;

  if ( i_idx < n ) {
    // Copy unaries to shared memory
    __shared__ float s_ub[ H ];
    __shared__ unsigned char s_ib[ H ];
    s_ub[ y ] = d_ub[ i_idx*H + y ];
    __syncthreads();

    // Conditioning step of ICM
    // Loop through binaries and add them to the unaries
    int j = 0;
    for (int i=0; i<m; i++) {

      if ( i == conditioning ) {
        continue;
      }

      s_ub[ y ] += d_bb[ H*H*j + d_codek[ i_idx + n*i ]*H + y ];
      j++; //
    }

    // Minimization step of ICM
    // Find the minimum after conditioning
    if ( y >= 128 ) { return; }
    __syncthreads();

    bool ismin = s_ub[ y ] > s_ub[ y+128 ];
    if ( ismin ) {
      s_ub[ y ] = s_ub[ y+128 ];
      s_ib[ y ] = y+128;
    } else {
      s_ib[ y ] = (unsigned char) y;
    }
    __syncthreads();

    if ( y >= 64 ) { return; }
    ismin = s_ub[ y ] > s_ub[ y+64 ];
    if ( ismin ) {
      s_ub[ y ] = s_ub[ y+64 ];
      s_ib[ y ] = s_ib[ y+64 ];
    }
    __syncthreads();

    if ( y >= 32 ) { return; }
    ismin = s_ub[ y ] > s_ub[ y+32 ];
    if ( ismin ) {
      s_ub[ y ] = s_ub[ y+32 ];
      s_ib[ y ] = s_ib[ y+32 ];
    }

    if ( y >= 16 ) { return; }
    ismin = s_ub[ y ] > s_ub[ y+16 ];
    if ( ismin ) {
      s_ub[ y ] = s_ub[ y+16 ];
      s_ib[ y ] = s_ib[ y+16 ];
    }

    if ( y >= 8 ) { return; }
    ismin = s_ub[ y ] > s_ub[ y+8 ];
    if ( ismin ) {
      s_ub[ y ] = s_ub[ y+8 ];
      s_ib[ y ] = s_ib[ y+8 ];
    }

    if ( y >= 4 ) { return; }
    ismin = s_ub[ y ] > s_ub[ y+4 ];
    if ( ismin ) {
      s_ub[ y ] = s_ub[ y+4 ];
      s_ib[ y ] = s_ib[ y+4 ];
    }

    if ( y >= 2 ) { return; }
    ismin = s_ub[ y ] > s_ub[ y+2 ];
    if ( ismin ) {
      s_ub[ y ] = s_ub[ y+2 ];
      s_ib[ y ] = s_ib[ y+2 ];
    }

    if ( y >= 1 ) { return; }

    // When we get here, only the 0th thread must be alive
    ismin = s_ub[ y ] > s_ub[ y+1 ];
    if ( ismin ) {
      s_ub[ y ] = s_ub[ y+1 ];
      s_ib[ y ] = s_ib[ y+1 ];
    }

    // Copy the new code back to GPU global memory
    d_codek[ i_idx + n*conditioning ] = s_ib[ y ];
  }
}


// C interface that we can call from Julia
extern "C"
{
  // Initializes the curand state
  void __global__ setup_kernel( int n, void* state ) {
    _setup_kernel( n, (curandState*) state );
  }

  // Perturbs the solution using reservoir sampling
  void __global__ perturb( void* state, unsigned char *codes, int n, int m, int k ) {
    _perturb( (curandState*) state, codes, n, m, k );
  }

  // Veccost optimized following CUDA's reduce guide at
  // http://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
  // requires us to hard-code vector dimensionality though
  void __global__ veccost(
    float *d_rx,
    float *d_codebooks,
    unsigned char *d_codes,
    float *d_veccost,
    int m,
    int n
  ) {
    _veccost(d_rx, d_codebooks, d_codes, d_veccost, m, n);
  }

  // Veccost maximizing load per thread. Slightly slower than veccost1, but does
  // does not require hard-coding of vector dimensionality.
  void __global__ veccost2(
    float *d_rx,
    float *d_codebooks,
    unsigned char *d_codes,
    float *d_veccost,
    int d,
    int m,
    int n
  ) {
    _veccost2(d_rx, d_codebooks, d_codes, d_veccost, d, m, n);
  }

  // Adds a vector to each column of a matrix. Used to add unary terms.
  void __global__ vec_add( float *matrix, float *vec, int n, int h ) {
    _vec_add( matrix, vec, n, h );
  }

  // ICM conditioning
  void __global__ condition_icm( float *d_ub, float *d_bb, unsigned char *d_codek, int n, int h) {
    _condition_icm( d_ub, d_bb, d_codek, n, h );
  }
  void __global__ condition_icm2( float *d_ub, float *d_bb, unsigned char *d_codek, int n, int h) {
    _condition_icm2( d_ub, d_bb, d_codek, n, h );
  }

  // ICM conditioning and minimization
  void __global__ condition_icm3(  float *d_ub, float *d_bb, unsigned char *d_codek, int conditioning, int m, int n) {
    _condition_icm3( d_ub, d_bb, d_codek, conditioning, m, n );
  }

}
