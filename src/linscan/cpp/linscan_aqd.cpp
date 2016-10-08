#include <algorithm>
#include <stdio.h>
#include <string.h>
#include "types.h"

using namespace std;

float inline sqr(float d) {
  return d * d;
}

/**
 * Performs kNN using linear scan in Asymmetric quatizer distance
 * (AQD) between codes and queries
 *
 * Inputs:
 *
 *   N: number of codes in the db
 *   NQ: number of queries to be answered
 *   B: number of bits in the db/query codes that should be taken into
 *      account in AQD distance
 *   K: number of results to be returned for each query ie, k in kNN
 *   codes: an array of UINT8 storing the db codes
 *   queries: an array of singles storing the query codes
 *   dim1codes: number of words in the database codes -- most likely
 *              dim1codes = B/8
 *   dim2codes: number of words in the query codes -- most likely
 *              dim2codes = B/8
 *
 * Outputs:
 *
 *   dists: double[K * NQ], squared distance to the nearest elements
 *          for each query
 *   res: int[K * N], stores the ids of K nearest neighbors for each
 *        query (zero-based)
 */
void _linscan_aqd_query(float* dists, UINT32* res, UINT8* codes, float* centers,
                       float* queries, int N, UINT32 NQ, int B, int K,
                       int dim1codes, int dim1queries, int subdim) {
  int B_over_8 = B / 8;
  float * pqueries = queries;
  pair<float,UINT32> * pairs;

  UINT32 * pres = res;
  float * pdists = dists;
  float * dis_from_q = 0;

  memset(dists, 0, K * NQ * sizeof(*dists));

  unsigned int i = 0;

  int buffer_size = (int)1e7;
  int npairs = min(N, buffer_size + K);

#pragma omp parallel shared(i) private(pairs, dis_from_q, pqueries, pres, pdists)
  {
    pairs = new pair<float,UINT32>[npairs];
    dis_from_q = new float[B_over_8 * (1 << 8)];

#pragma omp for
    for (i = 0; i < NQ; i++) {
      pqueries = queries + (UINT64)i * (UINT64)dim1queries;
      pres = res + (UINT64)i * (UINT64)K;
      pdists = dists + (UINT64)i * (UINT64)(K);

      for (int k = 0; k < B_over_8; k++) {
        for (int r = 0; r < (1 << 8); r++) {
          int t = k * (1 << 8) + r;
          dis_from_q[t] = 0;
          for (int s = 0; s < subdim; s++)
            dis_from_q[t] +=
              sqr(centers[t * subdim + s] - pqueries[k * subdim + s]);
        }
      }

      UINT8 *pcodes = codes;
      int from = 0;
      while (from < N) {
        int offset = 0;
        if (from > 0)
          offset = K;
        for (int j=0 + offset;
             j < min(N, from + buffer_size + (K - offset)) - from + offset;
             j++, pcodes += dim1codes) {
          pairs[j].first = 0;
          for (int k = 0; k < B_over_8; k++)
            pairs[j].first += dis_from_q[k * (1<<8) + pcodes[k]];
          pairs[j].second = j + from - offset;
        }
        from = min(N, from + buffer_size + (K - offset));
        partial_sort(pairs, pairs + K, pairs + npairs);
      }

      for (int j = 0; j < K; j++) {
        pres[j] = pairs[j].second;
        pdists[j] = pairs[j].first;
      }
    }
    delete [] pairs;
    delete [] dis_from_q;
  }
}

// C interface that we can call from Julia
extern "C"
{
	void linscan_aqd_query(float* dists, UINT32* res,
		  UINT8* codes, float* centers,
    	float* queries, int N, UINT32 NQ, int B, int K,
        int dim1codes, int dim1queries, int subdim) {
		_linscan_aqd_query( dists, res, codes, centers, queries,
			N, NQ, B, K, dim1codes, dim1queries, subdim);
	}
}
