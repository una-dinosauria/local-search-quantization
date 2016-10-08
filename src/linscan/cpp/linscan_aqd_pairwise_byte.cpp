#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <omp.h>

using namespace std;

/* Asymmetric Distance computation (ADC) for a method that passed norms appart */

/**
 * Performs kNN search using a linear scan
 */
void _linscan_aqd_query_extra_byte(
	float* dists,         // out. The estimated distances from the queries to the db
	int*   idx,           // out. The estimated indices of the k nearest neighburs
	unsigned char *codes, // in. d-by-m matrix with the encoded database
	float* queries,       // in. d-by-n matrix with queries
	float* codebooks,     // in. d-by-(m*h) big matrix with concatenated codebooks
	float* dbnorms,       // in. The norms of the database entries
	int nqueries,         // in. Total number of queries
	int ncodes,           // in. Total number of codes
	int m, 				  // in. The number of codebooks to use
	int h, 				  // in. The number of entries per codebook (assuming constant)
	int d,  			  // in. The dimensionality of the data
	int nn)	  			  // in. Number of nearest neighbours to return
{

    // ===== Compute the distance from each query to the database =====
    int total_cb_entries = m*h;

		int buffer_size = (int)1e7;
	  int npairs = min(ncodes, buffer_size + nn);

    #pragma omp parallel for
    for( int i=0; i<nqueries; i++ ) {       // Loop over queries.

			// ===== Create table entries =====
      float* query  = queries + i*d;      		     // Create a pointer for this query
      float* tentry = new float[ total_cb_entries ](); // For storing the table entries

      for( int j=0; j<total_cb_entries; j++ ) {   	// Loop over codebook entries.
          float* centry = codebooks + j*d; 			// Create a pointer for this codebook entry.

          for ( int k=0; k<d; k++) {             // Loop over dimensions.
              tentry[ j ] -= 2*query[k]*centry[k]; // Compute the dot product.
          }
      }

			pair<float,int> * pairs = new pair<float,int>[ npairs ]();

      // ===== Distance computation =====

			// Allocate space for distance-index pairs.
			unsigned char* code = codes;

			int from = 0;
			int normidx = 0;
			while (from < ncodes) {
				int offset = 0;
				if (from > 0) {
					offset = nn;
				}

        for (long j=0 + offset;                // Loop through database codes.
						j< min(ncodes, from + buffer_size + (nn - offset)) - from + offset;
						j++, code += m, normidx++) {

					pairs[j].first  = 0;
          for (int k=0; k<m; k++) {  // Loop through vector codes.
            pairs[j].first += tentry[ h*k + code[k] ];// Table lookup and sum.
          }
          pairs[j].first += dbnorms[ normidx ]; // Add the database norm (+ ||b^2||)

					pairs[j].second = j + 1  + from - offset; // Save the index -- ONE-BASED!
        }

				from = min(ncodes, from + buffer_size + (nn - offset));

				// Sort the distances
        partial_sort(pairs, pairs + nn, pairs + npairs);
			}

      for (long j=0; j<nn; j++) {
          dists[i*nn + j] = pairs[j].first;
          idx  [i*nn + j] = pairs[j].second;
      }
      delete [] pairs;
      delete [] tentry;
    }

    return;
}

extern "C"
{
	void linscan_aqd_query_extra_byte( float* dists, int* idx,
		unsigned char *codes, float* queries, float* codebooks, float* dbnorms,
		int nqueries, int ncodes, int m, int h, int d, int nn ) {

		_linscan_aqd_query_extra_byte( dists, idx,
			codes, queries, codebooks, dbnorms,
			nqueries, ncodes, m, h, d, nn );
	}
}
