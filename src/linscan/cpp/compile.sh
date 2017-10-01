
# Compile C methods for linear scan

# PQ / OPQ-based
# g++ -O3 -shared -fPIC linscan_aqd.cpp -o linscan_aqd.so -fopenmp
g++ -O3 -shared -fPIC linscan_aqd.cpp -o linscan_aqd.so

# Allocating one extra byte for the dbnorm
# g++ -O3 -shared -fPIC linscan_aqd_pairwise_byte.cpp -o linscan_aqd_pairwise_byte.so -fopenmp
g++ -O3 -shared -fPIC linscan_aqd_pairwise_byte.cpp -o linscan_aqd_pairwise_byte.so
