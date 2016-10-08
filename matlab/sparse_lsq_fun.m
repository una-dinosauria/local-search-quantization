function [ Aprod ] = sparse_lsq_fun( A, x, n, d, mode )
%SPARSE_LSQ_FUN A function to pass to spgl1 that computes Bc for sparse
% quantization. This is explained in the last paragraph of page 7 of the paper.

[n, mh] = size( A );

if mode == 1
  % x is mh*d long vector.
  xdiv = splitarray( x, d );

  % The product is n*d
  Aprod = zeros( n*d, 1 );

  for i = 1:d
    Aprod( (i-1)*n+1:(i*n) ) = A * xdiv{i};
  end

end

if mode == 2
  % x is n*d long vector.
  xdiv = splitarray( x, d );

  % The product is n*d
  Aprod = zeros( mh*d, 1 );

  At = A';
  for i = 1:d
    Aprod( (i-1)*mh+1:(i*mh) ) = At * xdiv{i};
  end

end

end
