function [ out ] = splitarray( x, nparts )
% Splits x (an array) into nparts of equal size. If not possible,
% the first splits get the extra bits.

  n       = numel(x);
  perpart = idivide( int32(n), nparts, 'floor' );
  xtra    = mod( n, nparts );
  out     = cell( nparts, 1 );

  % #fills the parts, which have 1 item more than the other parts,
  % #so these parts have size perpart + 1
  glidx = 1;
  for i = 1:xtra
    out{i} = x( glidx : (glidx+perpart) );
    glidx  = glidx+perpart+1;
  end

  for i = (xtra+1):nparts
    out{i} = x( glidx : (glidx+perpart-1) );
    glidx  = glidx+perpart;
  end

end

