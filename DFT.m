function X = DFT(x)
    x = x(:);                          % column vector length N
    N = numel(x);
    k = (0:N-1).';                     % N×1
    n = 0:N-1;                         % 1×N
    W = exp(-2j*pi*k.*n/N);            % Vandermonde matrix
    X = W * x;                         % forward DFT
end