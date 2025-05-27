function err = computeMSE(clean, proc)
% computeMSE  Mean‐squared error between clean and processed signals
%   err = computeMSE(clean, proc, N) trims off the first and last N/2
%   samples (where the STFT/OLA or other windowing introduces edge artifacts),
%   then returns 
%       err = mean( (clean_trim - proc_trim).^2 ).
%
%   Inputs:
%     clean  — 1×T or T×1 "clean" (noise-free) reference
%     proc   — same size as clean, the processed (denoised) signal
%     N      — frame length you used in your STFT (e.g. 512)
%
%   Output:
%     err    — scalar MSE over the central region

    % ensure row-vectors
    clean = clean(:).';  
    proc  = proc ( : ).';  
    assert(numel(clean)==numel(proc), 'Signals must be same length');

    err = mean( (clean - proc).^2 );

    % half = floor(N/2);
    % % trim off the first/last half-window to remove edge distortion
    % clean_trim = clean(half+1:end-half);
    % proc_trim  =  proc(half+1:end-half);

    % err = mean( (clean_trim - proc_trim).^2 );
end