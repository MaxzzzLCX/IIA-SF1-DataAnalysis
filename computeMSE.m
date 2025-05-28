
function err = computeMSE(clean, proc, N, L)
    % Corrected MSE calculation that accounts for windowing artifacts
    
    T = numel(clean);
    assert(numel(proc) == T, "Signals must match length");
    
    % Process clean signal through same windowing for fair comparison
    clean_processed = framewiseIdentity(clean, N, L);
    
    % Trim both signals to remove edge effects
    startIdx = L + 1;
    endIdx = T - L;
    
    % Make sure we don't exceed array bounds
    endIdx = min(endIdx, length(clean_processed));
    
    if startIdx >= endIdx
        error('Signals too short for this trim amount');
    end
    
    x_trim = clean_processed(startIdx:endIdx);
    y_trim = proc(startIdx:endIdx);
    
    err = mean((x_trim - y_trim).^2);
end

% Helper function that implements identity processing for fair MSE cmparison
function y_out = framewiseIdentity(x_in, N, overlap)
    y_out = zeros(size(x_in));
    x = buffer(x_in, N, overlap);
    [N_samps, N_frames] = size(x);
    x_w = repmat(hanning(N), 1, N_frames) .* x;
    
    for frame_no = 1:N_frames-2
        X_w = fft(x_w(:, frame_no));
        Y_w = X_w;  % Identity - no processing
        Y_w(N:-1:N/2+2) = conj(Y_w(2:N/2));
        y_w = real(ifft(Y_w));
        
        start_idx = (frame_no-1)*overlap + 1;
        end_idx = start_idx + N - 1;
        
        if end_idx <= length(y_out)
            y_out(start_idx:end_idx) = y_out(start_idx:end_idx) + y_w';
        end
    end
end

%{
function err = computeMSE(clean, proc, N, L)
    % clean — 1×T original signal
    % proc  — 1×T reconstructed signal (same length)
    % overlap — the P frames of buffer overlap

    T = numel(clean);
    assert(numel(proc)==T, "Signals must match length");

    % chop off the first & last 'overlap' samples
    startIdx = L + 1;
    endIdx   = T - L;

    x_trim = clean(startIdx : endIdx);
    y_trim = proc (startIdx : endIdx);

    err = mean( (x_trim - y_trim).^2 );
end
%}



%{
function err = computeMSE(clean, proc, N, overlap)
% computeCentralMSE  MSE on the central region actually filtered
%   err = computeCentralMSE(clean, proc, N, overlap) returns
%     mean((clean(start:end) - proc(start:end)).^2)
%   where start and end are chosen so that:
%     * the first N/2 samples (analysis edge) are removed
%     * the last unprocessed & synthesis-edge samples are removed
%
    clean = clean(:).';  
    proc  = proc ( : ).';  
    T = numel(clean);
    assert(T==numel(proc), "Lengths must match");

    % hop size
    H = N - overlap;
    % how many frames did buffer produce?
    M = ceil((T - overlap)/ H);
    % find the last frame index you actually processed:
    lastProcFrame = M - 2;           % matches your loop 1:(N_frames-2)
    % compute the last sample covered by that frame:
    lastSample = (lastProcFrame-1)*H + N;

    % now trim:
    startIdx = floor(N/2) + 1;       % cut away first half-window
    endIdx   = lastSample  - floor(N/2);  % cut away trailing half-window

    % sanity
    if endIdx <= startIdx
        error("No central region left! Check N, overlap, or loop range.");
    end

    % compute MSE on that slice
    err = mean( (clean(startIdx:endIdx) - proc(startIdx:endIdx)).^2 );
end

%}

%{
function err = computeMSE(clean, proc, N)
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

    half = floor(N/2);
    % trim off the first/last half-window to remove edge distortion
    clean_trim = clean(half+1:end-half);
    proc_trim  =  proc(half+1:end-half);

    err = mean( (clean_trim - proc_trim).^2 );
end

%}