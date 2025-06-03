
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
