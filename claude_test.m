function y_hat = framewiseWienerCorrected(y_in, N, overlap, noise_mag)
    % Corrected Wiener filter implementation
    
    y_hat = zeros(size(y_in));  % Initialize properly
    y = buffer(y_in, N, overlap);
    [N_samps, N_frames] = size(y);
    y_w = repmat(hanning(N), 1, N_frames) .* y;
    
    for frame_no = 1:N_frames-2
        % FFT of windowed frame (only once!)
        Y_w = fft(y_w(:, frame_no));
        
        % Apply Wiener filter
        G = WienerFilterCorrected(Y_w, noise_mag);
        Y_hat_w = G .* Y_w;
        
        % Ensure Hermitian symmetry
        Y_hat_w(N:-1:N/2+2) = conj(Y_hat_w(2:N/2));
        
        % IFFT
        y_hat_w = real(ifft(Y_hat_w));
        
        % Overlap-add (no window compensation needed here)
        start_idx = (frame_no-1)*overlap + 1;
        end_idx = start_idx + N - 1;
        
        if end_idx <= length(y_hat)
            y_hat(start_idx:end_idx) = y_hat(start_idx:end_idx) + y_hat_w';
        end
    end
end

function filter = WienerFilterCorrected(Y, noise_mag)
    % Corrected Wiener filter calculation
    % Input Y is already the FFT of the windowed signal
    
    S_Y = abs(Y).^2;  % Power spectral density of noisy signal
    S_N = noise_mag^2 * ones(size(Y));  % Noise PSD (white noise assumption)
    
    % Improved signal power estimation
    alpha = 1.0;  % Over-subtraction factor (can tune this)
    beta = 0.01;  % Spectral floor factor (can tune this)
    
    S_X = max(beta * S_Y, S_Y - alpha * S_N);  % Signal PSD estimate
    
    filter = S_X ./ (S_X + S_N);  % Wiener filter
end

function err = computeMSE_corrected(clean, proc, N, L)
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

function y_out = framewiseIdentity(x_in, N, overlap)
    % Identity processing for fair MSE comparison
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

%% Test the corrected implementation
function testCorrectedImplementation()
    clc; clearvars;
    
    % Test parameters
    N = 512; L = 256; T = 10000;
    
    % Create test signals
    x_clean = 0.5*cos((1:T)*pi/4) + sin((1:T)*pi/100);
    noise_mag = 0.01;
    rng(42);
    x_noisy = x_clean + noise_mag * randn(size(x_clean));
    
    fprintf('=== TESTING CORRECTED WIENER FILTER ===\n');
    
    % Apply original and corrected filters
    y_original = framewiseWienerOriginal(x_noisy, N, L, noise_mag);
    y_corrected = framewiseWienerCorrected(x_noisy, N, L, noise_mag);
    
    % Fair MSE comparison
    mse_noise = computeMSE_corrected(x_clean, x_noisy, N, L);
    mse_original = computeMSE_corrected(x_clean, y_original, N, L);
    mse_corrected = computeMSE_corrected(x_clean, y_corrected, N, L);
    
    fprintf('MSE - Noisy signal: %.6f\n', mse_noise);
    fprintf('MSE - Original Wiener: %.6f\n', mse_original);
    fprintf('MSE - Corrected Wiener: %.6f\n', mse_corrected);
    
    if mse_corrected < mse_noise
        fprintf('SUCCESS! Corrected Wiener improvement: %.2f dB\n', ...
            10*log10(mse_noise/mse_corrected));
    end
    
    if mse_corrected < mse_original
        fprintf('Corrected vs Original improvement: %.2f dB\n', ...
            10*log10(mse_original/mse_corrected));
    end
    
    % Frequency domain analysis
    fprintf('\n--- Frequency Analysis ---\n');
    
    % Focus on a specific frequency range where we expect improvement
    X_clean = fft(x_clean);
    X_noisy = fft(x_noisy);
    X_corrected = fft(y_corrected(1:length(x_clean)));
    
    % Look at noise reduction in high frequency region (where noise dominates)
    high_freq_bins = round(T*0.1):round(T*0.4);  % High frequency region
    
    noise_power_before = mean(abs(X_noisy(high_freq_bins)).^2);
    noise_power_after = mean(abs(X_corrected(high_freq_bins)).^2);
    
    fprintf('High frequency noise power before: %.6f\n', noise_power_before);
    fprintf('High frequency noise power after: %.6f\n', noise_power_after);
    fprintf('High frequency noise reduction: %.2f dB\n', ...
        10*log10(noise_power_before/noise_power_after));
end

function y_hat = framewiseWienerOriginal(y_in, N, overlap, noise_mag)
    % Your original implementation for comparison
    y_hat = 0*y_in;
    y = buffer(y_in, N, overlap);
    [N_samps, N_frames] = size(y);
    y_w = repmat(hanning(N), 1, N_frames) .* y;
    
    for frame_no = 1:N_frames-2
        Y_w(:, frame_no) = fft(y_w(:, frame_no));
        G = WienerFilterOriginal(y_w(:, frame_no), noise_mag);
        Y_hat_w(:, frame_no) = G .* Y_w(:, frame_no);
        Y_hat_w(N:-1:N/2+2, frame_no) = conj(Y_hat_w(2:N/2, frame_no));
        y_hat_w(:, frame_no) = ifft(Y_hat_w(:, frame_no));
        
        window_compensation = sum(hanning(N)) / N;
        y_hat_w(:, frame_no) = y_hat_w(:, frame_no) / window_compensation;
        
        y_hat((frame_no-1)*overlap+1:(frame_no-1)*overlap+N) = ...
            y_hat((frame_no-1)*overlap+1:(frame_no-1)*overlap+N) + y_hat_w(:, frame_no)';
    end
end

function filter = WienerFilterOriginal(signal_with_noise, noise_mag)
    Y = fft(signal_with_noise);  % Double FFT - this is the bug!
    S_Y = abs(Y).^2;
    S_N = noise_mag^2 * ones(size(Y));
    S_X = max(0, S_Y - S_N);
    filter = S_X ./ (S_X + S_N);
end

% Run the test
testCorrectedImplementation();