function filter = WienerFilterNonWhite(signal_with_noise, noise_type, noise_params)
    % Adaptive Wiener filter for non-white noise
    %
    % Inputs:
    %   signal_with_noise - Time domain noisy signal
    %   noise_type - 'white', 'ar1', 'pink', 'bandpass', etc.
    %   noise_params - Parameters specific to noise type
    
    Y = fft(signal_with_noise);
    N = length(Y);
    S_Y = abs(Y).^2;
    
    % Generate appropriate noise power spectrum
    S_N = generateAR1NoisePSD(N, noise_params.alpha, noise_params.sigma);
    
    % Wiener filter calculation
    S_X = max(0.01 * S_Y, S_Y - S_N);  % Small spectral floor
    filter = S_X ./ (S_X + S_N);
end

function S_N = generateAR1NoisePSD(N, alpha, sigma)
    % AR(1) noise - theoretical or empirical PSD
    
    % Method 1: Theoretical AR(1) PSD
    % H(ω) = σ² / |1 - α·e^(-jω)|²
    
    freqs = 2*pi*(0:N-1)/N;  % Normalized frequencies
    H_theoretical = sigma^2 ./ abs(1 - alpha * exp(-1j * freqs)).^2;
    
    % Apply windowing effect
    window_power = sum(hanning(N).^2) / N;
    S_N_theoretical = H_theoretical' * window_power;
    
    % Method 2: Empirical approach (more robust)
    num_realizations = 50;  % More realizations for colored noise
    S_N_empirical = zeros(N, 1);
    
    for i = 1:num_realizations
        % Generate AR(1) noise
        ar_noise = zeros(N, 1);
        for t = 2:N
            ar_noise(t) = alpha * ar_noise(t-1) + sigma * randn();
        end
        
        windowed_noise = hanning(N) .* ar_noise;
        S_N_empirical = S_N_empirical + abs(fft(windowed_noise)).^2;
    end
    
    S_N_empirical = S_N_empirical / num_realizations;
    
    % Use empirical estimate (more accurate for windowed case)
    S_N = S_N_empirical;
end