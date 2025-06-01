%% NEW FUNCTION: Adaptive Framewise Filter
function [y_hat, noise_history] = FramewiseFilterAdaptive(y_in, N, overlap, noise_type, filter_type, window_size, noise_percentile)
    % Framewise filter with adaptive noise estimation using minimum statistics
    
    y_hat = zeros(size(y_in));
    y = buffer(y_in, N, overlap);
    [N_samps, N_frames] = size(y);
    y_w = repmat(hanning(N), 1, N_frames) .* y;
    
    % Initialize adaptive noise tracking
    frame_energies = [];
    noise_estimates = [];
    frame_numbers = [];
    window_size = window_size;        % Look back 50 frames
    noise_percentile = noise_percentile;   % Bottom 10% assumed to be noise
    initial_frames = 10;     % Use first 10 frames to bootstrap
    
    fprintf('Processing %d frames with adaptive noise estimation...\n', N_frames-2);
    
    for frame_no = 1:N_frames-2
        current_frame = y_w(:, frame_no);
        
        % Calculate frame energy
        frame_energy = sum(current_frame.^2) / N;
        frame_energies(end+1) = frame_energy;
        
        % Adaptive noise estimation using minimum statistics
        if frame_no <= initial_frames
            % Bootstrap: use minimum of available frames
            noise_power = prctile(frame_energies, noise_percentile);
        else
            % Use sliding window minimum statistics
            start_idx = max(1, length(frame_energies) - window_size + 1);
            recent_energies = frame_energies(start_idx:end);
            noise_power = prctile(recent_energies, noise_percentile);
        end
        
        noise_mag = sqrt(noise_power);
        
        % Store for plotting
        noise_estimates(end+1) = noise_mag;
        frame_numbers(end+1) = frame_no;
        
        % Apply filtering with current noise estimate
        Y_w = fft(current_frame);
        
        switch lower(filter_type)
            case "wiener"
                G = WienerFilter(current_frame, noise_mag, "white");
            case "spectral_subtraction"
                G = SpectralSubtraction(current_frame, noise_mag, "white");
            otherwise
                error('Unknown filter type: %s', filter_type);
        end
        
        % Apply filter and reconstruct
        Y_hat_w = G .* Y_w;
        Y_hat_w(N:-1:N/2+2) = conj(Y_hat_w(2:N/2));
        y_hat_w = real(ifft(Y_hat_w));
        
        % Overlap-add
        start_idx = (frame_no-1)*overlap + 1;
        end_idx = start_idx + N - 1;
        if end_idx <= length(y_hat)
            y_hat(start_idx:end_idx) = y_hat(start_idx:end_idx) + y_hat_w';
        end
        
        % Print progress every 50 frames
        if mod(frame_no, 50) == 0
            fprintf('Frame %d: noise_mag = %.6f\n', frame_no, noise_mag);
        end
    end
    
    % Return noise history for analysis
    noise_history.frame_numbers = frame_numbers;
    noise_history.noise_estimates = noise_estimates;
    noise_history.frame_energies = frame_energies;
    
    fprintf('Adaptive processing complete. Noise varied from %.6f to %.6f\n', ...
        min(noise_estimates), max(noise_estimates));
end

function filter = WienerFilter(signal_with_noise, noise_mag, noise_type)

    Y = fft(signal_with_noise); % FFT of signal + noise
    N = length(Y);

    S_Y = abs(Y).^2;                   % power of audio with noise
    
    % With averaged estimate:
    if noise_type == "white"
        S_N_total = zeros(N, 1);
        for i = 1:10  % Average 10 realizations
            noise_frame = noise_mag * randn(N, 1);
            windowed_noise = hanning(N) .* noise_frame;
            S_N_total = S_N_total + abs(fft(windowed_noise)).^2;
        end
        S_N = S_N_total / 10;  % Much more stable
    
        
    elseif noise_type == "AR1"
        S_N = generateAR1NoisePSD(N, 0.9, 0.01);
        % S_N = generateAR1NoisePSD(N, noise_params.alpha, noise_params.sigma);
    end
    S_X = max(0, S_Y - S_N);           % power of signal
        
    filter = S_X ./ (S_X + S_N);       % Returns a wiener filter of length same as the signal

end


function filter = SpectralSubtraction(signal_with_noise, noise_mag, noise_type)

    Y = abs(fft(signal_with_noise)); % FFT of signal + noise
    N = length(Y);
    
    % With averaged estimate:
    if noise_type == "white"
        S_N_total = zeros(N, 1);
        for i = 1:10  % Average 10 realizations
            noise_frame = noise_mag * randn(N, 1);
            windowed_noise = hanning(N) .* noise_frame;
            S_N_total = S_N_total + abs(fft(windowed_noise)).^2;
        end
        S_N = S_N_total / 10;  % Much more stable
    elseif noise_type == "AR1"
        S_N = generateAR1NoisePSD(N, 0.9, 0.01);
        % S_N = generateAR1NoisePSD(N, noise_params.alpha, noise_params.sigma);
    end
    
    filter = max(((Y - sqrt(S_N)) ./ Y), 0); % Spectral Reduction Filter 
   
end


function filter = PowerSubtraction(signal_with_noise, noise_mag, noise_type)

    Y = abs(fft(signal_with_noise)); % FFT of signal + noise
    S_Y = Y .^ 2;
    N = length(Y);
    
    % With averaged estimate:
    if noise_type == "white"
        S_N_total = zeros(N, 1);
        for i = 1:10  % Average 10 realizations
            noise_frame = noise_mag * randn(N, 1);
            windowed_noise = hanning(N) .* noise_frame;
            S_N_total = S_N_total + abs(fft(windowed_noise)).^2;
        end
        S_N = S_N_total / 10;  % Much more stable
    elseif noise_type == "AR1"
        S_N = generateAR1NoisePSD(N, 0.9, 0.01);
        % S_N = generateAR1NoisePSD(N, noise_params.alpha, noise_params.sigma);
    end
    
    filter = max(((S_Y - S_N) ./ S_Y), 0) .^ (1/2); % Spectral Reduction Filter 
   
end