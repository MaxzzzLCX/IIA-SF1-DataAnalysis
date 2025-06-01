
function y_hat = FramewiseFilter(y_in, N, overlap, noise_mag, noise_type, filter_type)
    
    y_hat=0*y_in;
    w = hanning(N)';

    y=buffer(y_in,N,overlap); % partitioning the sigal into length 512 frames with 256 overlap
    [N_samps,N_frames]=size(y);
    y_w=repmat(hanning(N),1,N_frames).*y; % applying hanning window of length N to each frame
    
    % % Gain Mask according to the frequency identified from FFT
    % G = attenFactor * ones(N,1); % Default all frequency as attentuation x0.1 (-20dB)
    % G(keepMask) = 1; % All identified freqs at 1

    for frame_no=1:N_frames-2
        
        Y_w(:,frame_no)=fft(y_w(:,frame_no)); % FFT of windowed frame
        
        if filter_type == "wiener"
            G = WienerFilter(y_w(:,frame_no), noise_mag, noise_type); % Calculate the Wiener filter of this frame of data
            % G = ones(size (Y_w(:,frame_no))); %DEBUG
        elseif filter_type == "spectral_subtraction"
            G = SpectralSubtraction(y_w(:,frame_no), noise_mag, noise_type);
        elseif filter_type == "power_subtraction"
            G = PowerSubtraction(y_w(:,frame_no), noise_mag, noise_type); 
        end

        Y_hat_w(:,frame_no)= G .* Y_w(:,frame_no); % Applying Gain Mask to X
        % Y_w(2:N/8,frame_no)=0.1*X_w(2:N/8,frame_no); % attenuate bin 2 to 64 (N/8) by -20dB (*0.1)
        % Y_w(N/4+1:N/2,frame_no)=0.2*X_w(N/4+1:N/2,frame_no); % attentuate bin 129 to 256 by -14dB (*0.2)
        Y_hat_w(N:-1:N/2+2,frame_no)=conj(Y_hat_w(2:N/2,frame_no)); % make negative freq as conj of bin 2 to 64
                                                            % symmetry keeps signal real valued
        
        y_hat_w(:,frame_no)=ifft(Y_hat_w(:,frame_no)); % inverse FFT
        

        % Overlap and resynthesis all the frames
        y_hat((frame_no-1)*overlap+1:(frame_no-1)*overlap+N)= ...
        y_hat((frame_no-1)*overlap+1:(frame_no-1)*overlap+N)+ y_hat_w(:,frame_no)';
    end
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