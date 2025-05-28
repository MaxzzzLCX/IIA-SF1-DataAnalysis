%{
function y_hat = framewiseWiener(y_in, N, overlap, noise_mag)
    
    y_hat=0*y_in;
    w = hanning(N)';
    ws = sqrt(hanning(N))';

    y=buffer(y_in,N,overlap); % partitioning the sigal into length 512 frames with 256 overlap
    [N_samps,N_frames]=size(y);
    y_w=repmat(ws',1,N_frames).*y; % applying hanning window of length N to each frame
    
    % % Gain Mask according to the frequency identified from FFT
    % G = attenFactor * ones(N,1); % Default all frequency as attentuation x0.1 (-20dB)
    % G(keepMask) = 1; % All identified freqs at 1

    for frame_no=1:N_frames-2
        
        Y_w(:,frame_no)=fft(y_w(:,frame_no)); % FFT of windowed frame
        
        % G = WienerFilter(y_w(:,frame_no), noise_mag, ws); % Calculate the Wiener filter of this frame of data
        G = ones(size (Y_w(:,frame_no))); %DEBUG

        Y_hat_w(:,frame_no)= G .* Y_w(:,frame_no); % Applying Gain Mask to X
        % Y_w(2:N/8,frame_no)=0.1*X_w(2:N/8,frame_no); % attenuate bin 2 to 64 (N/8) by -20dB (*0.1)
        % Y_w(N/4+1:N/2,frame_no)=0.2*X_w(N/4+1:N/2,frame_no); % attentuate bin 129 to 256 by -14dB (*0.2)
        Y_hat_w(N:-1:N/2+2,frame_no)=conj(Y_hat_w(2:N/2,frame_no)); % make negative freq as conj of bin 2 to 64
                                                            % symmetry keeps signal real valued
        y_hat_w(:,frame_no)=real(ifft(Y_hat_w(:,frame_no))); % inverse FFT
                                                             % taking real() because can still have roundoff
        % Overlap and resynthesis all the frames
        y_hat((frame_no-1)*overlap+1:(frame_no-1)*overlap+N)= ...
        y_hat((frame_no-1)*overlap+1:(frame_no-1)*overlap+N)+ ws.*y_hat_w(:,frame_no)';
    end
end

function filter = WienerFilter(signal_with_noise, noise_mag, w)

    Y = fft(signal_with_noise); % FFT of signal + noise

    S_Y = abs(Y).^2;                   % power of audio with noise
    noise_psd = noise_mag^2 * sum(w.^2);   % scalar estimate
    S_N = noise_psd * ones(size(Y)); % power of white noise
    S_X = max(0, S_Y - S_N);           % power of signal
    
    filter = S_X ./ (S_X + S_N);       % Returns a wiener filter of length same as the signal
end

%}


function y_hat = framewiseWiener(y_in, N, overlap, noise_mag, noise_type)
    
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
        
        G = WienerFilter(y_w(:,frame_no), noise_mag, noise_type); % Calculate the Wiener filter of this frame of data
        % G = ones(size (Y_w(:,frame_no))); %DEBUG

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
    
    % NEW: DEBUGGING; Method 1
    %{
    noise_frame = noise_mag * randn(N, 1);  
    windowed_noise = hanning(N) .* noise_frame; % Apply Hanning window (same as your processing)
    % Take FFT (same as your processing)
    windowed_noise_fft = fft(windowed_noise);
    S_N = abs(windowed_noise_fft) .^ 2;
    %}

    % DEBUGGING: Method 2
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
