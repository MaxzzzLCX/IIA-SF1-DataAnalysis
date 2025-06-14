%% SF1 - Data Analysis Week 2

%% Task 2.1 FFT v.s. STFT
clc, clearvars

% (1) STFT Implementation
% input signal (cos and sin) with gaussian noise
x_in=0.5*cos([1:10000]*pi/4)+sin([1:10000]*pi/100)+randn(1,10000); 
y_out=0*x_in;
N=512;
overlap=256; % N/2 overlap length
x=buffer(x_in,N,overlap); % partitioning the sigal into length 512 frames with 256 overlap
[N_samps,N_frames]=size(x);
x_w=repmat(hanning(N),1,N_frames).*x; % applying hanning window of length N to each frame

for frame_no=1:N_frames-2
    X_w(:,frame_no)=fft(x_w(:,frame_no)); % FFT of windowed frame
    Y_w(:,frame_no)=X_w(:,frame_no);
    Y_w(2:N/8,frame_no)=0.1*X_w(2:N/8,frame_no); % attenuate bin 2 to 64 (N/8) by -20dB (*0.1)
    Y_w(N/4+1:N/2,frame_no)=0.2*X_w(N/4+1:N/2,frame_no); % attentuate bin 129 to 256 by -14dB (*0.2)
    Y_w(N:-1:N/2+2,frame_no)=conj(Y_w(2:N/2,frame_no)); % make negative freq as conj of bin 2 to 64
                                                        % symmetry keeps signal real valued
    y_w(:,frame_no)=ifft(Y_w(:,frame_no)); % inverse FFT

    % Overlap and resynthesis all the frames
    y_out((frame_no-1)*overlap+1:(frame_no-1)*overlap+N)=...
    y_out((frame_no-1)*overlap+1:(frame_no-1)*overlap+N)+y_w(:,frame_no)';
end

% Convert frequency bin into frequencies
SignalN = length(x_in);


freqs = linspace(0,pi,SignalN/2+1);
% freqs = (0:(SignalN-1)/2)/SignalN * 2*pi;

figure;

subplot(2,1,1)
% Plot the processed signal spectrum
Y = fftshift((abs(fft(y_out))));
Y_pos = Y(SignalN/2:SignalN);
plot(freqs, Y_pos);
% semilogy(freqs, Y_pos);
ylim([0,3000])
xlim([0,pi])
xlabel('Frequency (bins)');
ylabel('Magnitude');
title('STFT Spectrum of Signal');

% (2) FFT Implementation
x_in=0.5*cos([1:10000]*pi/4)+sin([1:10000]*pi/100)+randn(1,10000); 
N = length(x_in);
w = hanning(N);
% x = x_in .* w'; % apply hanning window
x = x_in; % not apply hanning window



y=0*x;
X =fft(x); % FFT of windowed frame
Y = X;
Y(2:N/8)=0.1*X(2:N/8); % attenuate bin 2 to 64 (N/8) by -20dB (*0.1)
Y(N/4+1:N/2)=0.2*X(N/4+1:N/2); % attentuate bin 129 to 256 by -14dB (*0.2)
Y(N:-1:N/2+2)=conj(Y(2:N/2)); % make negative freq as conj of bin 2 to 64
                                                    % symmetry keeps signal real valued
y =ifft(Y); % inverse FFT

subplot(2,1,2)
% Plot the processed signal spectrum
Y = fftshift((abs(fft(y))));
Y_pos = Y(N/2:N);
plot(freqs, Y_pos);
% semilogy(freqs, Y_pos);
ylim([0,3000])
xlim([0,pi])
xlabel('Frequency (bins)');
ylabel('Magnitude');
title('FFT Spectrum of Output Signal');


% Plot time domain
figure
subplot(3,1,1)
plot(y_out)
title("STFT Signal")
xlabel("Timestep")
ylabel("Amplitude")

subplot(3,1,2)
plot(y)
title("FFT Signal")
xlabel("Timestep")
ylabel("Amplitude")

subplot(3,1,3)
plot(y-y_out)
title("Difference")
xlabel("Timestep")
ylabel("Amplitude")

%%  Time-domain equivalent of the FFT mask
clc, clearvars

% (1) STFT Implementation
% input signal (cos and sin) with gaussian noise
x_in=0.5*cos([1:10000]*pi/4)+sin([1:10000]*pi/100)+randn(1,10000); 
y_out=0*x_in;
N=512;
overlap=256; % N/2 overlap length
x=buffer(x_in,N,overlap); % partitioning the sigal into length 512 frames with 256 overlap
[N_samps,N_frames]=size(x);
x_w=repmat(hanning(N),1,N_frames).*x; % applying hanning window of length N to each frame

for frame_no=1:N_frames-2
    X_w(:,frame_no)=fft(x_w(:,frame_no)); % FFT of windowed frame
    Y_w(:,frame_no)=X_w(:,frame_no);
    Y_w(2:N/8,frame_no)=0.1*X_w(2:N/8,frame_no); % attenuate bin 2 to 64 (N/8) by -20dB (*0.1)
    Y_w(N/4+1:N/2,frame_no)=0.2*X_w(N/4+1:N/2,frame_no); % attentuate bin 129 to 256 by -14dB (*0.2)
    Y_w(N:-1:N/2+2,frame_no)=conj(Y_w(2:N/2,frame_no)); % make negative freq as conj of bin 2 to 64
                                                        % symmetry keeps signal real valued
    y_w(:,frame_no)=ifft(Y_w(:,frame_no)); % inverse FFT

    % Overlap and resynthesis all the frames
    y_out((frame_no-1)*overlap+1:(frame_no-1)*overlap+N)=...
    y_out((frame_no-1)*overlap+1:(frame_no-1)*overlap+N)+y_w(:,frame_no)';
end

% Convert frequency bin into frequencies
SignalN = length(x_in);

freqs = linspace(0,pi,SignalN/2+1);
% freqs = (0:(SignalN-1)/2)/SignalN * 2*pi;

figure;

subplot(2,1,1)
% Plot the processed signal spectrum
Y = fftshift((abs(fft(y_out))));
Y_pos = Y(SignalN/2:SignalN);
plot(freqs, Y_pos);
% semilogy(freqs, Y_pos);
ylim([0,3000])
xlim([0,pi])
xlabel('Frequency (bins)');
ylabel('Magnitude');
title('STFT Filter of Signal');




% ---------------------------------------------------------------
x          = 0.5*cos((1:10000)*pi/4) + sin((1:10000)*pi/100) + randn(1,10000);
N          = 512;                        % frame length used before
signalN    = length(x);

% --- build the spectral mask G[k] --------------------------------
G = ones(N,1);          % default pass-band gain = 1
G( 2       :  N/8)   = 0.1;       %   0  – Fs/8   → ×0.1
G( N/4+1   :  N/2)   = 0.2;       % Fs/4 – Fs/2   → ×0.2
G(N:-1:N/2+2) = conj(G(2:N/2));   % ensure Hermitian symmetry

% --- impulse response (IFFT of the mask) -------------------------
g = ifft(G,'symmetric');           % real length-512 FIR
g = fftshift(g);                   % centre the impulse
g = circshift(g,1);                % make it causal: tap 0 at index 1

% OPTIONAL: visual check of its magnitude response ----------------
%{
f = (0:N-1)/N * pi;                % rad/sample axis
plot(f, abs(G)), grid on
xlabel('\omega'), ylabel('|G(\omega)|')
%}

% --- linear convolution with the whole signal --------------------
y = conv(x, g, 'same');            % 'same' keeps output length = length(x)

freqs = linspace(0,pi,signalN/2+1);
Y = fftshift((abs(fft(y))));
Y_pos = Y(length(Y)/2:length(Y));
subplot(2,1,2)
plot(freqs,Y_pos)
ylim([0,3000])
xlim([0,pi])
xlabel('Frequency (bins)');
ylabel('Magnitude');
title('Time-Domain Filter');




% Plot time domain
figure
% subplot(2,1,1)
% plot(y_out)
% title("STFT Filter")
% xlabel("Timestep")
% ylabel("Amplitude")

plot(y, 'DisplayName', 'Time Domain'); hold on
plot(y_out, 'DisplayName', 'STFT Filter')
title("TIme domain and STFT Comparison")
xlabel("Timestep")
ylabel("Amplitude")
legend('show')




% --- quick comparison with the OLA result ------------------------
% (Assume you have y_stft from your previous code)
% sound([x ; y_stft ; y], fs)      % original / OLA / direct-FIR






%% Task 2.2 - Own Noise Reduction - Ad Hoc Method


% Adapt the given code from Task 2.1 into a Noise Reduction function
% Define the noise reduction function
function y_out = noiseReduction(x_in, N, overlap, keepMask, attenFactor)
    
    y_out=0*x_in;

    x=buffer(x_in,N,overlap); % partitioning the sigal into length 512 frames with 256 overlap
    [N_samps,N_frames]=size(x);
    x_w=repmat(hanning(N),1,N_frames).*x; % applying hanning window of length N to each frame
    
    % Gain Mask according to the frequency identified from FFT
    G = attenFactor * ones(N,1); % Default all frequency as attentuation x0.1 (-20dB)
    G(keepMask) = 1; % All identified freqs at 1


    for frame_no=1:N_frames-2
        
        X_w(:,frame_no)=fft(x_w(:,frame_no)); % FFT of windowed frame
        Y_w(:,frame_no)=G .* X_w(:,frame_no); % Applying Gain Mask to X
        % Y_w(2:N/8,frame_no)=0.1*X_w(2:N/8,frame_no); % attenuate bin 2 to 64 (N/8) by -20dB (*0.1)
        % Y_w(N/4+1:N/2,frame_no)=0.2*X_w(N/4+1:N/2,frame_no); % attentuate bin 129 to 256 by -14dB (*0.2)
        Y_w(N:-1:N/2+2,frame_no)=conj(Y_w(2:N/2,frame_no)); % make negative freq as conj of bin 2 to 64
                                                            % symmetry keeps signal real valued
        y_w(:,frame_no)=ifft(Y_w(:,frame_no)); % inverse FFT
    
        % Overlap and resynthesis all the frames
        y_out((frame_no-1)*overlap+1:(frame_no-1)*overlap+N)= ...
        y_out((frame_no-1)*overlap+1:(frame_no-1)*overlap+N)+y_w(:,frame_no)';
    end

end

function keepMask = buildKeepMask(peaksHz, fs, N, bwBins)
% peaksHz : vector of peak frequencies (Hz)
% fs      : sample rate
% N       : frame length (same N as FFT)
% bwBins  : half-bandwidth in bins to keep around each peak (e.g. 1)
% --------------------------------------------------------------------
    if nargin < 4, bwBins = 1; end

    keepMask = false(N,1);                     % start with all zeros
    binFreq  = (0:N-1).' * fs / N;             % Hz axis (one full period)

    for f = peaksHz(:).'
        % wrap frequency into 0…fs range
        f = mod(f, fs);
        % nearest bin index
        [~,k] = min(abs(binFreq-f));
        % protect ±bwBins bins around that peak
        rng = mod((k-bwBins):(k+bwBins), N) + 1;
        keepMask(rng) = true;
    end
end



clc, clearvars
[data, rate] = audioread("/Users/maxlyu/Desktop/Part IIA/SF1/Audio examples for weeks 1-2-20250516/f1lcapae.wav");
% Play the audio signal
% sound(data, rate)

%{
Testing for "naive" noise reduction from task2.1

% Picks a pure note from audio, apply Hamming window
segment = data(1:60000); 
subplot(1,3,1)
plot(segment)
ylim([-0.3,0.3])

subplot(1,3,2)
segment_noise = segment + 0.1*randn(size(segment));
plot(segment_noise);
ylim([-0.3,0.3])
% sound(segment_noise, rate);

subplot(1,3,3)
% Reshape input signal into (1,N) shape for noiseReduction algorithm
segment_noise = segment_noise';
segment_reduced = noiseReduction(segment_noise, 512, 256);
plot(segment_reduced);
ylim([-0.3,0.3])

sound(segment_noise, rate);
sound(segment_reduced, rate);

%}

segment = data(1:20000)'; % reshape for noiseReduction input shape
sound(segment, rate)
segment_noise = segment + 0.1*randn(size(segment));
% segment_reduced = noiseReduction(segment_noise, 512, 256, keepMask, 0.1);


% Do FFT on signals, signal+noise, signal_reduced_noise
SignalN = length(segment); % length of signal
fft_sig = fftshift(abs(fft(segment)));
fft_noise = fftshift(abs(fft(segment_noise)));
% fft_reduced = fftshift(abs(fft(segment_reduced)));

fft_sig_pos = fft_sig(SignalN/2:SignalN);
fft_noise_pos = fft_noise(SignalN/2:SignalN);
% fft_reduced_pos = fft_reduced(SignalN/2:SignalN);

freqs = (0:length(fft_noise_pos)-1) * rate / SignalN;


minPkHeight = max(fft_noise_pos)*0.25;
minPkDist = 1*SignalN/rate;
[peaks,locs] = findpeaks( ...
        fft_noise_pos, ...
        'MinPeakHeight',  minPkHeight, ...
        'MinPeakDistance',minPkDist);

peaksHz = freqs(locs); % Find the freq. of the peaks in Hz

% Calculate the Gain Mask needed for these peak frequency components
keepMask = buildKeepMask(peaksHz, rate, 512, 1);

segment_reduced = noiseReduction(segment_noise, 512, 256, keepMask, 0.1);
fft_reduced = fftshift(abs(fft(segment_reduced)));
fft_reduced_pos = fft_reduced(SignalN/2:SignalN);


subplot(1,3,1)
plot(fft_sig_pos)
ylim([0,300])

subplot(1,3,2)
plot(fft_noise_pos), hold on
% plot(locs, peaks, 'rx', 'MarkerSize',8, 'LineWidth',1.5)
ylim([0,300])

subplot(1,3,3)
plot(fft_reduced_pos)
ylim([0,300])



%% Task 2.2 - Wiener Filter

clc, clearvars
[data, rate] = audioread("/Users/maxlyu/Desktop/Part IIA/SF1/Audio examples for weeks 1-2-20250516/f1lcapae.wav");
% Play the audio signal
% sound(data, rate)


x = data(1:20000)'; % reshape for noiseReduction input shape
sound(x, rate)

% Noise Magnitude
noise_mag = 0.01;
y = x + noise_mag * randn(size(x)); % audio with noise added


% Do FFT on signals, signal+noise
SignalN = length(x); 
X = fft(x);       % FFT of pure signal
Y = fft(y); % FFT of signal + noise

S_Y = abs(Y).^2;                   % power of audio with noise
S_N = noise_mag^2 * ones(size(Y)); % power of white noise
S_X = max(0, S_Y - S_N);           % power of signal

wienerFilter = S_X ./ (S_X + S_N);

X_hat = wienerFilter .* Y;

x_hat = real(ifft(X_hat));

% Reformat the FFT for plotting
X_pos = fftshift(abs(X)); X_pos = X_pos(SignalN/2:SignalN);
Y_pos = fftshift(abs(Y)); Y_pos = Y_pos(SignalN/2:SignalN);
X_hat_pos = fftshift(abs(X_hat)); X_hat_pos = X_hat_pos(SignalN/2:SignalN);

freqs = (0:length(X_pos)-1) * rate / SignalN;

subplot(1,3,1)
plot(freqs, X_pos)
title("Signal of audio before noise")
ylim([0,300])

subplot(1,3,2)
plot(freqs, Y_pos)
title("Signal with noise")
ylim([0,300])

subplot(1,3,3)
plot(freqs, X_hat_pos)
title("Signal after filtering")
ylim([0,300])


%% Task 2.2 Wiener Filter in STFT (Frame-wise)

clc, clearvars
[data, rate] = audioread("/Users/maxlyu/Desktop/Part IIA/SF1/Audio examples for weeks 1-2-20250516/f1lcapae.wav");

N = 512; L = 256; % 512 Frame size, 256 Overlap
noise_type = "white";


SignalN = 30000;
x = data(1:SignalN)'; % reshape for noiseReduction input shape
% sound(x, rate)

noise_mag = 0.05; % Noise Magnitude
y = x + noise_mag * randn(size(x)); % audio with noise added


% Apply the frame-wise Wiener filter to the noisy signal
y_hat = framewiseWiener(y, 512, 256, noise_mag, noise_type); % see framewiseWiener.m

% Play the filtered audio signal
% sound(y_hat, rate);

% Plot the original and filtered signals for comparison
figure;
subplot(3,1,1);
plot(x);
title('Original Signal (No Noise)');
xlabel('Sample Index');
ylabel('Amplitude'); ylim([-0.5,0.5])

subplot(3,1,2);
plot(y);
title('Signal with Noise');
xlabel('Sample Index');
ylabel('Amplitude'); ylim([-0.5,0.5])

subplot(3,1,3);
plot(y_hat);
title('Filtered Signal');
xlabel('Time Index');
ylabel('Amplitude'); ylim([-0.5,0.5])

sgtitle(sprintf('Frame-wise Wiener Filter, N=%d, L=%d', N, L))

% Plot the original and filtered signals for comparison
figure
% title("Frame-wise Winer Filter, N=")




X = fftshift(abs(fft(x)));
X_pos = X(length(X)/2:length(X));
freqs = (0:length(X_pos)-1) * rate / SignalN;

subplot(3,1,1);
X_dB = 20*log10(X_pos);
% plot(freqs, X_pos);
plot(freqs, X_dB)
title('Original Signal (No Noise)');
xlabel('Frequency (Hz)');
ylabel('Amplitude (dB)');

Y = fftshift(abs(fft(y)));
Y_pos = Y(length(Y)/2:length(Y));
Y_dB = 20*log10(Y_pos);
subplot(3,1,2);
% plot(freqs, Y_pos);
plot(freqs, Y_dB)
title('Signal with Noise');
xlabel('Frequency (Hz)');
ylabel('Amplitude (dB)');

Y_hat = fftshift(abs(fft(y_hat)));
Y_hat_pos = Y(length(Y)/2:length(Y_hat));
Y_hat_dB = 20*log10(Y_hat_pos);
subplot(3,1,3);
% plot(freqs, Y_hat_pos);
plot(freqs, Y_hat_dB)
title('Filtered Signal');
xlabel('Frequency (Hz)');
ylabel('Amplitude (dB)');

sgtitle(sprintf('Frame-wise Wiener Filter, N=%d, L=%d', N, L))

error_noise = computeMSE(x, y, N, L);
error_reduced = computeMSE(x, y_hat, N, L);
fprintf("MSE of Noisy Signal %.8f\n", error_noise)
fprintf("MSE of Filtered Signal %.8f\n", error_reduced)





%% Task 2.2 Non-White Noise - Original Approach

clc, clearvars
[data, rate] = audioread("/Users/maxlyu/Desktop/Part IIA/SF1/Audio examples for weeks 1-2-20250516/f1lcapae.wav");

N = 512; L = 256; % 512 Frame size, 256 Overlap


SignalN = 30000;
x = data(1:SignalN)'; % reshape for noiseReduction input shape
% sound(x, rate)

noise_mag = 0.05; % Noise Magnitude
non_white_noise = generateAR1Noise(SignalN, 0.9, 0.03);
y = x + non_white_noise; % audio with noise added


% Apply the frame-wise Wiener filter to the noisy signal
y_hat = framewiseWiener(y, 512, 256, noise_mag, "AR1"); % see framewiseWiener.m
% y_hat = FramewiseFilter(y, N, L, noise_mag, "AR1", "spectral_subtraction");

% Play the filtered audio signal
% sound(y_hat, rate);

% Plot the original and filtered signals for comparison
figure;
subplot(3,1,1);
plot(x);
title('Original Signal (No Noise)');
xlabel('Sample Index');
ylabel('Amplitude'); ylim([-0.5,0.5])

subplot(3,1,2);
plot(y);
title('Signal with Noise');
xlabel('Sample Index');
ylabel('Amplitude'); ylim([-0.5,0.5])

subplot(3,1,3);
plot(y_hat);
title('Filtered Signal');
xlabel('Time Index');
ylabel('Amplitude'); ylim([-0.5,0.5])

sgtitle(sprintf('Frame-wise Wiener Filter on AR(1) Noise(\\alpha=%.1f,\\sigma=%.2f), N=%d, L=%d',0.9,0.01, N, L))

% Plot the original and filtered signals for comparison
figure
% title("Frame-wise Winer Filter, N=")




X = fftshift(abs(fft(x)));
X_pos = X(length(X)/2:length(X));
freqs = (0:length(X_pos)-1) * rate / SignalN;

subplot(3,1,1);
X_dB = 20*log10(X_pos);
% plot(freqs, X_pos);
plot(freqs, X_dB)
title('Original Signal (No Noise)');
xlabel('Frequency (Hz)');
ylabel('Amplitude (dB)');

Y = fftshift(abs(fft(y)));
Y_pos = Y(length(Y)/2:length(Y));
Y_dB = 20*log10(Y_pos);
subplot(3,1,2);
% plot(freqs, Y_pos);
plot(freqs, Y_dB)
title('Signal with Noise');
xlabel('Frequency (Hz)');
ylabel('Amplitude (dB)');

Y_hat = fftshift(abs(fft(y_hat)));
Y_hat_pos = Y(length(Y)/2:length(Y_hat));
Y_hat_dB = 20*log10(Y_hat_pos);
subplot(3,1,3);
% plot(freqs, Y_hat_pos);
plot(freqs, Y_hat_dB)
title('Filtered Signal');
xlabel('Frequency (Hz)');
ylabel('Amplitude (dB)');

sgtitle(sprintf('Frame-wise Wiener Filter on AR(1) Noise(\\alpha=%.1f,\\sigma=%.2f), N=%d, L=%d',0.9,0.01, N, L))

error_noise = computeMSE(x, y, N, L);
error_reduced = computeMSE(x, y_hat, N, L);
fprintf("MSE of Noisy Signal %.8f\n", error_noise)
fprintf("MSE of Filtered Signal %.8f\n", error_reduced)


%% Task 2.2 Non-White Noise - Adaptive Scheme

clc, clearvars
[data, rate] = audioread("/Users/maxlyu/Desktop/Part IIA/SF1/Audio examples for weeks 1-2-20250516/f1lcapae.wav");

N = 512; L = 256; % 512 Frame size, 256 Overlap
alpha = 0.9; sigma = 0.03;


SignalN = 30000;
x = data(1:SignalN)'; % reshape for noiseReduction input shape
% sound(x, rate)

noise_mag = 0.05; % Noise Magnitude
non_white_noise = generateAR1Noise(SignalN, alpha, sigma);
y = x + non_white_noise; % audio with noise added


% Apply the frame-wise Wiener filter to the noisy signal
y_hat = framewiseWiener(y, 512, 256, noise_mag, "AR1"); % see framewiseWiener.m
% y_hat = FramewiseFilterAdaptive(y, N, L, "AR1", "spectral_subtraction", 30, 10); % see FramewiseFilterAdaptive.m

% Play the filtered audio signal
% sound(y_hat, rate);

% Plot the original and filtered signals for comparison
figure;
subplot(3,1,1);
plot(x);
title('Original Signal (No Noise)');
xlabel('Sample Index');
ylabel('Amplitude'); ylim([-0.5,0.5])

subplot(3,1,2);
plot(y);
title('Signal with Noise');
xlabel('Sample Index');
ylabel('Amplitude'); ylim([-0.5,0.5])

subplot(3,1,3);
plot(y_hat);
title('Filtered Signal');
xlabel('Time Index');
ylabel('Amplitude'); ylim([-0.5,0.5])

sgtitle(sprintf('Frame-wise Winer Filter on AR(1) Noise(\\alpha=%.1f,\\sigma=%.2f), N=%d, L=%d',0.9,0.01, N, L))

% Plot the original and filtered signals for comparison
figure
% title("Frame-wise Winer Filter, N=")




X = fftshift(abs(fft(x)));
X_pos = X(length(X)/2:length(X));
freqs = (0:length(X_pos)-1) * rate / SignalN;

subplot(3,1,1);
X_dB = 20*log10(X_pos);
% plot(freqs, X_pos);
plot(freqs, X_dB)
title('Original Signal (No Noise)');
xlabel('Frequency (Hz)');
ylabel('Amplitude (dB)');

Y = fftshift(abs(fft(y)));
Y_pos = Y(length(Y)/2:length(Y));
Y_dB = 20*log10(Y_pos);
subplot(3,1,2);
% plot(freqs, Y_pos);
plot(freqs, Y_dB)
title('Signal with Noise');
xlabel('Frequency (Hz)');
ylabel('Amplitude (dB)');

Y_hat = fftshift(abs(fft(y_hat)));
Y_hat_pos = Y(length(Y)/2:length(Y_hat));
Y_hat_dB = 20*log10(Y_hat_pos);
subplot(3,1,3);
% plot(freqs, Y_hat_pos);
plot(freqs, Y_hat_dB)
title('Filtered Signal');
xlabel('Frequency (Hz)');
ylabel('Amplitude (dB)');

sgtitle(sprintf('Frame-wise Winer Filter on AR(1) Noise(\\alpha=%.1f,\\sigma=%.2f), N=%d, L=%d',0.9,0.01, N, L))

error_noise = computeMSE(x, y, N, L);
error_reduced = computeMSE(x, y_hat, N, L);
fprintf("MSE of Noisy Signal %.8f\n", error_noise)
fprintf("MSE of Filtered Signal %.8f\n", error_reduced)

%% Task 2.2 Comparing Different Frame-wise Filters in STFT, with new funciton

clc, clearvars
[data, rate] = audioread("/Users/maxlyu/Desktop/Part IIA/SF1/Audio examples for weeks 1-2-20250516/f1lcapae.wav");

N = 256; L = 224; % 512 Frame size, 256 Overlap
noise_type = "white";
filter_type = "wiener";


SignalN = 30000;
x = data(1:SignalN)'; % reshape for noiseReduction input shape
% sound(x, rate)

noise_mag = 0.05; % Noise Magnitude
y = x + noise_mag * randn(size(x)); % audio with noise added


% Apply the frame-wise Wiener filter to the noisy signal
y_hat = FramewiseFilter(y, 512, 256, noise_mag, noise_type, filter_type); % see framewiseWiener.m

% Play the filtered audio signal
% sound(y_hat, rate);

% Plot the original and filtered signals for comparison
figure;
subplot(3,1,1);
plot(x);
title('Original Signal (No Noise)');
xlabel('Sample Index');
ylabel('Amplitude'); ylim([-0.5,0.5])

subplot(3,1,2);
plot(y);
title('Signal with Noise');
xlabel('Sample Index');
ylabel('Amplitude'); ylim([-0.5,0.5])

subplot(3,1,3);
plot(y_hat);
title('Filtered Signal');
xlabel('Time Index');
ylabel('Amplitude'); ylim([-0.5,0.5])

sgtitle(sprintf('Frame-wise Wiener Filter, N=%d, L=%d', N, L))

% Plot the original and filtered signals for comparison
figure
% title("Frame-wise Winer Filter, N=")




X = fftshift(abs(fft(x)));
X_pos = X(length(X)/2:length(X));
freqs = (0:length(X_pos)-1) * rate / SignalN;

subplot(3,1,1);
X_dB = 20*log10(X_pos);
% plot(freqs, X_pos);
plot(freqs, X_dB)
title('Original Signal (No Noise)');
xlabel('Frequency (Hz)');
ylabel('Amplitude (dB)');

Y = fftshift(abs(fft(y)));
Y_pos = Y(length(Y)/2:length(Y));
Y_dB = 20*log10(Y_pos);
subplot(3,1,2);
% plot(freqs, Y_pos);
plot(freqs, Y_dB)
title('Signal with Noise');
xlabel('Frequency (Hz)');
ylabel('Amplitude (dB)');

Y_hat = fftshift(abs(fft(y_hat)));
Y_hat_pos = Y(length(Y)/2:length(Y_hat));
Y_hat_dB = 20*log10(Y_hat_pos);
subplot(3,1,3);
% plot(freqs, Y_hat_pos);
plot(freqs, Y_hat_dB)
title('Filtered Signal');
xlabel('Frequency (Hz)');
ylabel('Amplitude (dB)');

sgtitle(sprintf('Frame-wise Wiener Filter, N=%d, L=%d', N, L))

error_noise = computeMSE(x, y, N, L);
error_reduced = computeMSE(x, y_hat, N, L);
fprintf("MSE of Noisy Signal %.8f\n", error_noise)
fprintf("MSE of Filtered Signal %.8f\n", error_reduced)

% Save processed data
mse_str = num2str(error_reduced, '%.6f');   
filename = sprintf('%s_%d_%d_%s.wav', filter_type, N, L, mse_str);
 
% audiowrite( filename, y_hat, rate );