%% SF1 - Data Analysis Week 2

%% Task 2.1
clc, clearvars

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

% TODO
freqs = linspace(0,2*pi,SignalN/2+1);
% freqs = (0:(SignalN-1)/2)/SignalN * 2*pi;

figure;
subplot(1,2,1)
% Plot the original signal spectrum
X = fftshift((abs(fft(x_in))));
X_pos = X(SignalN/2:SignalN);
plot(freqs, X_pos);
xlabel('Frequency (bins)');
ylabel('Magnitude');
ylim([0,6000])
xlim([0,2*pi])
title('FFT Spectrum of Input Signal')

subplot(1,2,2)
% Plot the processed signal spectrum
Y = fftshift((abs(fft(y_out))));
Y_pos = Y(SignalN/2:SignalN);
plot(freqs, Y_pos);
ylim([0,6000])
xlim([0,2*pi])
xlabel('Frequency (bins)');
ylabel('Magnitude');
title('FFT Spectrum of Input Signal');


%% Task 2.2 - Own Noise Reduction


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
