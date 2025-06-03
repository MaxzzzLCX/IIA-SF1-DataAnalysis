%% Task 2.4 Noise Reduction with Adaptive Noise Estimation
clc, clearvars
[data, rate] = audioread("/Users/maxlyu/Desktop/Part IIA/SF1/Noisy audio for denoising tests-20250528/dipper.wav");
data = data(:,1);
N = 512; L = 256; % 512 Frame size, 256 Overlap
window_size = 30; noise_percentile = 10;

noise_type = "white";
filter_type = "spectral_subtraction";
audio_name = "fastn";
SignalN = length(data);
y = data(1:SignalN)'; % reshape for noiseReduction input shape

% CHOICE: Static vs Adaptive noise estimation
use_adaptive_noise = true;  % Set to false for your original approach

if use_adaptive_noise
    fprintf('Using ADAPTIVE noise estimation\n');
    % Apply the frame-wise filter with adaptive noise estimation
    [y_hat, noise_history] = FramewiseFilterAdaptive(y, 512, 256, noise_type, filter_type, window_size, noise_percentile);
    
    % Plot noise evolution
    figure;
    plot(noise_history.frame_numbers, noise_history.noise_estimates);
    title('Adaptive Noise Estimation Over Time: Fast n');
    xlabel('Frame Number');
    ylabel('Estimated Noise RMS');
    grid on;
    
else
    fprintf('Using STATIC noise estimation\n');
    % Your original approach
    noise_mag = 2*estimateNoisePower(y, N); % Use this for SOFT noise clips
    fprintf('Static noise estimate: %.6f\n', noise_mag);
    y_hat = FramewiseFilter(y, 512, 256, noise_mag, noise_type, filter_type);
end

% Rest of your plotting code remains the same
figure;
subplot(2,1,1);
plot(y);
title('Signal with Noise');
xlabel('Sample Index');
ylabel('Amplitude'); ylim([-0.5,0.5])
subplot(2,1,2);
plot(y_hat);
title('Filtered Signal');
xlabel('Time Index');
ylabel('Amplitude'); ylim([-0.5,0.5])

if use_adaptive_noise
    sgtitle(sprintf('"Fast n": Adaptive Spectral Subtraction (W=%d, per.=%d, N=%d, L=%d)', window_size, noise_percentile, N, L))
else
    sgtitle(sprintf('"Male Soft": Static Spectral Subtraction (N=%d, L=%d)', N, L))
end

% Frequency domain plot
figure
Y = fftshift(abs(fft(y)));
Y_pos = Y(length(Y)/2:length(Y));
Y_dB = 20*log10(Y_pos);
freqs = (0:length(Y_pos)-1) * rate / SignalN;
subplot(2,1,1);
plot(freqs, Y_dB)
title('Signal with Noise');
xlabel('Frequency (Hz)');
ylabel('Amplitude (dB)');

Y_hat = fftshift(abs(fft(y_hat)));
Y_hat_pos = Y_hat(length(Y_hat)/2:length(Y_hat));  % Fixed bug: was using Y instead of Y_hat
Y_hat_dB = 20*log10(Y_hat_pos);
subplot(2,1,2);
plot(freqs, Y_hat_dB)
title('Filtered Signal');
xlabel('Frequency (Hz)');
ylabel('Amplitude (dB)');

if use_adaptive_noise
    sgtitle(sprintf('Adaptive Noise Estimation: Spectral Subtraction, N=%d, L=%d', N, L))
else
    sgtitle(sprintf('Static Noise Estimation: Spectral Subtraction Filter, N=%d, L=%d', N, L))
end

% Save processed data
if use_adaptive_noise
    filename = sprintf('%s_%d_%d_%s_%d_%d.wav', audio_name, window_size, noise_percentile, filter_type, N, L);
else
    filename = sprintf('%s_%s_static_%d_%d.wav', audio_name, filter_type, N, L);
end
audiowrite(filename, y_hat, rate);

