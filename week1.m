%% SF1 Data Analysis - Week 1
%% Section 1 DFT v.s. FFT
clc, clearvars
lengths = [10,25,50,75,100,250,500,750,1000,2500,5000,7500,10000,25000]; %lengths of signal
% disp(lengths)
dft_times = zeros(1,length(lengths));
fft_times = zeros(1,length(lengths));

index = 1;
for N = lengths
    % Compute the DFT of the signal
    x = rand(1,N);
    tic
    dftResult = abs(DFT(x));
    dft_times(index)=toc; %store time

    % Compute the FFT of the signal
    tic
    fftResult = abs(fft(x));
    fft_times(index) = toc;
    
    index = index + 1;
    
end

figure
loglog(lengths, dft_times, "bo-")
hold on
loglog(lengths, fft_times, "ro-")
grid on

% Plotting the Theoretical Complexities of DFT and FFT
% Take the last point from DFT/FFT, draw the lines backward
DFT_last = dft_times(end);
FFT_last = fft_times(end);
N_last = lengths(end);

hold on
% Theoretical DFT complexity O(N^2)
N_ext = linspace(10, N_last, 100);
DFT_complexity = (N_ext.^2) .* (DFT_last / (N_last^2)); % Scale to last point
loglog(N_ext, DFT_complexity, 'b--', 'DisplayName', 'DFT Complexity O(N^2)');

hold on
% Theoretical FFT complexity O(N log N)
FFT_complexity = (N_ext .* log2(N_ext)) .* (FFT_last / (N_last * log2(N_last))); % Scale to last point
loglog(N_ext, FFT_complexity, 'r--', 'DisplayName', 'FFT Complexity O(N log N)');


legend("DFT", "FFT", "DFT (Theoretical)", "FFT (Theoretical")
xlabel("log(N)")
ylabel("log(t)")
title("DFT vs FFT Runtime (log-log scale)")



%% Section 2 Hamming Filter; 
clc, clearvars
Ns = [50, 100, 200];
normalized_freq = 1;

for i = 1:length(Ns)
    N = Ns(i);
    zero_pad_len = 0;
    indexes = linspace(-N/2,N/2-1,N); 
    frequencies = indexes/N * normalized_freq * 2*pi;
    
    x = cos(indexes*normalized_freq);
    w = hamming(N, "periodic")';
    windowed_x = x .* w;
    FFT = abs(fft(windowed_x, N+zero_pad_len));
    FFTShifted = fftshift(FFT);

    subplot(1, 3, i)
    semilogy(frequencies, FFTShifted)
    xlabel("Normalized Freq.")
    ylabel("log(|Xp|)")
    ylim([0.01, 100])
    grid on
    title(['N = ', num2str(N)])
end
sgtitle('FFT of Hamming Windowed Sine Wave (Normalized Frequency 1) with Different N Values');

%% Section 1.3  Sine, Vary N
clc, clearvars
Ns = [50 100 200];
normalized_freq = 2;             % cycles / sample
zero_pad_len    = 0;               

for i = 1:numel(Ns)
    N       = Ns(i);
    indexes = linspace(-N/2, N/2-1, N);        % sample index
    freqs   = (indexes/N)*2*pi;                % normalized freq

    x       = sin(normalized_freq * indexes ); % generate signal
    w       = hamming(N).';                    % hamming window
    xw      = x .* w;                          % windowing signal

    FFT     = abs(fft(xw, N + zero_pad_len));
    FFT_s   = fftshift(FFT);                   % centre DC

    subplot(1,3,i)
    semilogy(freqs, FFT_s)
    xlabel('Normalized Frequency')
    ylabel('log(|X_p|)')
    ylim([0.01, 100])
    grid on
    title(['N = ' num2str(N)])
end
sgtitle('FFT of Hamming Windowed Sine Wave (Normalized Frequency 2) with Different N Values');

%% Section 1.3 Compare Sine, Cosine, Exp
clc, clearvars
normalized_freq = 2;             % cycles / sample
zero_pad_len    = 0;               % set >0 if you want oversampling


N       = 100;
indexes = linspace(-N/2, N/2-1, N);          % sample index
freqs   = (indexes/N)*2*pi;                  % normalized freq

x_sin = sin(normalized_freq * indexes);
x_cos = cos(normalized_freq * indexes);
x_exp = exp(1j*normalized_freq * indexes);

w       = hamming(N).';
x_sin_w = x_sin .* w;
x_cos_w = x_cos .* w;
x_exp_w = x_exp .* w;

FFT_sin = fftshift(abs(fft(x_sin_w, N + zero_pad_len)));
FFT_cos = fftshift(abs(fft(x_cos_w, N + zero_pad_len)));
FFT_exp = fftshift(abs(fft(x_exp_w, N + zero_pad_len)));

subplot(1,3,1)
semilogy(freqs, FFT_sin)
xlabel('Normalized Frequency')
ylabel('log(|X_p|)')
grid on
title("Sine")

subplot(1,3,2)
semilogy(freqs, FFT_cos)
xlabel('Normalized Frequency')
ylabel('log(|X_p|)')
grid on
title("Cosine")

subplot(1,3,3)
semilogy(freqs, FFT_exp)
xlabel('Normalized Frequency')
ylabel('log(|X_p|)')
grid on
title("Exponential")

sgtitle('FFT of Hamming Windowed Sine, Cosine, Exp.');

%% Section 1.3 Compare Windows
clc, clearvars
normalized_freq = 2;             
zero_pad    = 8;    % perform x8 zero padding 


N       = 100;
indexes = linspace(-N/2, N/2-1, N);          % sample index

x_sin = sin(normalized_freq * indexes);

w_hamming = hamming(N).'; % hamming window
w_hanning = hanning(N).'; % hanning window

x_rect = x_sin; 
x_ham = x_sin .* w_hamming;
x_han = x_sin .* w_hanning;

FFT_rect = fftshift(abs(fft(x_rect, N*zero_pad)));
FFT_ham = fftshift(abs(fft(x_ham, N*zero_pad)));
FFT_han = fftshift(abs(fft(x_han, N*zero_pad)));


% Main-lobe Widths 
M       = N*zero_pad;           % new length after padding
k       = -M/2 : M/2-1;         % indexes after shifting
omega   = k/M * 2*pi;           % frequencies after shifting

w_rect_rad = mainlobe_width(FFT_rect, omega);
w_ham_rad  = mainlobe_width(FFT_ham,  omega);
w_han_rad  = mainlobe_width(FFT_han,  omega);

fprintf('\nnull-to-null main-lobe width:\n');
fprintf('  Rectangular : %4d bins  = %.4f rad/sample\n', w_rect_rad.bins, w_rect_rad.rad);
fprintf('  Hamming     : %4d bins  = %.4f rad/sample\n', w_ham_rad.bins,  w_ham_rad.rad);
fprintf('  Hann        : %4d bins  = %.4f rad/sample\n', w_han_rad.bins,  w_han_rad.rad);



subplot(1,3,1)
semilogy(omega, FFT_rect)
xlabel('Normalized Frequency')
ylabel('log(|X_p|)')
ylim([10e-5, 10e2])
grid on
title("Rectangular Window")
text(0, max(FFT_rect)*0.2, sprintf('Width = %.4f rad', w_rect_rad.rad), ...
     'HorizontalAlignment','center','FontWeight','bold')

subplot(1,3,2)
semilogy(omega, FFT_ham)
xlabel('Normalized Frequency')
ylabel('log(|X_p|)')
ylim([10e-5, 10e2])
grid on
title("Hamming Window")
text(0, max(FFT_rect)*0.2, sprintf('Width = %.4f rad', w_ham_rad.rad), ...
     'HorizontalAlignment','center','FontWeight','bold')


subplot(1,3,3)
semilogy(omega, FFT_han)
xlabel('Normalized Frequency')
ylabel('log(|X_p|)')
ylim([10e-5, 10e2])
grid on
title("Hanning Window")
text(0, max(FFT_rect)*0.2, sprintf('Width = %.4f rad', w_han_rad.rad), ...
     'HorizontalAlignment','center','FontWeight','bold')


sgtitle('FFT of Different Windows');


% Calculates the mainlobe width
function out = mainlobe_width(FFTmag, omega_axis)


    [~, k0] = max(FFTmag);              % index of global peak

    % search left until magnitude starts rising (first minimum)
    kl = k0;
    while kl > 2 && FFTmag(kl-1) < FFTmag(kl)
        kl = kl - 1;
    end

    % search right
    kr = k0;
    while kr < numel(FFTmag)-1 && FFTmag(kr+1) < FFTmag(kr)
        kr = kr + 1;
    end

    out.bins = kr - kl;                                % width in FFT bins
    out.rad  = abs(omega_axis(kr) - omega_axis(kl));   % width in rad/sample
end


%% Section 1.3 Zero Padding
clc, clearvars
normalized_freq = 2;             
zero_pad_factors    = [1,2,4,8];  % factor of zero_pad


N       = 100;
indexes = linspace(-N/2, N/2-1, N);          % sample index
x_sin = sin(normalized_freq * indexes);

% rectangular window
w_hamming       = hamming(N).';
x_ham = x_sin .* w_hamming;

for i=1:4
    zero_pad = zero_pad_factors(i);
    FFT = fftshift(abs(fft(x_ham, N * zero_pad)));
    
    new_N = N * zero_pad;   % new length after padding
    indexes = linspace(-new_N/2, new_N/2-1, new_N);          
    freqs   = (indexes/new_N)*2*pi;          
    
    subplot(1,4,i)
    semilogy(freqs, FFT)
    xlabel('Normalized Frequency')
    ylabel('log(|X_p|)')
    grid on
    title([zero_pad "x Zero Padding"])

end


sgtitle('FFT of Hamming Windowed Sine Wave with Different Zero-Padding');

%% Section 1.3 Guassian Noise

clc, clearvars
N = 100;
normalized_freq = 2;             
zero_pad_len    = 0;            

indexes = linspace(-N/2, N/2-1, N);          
freqs   = (indexes/N)*2*pi;                

x       = sin(normalized_freq * indexes );
w       = hamming(N).';


noise_coefficients = [0,0.2,1,3];
for i=1:4
    noise_coefficient = noise_coefficients(i);

    noise = randn(N, 1)*noise_coefficient;

    % Add noise to the windowed signal
    noisy_x = x + noise';
    noisy_xw = noisy_x .* w; % windowed noisy signal
    
    % Compute the FFT of the noisy signal
    noisyX = fftshift(abs(fft(noisy_xw, N + zero_pad_len)));
    
    % Plot the FFT of the noisy signal
    subplot(1,4,i)
    semilogy(freqs, noisyX)
    xlabel("Normalized Frequency")
    ylabel("log(Xp)")
    ylim([0.01, 100])
    title(['Noise Coefficient = ' num2str(noise_coefficient)])

end

sgtitle('Gaussian Noise');


%% Section 3 - Q6 Pure Note
clc, clearvars
[data, rate] = audioread("/Users/maxlyu/Documents/MATLAB/IIA-SF1-DataAnalysis/Audio examples for weeks 1-2-20250516/piano_clean.wav");
% Play the audio signal
% sound(data, rate)

% Picks a pure note from audio, apply Hamming window
% segment = data(2800:4200); % First pure note 438Hz
segment = data(3200:3800); %[Checkpoint]
% segment = data(3400:3600); % First pure note 438Hz
sound(segment, rate);
% segment = data(6000:6200); % Transient?
% segment = data(6200:6400); % Second pure note 987Hz
windowed = segment .* hamming(length(segment));
% sound(segment, rate)

% Plot the windowed audio
plot(windowed);
xlabel('Sample Number');
ylabel('Amplitude');
title('Audio Signal Waveform of Pure Note');


% Plot the FFT of the audio segment
N = length(windowed);
zero_pad_len = 0;
fftSegment = abs(fft(windowed, N + zero_pad_len));
fftSegmentShifted = fftshift(fftSegment);

bins = linspace(-N/2, N/2, N);
freqs = bins/N * rate;

figure;
plot(freqs, fftSegmentShifted);
xlabel('Frequency Bin');
ylabel('Magnitude');
title('FFT of Audio Segment of Pure Note');
grid on;

% [A, index] = max(fftSegmentShifted);
% hold on
% 
% freq = index/N * rate;
% disp(freq)
% 
% plot(index, A, "o");
% text_label = sprintf('Max Freq is %d Hz', freq);
% text(index, A, text_label);

%% Q6 - Transient Note
clc, clearvars
[data, rate] = audioread("/Users/maxlyu/Documents/MATLAB/IIA-SF1-DataAnalysis/Audio examples for weeks 1-2-20250516/armst_37_orig.wav");
% Play the audio signal
% sound(data, rate)

% Picks a pure note from audio, apply Hamming window
% segment = data(12900:13100);
segment = data(500:800); % Transient?


sound(segment, rate);
% segment = data(6000:6200); % Transient?
% segment = data(6200:6400); % Second pure note 987Hz
windowed = segment .* hamming(length(segment));
% sound(segment, rate)

% Plot the windowed audio
plot(windowed);
xlabel('Sample Number');
ylabel('Amplitude');
title('Audio Signal Waveform of Transient Note');


% Plot the FFT of the audio segment
N = length(windowed);
zero_pad_len = 0;
fftSegment = abs(fft(windowed, N + zero_pad_len));
fftShiftedSegment = fftshift(fftSegment);
bins = linspace(-N/2, N/2, N);
freqs = bins/N * rate;

figure;
plot(freqs, fftShiftedSegment);
xlabel('Frequency Bin');
ylabel('Magnitude');
title('FFT of Audio Segment of Transient Note');
grid on;

[A, index] = max(fftSegment);
hold on
% 
% 
% 
% freq = index/N * rate;
% disp(freq)
% 
% plot(index, A, "o");
% text_label = sprintf('Max Freq is %d Hz', freq);
% text(index, A, text_label);

%% Q6 - Transient Note - New Clip
clc, clearvars
[data, rate] = audioread("/Users/maxlyu/Documents/MATLAB/IIA-SF1-DataAnalysis/Audio examples for weeks 1-2-20250516/grosse_original.wav");
% Play the audio signal
% sound(data, rate)

% Picks a pure note from audio, apply Hamming window
segment = data(75000:75500); % [Checkpoint]

% segment = data(197000:197200); % Transient?

sound(segment, rate);

windowed = segment .* hamming(length(segment));

% Plot the windowed audio
plot(windowed);
xlabel('Sample Number');
ylabel('Amplitude');
title('Audio Signal Waveform of Transient Note');


% Plot the FFT of the audio segment
N = length(windowed);
zero_pad_len = 0;
fftSegment = abs(fft(windowed, N + zero_pad_len));
fftShiftedSegment = fftshift(fftSegment);
bins = linspace(-N/2, N/2, N);
freqs = bins/N * rate;

figure;
plot(freqs, fftShiftedSegment);
xlabel('Frequency Bin');
ylabel('Magnitude');
title('FFT of Audio Segment of Transient Note');
grid on;


%% Q6 -  Consonant

clc, clearvars
[data, rate] = audioread("/Users/maxlyu/Documents/MATLAB/IIA-SF1-DataAnalysis/Audio examples for weeks 1-2-20250516/f1lcapae.wav");
% Play the audio signal
% sound(data, rate)

% Picks a pure note from audio, apply Hamming window
segment = data(60000:80000); % "p" from cap
% segment = data(62500:62800); % "p" from cap [Checkpoint]


sound(segment, rate);
% segment = data(6000:6200); % Transient?
% segment = data(6200:6400); % Second pure note 987Hz
windowed = segment .* hamming(length(segment));
% sound(segment, rate)

% Plot the windowed audio
plot(windowed);
xlabel('Sample Number');
ylabel('Amplitude');
title('Audio Signal Waveform of Consonant');


% Plot the FFT of the audio segment
N = length(windowed);
zero_pad_len = 0;
fftSegment = abs(fft(windowed, N + zero_pad_len));
fftSegmentShifted = fftshift(fftSegment);

bins = linspace(-N/2, N/2, N);
freqs = bins/N * rate;

figure;
% plot(freqs, fftSegmentShifted);
semilogy(freqs, fftSegmentShifted);
xlabel('Frequency Bin');
ylabel('Magnitude');
ylim([0,1]);
title('FFT of Audio Segment of Consonant');
grid on;

[A, index] = max(fftSegment);
hold on

freq = index/N * rate;
disp(freq)

% plot(index, A, "o");
% text_label = sprintf('Max Freq is %d Hz', freq);
% text(index, A, text_label);


%% Q6 - Vowel

clc, clearvars
[data, rate] = audioread("/Users/maxlyu/Documents/MATLAB/IIA-SF1-DataAnalysis/Audio examples for weeks 1-2-20250516/f1lcapae.wav");
% Play the audio signal
% sound(data, rate)

% Picks a pure note from audio, apply Hamming window
% segment = data(55000:58000); % "Ca" from cap
segment = data(56000:58000); % "Ca" from cap


sound(segment, rate);
% segment = data(6000:6200); % Transient?
% segment = data(6200:6400); % Second pure note 987Hz
windowed = segment .* hamming(length(segment));
% sound(segment, rate)

% Plot the windowed audio
plot(windowed);
xlabel('Sample Number');
ylabel('Amplitude');
title('Audio Signal Waveform of Vowel');


% Plot the FFT of the audio segment
N = length(windowed);
zero_pad_len = 0;
fftSegment = abs(fft(windowed, N + zero_pad_len));
fftSegmentShifted = fftshift(fftSegment);

bins = linspace(-N/2, N/2, N);
freqs = bins/N * rate;

figure;
plot(freqs, fftSegmentShifted);
semilogy(freqs, fftSegmentShifted)
xlabel('Frequency Bin');
ylabel('Magnitude (log scale)');
title('FFT of Audio Segment of Vowel');
grid on;
% 
% [A, index] = max(fftSegmentShifted);
% hold on
% 
% freq = index/N * rate;
% disp(freq)
% 
% max_x_value = freqs(index);
% plot(max_x_value, A, "o");
% text_label = sprintf('Max Freq is %d Hz', freq);
% text(max_x_value, A, text_label);

%% Q6 - Vowel New

clc, clearvars
[data, rate] = audioread("/Users/maxlyu/Documents/MATLAB/IIA-SF1-DataAnalysis/Audio examples for weeks 1-2-20250516/f1lcapae.wav");
% Play the audio signal
% sound(data, rate)

% Picks a pure note from audio, apply Hamming window
segment = data(56000:58000); % "Ca" from cap
sound(segment, rate);

windowed = segment .* hamming(length(segment));

% Plot the windowed audio
plot(windowed);
xlabel('Sample Number');
ylabel('Amplitude');
title('Audio Signal Waveform of Vowel');


% Plot the FFT of the audio segment
N = length(windowed);
zero_pad_len = 0;
fftSegment = abs(fft(windowed, N + zero_pad_len));
fftSegmentShifted = fftshift(fftSegment);

fft_plot = fftSegmentShifted(N/2:N);
bins = linspace(0, N/2, N/2+1);
freqs = bins/N * rate;

figure;
semilogy(freqs, fft_plot);
xlim([0,5000]);
xlabel('Frequency Bin');
ylabel('Magnitude (log-scale)');
title('FFT of Audio Segment of Vowel');
grid on;


%% Q6 - Consonant New

clc, clearvars
[data, rate] = audioread("/Users/maxlyu/Documents/MATLAB/IIA-SF1-DataAnalysis/Audio examples for weeks 1-2-20250516/f1lcapae.wav");
% Play the audio signal
% sound(data, rate)

% Picks a pure note from audio, apply Hamming window
segment = data(60000:65000); % "p" from cap

sound(segment, rate);

windowed = segment .* hanning(length(segment));

% Plot the windowed audio
plot(windowed);
xlabel('Sample Number');
ylabel('Amplitude');
title('Audio Signal Waveform of Consonant');


% Plot the FFT of the audio segment
N = length(windowed);
zero_pad_len = 0;
fftSegment = abs(fft(windowed, N + zero_pad_len));
fftSegmentShifted = fftshift(fftSegment);

fft_plot = fftSegmentShifted(N/2:N);
bins = linspace(0, N/2, N/2+1);
freqs = bins/N * rate;

figure;
semilogy(freqs, fft_plot);
xlim([0,4000]);
xlabel('Frequency Bin');
ylabel('Magnitude (log-scale)');
title('FFT of Audio Segment of Consonant');
grid on;


%% Q6 - Pure Note New

clc, clearvars
[data, rate] = audioread("/Users/maxlyu/Documents/MATLAB/IIA-SF1-DataAnalysis/Audio examples for weeks 1-2-20250516/piano_clean.wav");

% Play the audio signal
% sound(data, rate)

% Picks a pure note from audio, apply Hamming window
segment = data(3200:3800); % a pure note of piano

sound(segment, rate);

windowed = segment .* hamming(length(segment));

% Plot the windowed audio
plot(windowed);
xlabel('Sample Number');
ylabel('Amplitude');
title('Audio Signal Waveform of Pure Note');


% Plot the FFT of the audio segment
N = length(windowed);
zero_pad_len = 0;
fftSegment = abs(fft(windowed, N + zero_pad_len));
fftSegmentShifted = fftshift(fftSegment);

fft_plot = fftSegmentShifted(N/2:N);
bins = linspace(0, N/2, N/2+1);
freqs = bins/N * rate;

figure;
semilogy(freqs, fft_plot);
xlim([0,4000]);
xlabel('Frequency Bin');
ylabel('Magnitude (log-scale)');
title('FFT of Audio Segment of Pure Note');
grid on;

%% Q6 - Transient Note New

clc, clearvars

[data, rate] = audioread("/Users/maxlyu/Documents/MATLAB/IIA-SF1-DataAnalysis/Audio examples for weeks 1-2-20250516/grosse_original.wav");

% Picks a pure note from audio, apply Hamming window
segment = data(74000:75000); % a transient note of violin

sound(segment, rate);

windowed = segment .* hanning(length(segment));

% Plot the windowed audio
plot(windowed);
xlabel('Sample Number');
ylabel('Amplitude');
title('Audio Signal Waveform of Transient Note');

% Plot the FFT of the audio segment
N = length(windowed);
zero_pad_len = 0;
fftSegment = abs(fft(windowed, N + zero_pad_len));
fftSegmentShifted = fftshift(fftSegment);

fft_plot = fftSegmentShifted(N/2:N);
bins = linspace(0, N/2, N/2+1);
freqs = bins/N * rate;

figure;
semilogy(freqs, fft_plot);
xlim([0,4000]);
xlabel('Frequency Bin');
ylabel('Magnitude (log-scale)');
title('FFT of Audio Segment of Transient Note');
grid on;
%% Q7

clc, clearvars
[data, rate] = audioread("/Users/maxlyu/Documents/MATLAB/IIA-SF1-DataAnalysis/Audio examples for weeks 1-2-20250516/organ.wav");
% Play the audio signal
channel_one_data = data(:,1);
channel_two_data = data(:,2);

% Picks a pure note from audio, apply Hamming window
segment = channel_one_data(10000:30000); 
sound(segment, rate);


sound(segment, rate);
% segment = data(6000:6200); % Transient?
% segment = data(6200:6400); % Second pure note 987Hz
windowed = segment .* hamming(length(segment));
% sound(segment, rate)

% Plot the windowed audio
plot(windowed);
xlabel('Sample Number');
ylabel('Amplitude');
title('Audio Signal Waveform of organ.wav');


% Plot the FFT of the audio segment
N = length(windowed);
zero_pad_len = 0;
fftSegment = abs(fft(windowed, N + zero_pad_len));
fftSegmentShifted = fftshift(fftSegment);

% bins = linspace(-N/2, N/2, N);
% freqs = bins/N * rate;

fft_plot = fftSegmentShifted(N/2:N);
bins = linspace(0, N/2, N/2+1);
freqs = bins/N * rate;

figure;
% plot(freqs, fft_plot);
semilogy(freqs, fft_plot)
xlabel('Frequency Bin');
xlim([0,2000])
ylabel('Magnitude');
title('FFT of organ.wav Clip');
grid on;



% Some algorithm?
search_len = 30; % The largest [serach_len] frequency components are considered
fft_length = length(fftSegmentShifted);
[As, indexes] = maxk(fftSegmentShifted(fft_length/2:fft_length),search_len);

freqs = indexes/N * rate;
disp(freqs)


frequency_identified = zeros(1,search_len);
index_identified = 1;
for i=1:search_len
    flag = true; % flag used to keep track of new frequency discovered
    f = freqs(i); % current frequency
    
    for j=1:index_identified

        % Compare the current frequency with all frequencies seen before
        % If close to (within 10%) other frequency identified already (or harmonics), declare seen
        if abs((f-frequency_identified(j))/f)<=0.1 || abs((f-2*frequency_identified(j))/f)<=0.1 || abs((f-4*frequency_identified(j))/f)<=0.1
            
            flag = false;
        end
        % If a frequency identified before is a multiple of this
        % Then this frequency should be the fundamental component instead
        if abs((2*f-frequency_identified(j))/f) < 0.1
            frequency_identified(j) = f;
            flag = false;
        end
    
    end
    
    % If current frequency is not identified before, add into frequency
    % list
    if flag == true
        % disp(i)
        % disp(f)
        % disp(frequency_identified)
        frequency_identified(index_identified) = f;
        index_identified = index_identified + 1;
    end
end

disp(frequency_identified)


%% Q7 New Method Find Peak

clc, clearvars
[data, rate] = audioread("/Users/maxlyu/Documents/MATLAB/IIA-SF1-DataAnalysis/Audio examples for weeks 1-2-20250516/organ.wav");
% Play the audio signal
channel_one_data = data(:,1);
channel_two_data = data(:,2);

% Use just channel one
segment = channel_one_data(10000:30000); 
sound(segment, rate);
N = length(segment);

% Apply FFT
X      = fft(segment);
mag    = abs(X)/N;
magPos = mag(1:floor(N/2)+1);           % only keep the positive fs
fAxis  = (0:floor(N/2))*rate/N;         % frequencies in Hz

% Detect spectral peaks with findpeaks
[minPkHeight, minPkDist] = deal( ...
    0.1 * max(magPos), ...            % 10% (-20dB) of max magnitude
    1*N/rate         );               % 1 Hz between peaks

[pkVals, locs] = findpeaks( ...
        magPos, ...
        'MinPeakHeight',  minPkHeight, ...
        'MinPeakDistance',minPkDist);

peakFreqs = fAxis(locs);              % convert indexes to frequency (Hz)

% Plot the spectrum and highlight the peaks
figure
plot(fAxis, magPos, 'LineWidth',0.8), hold on
plot(peakFreqs, pkVals, 'rx', 'MarkerSize',8, 'LineWidth',1.5)
grid on, xlim([0 4000])
xlabel('Frequency  (Hz)')
ylabel('Magnitude  (linear)')
title('FFT magnitude spectrum with detected peaks')
legend('Spectrum','Detected peaks')
hold off

% Algorithm to exclude harmonics or close frequencies
search_len = length(peakFreqs);
frequency_identified = zeros(1,search_len);
index_identified = 1;
for i=1:search_len
    flag = true; % flag used to keep track of new frequency discovered
    f = peakFreqs(i); % current frequency
    
    for j=1:index_identified

        % Compare the current frequency with all frequencies seen before
        % If close to (within 10%) other frequency identified already (or harmonics), declare seen
        if abs((f-frequency_identified(j))/f)<=0.1 || abs((f-2*frequency_identified(j))/f)<=0.1 || abs((f-4*frequency_identified(j))/f)<=0.1
            
            flag = false;
        end
        % If a frequency identified before is a multiple of this
        % Then this frequency should be the fundamental component instead
        if abs((2*f-frequency_identified(j))/f) < 0.1
            frequency_identified(j) = f;
            flag = false;
        end
    
    end
    
    % If current frequency is not identified before, add into frequency list
    if flag == true
        % disp(i)
        % disp(f)
        % disp(frequency_identified)
        frequency_identified(index_identified) = f;
        index_identified = index_identified + 1;
    end
end

disp(frequency_identified)