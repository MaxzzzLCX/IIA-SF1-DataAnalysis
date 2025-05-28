%% Debug what's happening in quiet frames
function debugQuietFrames()
    clc; clearvars;
    
    % Recreate your test signal
    N = 512; overlap = 256;
    T = 30000;
    
    % Create signal with clear quiet start (like your plot)
    x_clean = zeros(1, T);
    x_clean(5000:25000) = 0.5*cos((1:20001)*pi/4) + sin((1:20001)*pi/100);
    
    noise_mag = 0.01;
    rng(42);
    x_noisy = x_clean + noise_mag * randn(size(x_clean));
    
    fprintf('=== DEBUGGING QUIET FRAME PROCESSING ===\n');
    
    % Buffer the signal (same as your code)
    y = buffer(x_noisy, N, overlap);
    [N_samps, N_frames] = size(y);
    y_w = repmat(hanning(N), 1, N_frames) .* y;
    
    % Analyze first few frames (should be quiet)
    fprintf('Frame Analysis:\n');
    for frame_no = 1:min(5, N_frames-2)
        frame_data = y(:, frame_no);
        windowed_data = y_w(:, frame_no);
        
        % Check frame content
        frame_rms = rms(frame_data);
        windowed_rms = rms(windowed_data);
        
        % What does your Wiener filter see?
        Y_w = fft(windowed_data);
        S_Y = abs(Y_w).^2;
        avg_S_Y = mean(S_Y);
        
        % Generate noise estimate (your method)
        noise_frame = noise_mag * randn(N, 1);
        windowed_noise = hanning(N) .* noise_frame;
        windowed_noise_fft = fft(windowed_noise);
        S_N = abs(windowed_noise_fft).^2;
        avg_S_N = mean(S_N);
        
        % Calculate Wiener gain
        S_X = max(0, S_Y - S_N);
        G = S_X ./ (S_X + S_N);
        avg_gain = mean(G);
        
        fprintf('Frame %d:\n', frame_no);
        fprintf('  Sample range: %d-%d\n', (frame_no-1)*overlap+1, (frame_no-1)*overlap+N);
        fprintf('  Frame RMS: %.6f\n', frame_rms);
        fprintf('  Windowed RMS: %.6f\n', windowed_rms);
        fprintf('  Avg S_Y: %.6f\n', avg_S_Y);
        fprintf('  Avg S_N: %.6f\n', avg_S_N);
        fprintf('  Avg Wiener gain: %.6f\n', avg_gain);
        
        % Check if frame is actually quiet
        if frame_rms < 2 * noise_mag
            fprintf('  Status: Should be QUIET\n');
        else
            fprintf('  Status: Contains SIGNAL\n');
        end
        fprintf('\n');
    end
    
    % Check what pure noise looks like
    fprintf('=== PURE NOISE REFERENCE ===\n');
    pure_noise = noise_mag * randn(N, 1);
    pure_windowed = hanning(N) .* pure_noise;
    pure_Y = fft(pure_windowed);
    pure_S_Y = abs(pure_Y).^2;
    pure_avg_S_Y = mean(pure_S_Y);
    
    fprintf('Pure noise avg S_Y: %.6f\n', pure_avg_S_Y);
    fprintf('Your frames avg S_Y: %.6f (first frame)\n', avg_S_Y);
    fprintf('Ratio: %.2f\n', avg_S_Y / pure_avg_S_Y);
    
    if avg_S_Y > 2 * pure_avg_S_Y
        fprintf('❌ PROBLEM: Your "quiet" frames have too much power!\n');
        fprintf('   This suggests they contain signal content, not just noise.\n');
    else
        fprintf('✅ Frames appear to be mostly noise\n');
    end
    
    % Test the fix: skip problematic first frames
    fprintf('\n=== SUGGESTED FIX ===\n');
    fprintf('Try starting processing from frame 3 or 4 instead of frame 1\n');
    fprintf('This avoids buffer padding artifacts.\n');
    
    % Visualize the issue
    figure;
    subplot(3,1,1);
    plot(x_clean(1:5000)); title('Clean Signal (First 5000 samples)');
    ylabel('Amplitude'); ylim([-0.1 0.1]);
    
    subplot(3,1,2);
    plot(x_noisy(1:5000)); title('Noisy Signal (First 5000 samples)');
    ylabel('Amplitude');
    
    subplot(3,1,3);
    % Show what's in the first few frames
    frame1_indices = 1:N;
    frame2_indices = overlap+1:overlap+N;
    frame3_indices = 2*overlap+1:2*overlap+N;
    
    plot(frame1_indices, y(:,1), 'r-', 'LineWidth', 1.5); hold on;
    plot(frame2_indices, y(:,2), 'g-', 'LineWidth', 1.5);
    plot(frame3_indices, y(:,3), 'b-', 'LineWidth', 1.5);
    title('Buffered Frames Content');
    legend('Frame 1', 'Frame 2', 'Frame 3');
    ylabel('Amplitude'); xlabel('Sample Index');
end

%% Test improved Wiener filter that skips problematic frames
function y_hat = framewiseWienerImproved(y_in, N, overlap, noise_mag)
    % Improved version that handles edge effects better
    
    y_hat = zeros(size(y_in));
    y = buffer(y_in, N, overlap);
    [N_samps, N_frames] = size(y);
    y_w = repmat(hanning(N), 1, N_frames) .* y;
    
    % Skip first few frames to avoid buffer padding issues
    start_frame = 3;  % Start from frame 3
    
    for frame_no = start_frame:N_frames-2
        Y_w = fft(y_w(:, frame_no));
        
        % Your existing Wiener filter
        noise_frame = noise_mag * randn(N, 1);
        windowed_noise = hanning(N) .* noise_frame;
        windowed_noise_fft = fft(windowed_noise);
        S_N = abs(windowed_noise_fft).^2;
        
        S_Y = abs(Y_w).^2;
        S_X = max(0, S_Y - S_N);
        G = S_X ./ (S_X + S_N);
        
        % Apply minimum gain threshold for quiet sections
        min_gain = 0.01;  % Prevent complete suppression
        G = max(min_gain, G);
        
        Y_hat_w = G .* Y_w;
        Y_hat_w(N:-1:N/2+2) = conj(Y_hat_w(2:N/2));
        y_hat_w = real(ifft(Y_hat_w));
        
        start_idx = (frame_no-1)*overlap + 1;
        end_idx = start_idx + N - 1;
        if end_idx <= length(y_hat)
            y_hat(start_idx:end_idx) = y_hat(start_idx:end_idx) + y_hat_w';
        end
    end
end

% Run the debug
debugQuietFrames();