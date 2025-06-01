function avg_noise_power = estimateNoisePower(audio_data, N)
    % Returns single scalar for white noise
    
    audio_buffered = buffer(audio_data, N, N/2);
    frame_energies = sum(audio_buffered.^2, 1) / N;
    threshold = prctile(frame_energies, 20);
    silent_frames = audio_buffered(:, frame_energies <= threshold);
    
    % Just take first silent frame and get average power
    first_silent = silent_frames(:, 1);
    Y = fft(first_silent);
    avg_noise_power = mean(abs(Y).^2);
end