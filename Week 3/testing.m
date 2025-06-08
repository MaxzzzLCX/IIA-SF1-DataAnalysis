%% Simple AR-Based Audio Analysis and Interpolation
clc; clear; close all;

fprintf('=== SIMPLE AR AUDIO INTERPOLATION ===\n');

% Load audio files (adjust paths as needed)
audio_files = {'/Users/maxlyu/Desktop/Part IIA/SF1/IIA-SF1-DataAnalysis/Week 3/armst_37_missing.wav', 
    '/Users/maxlyu/Desktop/Part IIA/SF1/IIA-SF1-DataAnalysis/Week 3/grosse_40_percent_missing.wav'};  % Replace with your actual filenames

for file_idx = 1:length(audio_files)
    if exist(audio_files{file_idx}, 'file')
        fprintf('\n--- Analyzing %s ---\n', audio_files{file_idx});
        
        % Load and prepare audio
        [audio_data, fs] = audioread(audio_files{file_idx});
        if size(audio_data, 2) > 1
            audio_data = mean(audio_data, 2);  % Convert to mono
        end
        
        % Step 1: Segment-wise AR model selection
        segment_models = performSegmentARAnalysis(audio_data, fs, audio_files{file_idx});
        
        % Step 2: Apply interpolation using segment-specific models
        performSegmentBasedInterpolation(audio_data, fs, segment_models, audio_files{file_idx});
        
    else
        fprintf('Warning: File %s not found. Skipping...\n', audio_files{file_idx});
    end
end

function segment_models = performSegmentARAnalysis(audio_data, fs, filename)
    % Perform AR model selection on each audio segment
    fprintf('\n  STEP 1: Segment-wise AR model selection...\n');
    
    % Segmentation parameters - use longer segments
    segment_duration = 0.1;  % 250ms segments (was 100ms)
    segment_samples = round(segment_duration * fs);
    overlap_ratio = 0.25;     % 25% overlap (less overlap)
    hop_size = round(segment_samples * (1 - overlap_ratio));
    
    % Calculate number of segments
    total_segments = floor((length(audio_data) - segment_samples) / hop_size) + 1;
    
    fprintf('    Processing %d segments of %.1fms each (%.1f%% overlap)...\n', ...
            total_segments, segment_duration*1000, overlap_ratio*100);
    
    % Storage for segment models
    segment_models = struct();
    segment_models.segment_starts = zeros(total_segments, 1);
    segment_models.segment_ends = zeros(total_segments, 1);
    segment_models.optimal_orders = zeros(total_segments, 1);
    segment_models.ar_parameters = cell(total_segments, 1);
    segment_models.log_marginal_likelihoods = zeros(total_segments, 1);
    segment_models.sigma_e_values = zeros(total_segments, 1);
    
    % Test AR orders - much wider range
    test_orders = 1:20;  % Extended range (was 2:15)
    
    % Analyze each segment
    for seg = 1:total_segments
        start_idx = (seg-1) * hop_size + 1;
        end_idx = min(start_idx + segment_samples - 1, length(audio_data));
        segment = audio_data(start_idx:end_idx);
        
        segment_models.segment_starts(seg) = start_idx;
        segment_models.segment_ends(seg) = end_idx;
        
        % Find optimal AR model for this segment
        [opt_order, ar_params, best_lml, sigma_e] = findOptimalARModel(segment, test_orders);
        
        segment_models.optimal_orders(seg) = opt_order;
        segment_models.ar_parameters{seg} = ar_params;
        segment_models.log_marginal_likelihoods(seg) = best_lml;
        segment_models.sigma_e_values(seg) = sigma_e;
        
        if mod(seg, 20) == 0 || seg == total_segments
            fprintf('      Processed %d/%d segments, recent orders: [%d, %d, %d]\n', ...
                    seg, total_segments, segment_models.optimal_orders(max(1,seg-2)), ...
                    segment_models.optimal_orders(max(1,seg-1)), segment_models.optimal_orders(seg));
        end
    end
    
    % Print summary statistics
    fprintf('\n    SEGMENT-WISE AR MODEL SUMMARY:\n');
    fprintf('      Mean optimal order: %.1f ± %.1f\n', ...
            mean(segment_models.optimal_orders), std(segment_models.optimal_orders));
    fprintf('      Order range: %d to %d\n', ...
            min(segment_models.optimal_orders), max(segment_models.optimal_orders));
    fprintf('      Mode order: %d\n', mode(segment_models.optimal_orders));
    fprintf('      Mean σₑ: %.4f ± %.4f\n', ...
            mean(segment_models.sigma_e_values), std(segment_models.sigma_e_values));
    
    % Show order distribution
    [order_counts, order_values] = hist(segment_models.optimal_orders, unique(segment_models.optimal_orders));
    fprintf('      Order distribution:\n');
    for i = 1:length(order_values)
        fprintf('        P=%d: %d segments (%.1f%%)\n', order_values(i), order_counts(i), ...
                order_counts(i)/total_segments*100);
    end
    
    % Visualize segment analysis
    visualizeSegmentAnalysis(segment_models, filename);
end

function [opt_order, ar_params, best_lml, sigma_e] = findOptimalARModel(segment, test_orders)
    % Find optimal AR model using simplified BIC approach
    
    if length(segment) < 100
        opt_order = 4;
        ar_params = [0.5; -0.3; 0.1; -0.05];
        best_lml = -inf;
        sigma_e = std(segment);
        return;
    end
    
    % Storage for results
    all_params = cell(size(test_orders));
    all_sigma_e = zeros(size(test_orders));
    bic_values = inf(size(test_orders));
    
    fprintf('    Testing orders: ');
    
    for i = 1:length(test_orders)
        P = test_orders(i);
        
        try
            % Estimate AR parameters
            [a_est, sigma_est, fit_success] = estimateAR_ML_robust(segment, P);
            
            if fit_success && ~any(isnan(a_est)) && sigma_est > 0 && all(isfinite(a_est))
                all_params{i} = a_est;
                all_sigma_e(i) = sigma_est;
                
                % Calculate BIC using very simple approach
                try
                    % Direct calculation without calling calculateResiduals
                    N = length(segment);
                    
                    % Construct prediction
                    prediction_error = 0;
                    valid_predictions = 0;
                    
                    for n = (P+1):N
                        if n <= length(segment) && (n-P) >= 1
                            % Get past P values
                            past_values = segment(n-P:n-1);
                            if length(past_values) == P && all(isfinite(past_values))
                                predicted = sum(a_est .* flip(past_values));
                                if isfinite(predicted)
                                    error = segment(n) - predicted;
                                    if isfinite(error)
                                        prediction_error = prediction_error + error^2;
                                        valid_predictions = valid_predictions + 1;
                                    end
                                end
                            end
                        end
                    end
                    
                    if valid_predictions > P && prediction_error > 0
                        mse = prediction_error / valid_predictions;
                        if mse > 0 && isfinite(mse)
                            % Simple BIC formula
                            log_likelihood = -valid_predictions/2 * log(2*pi*mse) - prediction_error/(2*mse);
                            if isfinite(log_likelihood)
                                bic_values(i) = -2*log_likelihood + P*log(valid_predictions);
                                if isfinite(bic_values(i))
                                    fprintf('P%d(%.1f) ', P, bic_values(i));
                                else
                                    fprintf('P%d(BIC-inf) ', P);
                                end
                            else
                                fprintf('P%d(LL-inf) ', P);
                            end
                        else
                            fprintf('P%d(MSE-bad:%.2e) ', P, mse);
                        end
                    else
                        fprintf('P%d(pred-fail:%d/%d) ', P, valid_predictions, N-P);
                    end
                    
                catch inner_err
                    fprintf('P%d(calc-err:%s) ', P, inner_err.message(1:min(15,end)));
                end
            else
                fprintf('P%d(fit-fail) ', P);
            end
            
        catch outer_err
            fprintf('P%d(outer-err:%s) ', P, outer_err.message(1:min(15,end)));
        end
    end
    fprintf('\n');
    
    % Find best order
    valid_bic = bic_values(isfinite(bic_values));
    
    if ~isempty(valid_bic)
        [best_bic, best_idx] = min(bic_values);
        
        if isfinite(best_bic)
            opt_order = test_orders(best_idx);
            ar_params = all_params{best_idx};
            sigma_e = all_sigma_e(best_idx);
            best_lml = -best_bic;
            
            fprintf('      Selected P=%d with BIC=%.2f\n', opt_order, best_bic);
        else
            fprintf('      BIC selection failed, using first valid AR fit\n');
            % Find first successful fit
            first_valid = find(~cellfun(@isempty, all_params), 1);
            if ~isempty(first_valid)
                opt_order = test_orders(first_valid);
                ar_params = all_params{first_valid};
                sigma_e = all_sigma_e(first_valid);
            else
                opt_order = 2;
                ar_params = [0.5; -0.3];
                sigma_e = std(segment);
            end
            best_lml = -inf;
        end
    else
        fprintf('      No valid BIC values, using fallback P=2\n');
        opt_order = 2;
        ar_params = [0.5; -0.3];
        sigma_e = std(segment);
        best_lml = -inf;
    end
end

function lml = calculateLogMarginalLikelihood(segment, P, prior_sigma, sigma_e)
    % Calculate log marginal likelihood for AR model
    
    N = length(segment);
    
    if N <= P + 10
        lml = -inf;
        return;
    end
    
    % Construct design matrix
    G = zeros(N-P, P);
    for i = 1:(N-P)
        G(i, :) = flip(segment(i:(i+P-1)));
    end
    y_vec = segment(P+1:N);
    
    % Prior parameters
    mu0 = zeros(P, 1);              % Prior mean
    C0 = (prior_sigma^2) * eye(P);  % Prior covariance
    C0_inv = inv(C0);
    
    try
        % Calculate precision matrices and MAP estimate
        Phi = (G'*G) + (sigma_e^(-2)) * C0_inv;
        Theta = (G'*y_vec) + (sigma_e^(-2)) * C0_inv * mu0;
        theta_MAP = Phi \ Theta;
        
        % Log marginal likelihood formula
        lml = -P/2 * log(2*pi) ...                                    % Prior normalization
              - 1/2 * log(det(C0)) ...                                % Prior determinant  
              - 1/2 * log(det(Phi)) ...                               % Posterior precision det
              - (N-P)/2 * log(2*pi*sigma_e^2) ...                    % Likelihood normalization
              - (y_vec'*y_vec + (sigma_e^(-2)) * (mu0' * C0_inv * mu0) - Theta'*theta_MAP) / (2*sigma_e^2);
        
    catch
        lml = -inf;
    end
end

function visualizeSegmentAnalysis(segment_models, filename)
    % Visualize segment analysis results
    
    figure('Position', [50, 50, 1600, 800]);
    
    % Plot 1: Optimal orders over time
    subplot(2, 3, 1);
    segment_centers = (segment_models.segment_starts + segment_models.segment_ends) / 2;
    plot(segment_centers, segment_models.optimal_orders, 'bo-', 'MarkerSize', 4);
    xlabel('Sample Index'); ylabel('Optimal AR Order');
    title('Optimal AR Order vs Time');
    grid on;
    
    % Plot 2: Histogram of optimal orders
    subplot(2, 3, 2);
    histogram(segment_models.optimal_orders, 'BinMethod', 'integers', 'FaceAlpha', 0.7);
    xlabel('AR Model Order'); ylabel('Count');
    title(sprintf('Distribution of Optimal Orders (μ=%.1f)', mean(segment_models.optimal_orders)));
    grid on;
    
    % Plot 3: Log marginal likelihood over time
    subplot(2, 3, 3);
    valid_lml = segment_models.log_marginal_likelihoods(isfinite(segment_models.log_marginal_likelihoods));
    valid_centers = segment_centers(isfinite(segment_models.log_marginal_likelihoods));
    plot(valid_centers, valid_lml, 'ro-', 'MarkerSize', 3);
    xlabel('Sample Index'); ylabel('Log Marginal Likelihood');
    title('Model Evidence vs Time');
    grid on;
    
    % Plot 4: Noise variance over time
    subplot(2, 3, 4);
    plot(segment_centers, segment_models.sigma_e_values, 'go-', 'MarkerSize', 3);
    xlabel('Sample Index'); ylabel('Noise Std (σₑ)');
    title('Estimated Noise Level vs Time');
    grid on;
    
    % Plot 5: Order vs Log Marginal Likelihood scatter
    subplot(2, 3, 5);
    valid_orders = segment_models.optimal_orders(isfinite(segment_models.log_marginal_likelihoods));
    scatter(valid_orders, valid_lml, 30, 'filled');
    xlabel('Optimal AR Order'); ylabel('Log Marginal Likelihood');
    title('Model Order vs Evidence');
    grid on;
    
    % Plot 6: Summary statistics
    subplot(2, 3, 6);
    axis off;
    
    % Calculate statistics
    order_counts = histcounts(segment_models.optimal_orders, min(segment_models.optimal_orders):max(segment_models.optimal_orders));
    [~, mode_idx] = max(order_counts);
    mode_order = min(segment_models.optimal_orders) + mode_idx - 1;
    
    summary_text = {
        'SEGMENT ANALYSIS SUMMARY:';
        '';
        sprintf('Total segments: %d', length(segment_models.optimal_orders));
        sprintf('Mean order: %.1f ± %.1f', mean(segment_models.optimal_orders), std(segment_models.optimal_orders));
        sprintf('Median order: %.1f', median(segment_models.optimal_orders));
        sprintf('Mode order: %d', mode_order);
        sprintf('Order range: [%d, %d]', min(segment_models.optimal_orders), max(segment_models.optimal_orders));
        '';
        sprintf('Mean σₑ: %.4f', mean(segment_models.sigma_e_values));
        sprintf('Mean log evidence: %.1f', mean(valid_lml));
    };
    
    text(0.05, 0.95, summary_text, 'Units', 'normalized', 'VerticalAlignment', 'top', ...
         'FontSize', 10);
    
    sgtitle(sprintf('Segment-wise AR Model Analysis: %s', filename));
end

function performSegmentBasedInterpolation(audio_data, fs, segment_models, filename)
    % Perform interpolation using segment-specific AR models
    fprintf('\n  STEP 2: Segment-based interpolation...\n');
    
    % Create missing data pattern as specified in handout:
    % "missing chunks of 100 data points out of each run of 1000"
    chunk_size = 1000;
    missing_size = 100;
    
    total_chunks = floor(length(audio_data) / chunk_size);
    fprintf('    Creating missing data pattern: %d/%d samples missing (%.1f%%)\n', ...
            missing_size, chunk_size, missing_size/chunk_size*100);
    
    % Create missing pattern
    missing_mask = false(length(audio_data), 1);
    for chunk = 1:total_chunks
        chunk_start = (chunk-1) * chunk_size + 1;
        chunk_end = min(chunk_start + chunk_size - 1, length(audio_data));
        
        % Random position within chunk for missing data
        if chunk_end - chunk_start + 1 >= missing_size
            missing_start = chunk_start + randi(chunk_end - chunk_start + 1 - missing_size);
            missing_end = missing_start + missing_size - 1;
            missing_mask(missing_start:missing_end) = true;
        end
    end
    
    fprintf('    Total missing samples: %d (%.1f%% of audio)\n', ...
            sum(missing_mask), sum(missing_mask)/length(missing_mask)*100);
    
    % Create corrupted audio
    corrupted_audio = audio_data;
    corrupted_audio(missing_mask) = 0;
    
    % Reconstruct using segment-based approach
    fprintf('    Reconstructing using segment-specific AR models...\n');
    reconstructed_audio = reconstructWithSegmentModels(audio_data, missing_mask, segment_models);
    
    % Also create baseline reconstruction using fixed AR order
    baseline_order = round(median(segment_models.optimal_orders));
    fprintf('    Creating baseline reconstruction (fixed P=%d)...\n', baseline_order);
    reconstructed_baseline = reconstructWithFixedOrder(audio_data, missing_mask, baseline_order);
    
    % Calculate performance metrics
    missing_samples = audio_data(missing_mask);
    adaptive_samples = reconstructed_audio(missing_mask);
    baseline_samples = reconstructed_baseline(missing_mask);
    
    mse_adaptive = mean((missing_samples - adaptive_samples).^2);
    mse_baseline = mean((missing_samples - baseline_samples).^2);
    
    fprintf('\n    INTERPOLATION RESULTS:\n');
    fprintf('      Segment-adaptive MSE: %.6f\n', mse_adaptive);
    fprintf('      Fixed-order MSE:      %.6f\n', mse_baseline);
    if mse_baseline > 0
        fprintf('      Improvement ratio:    %.2fx\n', mse_baseline/mse_adaptive);
    end
    
    % Save audio files
    [~, base_name, ~] = fileparts(filename);
    
    % Normalize to prevent clipping
    all_audio = [corrupted_audio(:); reconstructed_audio(:); reconstructed_baseline(:)];
    max_val = max(abs(all_audio));
    if max_val > 0
        norm_factor = 0.95 / max_val;
    else
        norm_factor = 1;
    end
    
    audiowrite([base_name '_corrupted.wav'], corrupted_audio * norm_factor, fs);
    audiowrite([base_name '_reconstructed_adaptive.wav'], reconstructed_audio * norm_factor, fs);
    audiowrite([base_name '_reconstructed_baseline.wav'], reconstructed_baseline * norm_factor, fs);
    
    fprintf('      Saved: %s_corrupted.wav\n', base_name);
    fprintf('      Saved: %s_reconstructed_adaptive.wav\n', base_name);
    fprintf('      Saved: %s_reconstructed_baseline.wav\n', base_name);
    
    % Visualize results
    visualizeInterpolationResults(audio_data, missing_mask, reconstructed_audio, ...
                                  reconstructed_baseline, segment_models, fs, filename);
end

function reconstructed = reconstructWithSegmentModels(original, missing_mask, segment_models)
    % Reconstruct audio using segment-specific AR models
    
    reconstructed = original;
    missing_segments = findMissingSegments(missing_mask);
    
    fprintf('      Processing %d missing segments...\n', size(missing_segments, 1));
    
    for seg_idx = 1:size(missing_segments, 1)
        start_miss = missing_segments(seg_idx, 1);
        end_miss = missing_segments(seg_idx, 2);
        
        % Find which analysis segment this missing part belongs to
        segment_idx = findCorrespondingSegment(start_miss, end_miss, segment_models);
        
        if segment_idx > 0
            % Use the AR model from this segment
            ar_order = segment_models.optimal_orders(segment_idx);
            ar_params = segment_models.ar_parameters{segment_idx};
            
            fprintf('        Gap %d (samples %d-%d): using segment %d, P=%d\n', ...
                    seg_idx, start_miss, end_miss, segment_idx, ar_order);
        else
            % Fallback: use median order and default parameters
            ar_order = round(median(segment_models.optimal_orders));
            ar_params = zeros(ar_order, 1);
            if ar_order >= 2
                ar_params(1:2) = [0.5; -0.3];
            end
            
            fprintf('        Gap %d (samples %d-%d): using fallback P=%d\n', ...
                    seg_idx, start_miss, end_miss, ar_order);
        end
        
        % Interpolate this gap
        interpolated_gap = interpolateGap(original, start_miss, end_miss, ar_params);
        reconstructed(start_miss:end_miss) = interpolated_gap;
    end
end

function segment_idx = findCorrespondingSegment(start_miss, end_miss, segment_models)
    % Find which analysis segment contains or is closest to the missing data
    
    gap_center = (start_miss + end_miss) / 2;
    
    % Find segment that contains this gap center
    for i = 1:length(segment_models.segment_starts)
        if gap_center >= segment_models.segment_starts(i) && gap_center <= segment_models.segment_ends(i)
            segment_idx = i;
            return;
        end
    end
    
    % If no containing segment, find closest one
    segment_centers = (segment_models.segment_starts + segment_models.segment_ends) / 2;
    [~, segment_idx] = min(abs(segment_centers - gap_center));
end

function interpolated_gap = interpolateGap(signal, start_miss, end_miss, ar_params)
    % Simple forward/backward interpolation as described in handout
    
    gap_length = end_miss - start_miss + 1;
    P = length(ar_params);
    N = length(signal);
    
    % Check if we have reasonable AR parameters
    if all(ar_params == 0) || any(isnan(ar_params))
        % Fallback: simple linear interpolation
        if start_miss > 1 && end_miss < N
            start_val = signal(start_miss - 1);
            end_val = signal(end_miss + 1);
            interpolated_gap = linspace(start_val, end_val, gap_length)';
        else
            interpolated_gap = zeros(gap_length, 1);
        end
        return;
    end
    
    % Forward prediction: x̂ₙᶠ = Σaᵢx̂ₙ₋ᵢᶠ + eₙᶠ
    forward_pred = zeros(gap_length, 1);
    for i = 1:gap_length
        current_idx = start_miss + i - 1;
        pred_data = zeros(P, 1);
        
        for j = 1:P
            look_back_idx = current_idx - j;
            if look_back_idx >= 1 && look_back_idx < start_miss
                pred_data(j) = signal(look_back_idx);  % Use original signal
            elseif look_back_idx >= start_miss && look_back_idx < current_idx
                pred_data(j) = forward_pred(look_back_idx - start_miss + 1);  % Use prediction
            else
                pred_data(j) = 0;  % Zero padding for boundary cases
            end
        end
        
        forward_pred(i) = sum(ar_params .* flip(pred_data));
        % Bound the prediction to prevent explosion
        forward_pred(i) = max(-1, min(1, forward_pred(i)));
    end
    
    % Backward prediction: x̂ₙᵇ = Σaᵢx̂ₙ₊ᵢᵇ + eₙᵇ  
    backward_pred = zeros(gap_length, 1);
    for i = gap_length:-1:1
        current_idx = start_miss + i - 1;
        pred_data = zeros(P, 1);
        
        for j = 1:P
            look_forward_idx = current_idx + j;
            if look_forward_idx <= N && look_forward_idx > end_miss
                pred_data(j) = signal(look_forward_idx);  % Use original signal
            elseif look_forward_idx <= end_miss && look_forward_idx > current_idx
                pred_data(j) = backward_pred(look_forward_idx - start_miss + 1);  % Use prediction
            else
                pred_data(j) = 0;  % Zero padding for boundary cases
            end
        end
        
        backward_pred(i) = sum(ar_params .* pred_data);
        % Bound the prediction to prevent explosion
        backward_pred(i) = max(-1, min(1, backward_pred(i)));
    end
    
    % Weighted combination: x̂ₙ = αx̂ₙᶠ + (1-α)x̂ₙᵇ
    % α varies smoothly from 1 to 0 as described in handout
    if gap_length == 1
        alpha = 0.5;
    else
        alpha = linspace(1, 0, gap_length)';
    end
    
    interpolated_gap = alpha .* forward_pred + (1 - alpha) .* backward_pred;
    
    % Final bounds check to ensure audio doesn't explode
    interpolated_gap = max(-1, min(1, interpolated_gap));
end

function reconstructed = reconstructWithFixedOrder(original, missing_mask, fixed_order)
    % Baseline reconstruction using fixed AR order
    
    reconstructed = original;
    missing_segments = findMissingSegments(missing_mask);
    
    % Estimate AR parameters from entire available signal (simplified)
    available_signal = original(~missing_mask);
    if length(available_signal) > fixed_order + 20
        try
            [ar_params, ~] = estimateAR_ML(available_signal, fixed_order);
        catch
            ar_params = zeros(fixed_order, 1);
            if fixed_order >= 2
                ar_params(1:2) = [0.5; -0.3];
            end
        end
    else
        ar_params = zeros(fixed_order, 1);
        if fixed_order >= 2
            ar_params(1:2) = [0.5; -0.3];
        end
    end
    
    for seg_idx = 1:size(missing_segments, 1)
        start_miss = missing_segments(seg_idx, 1);
        end_miss = missing_segments(seg_idx, 2);
        
        interpolated_gap = interpolateGap(original, start_miss, end_miss, ar_params);
        reconstructed(start_miss:end_miss) = interpolated_gap;
    end
end

function visualizeInterpolationResults(original, missing_mask, adaptive, baseline, segment_models, fs, filename)
    % Visualize interpolation results with segment information
    
    figure('Position', [250, 250, 1600, 1000]);
    
    % Select representative segment for detailed view
    segment_start = round(length(original) * 0.3);
    segment_length = round(0.3 * fs);  % 300ms
    segment_end = min(segment_start + segment_length, length(original));
    
    t = (segment_start:segment_end) / fs;
    
    % Plot 1: Detailed time-domain view
    subplot(2, 2, 1);
    plot(t, original(segment_start:segment_end), 'k-', 'LineWidth', 1.5, 'DisplayName', 'Original');
    hold on;
    
    seg_missing = missing_mask(segment_start:segment_end);
    if any(seg_missing)
        t_missing = t(seg_missing);
        missing_indices = segment_start + find(seg_missing) - 1;
        plot(t_missing, original(missing_indices), 'rx', ...
             'MarkerSize', 6, 'LineWidth', 2, 'DisplayName', 'Missing');
        plot(t_missing, adaptive(missing_indices), 'go', ...
             'MarkerSize', 4, 'DisplayName', 'Adaptive');
        plot(t_missing, baseline(missing_indices), 'bo', ...
             'MarkerSize', 3, 'DisplayName', 'Fixed Order');
    end
    
    xlabel('Time (s)'); ylabel('Amplitude');
    title('Detailed View: Interpolation Comparison');
    legend('show', 'Location', 'best');
    grid on;
    
    % Plot 2: Power spectral density comparison
    subplot(2, 2, 2);
    try
        [psd_orig, f] = pwelch(original, [], [], [], fs);
        [psd_adaptive, ~] = pwelch(adaptive, [], [], [], fs);
        [psd_baseline, ~] = pwelch(baseline, [], [], [], fs);
        
        semilogx(f, 10*log10(psd_orig), 'k-', 'LineWidth', 2, 'DisplayName', 'Original');
        hold on;
        semilogx(f, 10*log10(psd_adaptive), 'g--', 'LineWidth', 1.5, 'DisplayName', 'Adaptive');
        semilogx(f, 10*log10(psd_baseline), 'b:', 'LineWidth', 1.5, 'DisplayName', 'Fixed Order');
        
        xlabel('Frequency (Hz)'); ylabel('PSD (dB)');
        title('Power Spectral Density Comparison');
        legend('show', 'Location', 'best');
        grid on;
    catch
        text(0.5, 0.5, 'PSD calculation failed', 'HorizontalAlignment', 'center');
    end
    
    % Plot 3: Error analysis
    subplot(2, 2, 3);
    if any(missing_mask)
        missing_samples = original(missing_mask);
        adaptive_error = abs(adaptive(missing_mask) - missing_samples);
        baseline_error = abs(baseline(missing_mask) - missing_samples);
        
        histogram(adaptive_error, 30, 'FaceAlpha', 0.7, 'DisplayName', 'Adaptive');
        hold on;
        histogram(baseline_error, 30, 'FaceAlpha', 0.7, 'DisplayName', 'Fixed Order');
        
        xlabel('Absolute Error'); ylabel('Count');
        title('Error Distribution');
        legend('show');
        grid on;
    end
    
    % Plot 4: Segment model orders used
    subplot(2, 2, 4);
    segment_centers = (segment_models.segment_starts + segment_models.segment_ends) / 2 / fs;
    plot(segment_centers, segment_models.optimal_orders, 'bo-', 'MarkerSize', 4);
    xlabel('Time (s)'); ylabel('AR Model Order');
    title('Segment-wise Optimal AR Orders');
    grid on;
    ylim([min(segment_models.optimal_orders)-0.5, max(segment_models.optimal_orders)+0.5]);
    
    sgtitle(sprintf('Segment-based AR Interpolation Results: %s', filename));
end

% Helper functions
function missing_segments = findMissingSegments(missing_mask)
    % Find contiguous segments of missing data
    diff_mask = diff([0; missing_mask; 0]);
    start_indices = find(diff_mask == 1);
    end_indices = find(diff_mask == -1) - 1;
    missing_segments = [start_indices, end_indices];
end

function [a_est, sigma_e_est, fit_success] = estimateAR_ML_robust(y, P)
    % Robust Maximum likelihood estimation of AR parameters
    N = length(y);
    fit_success = false;
    
    if N <= P + 20 || P <= 0
        a_est = zeros(P, 1);
        sigma_e_est = std(y) + 1e-6;
        return;
    end
    
    try
        % Construct design matrix
        G = zeros(N-P, P);
        for i = 1:(N-P)
            G(i, :) = flip(y(i:(i+P-1)));
        end
        y_vec = y(P+1:N);
        
        % Check for data issues
        if any(~isfinite(G(:))) || any(~isfinite(y_vec))
            fprintf('[Data has NaN/Inf] ');
            a_est = zeros(P, 1);
            sigma_e_est = std(y) + 1e-6;
            return;
        end
        
        % Check matrix condition
        GtG = G' * G;
        condition_number = cond(GtG);
        
        if condition_number > 1e15
            fprintf('[Matrix ill-conditioned: %.1e] ', condition_number);
            a_est = zeros(P, 1);
            sigma_e_est = std(y) + 1e-6;
            return;
        end
        
        % Regularized solution for numerical stability
        if rank(GtG) < P || condition_number > 1e10
            lambda = 1e-8 * trace(GtG) / P;  % Small regularization
            a_est = (GtG + lambda * eye(P)) \ (G' * y_vec);
        else
            a_est = GtG \ (G' * y_vec);
        end
        
        % Check solution validity
        if any(~isfinite(a_est))
            fprintf('[Solution has NaN/Inf] ');
            a_est = zeros(P, 1);
            sigma_e_est = std(y) + 1e-6;
            return;
        end
        
        % Check for explosive solutions
        if any(abs(a_est) > 10)
            fprintf('[Explosive coeffs: max=%.2f] ', max(abs(a_est)));
            a_est = zeros(P, 1);
            sigma_e_est = std(y) + 1e-6;
            return;
        end
        
        % Estimate noise variance with bounds
        residuals = y_vec - G * a_est;
        if any(~isfinite(residuals))
            fprintf('[Residuals have NaN/Inf] ');
            a_est = zeros(P, 1);
            sigma_e_est = std(y) + 1e-6;
            return;
        end
        
        sigma_e_est = sqrt(mean(residuals.^2));
        sigma_e_est = max(sigma_e_est, 1e-8);  % Minimum noise floor
        
        fit_success = true;
        
    catch ME
        fprintf('[Exception: %s] ', ME.message(1:min(20,end)));
        a_est = zeros(P, 1);
        sigma_e_est = std(y) + 1e-6;
        fit_success = false;
    end
end