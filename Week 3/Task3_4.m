%% Task 3.4 - Bayesian Signal Detection

% To find a signal buried at some offset {0,1,...85}
% We set up a linear gaussian model for each offset -> 86 models
% Perform model selection of the 86 models

clc; clearvars;

% Load the hidden data
load('hidden_data.mat'); % Assumes this contains variable 'y'

% Given parameters from the data generation
N = 100;
sigma_n = 2;    % Noise standard deviation
sigma_theta = 1; % Signal amplitude prior std dev
signal = [-3 5 -2 4 1 3 5 -1 2 4 6 5 -2 -2 1]; % Known signal pattern
signal_length = length(signal);

% If hidden_data.mat doesn't exist, generate test data
if ~exist('y', 'var')
    fprintf('Generating test data...\n');
    y = generateHiddenData(N, sigma_n, sigma_theta, signal);
end

% y = generateHiddenData(N, sigma_n, sigma_theta, signal, 12);


fprintf('=== BAYESIAN SIGNAL DETECTION ===\n');
fprintf('Data length: %d\n', length(y));
fprintf('Signal pattern length: %d\n', signal_length);
fprintf('Noise std dev: %.2f\n', sigma_n);
fprintf('Signal amplitude prior std dev: %.2f\n', sigma_theta);

% Find the hidden signal using Bayesian model selection
[best_offset, posterior_probs, theta_estimates, log_marginals] = ...
    detectHiddenSignal(y, signal, sigma_n, sigma_theta);

% Display results
fprintf('\n=== DETECTION RESULTS ===\n');
fprintf('Most probable offset: %d\n', best_offset);
fprintf('Posterior probability: %.4f\n', max(posterior_probs));
fprintf('Estimated theta: %.3f\n', theta_estimates(best_offset + 1));

% Calculate null hypothesis probability
null_log_ml = calculateNullModel(y, sigma_n);
all_log_ml = [log_marginals', null_log_ml];
all_probs = exp(all_log_ml - max(all_log_ml));
all_probs = all_probs / sum(all_probs);
null_prob = all_probs(end);

fprintf('Null hypothesis probability: %.6f\n', null_prob);
fprintf('Evidence ratio (signal/null): %.2f\n', max(posterior_probs)/null_prob);

% Plot results
plotResults(y, signal, best_offset, posterior_probs, theta_estimates, null_prob);

% Test sensitivity to prior parameters
testPriorSensitivity(y, signal, sigma_n);

% Main detection function
function [best_offset, posterior_probs, theta_estimates, log_marginals] = ...
    detectHiddenSignal(y, signal, sigma_n, sigma_theta)
    
    N = length(y);
    signal_length = length(signal);
    max_offset = N - signal_length; % Maximum valid offset
    
    log_marginals = zeros(max_offset + 1, 1);
    theta_estimates = zeros(max_offset + 1, 1);
    
    fprintf('Computing marginal likelihoods for %d offset positions...\n', max_offset + 1);
    
    % Test each possible offset
    for offset = 0:max_offset

        % Model: y = G*theta + e
        % (G is length N, with 16 elements nonzero)
        % (theta is a scalar) - the parameter to infer

        % Create design matrix for this offset
        G = zeros(N, 1);
        G(offset + 1:offset + signal_length) = signal';
        
        % Calculate marginal likelihood and MAP estimate
        [log_ml, theta_map] = calculateModelEvidence(y, G, 0, sigma_theta^2, sigma_n);
        
        log_marginals(offset + 1) = log_ml;
        theta_estimates(offset + 1) = theta_map;
        
    end
    
    % Convert to normalized posterior probabilities
    posterior_probs = exp(log_marginals - max(log_marginals));
    posterior_probs = posterior_probs / sum(posterior_probs);
    
    % Find MAP offset
    [~, best_idx] = max(posterior_probs);
    best_offset = best_idx - 1; % Convert back to 0-based indexing
end

function [log_ml, theta_map] = calculateModelEvidence(y, G, mu0, sigma0, sigma_n)
    % Calculate log marginal likelihood for linear Gaussian model
    % y = G*theta + noise, theta ~ N(mu0, sigma0_sq)
    
    N = length(y);
    
    if all(G == 0) % Handle null model case
        log_ml = -N/2 * log(2*pi*sigma_n^2) - sum(y.^2)/(2*sigma_n^2);
        theta_map = 0;
        return;
    end
    
    % Posterior calculations
    Phi = (G' * G) / sigma_n^2 + 1/(sigma0^2);
    Theta = (G' * y) / sigma_n^2 + mu0/(sigma0^2);
    theta_map = Theta / Phi;
    
    % Log marginal likelihood (Laplace approximation)
    log_ml = -1/2 * log(2*pi*(sigma0^2)) ...           % Prior normalization
             -1/2 * log(2*pi/Phi) ...                  % Posterior normalization
             -(N-1)/2 * log(2*pi*sigma_n^2) ...        % Likelihood normalization
             -(sum(y.^2) + mu0^2/(sigma0^2) - Theta^2/Phi)/(2*sigma_n^2); % Fit term
end

function null_log_ml = calculateNullModel(y, sigma_n)
    % Calculate marginal likelihood for null model: y = noise only
    N = length(y);
    null_log_ml = -N/2 * log(2*pi*sigma_n^2) - sum(y.^2)/(2*sigma_n^2);
end

function plotResults(y, signal, best_offset, posterior_probs, theta_estimates, null_prob)
    figure('Position', [100, 100, 1200, 800]);
    
    % Plot 1: Data and detected signal
    subplot(2,1,1);
    plot(y, 'b-', 'LineWidth', 1); hold on;
    
    % Overlay detected signal
    if best_offset >= 0
        signal_overlay = zeros(size(y));
        signal_start = best_offset + 1;
        signal_end = min(signal_start + length(signal) - 1, length(y));
        signal_overlay(signal_start:signal_end) = signal(1:(signal_end-signal_start+1)) * theta_estimates(best_offset + 1);
        plot(signal_overlay, 'r-', 'LineWidth', 2);
    end
    
    xlabel('Sample Index'); ylabel('Amplitude');
    title(sprintf('Data and Detected Signal (Offset = %d)', best_offset));
    legend('Observed Data', 'Detected Signal');
    grid on;
    
    % Plot 2: Posterior probabilities for offset
    subplot(2,1,2);
    offsets = 0:length(posterior_probs)-1;
    bar(offsets, posterior_probs, 'FaceColor', [0.3, 0.6, 0.9]);
    xlabel('Offset Position'); ylabel('Posterior Probability');
    title('Posterior Distribution of Signal Offset');
    
    
    % Mark the most probable offset
    hold on;
    [max_prob, max_idx] = max(posterior_probs);
    plot(max_idx-1, max_prob, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'red');
    legend("", "MAP Estimate of offset");
    grid on;

    sgtitle('Bayesian Signal Detection Results');
    
    % Plot 3: Theta estimates vs offset
    figure;
    plot(offsets, theta_estimates, 'g-', 'LineWidth', 1.5); hold on;
    plot(best_offset, theta_estimates(best_offset + 1), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'red');
    xlabel('Offset Position'); ylabel('Estimated θ');
    title('MAP Estimates of Signal Amplitude');
    legend("",sprintf("MAP theta at offset=%d is %.3f", best_offset, theta_estimates(best_offset + 1)))
    grid on;
    
    % Plot 4: Model comparison (signal vs null)
    figure;
    signal_prob = 1 - null_prob;
    bar_data = [signal_prob, null_prob];
    bar(bar_data, 'FaceColor', [0.7, 0.7, 0.7]);
    set(gca, 'XTickLabel', {'Signal Present', 'Null (No Signal)'});
    ylabel('Probability');
    title('Signal Detection: Signal vs Null Hypothesis');
    legend(sprintf("Prob. of Null Hypothesis = %.4f", null_prob))
    grid on;
    
    % sgtitle('Bayesian Signal Detection Results');
end

function testPriorSensitivity(y, signal, sigma_n)
    fprintf('\n=== TESTING PRIOR SENSITIVITY ===\n');
    
    % Test different values of sigma_theta
    sigma_theta_values = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10, 50, 100];
    
    results = [];
    for i = 1:length(sigma_theta_values)
        sigma_theta = sigma_theta_values(i);
        
        [best_offset, posterior_probs, theta_estimates, ~] = ...
            detectHiddenSignal(y, signal, sigma_n, sigma_theta);
        
        null_log_ml = calculateNullModel(y, sigma_n);
        signal_log_ml = max(log(posterior_probs));
        
        % Calculate Bayes factor (signal vs null)
        bayes_factor = exp(signal_log_ml - null_log_ml);
        
        results = [results; sigma_theta, best_offset, max(posterior_probs), ...
                  theta_estimates(best_offset + 1), bayes_factor];
        
        fprintf('σ_θ = %.1f: Offset = %2d, Prob = %.4f, θ = %6.3f, BF = %8.2f\n', ...
                sigma_theta, best_offset, max(posterior_probs), ...
                theta_estimates(best_offset + 1), bayes_factor);
    end
    
    % Plot sensitivity analysis
    figure('Position', [200, 200, 1000, 600]);
    
    subplot(2,2,1);
    semilogx(results(:,1), results(:,2), 'bo-', 'LineWidth', 2, 'MarkerSize', 6);
    xlabel('Prior σ_θ'); ylabel('Best Offset');
    title('Detected Offset vs Prior');
    grid on;
    
    subplot(2,2,2);
    semilogx(results(:,1), results(:,3), 'ro-', 'LineWidth', 2, 'MarkerSize', 6);
    xlabel('Prior σ_θ'); ylabel('Max Posterior Probability');
    title('Detection Confidence vs Prior');
    grid on;
    
    subplot(2,2,3);
    semilogx(results(:,1), results(:,4), 'go-', 'LineWidth', 2, 'MarkerSize', 6);
    xlabel('Prior σ_θ'); ylabel('Estimated θ');
    title('Amplitude Estimate vs Prior');
    grid on;
    
    subplot(2,2,4);
    loglog(results(:,1), results(:,5), 'mo-', 'LineWidth', 2, 'MarkerSize', 6);
    xlabel('Prior σ_θ'); ylabel('Bayes Factor (Signal/Null)');
    title('Evidence Strength vs Prior');
    grid on;
    
    sgtitle('Sensitivity to Prior Parameter σ_θ');
end

function y = generateHiddenData(N, sigma_n, sigma_theta, signal, offset)
    % Generate test data (for when hidden_data.mat is not available)
    
    noise = sigma_n * randn(N, 1);
    theta = sigma_theta * randn(1);

    if nargin < 5  % Check if 'offset' is provided
        offset = round(85 * rand(1));
    end
    
    y = noise;
    signal_offset = zeros(N, 1);
    signal_offset(offset+1:offset+length(signal)) = signal' * theta;
    y = y + signal_offset;
    
    fprintf('Generated test data with offset = %d, theta = %.3f\n', offset, theta);
end
