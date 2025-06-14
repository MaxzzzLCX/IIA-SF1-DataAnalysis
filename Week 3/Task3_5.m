%% Task 3.5  - Auto-Regressive (AR) Model

clc, clearvars

% AR process of order P can be written as:
% y = Ga + e, 
% where y = [y_{P+1}, ..., y_N] (vector of last N-P signals)
% e is 1 x (N-P) 
% G is (N-P) x P matrix


P = 2;
N = 100;
% Generate a sample AR process signal
a = [0.5, -0.3];
first_signals = randn(P, 1);
y = generate_AR_signal(N, a, first_signals);

% y = randn(N, 1); % Example signal, replace with actual AR process if needed
G = compute_G_matrix(y, N, P);

function y = generate_AR_signal(N, a, first_signals, sigma_e)
    % first_signals are the first elements [x1, x2, ..., xP] of x
       
    P = length(a);
    y = zeros(N,1);
    y(1:P,1) = first_signals;

    for n=P+1:N
        % Iteratively calculate the next element
        y(n,1) = dot(a, flip(y(n-P:n-1))) + sigma_e * randn; % convolution of AR model

    end  
end

function G = compute_G_matrix(y, N, P)
    
    G = zeros(N-P, P);
    for i = 1:(N-P)
        G(i, :) = flip(y(i:(i+P-1)));
    end
end



%% Task 3.5 - AR Model: DFT and Power Spectrum Analysis
clc, clearvars

% AR(2) process parameters
P = 2;
N = 1000;  % Use longer signal for better spectral analysis
sigma_e = 0.01; % standard deviation of noise

% Test different AR parameters
fprintf('=== AR MODEL SPECTRUM ANALYSIS ===\n');

% Case 1: Stable AR(2) process
a1 = [0.5, -0.3];
fprintf('Case 1 - Stable AR(2): a = [%.1f, %.1f]\n', a1);
analyzeARSpectrum(N, a1, sprintf('Stable AR(2) a = [%.1f, %.1f]\n', a1), sigma_e);


% Case 2: Border Case AR(2) process - |pole|=1
a2 = [0.5, 0.5];
fprintf('Case 2 - Edge Case AR(2): a = [%.1f, %.1f]\n', a2);
analyzeARSpectrum(N, a2, sprintf('Edge Case AR(2) a = [%.1f, %.1f]\n', a2), sigma_e);

% Case 3: Unstable AR(2) process 
a3 = [-0.4, 0.7];
fprintf('Case 3 - Untable AR(2): a = [%.1f, %.1f]\n', a3);
analyzeARSpectrum(N, a3, sprintf('Untable AR(2) a = [%.1f, %.1f]\n', a3), sigma_e);

% Case 4: Unstable AR(2) process 
a4 = [0.5, 0.7];
fprintf('Case 4 - Untable AR(2): a = [%.1f, %.1f]\n', a4);
analyzeARSpectrum(N, a4, sprintf('Untable AR(2) a = [%.1f, %.1f]\n', a4), sigma_e);


function analyzeARSpectrum(N, a, title_str, sigma_e)
    P = length(a);
    
    % Generate AR signal
    first_signals = sigma_e * randn(P, 1);
    y = generate_AR_signal(N, a, first_signals, sigma_e);
    
    % Calculate empirical power spectrum using DFT
    Y = fft(y);
    empirical_psd = abs(Y).^2 / N;  % Periodogram estimate
    
    % Calculate theoretical power spectrum
    freq = (0:N-1)/N *2*pi;  % Normalized frequencies [-pi, pi]
    theoretical_psd = calculateTheoreticalPSD(a, freq, sigma_e);
    
    % Check pole positions for stability
    [poles, is_stable] = checkStability(a);
    
    % Plot results
    % figure('Position', [100, 100, 1200, 800]);
    
    % Plot 1A: Time series
    figure;
    subplot(2,1,1)
    plot(y(1:min(200, N)), 'b-', 'LineWidth', 1);
    xlabel('Sample Index'); ylabel('Amplitude');
    title(sprintf('%s - Time Series', title_str));
    grid on;

    % Plot 1B: Pole-zero plot
    subplot(2, 1, 2);
    plotPoleZero(a);
    

    % Plot 2A: Empirical vs Theoretical PSD (linear scale)
    figure;
    subplot(2, 1, 1);
    % Note: for real signals, only plot positive frequencies
    plot(freq(1:N/2), empirical_psd(1:N/2), 'b-', 'LineWidth', 1); hold on;
    plot(freq(1:N/2), theoretical_psd(1:N/2), 'r--', 'LineWidth', 2);
    xlabel('Normalized Frequency'); ylabel('Power Spectral Density');
    title('PSD: Empirical vs Theoretical');
    legend('Empirical (DFT)', 'Theoretical', 'Location', 'best');
    grid on;
    
    % Plot 2B: PSD in dB scale
    subplot(2, 1, 2);
    plot(freq(1:N/2), 10*log10(empirical_psd(1:N/2)), 'b-', 'LineWidth', 1); hold on;
    plot(freq(1:N/2), 10*log10(theoretical_psd(1:N/2)), 'r--', 'LineWidth', 2);
    xlabel('Normalized Frequency'); ylabel('Power (dB)');
    title('PSD in dB Scale');
    legend('Empirical', 'Theoretical', 'Location', 'best');
    grid on;
    
    
    % This following part about autocorrleation is not asked by handout
    %{
    % Plot 3: Autocorrelation
    figure;
    [autocorr_emp, lags] = xcorr(y, 20, 'biased');
    autocorr_theor = calculateTheoreticalAutocorr(a, lags);
    
    stem(lags, autocorr_emp, 'b-', 'LineWidth', 1); hold on;
    plot(lags, autocorr_theor, 'ro-', 'LineWidth', 2);
    xlabel('Lag'); ylabel('Autocorrelation');
    title('Autocorrelation Function');
    legend('Empirical', 'Theoretical', 'Location', 'best');
    grid on;
    %}
   
    
    % Print numerical results
    fprintf('  Poles: %.3f ± %.3fj\n', real(poles(1)), abs(imag(poles(1))));
    fprintf('  Pole magnitudes: [%.3f, %.3f]\n', abs(poles));
    fprintf('  Stable: %s\n', getStabilityText(is_stable));
end

function theoretical_psd = calculateTheoreticalPSD(a, freq, sigma_e)
    % Calculate theoretical power spectrum for AR process
    % H(z) = 1 / (1 - a1*z^-1 - a2*z^-2 - ... - ap*z^-p)
    % PSD = sigma_e^2 * |H(e^jw)|^2
  
    
    % Calculate H(e^jw) for each frequency
    H_mag_sq = zeros(size(freq));
    
    for k = 1:length(freq)
        w = freq(k);
        
        % Calculate denominator: 1 - sum(a_i * exp(-j*w*i))
        denom = 1;
        for i = 1:length(a)
            denom = denom - a(i) * exp(-1j * w * i);
        end
        
        % |H(e^jw)|^2 = 1 / |denominator|^2
        H_mag_sq(k) = 1 / (abs(denom)^2);
    end
    
    theoretical_psd = sigma_e^2 * H_mag_sq;
end

function [poles, is_stable] = checkStability(a)
    % Find poles of AR filter and check stability
    % AR filter: H(z) = 1 / (1 - a1*z^-1 - a2*z^-2 - ...)
    % Characteristic polynomial: z^p - a1*z^(p-1) - a2*z^(p-2) - ... - ap = 0
    
    P = length(a);
    
    % Construct characteristic polynomial coefficients
    % For z^p - a1*z^(p-1) - a2*z^(p-2) - ... - ap = 0
    poly_coeffs = [1, -a];  % [1, -a1, -a2, ..., -ap]
    
    % Find roots (poles)
    poles = roots(poly_coeffs);
    
    % Check stability: all poles must be inside unit circle
    pole_magnitudes = abs(poles);
    is_stable = all(pole_magnitudes < 1);
    
    fprintf('    Pole analysis:\n');
    for i = 1:length(poles)
        fprintf('      Pole %d: %.3f %+.3fj, |pole| = %.3f\n', ...
                i, real(poles(i)), imag(poles(i)), abs(poles(i)));
    end
    fprintf('    Stability: %s\n', getStabilityText(is_stable));
end

function stability_text = getStabilityText(is_stable)
    % Helper function to get stability text
    if is_stable
        stability_text = 'STABLE';
    else
        stability_text = 'UNSTABLE';
    end
end

function plotPoleZero(a)
    % Plot pole-zero diagram
    [poles, is_stable] = checkStability(a);
    
    % Plot unit circle
    theta = linspace(0, 2*pi, 100);
    plot(cos(theta), sin(theta), 'k--', 'LineWidth', 1); hold on;
    
    % Plot poles
    plot(real(poles), imag(poles), 'rx', 'MarkerSize', 10, 'LineWidth', 3);
    
    % Plot zeros (at origin for AR process)
    plot(0, 0, 'bo', 'MarkerSize', 8, 'LineWidth', 2);
    
    xlabel('Real Part'); ylabel('Imaginary Part');
    title('Pole-Zero Plot');
    legend('Unit Circle', 'Poles', 'Zeros', 'Location', 'best');
    grid on; axis equal;
    xlim([-2, 2]); ylim([-2, 2]);
    
    % Add stability indication
    if is_stable
        text(0.7, 1.7, 'STABLE', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'green');
    else
        text(0.7, 1.7, 'UNSTABLE', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'red');
    end
end




%% Task 3.5 - AR Parameter Estimation: ML and Bayesian Methods

clc; clear; close all;

fprintf('=== AR PARAMETER ESTIMATION: ML vs BAYESIAN ===\n');

% Generate true AR signal for estimation experiments
% true_a = [0.5, 0.3]; % True AR(2)
% true_a = [0.5, -0.3, 0.2];  % True AR(3) coefficients
% true_a = [0.5, -0.4, 0.3, -0.2, 0.1, -0.1, 0.5, -0.4, 0.3, -0.2, 0.1, -0.1]; %AR(12)
% true_a = [0.5, -0.4, 0.3, -0.2, 0.1, -0.1, 0.5, -0.4, 0.3, -0.2, 0.1, -0.1, 0.5, -0.4, 0.3, -0.2, 0.1, -0.1]; % AR(18)

% randomly generate
P = 15; prior_sigma=0.01;
true_a = prior_sigma * randn(1, P);

true_P = length(true_a);
true_sigma_e = 1.0;
N = 10000;  % Signal length

fprintf('True AR coefficients: [%.2f, %.2f, %.2f]\n', true_a);
fprintf('True model order: P = %d\n', true_P);
fprintf('True noise variance: σₑ² = %.2f\n', true_sigma_e^2);

% Generate true AR signal
% rng(42);  % For reproducibility
first_signals = randn(true_P, 1);
y = generate_AR_signal(N, true_a, first_signals, true_sigma_e);

% Run comprehensive parameter estimation analysis
% analyzeParameterEstimation(y, N, true_a, true_sigma_e);
analyzeModelOrderSelection(y, N, true_a, true_sigma_e);
% comparePriorEffects(y, N, true_a, true_sigma_e);

function analyzeParameterEstimation(y, N, true_a, true_sigma_e)
    fprintf('\n--- Parameter Estimation Comparison ---\n');
    
    P = length(true_a);
    
    % ML Estimation
    [a_ml, ~] = estimateAR_ML(y, P);
    
    % Bayesian Estimation with different priors
    prior_vars = [0.01, 0.1, 1.0];  % Different prior variances
    prior_mean = zeros(P, 1);
    
    figure('Position', [50, 50, 1600, 1000]);
    
    fprintf('ML Estimates:\n');
    fprintf('  Coefficients: [%.3f, %.3f, %.3f]\n', a_ml);
    % fprintf('  Noise variance: %.3f\n', sigma_e_ml^2);
    
    % Plot ML results
    subplot(4, 4, 1);
    bar([true_a; a_ml']');
    xlabel('Coefficient Index'); ylabel('Value');
    title('ML vs True Coefficients');
    legend('True', 'ML', 'Location', 'best');
    grid on;
    
    % Calculate and plot residuals for ML
    residuals_ml = calculateResiduals(y, a_ml);
    mse_ml = mean(residuals_ml.^2);
    
    subplot(4, 4, 2);
    plot(residuals_ml, 'b-', 'LineWidth', 1);
    xlabel('Sample'); ylabel('Residual');
    title(sprintf('ML Residuals (MSE: %.3f)', mse_ml));
    grid on;
    
    % PSD comparison for ML
    subplot(4, 4, 3);
    % plotPSDComparison(y, a_ml, sigma_e_ml, 'ML Estimate');
    plotPSDComparison(y, a_ml, true_sigma_e, 'ML Estimate');
    
    % Bayesian estimates with different priors
    for i = 1:length(prior_vars)
        prior_var = prior_vars(i);
        [a_bayes, sigma_e_bayes, ~] = estimateAR_Bayesian(y, P, prior_var, prior_mean, true_sigma_e);
        
        fprintf('\nBayesian (prior var = %.1f):\n', prior_var);
        fprintf('  Coefficients: [%.3f, %.3f, %.3f]\n', a_bayes);
        % fprintf('  Noise variance: %.3f\n', sigma_e_bayes^2);
        
        % Plot Bayesian results
        subplot(4, 4, 4*i + 1);
        bar([true_a; a_ml'; a_bayes']');
        xlabel('Coefficient Index'); ylabel('Value');
        title(sprintf('Prior var = %.1f', prior_var));
        legend('True', 'ML', 'Bayesian', 'Location', 'best');
        grid on;
        
        % Calculate residuals for Bayesian
        residuals_bayes = calculateResiduals(y, a_bayes);
        mse_bayes = mean(residuals_bayes.^2);
        
        subplot(4, 4, 4*i + 2);
        plot(residuals_bayes, 'r-', 'LineWidth', 1);
        xlabel('Sample'); ylabel('Residual');
        title(sprintf('Bayes Residuals (MSE: %.3f)', mse_bayes));
        grid on;
        
        % PSD comparison for Bayesian
        subplot(4, 4, 4*i + 3);
        % plotPSDComparison(y, a_bayes, sigma_e_bayes, sprintf('Bayesian (σ²=%.1f)', prior_var));
        plotPSDComparison(y, a_bayes, true_sigma_e, sprintf('Bayesian (σ²=%.1f)', prior_var));

        % Prior effect visualization
        subplot(4, 4, 4*i + 4);
        coeff_diff = a_bayes - a_ml;
        bar(coeff_diff);
        xlabel('Coefficient Index'); ylabel('Difference (Bayes - ML)');
        title('Prior Effect');
        grid on;
    end
    
    sgtitle('AR Parameter Estimation: ML vs Bayesian with Different Priors');
end

function analyzeModelOrderSelection(y, N, true_a, true_sigma_e)
    fprintf('\n--- Bayesian Model Order Selection Analysis ---\n');
    
    max_order = 30;  % Test up to order 15
    orders = 1:max_order;
    
    mse_ml = zeros(size(orders));
    mse_bayes = zeros(size(orders));
    log_marginal_likelihoods = zeros(size(orders));
    
    % Prior settings for Bayesian model selection
    prior_sigma = 1.0;  % Prior standard deviation
    
    % figure('Position', [100, 100, 1600, 1000]);
    
    for P = orders
        fprintf('Testing order P = %d\n', P);
        
        % Prior parameters
        prior_mean = zeros(P, 1);
        prior_cov = (prior_sigma^2) * eye(P);
        
        % ML estimation
        [a_ml, sigma_e_ml] = estimateAR_ML(y, P);
        residuals_ml = calculateResiduals(y, a_ml);
        mse_ml(P) = mean(residuals_ml.^2);
        
        % Bayesian estimation
        [a_bayes, sigma_e_bayes, ~] = estimateAR_Bayesian(y, P, prior_sigma^2, prior_mean, true_sigma_e);
        residuals_bayes = calculateResiduals(y, a_bayes);
        mse_bayes(P) = mean(residuals_bayes.^2);
        
        % Calculate marginal likelihood for Bayesian model selection
        G = zeros(N-P, P);
        for i = 1:(N-P)
            G(i, :) = flip(y(i:(i+P-1)));
        end
        y_vec = y(P+1:N);
        
        if P == 0 || isempty(G)
            % Handle P=0 case (just noise)
            log_marginal_likelihoods(P) = log_marginal_likelihood(y_vec, 0, [], [], true_sigma_e);
        else
            log_marginal_likelihoods(P) = log_marginal_likelihood(y_vec, G, prior_mean, prior_cov, true_sigma_e);
        end
        
        fprintf('  P=%d: Log Marginal Likelihood = %.2f\n', P, log_marginal_likelihoods(P));
    end
    
    % Plot MSE vs model order
    figure;
    plot(orders, mse_ml, 'bo-', 'LineWidth', 2, 'MarkerSize', 6); hold on;
    plot(orders, mse_bayes, 'ro-', 'LineWidth', 2, 'MarkerSize', 6); hold on;
    xline(length(true_a), 'k--', 'True Order', 'LineWidth', 2);
    xlabel('Model Order P'); ylabel('MSE');
    title(sprintf('MSE vs Model Order, N=%d', N));
    legend('ML', 'Bayesian', 'True Order', 'Location', 'best');
    grid on;
    

    figure;
    % Plot Marginal Likelihood (the key plot!)
    subplot(3, 1, 1);
    plot(orders, log_marginal_likelihoods, 'g^-', 'LineWidth', 2, 'MarkerSize', 8);
    xline(length(true_a), 'k--', 'True Order', 'LineWidth', 2);
    xlabel('Model Order P'); ylabel('Log Marginal Likelihood');
    title('Bayesian Model Evidence');
    grid on;
    
    % Convert to model probabilities (if desired)
    subplot(3, 1, 2);
    % Normalize to get model probabilities
    max_lml = max(log_marginal_likelihoods);
    log_probs = log_marginal_likelihoods - max_lml;
    probs = exp(log_probs);
    probs = probs / sum(probs);
    
    bar(orders, probs, 'FaceColor', [0.2 0.7 0.2]);
    xline(length(true_a), 'k--', 'True Order', 'LineWidth', 2);
    xlabel('Model Order P'); ylabel('Model Probability');
    title('Posterior Model Probabilities');
    grid on;

    
    % Bayes Factors relative to true model
    subplot(3, 1, 3);
    true_order_idx = length(true_a);
    bayes_factors = exp(log_marginal_likelihoods - log_marginal_likelihoods(true_order_idx));
    semilogy(orders, bayes_factors, 'mo-', 'LineWidth', 2, 'MarkerSize', 6);
    xline(length(true_a), 'k--', 'True Order', 'LineWidth', 2);
    ylim([1e-10, 10]);
    xlabel('Model Order P'); ylabel('Bayes Factor vs True Order');
    title(sprintf('Bayes Factors (vs P=%d)', true_order_idx));
    grid on;

    sgtitle(sprintf('Bayesian Model Order Selection Analysis, N=%d', N));
    

    figure;
    % Detailed view around true order
    true_order = length(true_a);
    plot_range = max(1, true_order-3):min(max_order, true_order+5);
    semilogy(plot_range, mse_ml(plot_range), 'bo-', 'LineWidth', 2); hold on;
    semilogy(plot_range, mse_bayes(plot_range), 'ro-', 'LineWidth', 2);
    xline(true_order, 'k--', 'LineWidth', 2);
    xlabel('Model Order P'); ylabel('MSE (log scale)');
    title('MSE Detail Around True Order');
    legend('ML', 'Bayesian', 'Location', 'best');
    grid on;
    
   

    
    % Find optimal orders
    [max_lml_val, opt_bayes] = max(log_marginal_likelihoods);
    [~, opt_mse_ml] = min(mse_ml);
    [~, opt_mse_bayes] = min(mse_bayes);
    [max_prob_val, opt_prob] = max(probs);
   
    
   
    
    
    
    
    fprintf('\nBayesian Model Selection Results:\n');
    fprintf('  True order: %d\n', length(true_a));
    fprintf('  Optimal order (marginal likelihood): %d\n', opt_bayes);
    fprintf('  Max marginal likelihood: %.2f\n', max_lml_val);
    fprintf('  Model probability for true order: %.1f%%\n', probs(true_order_idx)*100);
    
    % Show Bayes factors
    fprintf('\nBayes Factors (relative to true order P=%d):\n', length(true_a));
    for i = 1:min(8, max_order)
        bf = bayes_factors(i);
        if bf > 1
            fprintf('  P=%d: BF = %.2f (%.1fx more likely)\n', i, bf, bf);
        else
            fprintf('  P=%d: BF = %.2f (%.1fx less likely)\n', i, bf, 1/bf);
        end
    end
end


function comparePriorEffects(y, N, true_a, true_sigma_e)
    fprintf('\n--- Prior Effects Comparison ---\n');
    
    P = length(true_a);
    prior_sigmas = [0.01, 0.1, 1.0, 10.0, 100.0];
    prior_mean = zeros(P,1);
    
    % ML estimate for reference
    [a_ml, ~] = estimateAR_ML(y, P);
    
    % figure('Position', [150, 150, 1400, 800]);
    
    coeff_estimates = zeros(length(prior_sigmas), P);
    mse_values = zeros(size(prior_sigmas));
    
    for i = 1:length(prior_sigmas)
        prior_sigma = prior_sigmas(i);
        [a_bayes, ~] = estimateAR_Bayesian(y, P, prior_sigma, prior_mean, true_sigma_e);
        coeff_estimates(i, :) = a_bayes;
        
        residuals = calculateResiduals(y, a_bayes);
        mse_values(i) = mean(residuals.^2);
    end
    
    % Plot coefficient evolution
    figure;
    subplot(2, 3, 1:3);
    % Define colors for each coefficient (reuse for higher orders)
    colors = {[0.0, 0.4, 0.7], [0.8, 0.2, 0.2], [0.2, 0.7, 0.2], [0.9, 0.5, 0.0], ...
              [0.6, 0.2, 0.8], [0.4, 0.8, 0.8], [0.8, 0.8, 0.2], [0.5, 0.5, 0.5]};
    
    % Plot each coefficient with consistent colors
    for j = 1:P
        color = colors{mod(j-1, length(colors)) + 1}; % Cycle through colors
        
        % Bayesian estimates (solid line with circles)
        semilogx(prior_sigmas, coeff_estimates(:, j), 'o-', 'LineWidth', 2, ...
                'MarkerSize', 6, 'Color', color, 'MarkerFaceColor', color, ...
                'DisplayName', sprintf('a_%d (Bayes)', j)); hold on;
        
        % True coefficients (thick dashed line, same color)
        semilogx(prior_sigmas, ones(size(prior_sigmas)) * true_a(j), '--', ...
                'LineWidth', 3, 'Color', color, ...
                'DisplayName', sprintf('a_%d (True)', j));
        
        % ML estimates (dotted line, same color)
        semilogx(prior_sigmas, ones(size(prior_sigmas)) * a_ml(j), ':', ...
                'LineWidth', 2.5, 'Color', color, ...
                'DisplayName', sprintf('a_%d (ML)', j));
    end
    
    xlabel('Prior Standard Deviation'); ylabel('Coefficient Value');
    title('Bayesian Coefficient Estimates vs Prior Standard Deviation');
    legend('show', 'Location', 'best', 'NumColumns', 2);
    grid on;
    %{
    for j = 1:P
        semilogx(prior_sigmas, coeff_estimates(:, j), 'o-', 'LineWidth', 2, ...
                'MarkerSize', 8, 'DisplayName', sprintf('a_%d', j)); hold on;
        semilogx(prior_sigmas, ones(size(prior_sigmas)) * true_a(j), '--', ...
                'LineWidth', 2, 'DisplayName', sprintf('True a_%d', j));
        semilogx(prior_sigmas, ones(size(prior_sigmas)) * a_ml(j), ':', ...
                'LineWidth', 2, 'DisplayName', sprintf('ML a_%d', j));
    end
    xlabel('Prior Variance'); ylabel('Coefficient Value');
    title('Bayesian Coefficient Estimates vs Prior Variance');
    legend('show', 'Location', 'best');
    grid on;
    %}
    
    % MSE vs prior variance
    subplot(2, 3, 4);
    semilogx(prior_sigmas, mse_values, 'ro-', 'LineWidth', 2, 'MarkerSize', 8);
    xlabel('Prior Standard Deviation'); ylabel('MSE');
    title('MSE vs Prior Standard Deviation');
    grid on;
    
    % Shrinkage effect
    subplot(2, 3, 5);
    shrinkage = abs(coeff_estimates - a_ml') ./ abs(a_ml');
    for j = 1:P
        semilogx(prior_sigmas, shrinkage(:, j), 'o-', 'LineWidth', 2, ...
                'DisplayName', sprintf('Coeff %d', j)); hold on;
    end
    xlabel('Prior Standard Deviation'); ylabel('Relative Shrinkage');
    title('Shrinkage Effect');
    legend('show');
    grid on;
    
    % Prior visualization
    subplot(2, 3, 6);
    x_range = -2:0.01:2;
    for i = [1, 3, 5]  % Show a few different priors
        prior_pdf = normpdf(x_range, 0, sqrt(prior_sigmas(i)));
        plot(x_range, prior_pdf, 'LineWidth', 2, ...
             'DisplayName', sprintf('σ² = %.2f', prior_sigmas(i))); hold on;
    end
    xlabel('Coefficient Value'); ylabel('Prior Density');
    title('Gaussian Priors');
    legend('show');
    grid on;
    
    sgtitle(sprintf('Effect of Gaussian Prior on Bayesian AR Estimation, N=%d', N));
    
    fprintf('Key observations:\n');
    fprintf('  - Small prior variance → strong shrinkage toward zero\n');
    fprintf('  - Large prior variance → approaches ML estimate\n');
    fprintf('  - Optimal prior balances bias and variance\n');
end

function [a_est, sigma_e_est] = estimateAR_ML(y, P)
    % Maximum Likelihood estimation using least squares
    N = length(y);

    G = compute_G_matrix(y, N, P);

    % % Construct data matrix G and response vector y_vec
    % G = zeros(N-P, P);
    % for i = 1:(N-P)
    %     G(i, :) = flip(y(i:(i+P-1)));
    % end

    y_vec = y(P+1:N);
    
    % ML estimate: a = (G'G)^(-1) G'y
    a_est = (G' * G) \ (G' * y_vec);
    
    % Estimate noise variance
    residuals = y_vec - G * a_est;
    sigma_e_est = sqrt(mean(residuals.^2));
end

function [a_map, sigma_e_est, posterior_cov] = estimateAR_Bayesian(y, P, prior_sigma, prior_mean, true_sigma_e)
    % Bayesian estimation following the handout formulation
    % Prior: θ ~ N(m_θ, C_θ)
    % Posterior: θ|x ~ N(θ^MAP, σ_e² Φ^(-1))
    
    if nargin < 4
        prior_mean = zeros(P, 1);  % Default zero mean prior
    end
    
    N = length(y);
    
    G = compute_G_matrix(y, N, P);

    % % Construct data matrix G and response vector x
    % G = zeros(N-P, P);
    % for i = 1:(N-P)
    %     G(i, :) = flip(y(i:(i+P-1)));
    % end

    x = y(P+1:N);  % Using x as in the handout
    
    % Estimate noise variance first (can be done iteratively)
    % For simplicity, use ML estimate initially
    [a_ml, sigma_e_est] = estimateAR_ML(y, P);
    
    % Prior parameters
    m_theta = prior_mean;              % Prior mean
    C_theta = prior_sigma^2 * eye(P);      % Prior covariance
    C_theta_inv = inv(C_theta);        % Prior precision
    
    % Following equations (19)-(21) from handout:
    % Φ = G^T G + σ_e^(-2) C_θ^(-1)
    Phi = G'*G + (1/true_sigma_e^2) * C_theta_inv;
    
    % Θ = G^T x + σ_e^(-2) C_θ^(-1) m_θ
    Theta = G'*x + (1/true_sigma_e^2) * C_theta_inv * m_theta;
    
    % θ^MAP = Φ^(-1) Θ
    a_map = Phi \ Theta;  % MAP estimate
    
    % Posterior covariance: C_θ^post = σ_e² Φ^(-1)
    posterior_cov = true_sigma_e^2 * inv(Phi);
    
    fprintf('    Bayesian estimation details:\n');
    fprintf('      Prior mean: [%.3f, %.3f, %.3f]\n', m_theta(1:min(3,P)));
    fprintf('      Prior variance: %.3f\n', prior_sigma^2);
    fprintf('      Estimated σ_e: %.3f\n', true_sigma_e);
    fprintf('      MAP estimate: [%.3f, %.3f, %.3f]\n', a_map(1:min(3,P)));
end

function residuals = calculateResiduals(y, a_est)
    % Calculate residuals using estimated AR parameters
    P = length(a_est);
    N = length(y);
    
    residuals = zeros(N-P, 1);
    for n = (P+1):N
        y_pred = dot(a_est, flip(y(n-P:n-1)));
        residuals(n-P) = y(n) - y_pred;
    end
end

function plotPSDComparison(y, a_est, sigma_e_est, title_str)
    % Compare empirical and estimated theoretical PSD
    N = length(y);
    
    % Empirical PSD
    Y = fft(y);
    empirical_psd = abs(Y).^2 / N;
    
    % Theoretical PSD from estimates
    freq = (0:N-1) / N;  % Normalized frequency [0, 1)
    theoretical_psd = calculateTheoreticalPSD(a_est, freq, sigma_e_est);
    
    plot(freq(1:N/2), empirical_psd(1:N/2), 'b-', 'LineWidth', 1); hold on;
    plot(freq(1:N/2), theoretical_psd(1:N/2), 'r--', 'LineWidth', 2);
    xlabel('Normalized Frequency [0, 0.5]'); ylabel('PSD');
    title(title_str);
    legend('Empirical', 'Estimated', 'Location', 'best');
    grid on;
end

function lml = log_marginal_likelihood(y, G, mu0, C0, sigma_e)
    N = length(y);
    P = length(C0); % Dimension of prior

    if G==0
        % Then use yn = en
        lml = - N/2*log(2*pi*sigma_e^2) - (y'*y)/(2*sigma_e^2);
    
    else
    Phi = (G'*G) + sigma_e^2 * inv(C0);
    Theta = (G'*y) + sigma_e^2 * (C0 \ mu0);

    theta_MAP = Phi \ Theta;
    
    lml = - P/2*log(2*pi) - 1/2*log(det(C0)) - 1/2*log(det(Phi)) ...
          - (N-P)/2 * log(2*pi*sigma_e^2) ...
          - (y'*y + sigma_e^2 * (mu0' *(C0\mu0)) - Theta'*theta_MAP) / (2*sigma_e^2);
    
    end

end



%% Task 3.5 Model Selection for Vowel, Consonant, Steady, Transient

clc, clearvars, close all;

content = "transient";

if content == "vowel"
    [data, rate] = audioread("/Users/maxlyu/Desktop/Part IIA/SF1/Audio examples for weeks 1-2-20250516/f1lcapae.wav");
    segment = data(55000:57000);
    % segment = data(321500:323000); %Jacky
elseif content == "consonant"
    [data, rate] = audioread("/Users/maxlyu/Desktop/Part IIA/SF1/Audio examples for weeks 1-2-20250516/f1lcapae.wav");
    segment = data(60000:62000);
    segment = data(319900:321230); %Jacky
elseif content == "transient"
    [data, rate] = audioread("/Users/maxlyu/Desktop/Part IIA/SF1/Audio examples for weeks 1-2-20250516/grosse_original.wav");
    % segment = data(74000:75000); % transient note of violin
    segment = data(74000:74600); % transient note of violin
    [data, rate] = audioread("/Users/maxlyu/Desktop/Part IIA/SF1/Audio examples for weeks 1-2-20250516/armst_37_orig.wav");
    segment = data(400:2800); % Jacky
elseif content == "steady"
    [data, rate] = audioread("/Users/maxlyu/Desktop/Part IIA/SF1/Audio examples for weeks 1-2-20250516/piano_clean.wav");
    segment = data(3200:3800); % pure note from piano
    [data, rate] = audioread("/Users/maxlyu/Desktop/Part IIA/SF1/Audio examples for weeks 1-2-20250516/armst_37_orig.wav");
    segment = data(16791:18356); % Jacky
end

% Calculate the model log-marginal-likelihood of P=1-20, selection model
ARModelSelection(segment, content);

function ARModelSelection(y, content)
    fprintf('\n--- Bayesian Model Order Selection Analysis ---\n');
    N = length(y);

    max_order = 20; 
    orders = 1:max_order;
    
    mse_ml = zeros(size(orders));
    mse_bayes = zeros(size(orders));
    log_marginal_likelihoods = zeros(size(orders));
    
    % Prior settings for Bayesian model selection
    prior_sigma = 0.15;  % Prior standard deviation
    
    % figure('Position', [100, 100, 1600, 1000]);
    
    for P = orders
        fprintf('Testing order P = %d\n', P);
        
        
        
        % ML estimation
        [a_ml, sigma_e_ml] = estimateAR_ML(y, P);
        residuals_ml = calculateResiduals(y, a_ml);
        mse_ml(P) = mean(residuals_ml.^2);

        % Prior parameters
        prior_mean = zeros(P, 1);
        prior_sigma = sigma_e_ml / 0.5;
        prior_cov = (prior_sigma^2) * eye(P);

        
        % Bayesian estimation
        [a_bayes, sigma_e_bayes, ~] = estimateAR_Bayesian(y, P, prior_sigma^2, prior_mean, sigma_e_ml);
        residuals_bayes = calculateResiduals(y, a_bayes);
        mse_bayes(P) = mean(residuals_bayes.^2);
        
        % Calculate marginal likelihood for Bayesian model selection
        G = zeros(N-P, P);
        for i = 1:(N-P)
            G(i, :) = flip(y(i:(i+P-1)));
        end
        y_vec = y(P+1:N);

        % THIS WAS MISSING - Calculate the actual log marginal likelihood
        if P == 0 || isempty(G)
            log_marginal_likelihoods(P) = log_marginal_likelihood(y_vec, 0, [], [], sigma_e_ml);
        else
            log_marginal_likelihoods(P) = log_marginal_likelihood(y_vec, G, prior_mean, prior_cov, sigma_e_ml);
        end
        
        
        fprintf('  P=%d: Log Marginal Likelihood = %.2f\n', P, log_marginal_likelihoods(P));
    end
    
    % Plot MSE vs model order
    figure;
    plot(orders, mse_ml, 'bo-', 'LineWidth', 2, 'MarkerSize', 6); hold on;
    plot(orders, mse_bayes, 'ro-', 'LineWidth', 2, 'MarkerSize', 6); hold on;
    % xline(length(true_a), 'k--', 'True Order', 'LineWidth', 2);
    xlabel('Model Order P'); ylabel('MSE');
    title(sprintf('MSE vs Model Order (%s audio clip)', content));
    legend('ML', 'Bayesian', 'True Order', 'Location', 'best');
    grid on;
    

    figure;
    % Plot Marginal Likelihood (the key plot!)
    subplot(2, 1, 1);
    plot(orders, log_marginal_likelihoods, 'g^-', 'LineWidth', 2, 'MarkerSize', 8);
    xlabel('Model Order P'); ylabel('Log Marginal Likelihood');
    title('Bayesian Model Evidence');
    grid on;
    
    % Convert to model probabilities (if desired)
    subplot(2, 1, 2);
    max_lml = max(log_marginal_likelihoods);
    log_probs = log_marginal_likelihoods - max_lml;
    probs = exp(log_probs);
    probs = probs / sum(probs);
    
    bar(orders, probs, 'FaceColor', [0.2 0.7 0.2]);
    xlabel('Model Order P'); ylabel('Model Probability');
    title('Posterior Model Probabilities');
    grid on;


    sgtitle(sprintf('Bayesian Model Order Selection Analysis (%s audio clip)', content));

    
    % Find optimal orders
    [max_lml_val, opt_bayes] = max(log_marginal_likelihoods);
    [~, opt_mse_ml] = min(mse_ml);
    [~, opt_mse_bayes] = min(mse_bayes);
    [max_prob_val, opt_prob] = max(probs);
   
    
  
    
    fprintf('\nBayesian Model Selection Results:\n');
    % fprintf('  True order: %d\n', length(true_a));
    fprintf('  Optimal order (marginal likelihood): %d\n', opt_bayes);
    fprintf('  Max marginal likelihood: %.2f\n', max_lml_val);
    % fprintf('  Model probability for true order: %.1f%%\n', probs(true_order_idx)*100);
    
end
