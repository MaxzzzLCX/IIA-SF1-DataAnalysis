%% Week 3 Tasks 3.1 - y = theta + e

clc, clearvars
    
% Model: yn = theta + en
% y = theta + e

% Prior - Gaussian N(mu0, sigma0^2)
mu0 = 0; 
sigma0 = 0.1;  % Prior standard deviation
theta = mu0 + sigma0 * randn(); % Draw theta from N(mu0, sigma0^2)

dim = 1; 
N = 1;
sigma_e = 1;  % Error standard deviation
G = ones(dim, 1);

y = generate_y(G, theta, dim, sigma_e);

theta_ML = ml(y, G);
[theta_MAP, sigma_post] = map(y, G, sigma_e, mu0, sigma0);

fprintf('True theta: %.3f\n', theta);
fprintf('ML estimate: %.3f\n', theta_ML);
fprintf('MAP estimate: %.3f\n', theta_MAP);
fprintf('MAP posterior std: %.3f\n', sigma_post);

% Plot likelihood and posterior
plotLikelihoodAndPosterior(y, G, sigma_e, mu0, sigma0, theta, theta_ML, theta_MAP, sigma_post);

% Explore different scenarios
exploreDifferentScenarios(theta);

%% Functions

function y = generate_y(G, theta, dim, sigma_e)
    % Generate observations: y = G*theta + e, where e ~ N(0, sigma_e^2)
    e = sigma_e * randn(dim, 1); % Error with std dev sigma_e
    y = G * theta + e; 
end

function theta_ml = ml(y, G)
    % Maximum Likelihood estimate: theta_ML = (G'G)^(-1) G'y
    theta_ml = (G' * G) \ (G' * y);
end

function [theta_MAP, sigma_post] = map(y, G, sigma_e, mu0, sigma0)
    % Maximum A Posteriori estimate with Gaussian prior
    % Prior: theta ~ N(mu0, sigma0^2)
    % Likelihood: y|theta ~ N(G*theta, sigma_e^2 * I)
    
    % Convert to precision (inverse variance) for computation
    precision_prior = 1 / sigma0^2;
    precision_likelihood = 1 / sigma_e^2;
    
    % Posterior precision and mean
    precision_post = (G' * G) * precision_likelihood + precision_prior;
    weighted_mean = (G' * y) * precision_likelihood + mu0 * precision_prior;
    
    theta_MAP = weighted_mean / precision_post;
    sigma_post = sqrt(1 / precision_post);  % Posterior standard deviation
end

function log_likelihood = logLikelihood(theta_vals, y, G, sigma_e)
    % Calculate log-likelihood for each theta value
    % L(theta) = (2*pi*sigma_e^2)^(-N/2) * exp(-1/(2*sigma_e^2) * ||y - G*theta||^2)
    
    N = length(y);
    log_likelihood = zeros(size(theta_vals));
    
    for i = 1:length(theta_vals)
        theta = theta_vals(i);
        residual = y - G * theta;
        log_likelihood(i) = -N/2 * log(2*pi*sigma_e^2) - 1/(2*sigma_e^2) * (residual' * residual);
    end
end

function log_posterior = logPosterior(theta_vals, y, G, sigma_e, mu0, sigma0)
    % Calculate log-posterior for each theta value
    % p(theta|y) ∝ p(y|theta) * p(theta)
    
    log_likelihood_vals = logLikelihood(theta_vals, y, G, sigma_e);
    log_posterior = zeros(size(theta_vals));
    
    for i = 1:length(theta_vals)
        theta = theta_vals(i);
        % Log prior: log p(theta) = -1/2 * log(2*pi*sigma0^2) - 1/(2*sigma0^2) * (theta - mu0)^2
        log_prior = -1/2 * log(2*pi*sigma0^2) - 1/(2*sigma0^2) * (theta - mu0)^2;
        log_posterior(i) = log_likelihood_vals(i) + log_prior;
    end
end

function plotLikelihoodAndPosterior(y, G, sigma_e, mu0, sigma0, true_theta, theta_ml, theta_map, sigma_post)
    % Plot likelihood function and posterior density
    
    % Define range of theta values to plot (based on prior and data)
    data_range = 3 * sigma_e;  % 3 std devs around data
    prior_range = 3 * sigma0;  % 3 std devs around prior mean
    plot_min = min([mu0 - prior_range, theta_ml - data_range, true_theta - data_range]);
    plot_max = max([mu0 + prior_range, theta_ml + data_range, true_theta + data_range]);
    theta_range = linspace(plot_min, plot_max, 1000);
    
    % Calculate likelihood and posterior
    log_likelihood_vals = logLikelihood(theta_range, y, G, sigma_e);
    log_posterior_vals = logPosterior(theta_range, y, G, sigma_e, mu0, sigma0);
    
    % Convert to regular scale (normalized for plotting)
    likelihood_vals = exp(log_likelihood_vals - max(log_likelihood_vals));
    posterior_vals = exp(log_posterior_vals - max(log_posterior_vals));
    
    % Prior values
    prior_vals = normpdf(theta_range, mu0, sigma0);
    prior_vals = prior_vals / max(prior_vals); % Normalize for plotting
    
    figure;
    plot(theta_range, likelihood_vals, 'b-', 'LineWidth', 2); hold on;
    plot(theta_range, prior_vals, 'g--', 'LineWidth', 2);
    plot(theta_range, posterior_vals, 'r-', 'LineWidth', 2);
    
    % Mark estimates
    plot(true_theta, 0, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k');
    plot(theta_ml, 0, 'bs', 'MarkerSize', 8, 'MarkerFaceColor', 'b');
    plot(theta_map, 0, 'r^', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
    
    xlabel('\theta');
    ylabel('Normalized Density');
    legend('Likelihood', 'Prior', 'Posterior', 'True \theta', 'ML estimate', 'MAP estimate', ...
           'Location', 'best');
    title(sprintf('N=%d, \\sigma_e=%.3f, Prior: N(%.1f,%.3f)', length(y), sigma_e, mu0, sigma0));
    grid on;
end

function exploreDifferentScenarios(true_theta)
    % Explore different N values and prior parameters
    
    N_values = [1, 10, 100, 1000];
    sigma_e = 1;  % Fixed error standard deviation
    
    % Scenario 1: Different N values with fixed prior
    figure;
    mu0 = 0; 
    sigma0 = 0.1;  % Prior standard deviation
    
    for i = 1:length(N_values)
        N = N_values(i);
        G = ones(N, 1);
        y = generate_y(G, true_theta, N, sigma_e);
        
        theta_ml = ml(y, G);
        [theta_map, sigma_post] = map(y, G, sigma_e, mu0, sigma0);
        
        subplot(2, 2, i);
        theta_range = linspace(-2, 2, 1000);
        
        % Calculate and plot
        log_likelihood_vals = logLikelihood(theta_range, y, G, sigma_e);
        log_posterior_vals = logPosterior(theta_range, y, G, sigma_e, mu0, sigma0);
        
        likelihood_vals = exp(log_likelihood_vals - max(log_likelihood_vals));
        posterior_vals = exp(log_posterior_vals - max(log_posterior_vals));
        prior_vals = normpdf(theta_range, mu0, sigma0);
        prior_vals = prior_vals / max(prior_vals);
        
        plot(theta_range, likelihood_vals, 'b-', 'LineWidth', 2); hold on;
        plot(theta_range, prior_vals, 'g--', 'LineWidth', 1);
        plot(theta_range, posterior_vals, 'r-', 'LineWidth', 2);
        
        plot(true_theta, 0, 'ko', 'MarkerSize', 6, 'MarkerFaceColor', 'k');
        plot(theta_ml, 0, 'bs', 'MarkerSize', 6, 'MarkerFaceColor', 'b');
        plot(theta_map, 0, 'r^', 'MarkerSize', 6, 'MarkerFaceColor', 'r');
        
        xlabel('\theta'); ylabel('Normalized Density');
        title(sprintf('N = %d', N));
        if i == 1
            legend('Likelihood', 'Prior', 'Posterior', 'True', 'ML', 'MAP', 'Location', 'best');
        end
        grid on;
        xlim([-2, 2]);
    end
    sgtitle(sprintf('Effect of Sample Size N (\\sigma_e=%.1f, \\sigma_0=%.1f)', sigma_e, sigma0));
    
    % Scenario 2: Different prior standard deviations
    figure;
    N = 1;
    G = ones(N, 1);
    y = generate_y(G, true_theta, N, sigma_e);
    sigma0_values = [0.1, 0.3, 1.0, 3.0];  % Prior standard deviations
    
    for i = 1:length(sigma0_values)
        sigma0 = sigma0_values(i);
        mu0 = 0;
        
        [theta_map, sigma_post] = map(y, G, sigma_e, mu0, sigma0);
        theta_ml = ml(y, G); % ML doesn't change
        
        subplot(2, 2, i);
        theta_range = linspace(-3, 3, 1000);
        
        log_likelihood_vals = logLikelihood(theta_range, y, G, sigma_e);
        log_posterior_vals = logPosterior(theta_range, y, G, sigma_e, mu0, sigma0);
        
        likelihood_vals = exp(log_likelihood_vals - max(log_likelihood_vals));
        posterior_vals = exp(log_posterior_vals - max(log_posterior_vals));
        prior_vals = normpdf(theta_range, mu0, sigma0);
        prior_vals = prior_vals / max(prior_vals);
        
        plot(theta_range, likelihood_vals, 'b-', 'LineWidth', 2); hold on;
        plot(theta_range, prior_vals, 'g--', 'LineWidth', 1);
        plot(theta_range, posterior_vals, 'r-', 'LineWidth', 2);
        
        plot(true_theta, 0, 'ko', 'MarkerSize', 6, 'MarkerFaceColor', 'k');
        plot(theta_ml, 0, 'bs', 'MarkerSize', 6, 'MarkerFaceColor', 'b');
        plot(theta_map, 0, 'r^', 'MarkerSize', 6, 'MarkerFaceColor', 'r');
        
        xlabel('\theta'); ylabel('Normalized Density');
        title(sprintf('\\sigma_0 = %.1f', sigma0));
        if i == 1
            legend('Likelihood', 'Prior', 'Posterior', 'True', 'ML', 'MAP', 'Location', 'best');
        end
        grid on;
        xlim([-3, 3]);
    end
    sgtitle(sprintf('Effect of Prior Std Dev (N=%d, \\sigma_e=%.1f)', N, sigma_e));
    
    % Scenario 3: Different Error Standard Deviations
    figure;
    N = 1;
    sigma0 = 0.1;  % Fixed prior std dev
    G = ones(N, 1);
    mu0 = 0;
    sigma_e_values = [0.1, 0.3, 1.0, 3.0];  % Error standard deviations
    
    for i = 1:length(sigma_e_values)
        sigma_e_current = sigma_e_values(i);
        
        % Generate new data with this noise level
        y_current = generate_y(G, true_theta, N, sigma_e_current);
        
        [theta_map, sigma_post] = map(y_current, G, sigma_e_current, mu0, sigma0);
        theta_ml = ml(y_current, G);
        
        subplot(2, 2, i);
        theta_range = linspace(-3, 3, 1000);
        
        log_likelihood_vals = logLikelihood(theta_range, y_current, G, sigma_e_current);
        log_posterior_vals = logPosterior(theta_range, y_current, G, sigma_e_current, mu0, sigma0);
        
        likelihood_vals = exp(log_likelihood_vals - max(log_likelihood_vals));
        posterior_vals = exp(log_posterior_vals - max(log_posterior_vals));
        prior_vals = normpdf(theta_range, mu0, sigma0);
        prior_vals = prior_vals / max(prior_vals);
        
        plot(theta_range, likelihood_vals, 'b-', 'LineWidth', 2); hold on;
        plot(theta_range, prior_vals, 'g--', 'LineWidth', 1);
        plot(theta_range, posterior_vals, 'r-', 'LineWidth', 2);
        
        plot(true_theta, 0, 'ko', 'MarkerSize', 6, 'MarkerFaceColor', 'k');
        plot(theta_ml, 0, 'bs', 'MarkerSize', 6, 'MarkerFaceColor', 'b');
        plot(theta_map, 0, 'r^', 'MarkerSize', 6, 'MarkerFaceColor', 'r');
        
        xlabel('\theta'); ylabel('Normalized Density');
        title(sprintf('\\sigma_e = %.1f', sigma_e_current));
        if i == 1
            legend('Likelihood', 'Prior', 'Posterior', 'True', 'ML', 'MAP', 'Location', 'best');
        end
        grid on;
        xlim([-3, 3]);
    end
    sgtitle(sprintf('Effect of Error Std Dev (N=%d, \\sigma_0=%.1f)', N, sigma0));
    
    fprintf('\nKey Observations:\n');
    fprintf('1. As N increases, both ML and MAP converge to true value, posterior std decreases\n');
    fprintf('2. As prior std (sigma0) increases, MAP approaches ML (weak prior)\n');
    fprintf('3. As error std (sigma_e) increases, likelihood becomes flatter, prior has more influence\n');
    fprintf('4. ML estimate is always the sample mean, independent of prior\n');
    fprintf('5. Posterior std depends on both prior std and likelihood precision\n');
end