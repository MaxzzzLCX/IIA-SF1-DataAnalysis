%% Week 3 - General Linear Gaussian Model Task 3.2

clc, clearvars


% Model: yn = theta1 + theta2*n + en  G=[1,1,..,1]    theta=[theta1,theta]^T
%                                       [1,2,..,N]^T

% Prior - Gaussian N(mu0, C0)
P = 2; % P represents the number of parameters theta
mu0 = zeros(P,1); sigma0 = 0.1; C0 = sigma0^2 * eye(P);

theta = mu0 + sqrtm(C0) * randn(size(mu0)); % Draw theta from N(mu0, sigma0^2)

N = 20;
sigma_e = 1;  % Error standard deviation

G = ones(N, 2); G(:,2) = linspace(1,N,N);

y = generate_y(G, theta, N, sigma_e);

theta_ML = ml(y, G);
[theta_MAP, C_post] = map(y, G, sigma_e, mu0, sigma0);

% Display results
fprintf('=== PARAMETER ESTIMATES ===\n');
fprintf('True theta:     [%.3f, %.3f]\n', theta(1), theta(2));
fprintf('ML estimate:    [%.3f, %.3f]\n', theta_ML(1), theta_ML(2));
fprintf('MAP estimate:   [%.3f, %.3f]\n', theta_MAP(1), theta_MAP(2));
fprintf('MAP posterior std: [%.3f, %.3f]\n', sqrt(diag(C_post)));

% Visualization
plotResults(y, G, theta, theta_ML, theta_MAP, N);

% Explore different scenarios
exploreDifferentScenarios();


%% Task 3.3
clc, clearvars

% Model 1: yn = en                      theta=0
% Model 2: yn = theta + en              G=[1,1..]^T, theta
% Model 3: yn = theta1 + theta2*n + en  G=[1,1,..,1]    theta=[theta1,theta]^T
%                                         [1,2,..,N]^T


sigma_e = 100;
sigma0 = 0.1;
N = 1000; Ndataset=10000;
accurate = zeros(3,3);

for i=1:Ndataset
    
    [SampleModel, y] = sample_from_random_model(N, sigma_e, sigma0);
    ChosenModel = model_selection(N, y, sigma_e, sigma0);
    
    % Keep track of confusion matrix
    accurate(SampleModel, ChosenModel) = accurate(SampleModel, ChosenModel) + 1 ;
end

rowSums = sum(accurate,2);
confus_matrix = accurate ./ rowSums;
disp(confus_matrix)


function [SampleModel, y] = sample_from_random_model(N, sigma_e, sigma0)
    
    % randomly select a model
    SampleModel = randi([1,3]);
    
    % (1) Sample a sequence of data from one of the models
    if SampleModel == 1
        % Model 1
        theta = 0; G=0;
        y = generate_y(G, theta, N, sigma_e);
    elseif SampleModel == 2
        % Model 2: yn = theta + en
    
        % Prior - Gaussian N(mu0, sigma0^2)
        P = 1;
        mu0 = 0; C0 = sigma0^2 * eye(P);
        theta = mu0 + sigma0 * randn(); % Draw theta from N(mu0, sigma0^2)
        
        %dim = 1; 
        G = ones(N, 1);
        
        y = generate_y(G, theta, N, sigma_e);
        
    elseif SampleModel == 3
        % Model 3: yn = theta1 + theta2*n + en
    
        % Prior - Gaussian N(mu0, C0)
        P = 2; % P represents the number of parameters theta
        mu0 = zeros(P,1); C0 = sigma0^2 * eye(P);
        
        theta = mu0 + sqrtm(C0) * randn(size(mu0)); % Draw theta from N(mu0, sigma0^2)
     
        
        G = ones(N, 2); G(:,2) = linspace(1,N,N);
        
        y = generate_y(G, theta, N, sigma_e);
    end

% disp(y)
end

function ChosenModel = model_selection(N, y, sigma_e, sigma0)

    % (2) Calculate marginal likelihood of data y on three models
    
    % Model 1: yn = en (pure random noise)
    G=0; theta=0; mu0=0; C0=0;
    lml_1 = log_marginal_likelihood(y, G, mu0, C0, sigma_e);
    
    
    % Model 2: yn = theta + en
    % Prior - Gaussian N(mu0, sigma0^2)
    P = 1;
    mu0 = 0; C0 = sigma0^2 * eye(P); 
    G = ones(N, 1);
    
    lml_2 = log_marginal_likelihood(y, G, mu0, C0, sigma_e);
    
    
    
    % Model 3: yn = theta1 + theta2*n + en
    
    % Prior - Gaussian N(mu0, C0)
    P = 2; % P represents the number of parameters theta
    mu0 = zeros(P,1); C0 = sigma0^2 * eye(P);
    
    G = ones(N, 2); G(:,2) = linspace(1,N,N);
    
    lml_3 = log_marginal_likelihood(y, G, mu0, C0, sigma_e);
    
    
    if lml_1 > lml_2 && lml_1 > lml_3
        ChosenModel = 1; % Model 1 is the best fit
    elseif lml_2 > lml_1 && lml_2 > lml_3
        ChosenModel = 2; % Model 2 is the best fit
    else
        ChosenModel = 3; % Model 3 is the best fit
    end

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


%% Functions 

function y = generate_y(G, theta, N, sigma_e)
    % Generate observations: y = G*theta + e, where e ~ N(0, sigma_e^2)
    e = sigma_e .* randn(N, 1); % Error with std dev sigma_e
    y = G * theta + e; 
end

function theta_ml = ml(y, G)
    % Maximum Likelihood estimate: theta_ML = (G'G)^(-1) G'y
    theta_ml = (G' * G) \ (G' * y);
end

function [theta_MAP, C_post] = map(y, G, sigma_e, mu0, C0)
    % Maximum A Posteriori estimate with Gaussian prior
    
    Phi = (G'*G) + sigma_e^2 * inv(C0);
    Theta = (G'*y) + sigma_e^2 * (C0 \ mu0);

    theta_MAP = Phi \ Theta;
    C_post = sigma_e^2 * inv(Phi);
    

end

function plotResults(y, G, theta, theta_ML, theta_MAP, N)
    % Plot data and fitted models
    
    n_vals = 1:N;
    y_true = G * theta;
    y_ML = G * theta_ML;
    y_MAP = G * theta_MAP;
    
    figure;
    plot(n_vals, y, 'ko', 'MarkerSize', 6, 'MarkerFaceColor', 'k'); hold on;
    plot(n_vals, y_true, 'g-', 'LineWidth', 2);
    plot(n_vals, y_ML, 'b--', 'LineWidth', 2);
    plot(n_vals, y_MAP, 'r:', 'LineWidth', 2);
    
    xlabel('n'); ylabel('y_n');
    legend('Observations', 'True Model', 'ML Fit', 'MAP Fit', 'Location', 'best');
    title('Linear Trend Model: y_n = \theta_1 + \theta_2 n + e_n');
    grid on;
    
end

function exploreDifferentScenarios()
    % Explore different N values and prior settings
    
    fprintf('\n=== EXPLORING DIFFERENT SCENARIOS ===\n');
    
    % True parameters for all scenarios
    P = 2; % P represents the number of parameters theta
    mu0 = zeros(P,1); sigma0 = 1; C0 = sigma0^2 * eye(P);
    % theta = mu0 + sqrtm(C0) * randn(size(mu0)); % Draw theta from N(mu0, sigma0^2)

    theta = [2.0; -0.1];  % [DC; slope]
    sigma_e = 1;
    
    % Scenario 1: Different sample sizes
    figure;
    N_values = [2, 10, 50, 200];  % Must be >= P=2 for ML
    mu0 = [0; 0];
    C0 = eye(2);  % Unit prior covariance
    
    for i = 1:length(N_values)
        N = N_values(i);
        
        % Generate data
        G = [ones(N, 1), (1:N)'];
        y = generate_y(G, theta, N, sigma_e);
        
        % Estimates
        theta_ML = ml(y, G);
        [theta_MAP, C_post] = map(y, G, sigma_e, mu0, C0);
        
        subplot(2, 2, i);
        n_vals = 1:N;
        plot(n_vals, y, 'ko', 'MarkerSize', 4, 'MarkerFaceColor', 'k'); hold on;
        plot(n_vals, G * theta, 'g-', 'LineWidth', 2);
        plot(n_vals, G * theta_ML, 'b--', 'LineWidth', 2);
        plot(n_vals, G * theta_MAP, 'r:', 'LineWidth', 2);
        
        xlabel('n'); ylabel('y_n');
        title(sprintf('N = %d', N));
        
        legend('Data', sprintf('True [%.3f,%.3f]', theta(1), theta(2)), ...
            sprintf('ML [%.3f,%.3f]', theta_ML(1), theta_ML(2)), ...
            sprintf('MAP [%.3f,%.3f]', theta_MAP(1), theta_MAP(2)), 'Location', 'best');
        
        grid on;
        
        % Print results
        fprintf('N=%d: ML=[%.3f,%.3f], MAP=[%.3f,%.3f], Post_std=[%.3f,%.3f]\n', ...
            N, theta_ML(1), theta_ML(2), theta_MAP(1), theta_MAP(2), ...
            sqrt(C_post(1,1)), sqrt(C_post(2,2)));
    end
    sgtitle('Effect of Sample Size N on Linear Trend Estimation');
    
    % Scenario 2: Different prior strengths
    figure;
    N = 10;
    G = [ones(N, 1), (1:N)'];
    y = generate_y(G, theta, N, sigma_e);
    
    sigma0_values = [0.01, 0.1, 1, 10.0];  % Prior standard deviations
    
    for i = 1:length(sigma0_values)
        sigma0 = sigma0_values(i);
        mu0 = [0; 0];  % Prior mean
        C0 = sigma0^2 * eye(2);  % Prior covariance
        
        theta_ML = ml(y, G);  % ML doesn't change
        [theta_MAP, C_post] = map(y, G, sigma_e, mu0, C0);
        
        subplot(2, 2, i);
        n_vals = 1:N;
        plot(n_vals, y, 'ko', 'MarkerSize', 4, 'MarkerFaceColor', 'k'); hold on;
        plot(n_vals, G * theta, 'g-', 'LineWidth', 2);
        plot(n_vals, G * theta_ML, 'b--', 'LineWidth', 2);
        plot(n_vals, G * theta_MAP, 'r:', 'LineWidth', 2);
        
        xlabel('n'); ylabel('y_n');
        title(sprintf('\\sigma_0 = %.2f', sigma0));
       
        legend('Data', sprintf('True [%.3f,%.3f]', theta(1), theta(2)), ...
        sprintf('ML [%.3f,%.3f]', theta_ML(1), theta_ML(2)), ...
        sprintf('MAP [%.3f,%.3f]', theta_MAP(1), theta_MAP(2)), 'Location', 'best');
        
       
        grid on;
        
        fprintf('Prior_std=%.1f: MAP=[%.3f,%.3f], Post_std=[%.3f,%.3f]\n', ...
            sigma0, theta_MAP(1), theta_MAP(2), sqrt(C_post(1,1)), sqrt(C_post(2,2)));
    end
    sgtitle('Effect of Prior Strength on MAP Estimation');
    
    % Scenario 3: Different error variance
    figure;
    sigma_e_values = [0.01, 0.1, 1, 10.0];  % error standard deviations
    mu0 = [0; 0]; sigma_0 = 1; C0 = eye(2);
    N = 10;
    G = [ones(N, 1), (1:N)'];
    y = generate_y(G, theta, N, sigma_e);
    
    
    for i = 1:length(sigma0_values)
        sigma_e = sigma_e_values(i);
        y = generate_y(G, theta, N, sigma_e);

        theta_ML = ml(y, G);  % ML doesn't change
        [theta_MAP, C_post] = map(y, G, sigma_e, mu0, C0);
        
        subplot(2, 2, i);
        n_vals = 1:N;
        plot(n_vals, y, 'ko', 'MarkerSize', 4, 'MarkerFaceColor', 'k'); hold on;
        plot(n_vals, G * theta, 'g-', 'LineWidth', 2);
        plot(n_vals, G * theta_ML, 'b--', 'LineWidth', 2);
        plot(n_vals, G * theta_MAP, 'r:', 'LineWidth', 2);
        
        xlabel('n'); ylabel('y_n');
        title(sprintf('\\sigma_e = %.2f', sigma_e));
       
        legend('Data', sprintf('True [%.3f,%.3f]', theta(1), theta(2)), ...
        sprintf('ML [%.3f,%.3f]', theta_ML(1), theta_ML(2)), ...
        sprintf('MAP [%.3f,%.3f]', theta_MAP(1), theta_MAP(2)), 'Location', 'best');
        
       
        grid on;
        
        fprintf('Prior_std=%.1f: MAP=[%.3f,%.3f], Post_std=[%.3f,%.3f]\n', ...
            sigma0, theta_MAP(1), theta_MAP(2), sqrt(C_post(1,1)), sqrt(C_post(2,2)));
    end
    sgtitle('Effect of Error STD on MAP Estimation');


    % Scenario 3: Underdetermined case (N < P)
    fprintf('\n=== UNDERDETERMINED CASE (N < P) ===\n');
    N = 1;  % Only 1 observation, but P=2 parameters
    G = [1, 1];  % Design matrix: y = theta1 + theta2*1 + e
    y = generate_y(G, theta, N, sigma_e);
    
    fprintf('With N=%d < P=%d:\n', N, 2);
    
    try
        theta_ML = ml(y, G);
        fprintf('ML estimate: [%.3f, %.3f] (using pseudo-inverse)\n', theta_ML(1), theta_ML(2));
    catch ME
        fprintf('ML failed: %s\n', ME.message);
    end
    
    % MAP should still work due to prior regularization
    mu0 = [0; 0];
    C0 = eye(2);
    [theta_MAP, C_post] = map(y, G, sigma_e, mu0, C0);
    fprintf('MAP estimate: [%.3f, %.3f]\n', theta_MAP(1), theta_MAP(2));
    fprintf('MAP posterior std: [%.3f, %.3f]\n', sqrt(C_post(1,1)), sqrt(C_post(2,2)));
    
    fprintf('\nNote: MAP works even when ML fails because prior provides regularization!\n');
end