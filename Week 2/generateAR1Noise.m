function noise = generateAR1Noise(N, alpha, sigma)
    % AR(1): n_t = alpha * n_{t-1} + sigma * w_t
    noise = zeros(1, N);
    for t = 2:N
        noise(t) = alpha * noise(t-1) + sigma * randn();
    end
end