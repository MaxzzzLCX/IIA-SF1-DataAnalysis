% From Moodel
% Code for Task 3.4 - "Bayesian finding a needle in a haystack"
clc, clearvars

N=100;

sigma_n=2; % noise sigma

sigma_theta=1; % prior sigma

signal=[-3 5 -2 4 1 3 5 -1 2 4 6 5 -2 -2 1]; % length 15 data

noise=sigma_n*randn(N,1);

theta=sigma_theta*randn(1);

offset=round(85*rand(1));

y=noise;

signal_offset=0*noise;
signal_offset(offset:offset+14)=signal'*theta;


y=y+signal_offset; % embed the signal'*theta at some section of the signal

ax=[1:N];
subplot(211)
plot(ax,y,ax,signal_offset)
subplot(212)
match_n=filter(signal(end:-1:1),1,noise);
match_s=filter(signal(end:-1:1),1,signal_offset);
match=filter(signal(end:-1:1),1,y);
plot(ax,match,ax,match_s,ax,match_n)