clc, clearvars, close all;
% ar_gap_repair
% ar_gap_repair_ML_approach(40000)

function ar_gap_repair
    % AR based packet-loss repair (100/1000 gap pattern)
    % --------------------------------------------------
    % Max Lyu – coursework, week 4
    
    % -------------- user parameters -------------------
    inFile        = '/Users/maxlyu/Desktop/Part IIA/SF1/IIA-SF1-DataAnalysis/Week 3/grosse_40_percent_missing.wav';   % <-- change to any mono WAV
    frameDur_ms   = 20;               % frame length
    hopFrac       = 0.50;             % 50 % overlap
    maxOrder      = 20;
    priorScale    = 0.5;              % σ_a² = σ_e² / priorScale
    addResidNoise = true;             % inject residual noise or not
    
    gapLen        = 100;              % missing 100 consecutive samples
    gapPeriod     = 1000;             % every 1000 samples (10 %)
    % --------------------------------------------------
    
    % read + normalise
    [x,fs] = audioread(inFile);        x = x(:,1);
    x = x / max(abs(x)+eps);           % avoid clip
    
    % A.  frame-wise AR order selection
    Lframe = round(frameDur_ms/1000*fs);
    Lhop   = round(Lframe*hopFrac);
    frames = buffer(x,Lframe,Lframe-Lhop,'nodelay').';    % M × L
    M = size(frames,1);
    
    orderSel  = zeros(M,1);            % best P for each frame
    aHat      = cell(M,1);             % AR coeffs
    sigmaEHat = zeros(M,1);            % σ̂_e per frame
    
    fprintf('Selecting AR order frame-wise …\n');
    for m = 1:M
        [Pbest,aBest,sigEbest] = selectOrder(frames(m,:).',maxOrder,priorScale);
        orderSel(m)  = Pbest;
        aHat{m}      = aBest;
        sigmaEHat(m) = sigEbest;
    end
    
    % B.  build gap mask and zero the lost packets
    N = numel(x);
    mask = true(N,1);
    for k = 1:gapPeriod:N-gapLen
        mask(k:k+gapLen-1) = false;
    end
    y = x;  
    % y(~mask) = 0;
    
    % C.  interpolate each gap
    yRec = y;
    gapStart = find(~mask & [true; mask(1:end-1)]);   % first idx of each gap
    
    for g = 1:numel(gapStart)
        n0   = gapStart(g);
        Lgap = gapLen;                                % fixed pattern
        idxG = n0:n0+Lgap-1;
    
        % nearest frame to left edge
        frameIdx = max(1, ceil(n0 / Lhop));
        P  = orderSel(frameIdx);
        a  = aHat{frameIdx};
        se = sigmaEHat(frameIdx);
    
        if n0<=P || n0+Lgap-1+P > N,  continue;  end  % gap touches border
    
        % ---- forward (causal) prediction
        xF = zeros(Lgap,1);  buf = yRec;
        for n = 1:Lgap
            idx = n0+n-1;
            hist = buf(idx-1:-1:idx-P);
            xF(n) = a.'*hist + addResidNoise*se*randn;
            buf(idx) = xF(n);
        end
    
        % ---- backward (anti-causal) prediction
        xB = zeros(Lgap,1);  buf = yRec;
        for n = 1:Lgap
            idx = n0+Lgap-n;
            hist = buf(idx+1:idx+P);
            xB(Lgap-n+1) = a.'*hist + addResidNoise*se*randn;
            buf(idx) = xB(Lgap-n+1);
        end
    
        % ---- cross-fade
        alpha = linspace(1,0,Lgap).';
        yRec(idxG) = alpha.*xF + (1-alpha).*xB;
    end
    
    % D.  metrics & write
    mseGap = mean((x(~mask)-yRec(~mask)).^2);
    mseAll = mean((x-yRec).^2);
    
    fprintf('\nMSE on lost samples : %.6f\n',mseGap);
    fprintf('MSE on full signal  : %.6f\n',mseAll);
    
    [~,name] = fileparts(inFile);
    outFile = sprintf('interp_%s_gap%do%do.wav',name,gapLen,gapPeriod);
    audiowrite(outFile,yRec,fs);
    fprintf('Repaired WAV written → %s\n',outFile);

end

function [bestP,aBest,sigBest] = selectOrder(y,maxP,priorScale)
    lml  = -inf(maxP,1);
    aAll = cell(maxP,1);  sigE = zeros(maxP,1);
    for P = 1:maxP
        [a_ml,sig_ml] = estimateAR_ML(y,P);
        priorSig      = sig_ml / priorScale;
        C0            = (priorSig^2)*eye(P);
        mu0           = zeros(P,1);

        N    = numel(y);
        G    = compute_G_matrix(y,N,P);
        logML = log_marginal_likelihood(y(P+1:end),G,mu0,C0,sig_ml);

        lml(P)  = logML;
        aAll{P} = a_ml;
        sigE(P) = sig_ml;
    end
    [~,bestP] = max(lml);
    aBest = aAll{bestP};
    sigBest = sigE(bestP);
end
% -------------------------------------------------------------------------
function [a_est, sigma_e_est] = estimateAR_ML(y, P)
    N = numel(y);
    G = compute_G_matrix(y,N,P);
    yv = y(P+1:N);
    a_est = (G' * G) \ (G' * yv);
    res   = yv - G * a_est;
    sigma_e_est = sqrt(mean(res.^2) + eps);
end
% -------------------------------------------------------------------------
function lml = log_marginal_likelihood(y, G, mu0, C0, sigma_e)
    N = numel(y);
    if isequal(G,0)                        % null model
        lml = -N/2*log(2*pi*sigma_e^2) - (y'*y)/(2*sigma_e^2);
        return
    end
    C0inv = inv(C0);
    Phi   = G'*G + sigma_e^2 * C0inv;
    Theta = G'*y + sigma_e^2 * (C0 \ mu0);
    thetaMAP = Phi \ Theta;
    P = numel(mu0);
    lml = -P/2*log(2*pi) - 0.5*log(det(C0)) - 0.5*log(det(Phi)) ...
          -(N-P)/2*log(2*pi*sigma_e^2) ...
          - (y'*y + sigma_e^2*mu0'*(C0\mu0) - Theta'*thetaMAP)/(2*sigma_e^2);
end
% -------------------------------------------------------------------------
function G = compute_G_matrix(y,N,P)
    G = zeros(N-P,P);
    for i = 1:N-P
        G(i,:) = flip(y(i:i+P-1));
    end
end

function yRec = LSAR_interpolate(y, mask, a)

    P   = numel(a);
    x   = y(:);                       % ensure column
    N   = numel(x);
    yRec = x;                         % will hold the reconstruction

    % ---------- build Toeplitz A -----------------------------------------
    G = zeros(N-P,P);
    for n = 1:N-P
        G(n,:) = flip(x(n:n+P-1));    % only to reuse compute_G_matrix
    end
    A = [ -G  eye(N-P) ];             % (N-P) × N complete matrix
    A(:,1:P) = -fliplr(cumsum(repmat(a.',N-P,1),2)); % faster Toeplitz

    % ---------- each contiguous gap separately ---------------------------
    dMask = diff([false; mask; false]);
    gapBeg = find(dMask == -1);       % entering a gap
    gapEnd = find(dMask == +1)-1;     % leaving a gap

    for g = 1:numel(gapBeg)
        idxU = gapBeg(g):gapEnd(g);           % unknown row indices
        idxK = setdiff(1:N, idxU);            % known sample indices

        Au  = A(:,idxU);
        Ak  = A(:,idxK);

        % LS / ML solution (49)
        xU  = - (Au.'*Au) \ (Au.' * Ak * x(idxK));

        yRec(idxU) = xU;
    end
end

ar_LSAR_repair("/Users/maxlyu/Desktop/Part IIA/SF1/IIA-SF1-DataAnalysis/Week 3/grosse_40_percent_missing.wav", "output.wav", 200, 0.5, 20)

function ar_LSAR_repair(inFile, outFile, frameDur_ms, hopFrac, maxOrder)
% LS/ML AR interpolation for pre-gapped audio
% ---------------------------------------------------------------
% inFile      : WAV that already contains the "holes" (zeros)
% outFile     : repaired WAV (pass '' to suppress writing)
% frameDur_ms : analysis frame length in ms   (default 20)
% hopFrac     : hop / frame                    (default 0.50)
% maxOrder    : maximum AR order tested        (default 20)

% ---------- defaults ----------
if nargin < 3, frameDur_ms = 20; end
if nargin < 4, hopFrac     = 0.50; end
if nargin < 5, maxOrder    = 20;  end
priorScale = 0.5;           % σ_a² = σ_e² / priorScale
thr        = 1e-5;          % "zero" threshold
% -------------------------------

% 1) read – x already contains missing packets (=0)
[x,fs] = audioread(inFile); x = x(:,1);
x = x / max(abs(x)+eps);

N    = numel(x);
mask = abs(x) > thr;        % true  = observed, false = missing
y    = x;                   % *do not* overwrite the gaps!

% 2) frame-wise order selection on *known* samples
Lframe   = round(frameDur_ms/1000*fs);
Lhop     = round(Lframe*hopFrac);
frames   = buffer(y,Lframe,Lframe-Lhop,'nodelay').';
maskFr   = buffer(mask,Lframe,Lframe-Lhop,'nodelay').';

M        = size(frames,1);
orderSel = zeros(M,1);
aHat     = cell(M,1);
sigEHat  = zeros(M,1);

for m = 1:M
    y_m   = frames(m,:).';
    m_msk = maskFr(m,:).';
    [P,a,sig] = selectOrder_valid(y_m,m_msk,maxOrder,priorScale);
    orderSel(m)=P;  aHat{m}=a; sigEHat(m)=sig;
end

% 3) LSAR repair --------------------------------------------------
yRec = y;
gapStart = find(~mask & [true; mask(1:end-1)]); % first idx of each gap
for g = 1:numel(gapStart)
    n0   = gapStart(g);
    n1   = find(mask(n0:end),1,'first')+n0-2;
    if isempty(n1), n1=N; end
    idxGap = n0:n1;

    frameIdx = max(1, ceil(n0 / Lhop));
    P   = orderSel(frameIdx);
    a   = aHat{frameIdx};

    theta = lsar_gap(a, yRec, idxGap, P);
    yRec(idxGap) = theta;          % insert ML/LSAR estimate
end


% 4) save + report
if ~isempty(outFile)
    audiowrite(outFile,yRec,fs);
end
fprintf('✓ LSAR repair complete ⇒ %s\n',outFile);
end

function [bestP,aBest,sigBest] = selectOrder_valid(y,mask,maxP,priorScale)
% order selection when some samples are missing
bestP=1; logMLmax=-inf;
for P=1:maxP
    [a_ml,sig_ml] = estimateAR_ML_valid(y,mask,P);
    priorSig = sig_ml/priorScale; C0 = (priorSig^2)*eye(P);
    mu0 = zeros(P,1);

    [G,xv] = build_G_valid(y,mask,P);
    lml = log_marginal_likelihood(xv,G,mu0,C0,sig_ml);

    if lml>logMLmax
        logMLmax=lml; bestP=P; aBest=a_ml; sigBest=sig_ml;
    end
end
end
% ------------------------------------------------------------------
function [a,sig] = estimateAR_ML_valid(y,mask,P)
[G,xv] = build_G_valid(y,mask,P);
a  = (G'*G)\(G'*xv);
res = xv-G*a; sig = sqrt(mean(res.^2)+eps);
end
% ------------------------------------------------------------------
function [G,xv] = build_G_valid(y,mask,P)
N = numel(y);
keep = mask; keep(1:P)=false;         % need P past samples
rows = find(keep);
rOk  = rows(rows>P);
xv   = y(rOk);

nR   = numel(rOk);
G    = zeros(nR,P);
for r = 1:nR
    i = rOk(r);
    G(r,:) = flip(y(i-P:i-1));
end
end
% ------------------------------------------------------------------
function A = build_A_LSAR(y,mask,a,idxGap)
% Build matrix A (Eq. 44) but only *rows* involving idxGap matter.
P = numel(a); N = numel(y); iGap = idxGap(:).';
rows = setdiff(P+1:N,idxGap);

A = zeros(numel(rows),N);
for r=1:numel(rows)
    n = rows(r);
    A(r,n) = 1;
    A(r,n-P:n-1) = -a(:)';
end
A = A(:,idxGap);  % keep columns of unknowns
end

function theta = lsar_gap(Acoef, xKnown, idxGap, P)
% LS/ML estimate for ONE gap (Eq. 49) – memory-safe
%
% Acoef   : AR coefficients a(1:P)         (column)
% xKnown  : full signal with zeros in the gap
% idxGap  : indices of the missing burst
% P       : AR order

N   = numel(xKnown);
ig  = idxGap(:).';                    % unknown sample indices
kg  = setdiff( (P+1:N), ig );         % rows where eqn is defined
Lk  = numel(kg);
Lg  = numel(ig);

% ---------- build sparse A(:,idxGap) ----------
% each row n uses columns n, n-1, …, n-P
rows = repmat((1:Lk).',1,P+1);        % row numbers
cols = bsxfun(@minus, kg.', 0:P);     % sample indices per row
vals = [ones(Lk,1) , -repmat(Acoef.',Lk,1)];

% keep only columns that are in the gap
sel        = ismember(cols, ig);
rows_sparse= rows(sel);
cols_sparse= cols(sel);
vals_sparse= vals(sel);

A_i = sparse(rows_sparse, ...
             discretize(cols_sparse, [ig-0.5 ig(end)+0.5]), ...
             vals_sparse, ...
             Lk, Lg);                 % rows × unknown-cols

% ---------- right-hand side  ----------
b = filter([0 -Acoef(:).'],1,xKnown); % innovations
b = b(kg);

% ---------- LS solution (Eq. 49) ----------
theta = -(A_i.'*A_i) \ (A_i.' * b);
end
