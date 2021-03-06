function [alpha,alphan,W,Wo] = denoising_nmf_2(V,D,options,Px,Pn,A)
%
%  0.5* || V -Ds*alphas - Dn*alphan ||^2 + lambda || alphas ||_1
%
% Ds is sparse non-negative dictionary for speech
% V is the magnitude spectrogram of the mixed signal

% number of atoms for noise dictionary


if nargin <6
    A = [];
end

Kn = getoptions(options,'Kn',5);

[N,M] = size(V);

Wo = mexNormalize(max(0.1+rand(N,Kn),0));
W = getoptions(options,'W',Wo);
alphan = max(0.1+rand(Kn,M),0)+0.01;

niter = getoptions(options,'iter',20);

eps = 1e-9;

in_iter = 1;

% precompute Lipshitz bound
mu = getoptions(options,'mu',0);
L = norm(D,2)^2+ mu^2*norm(A,2)^2 + mu^2;

options.Dsq = D'*D;

for i=1:niter
    
    % minimize over [alphas, alphan]
    options.t0 = .5 * (1/(L +norm(W,2)^2 ));
    
    alpha = mexLasso(max(V-W*alphan,0),D,options);
    %alpha = nmf_semisup(max(V-W*alphan,0),D,zeros(size(W)),[],options);
    
    c = compute_obj(V,[alpha;alphan],D,W,options);
    
    a(i,:) = [norm(Px - D*alpha,'fro')^2/norm(Px,'fro')^2 norm(Pn - W*alphan,'fro')^2/norm(Px,'fro')^2 norm(V -D*alpha - W*alphan,'fro')^2/norm(V,'fro')^2 c];
    
    options.H = [alpha;alphan];
    
    % minimize Dn
    for j=1:in_iter
        
        % W
        V_ap = [D,W]*[alpha;alphan];
        W = W .* ( V*alphan')./(V_ap*alphan');

        W = mexNormalize(W);
        W(W(:)<eps) = 0;
        
        
        % alpha_n
        V_ap = [D,W]*[alpha;alphan];
        alphan = alphan.* (W'*V)./(W'*V_ap + eps);
        
    end

    
end

keyboard
