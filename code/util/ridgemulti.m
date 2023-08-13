function [ws cmode] = ridgemulti(X, Y, as, dmod)
% function ws = ridgemulti(X, Y, as, [dmod])
% A simple multi-input multi-output ridge regressor
%    Input:
%          X: stimuli (SxN matrix, S is sample size, N is feature
%          size)
%          Y: responses (SxV matrix, V is model size)
%         as: an array of regularization parameters (Ax1 vector)
%       dmod: mode for regularization
%             (for backword compatibility; use dmod=1 (default))
%    Output:
%        ws: estimated weights (NxVxA matrix)
%
% Reference: see Tikhonov regularization in Wikipedia
% SN, June 2009
%

if ~exist('dmod', 'var')
    dmod = 1;
end

if size(X,2)>size(X,1)
    cmode = 1;
    cnum = size(X,1);
elseif size(X,1)==size(X,2) && isequal(X,X')
    fprintf('X is symmetric. Assuming that this is a covariance matrix.\n');
    cmode = 2;
    cnum = size(X,1);
else
    cmode = 0;
    cnum = size(X,2);
end

fprintf('%d samples, %d channels\n',size(X,1),size(X,2));
fprintf('calculating ridge regression (%dx%d)...', cnum, cnum);

switch cmode
    case 1
        [U S] = eig(X*X');
    case 0
        [U S] = eig(X'*X);
    case 2
        [U S] = eig(X);
end

ds = diag(S);

ws = zeros(size(X,2),size(Y,2),length(as),'single');

if cmode
    U1=X'*U;
    U2=U'*Y;
else
    Uxy=U'*X'*Y;
end

for ii=1:length(as)
    if dmod==0
        Sd = diag(ds./(ds.^2+as(ii).^2));
    else
        Sd = diag(1./(ds+as(ii)));
    end
    if cmode
        ws(:,:,ii) = U1*Sd*U2;
    else
        ws(:,:,ii) = U*Sd*Uxy;
    end
end

fprintf('done.\n');

return
