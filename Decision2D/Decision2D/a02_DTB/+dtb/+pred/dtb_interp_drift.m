function D = dtb_interp_drift(drift0, t, ub, lb, y, p0, notabs_flag, ...
        varargin)
% Interpolates results to allow many (>100) distinct drifts.
%
% D = dtb_interp_drift(drift0, t, ub, lb, y, p0, notabs_flag, ...)
%    
% OPTIONS:
% ... % Can use any function with the same input/output.
% 'dtb_fun', @dtb.pred.spectral_dtbAA
% 'n_drift', 100
%
% See also: dtb.pred.spectral_dtbAA

% 2016 (c) Yul Kang. hk2699 at columbia dot edu.

S = varargin2S(varargin, {
    ... % Can use any function with the same input/output.
    'dtb_fun', @dtb.pred.spectral_dtbAA
    'n_drift', 100
    });

if nargin < 7
    notabs_flag = false;
end

assert(isvector(drift0));
if numel(drift0) <= S.n_drift
    % Do not interpolate in numel(drift0) is not bigger.
    D = S.dtb_fun(drift0, t, ub, lb, y, p0, notabs_flag);
    return;
end

min_drift = min(drift0);
max_drift = max(drift0);
drift = linspace(min_drift, max_drift, S.n_drift);

% tic; % DEBUG
D = S.dtb_fun(drift, t, ub, lb, y, p0, notabs_flag);
% disp('dtb finished'); % DEBUG
% toc; % DEBUG

% Interpolate D.up.pdf_t and lo.pdf_t.
% Remove p, mean_t, cdf_t, since they won't be interpolated and
% thus invalid for now.
for side = {'lo', 'up'}
    D.(side{1}) = rmfield(D.(side{1}), {'p', 'mean_t', 'cdf_t'});

    D.(side{1}).pdf_t = ...
        max(interp1(drift(:), D.(side{1}).pdf_t', drift0(:), 'spline')', 0); %#ok<UDIM>
end
% disp('td_pdf interpolation finished'); % DEBUG
% toc; % DEBUG

% Interpolate notabs.pdf if required.
if notabs_flag
    for f = {'pdf', 'pos_t', 'neg_t'}
        D.notabs.(f{1}) = ...
            max(interp1(drift, D.notabs.(f{1}), drift0, 'spline'), 0);
    end
%     disp('notabs interpolation finished'); % DEBUG
%     toc; % DEBUG
end
end
