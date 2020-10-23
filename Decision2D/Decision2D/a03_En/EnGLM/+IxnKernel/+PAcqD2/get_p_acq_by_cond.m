function [p_acq, ci_p_acq, res_hard, res_easy] = get_p_acq_by_cond( ...
    en_hard, ch_hard, en_easy, ch_easy, roni_easy, roni_hard, varargin)
% [p_acq, ci_p_acq, res_hard, res_easy] ...
% = get_p_acq_by_cond(en_hard, ch_hard, en_easy, ch_easy)
%
% OPTIONS:
% 'lapse0', logit(5e-2)
% 'lapse_min', logit(1e-2)
% 'lapse_max', logit(1 - 1e-2)
% 'pacq_kind', 'lapse' % 'lapse12'|'lapse'|'slope'
% : 
%
% OUTPUT:
% ci_p_acq([LB, UB])

S = varargin2S(varargin, {
    'lapse0', logit(5e-2)
    'lapse_min', logit(1e-2)
    'lapse_max', logit(1 - 1e-2)
    'pacq_kind', 'lapse' % 'lapse12'|'lapse'|'slope'
    ... % 'lapse12': compare lapse rates from both easy and hard
    ... % 'lapse': get lapse rates from hard, fixing slope to easy's
    ... % 'slope': compare slope of hard vs easy
    });
C = S2C(S);

if nargin < 5
    roni_easy = [];
end
if nargin < 6
    roni_hard = [];
end

% [b_easy, res_easy] = bml.stat.glmfit_lapse(standardize(en_easy), ch_easy, C{:});

incl = ~all(isnan(en_easy), 2) & ~isnan(ch_easy);
en_easy = en_easy(incl, :);
ch_easy = ch_easy(incl, :);
if ~isempty(roni_easy)
    roni_easy = roni_easy(incl, :);
end

switch S.pacq_kind
    case 'lapse12'
        error('Not implemented yet!');
        
        res_easy = glmwrap( ...
            standardize([en_easy, roni_easy]), ch_easy, 'binomial');
        b_easy = res_easy.b;
        b_easy(end+1) = logit(1e-5);
        res_easy.b = b_easy;
        
        C = varargin2C(C, {
            ... 'bias0', b_easy(1)
            ... 'bias_min', b_easy(1)
            ... 'bias_max', b_easy(1)
            'slope0', b_easy(2)
            'slope_min', b_easy(2)
            'slope_max', b_easy(2)
            });

        incl = ~all(isnan(en_hard), 2) & ~isnan(ch_hard);
        en_hard = en_hard(incl, :);
        ch_hard = ch_hard(incl, :);
        if ~isempty(roni_hard)
            roni_hard = roni_hard(incl, :);
        end

        [b_hard, res_hard] = bml.stat.glmfit_lapse( ...
            standardize([en_hard, roni_hard]), ch_hard, C{:});

        % p_acq = 1 - invLogit(b_hard(end)); % 1 - lapse
        p_acq = (1 - invLogit(b_hard(end))) / (1 - invLogit(b_easy(end)));

        ci_p_acq = ...
            1 - invLogit([
                    b_hard(end) - res_hard.se(end), ...
                    b_hard(end) + res_hard.se(end)
                ]);
            
    case 'lapse'
        res_easy = glmwrap( ...
            standardize([en_easy, roni_easy]), ch_easy, 'binomial');
        b_easy = res_easy.b;
        b_easy(end+1) = logit(1e-5);
        res_easy.b = b_easy;
        
        C = varargin2C(C, {
            ... 'bias0', b_easy(1)
            ... 'bias_min', b_easy(1)
            ... 'bias_max', b_easy(1)
            'slope0', b_easy(2:(end-1))
            'slope_min', b_easy(2:(end-1))
            'slope_max', b_easy(2:(end-1))
            });

        incl = ~all(isnan(en_hard), 2) & ~isnan(ch_hard);
        en_hard = en_hard(incl, :);
        ch_hard = ch_hard(incl, :);
        if ~isempty(roni_hard)
            roni_hard = roni_hard(incl, :);
        end

        [b_hard, res_hard] = bml.stat.glmfit_lapse( ...
            standardize([en_hard, roni_hard]), ch_hard, C{:});

        % p_acq = 1 - invLogit(b_hard(end)); % 1 - lapse
        p_acq = (1 - invLogit(b_hard(end))) / (1 - invLogit(b_easy(end)));

        ci_p_acq = ...
            1 - invLogit([
                    b_hard(end) - res_hard.se(end), ...
                    b_hard(end) + res_hard.se(end)
                ]);        
            
    case 'slope'
        res_easy = glmwrap( ...
            standardize([en_easy, roni_easy]), ch_easy, 'binomial');
        b_easy = res_easy.b;
        
        incl = ~all(isnan(en_hard), 2) & ~isnan(ch_hard);
        en_hard = en_hard(incl, :);
        ch_hard = ch_hard(incl, :);
        if ~isempty(roni_hard)
            roni_hard = roni_hard(incl, :);
        end
        
        res_hard = glmwrap( ...
            standardize([en_hard, roni_hard]), ch_hard, 'binomial');
        b_hard = res_hard.b;

        p_acq = b_hard(2) / b_easy(2);

        ci_p_acq = [nan, nan]; % Not supported yet

    otherwise
        error('Unsupported pacq_kind:%s\n', S.pacq_kind);
end

    
