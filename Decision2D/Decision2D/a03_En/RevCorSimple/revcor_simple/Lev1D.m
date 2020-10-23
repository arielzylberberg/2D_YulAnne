classdef Lev1D < matlab.mixin.Copyable
methods (Static)
    function [slope, se_slope, bias, se_bias, res, S] = slope_by_time( ...
            en, ch, wt, varargin)
        % Get slope & bias for each time bin
        %
        % [slope, se_slope, bias, se_bias, res] = slope_by_time(en, ch)
        %
        % INPUT:
        % en(trial, t): momentary evidence. May come from pool_time
        % ch(trial): boolean choice.
        % wt(trial, t): # frames in the trial and time bin
        % en & ch are from pool_time()
        %
        % OUTPUT:
        % slope(t)
        % se_slope(t)
        % bias(t)
        % res{t}: stat from glmfit.
        S = varargin2S(varargin, {
            'cond', []
            'lev', 'mean' % 'beta'|'logit'|'mean'
            ... 'beta': beta coefficients from logisitc regression 
            ... 'logit': beta * en % SEM may not be correct!
            });
        
        n_tr = size(en, 1);
        nt = size(en, 2);
        slope = zeros(1, nt);
        se_slope = zeros(1, nt); 
        bias = zeros(1, nt);
        se_bias = zeros(1, nt);
        res = cell(1, nt);
        if nargin < 3 || isempty(wt)
            wt = ones(n_tr, nt);
        end        
        if ~isempty(S.cond)
            [~,~,d_cond] = unique(S.cond);
        else
            d_cond = [];
        end
        for t = 1:nt
            switch S.lev
                case 'mean'
                    ch1 = sign(ch - 0.5);
                    en1 = en(:,t);
                    
                    bias(t) = nanmean(en1);
                    se_bias(t) = nansem(en1);
                    
                    if ~isempty(S.cond)
                        bias_by_cond = accumarray(d_cond, en1, ...
                            [], @nanmean);
                        bias_cond = bias_by_cond(d_cond);
                        en_ch1 = (en1 - bias_cond) .* ch1;
                    else
                        en_ch1 = (en1 - bias(t)) .* ch1;
                    end
                    slope(t) = nanmean(en_ch1);
                    se_slope(t) = nansem(en_ch1);
                    % se_slope is not correct for now
                    res1 = struct;
                    
                case 'beta'
                    [b, ~, res1] = glmfit(en(:,t), ch, 'binomial', ...
                        'Weights', wt(:,t));

                    slope(t) = b(2);
                    se_slope(t) = res1.se(2);
                    bias(t) = b(1);
                    se_bias(t) = res1.se(1);
                case 'logit'
                    [b, ~, res1] = glmfit(en(:,t), ch, 'binomial', ...
                        'Weights', wt(:,t));

                    ch1 = sign(ch - 0.5);
                    en_ch1 = en(:,t) .* ch1;
                    slope(t) = b(2) * nanmean(en_ch1);
                    se_slope(t) = b(2) * nansem(en_ch1);
                    bias(t) = b(1) * nanmean(en_ch1);
                    se_bias(t) = b(1) * nansem(en_ch1);
            end
            res{t} = res1;
        end
    end
    function [en, wt, t, S] = pool_time(en0, varargin)
        % [en, wt, t, S] = pool_time(en0, varargin)
        %
        % INPUT:
        % en0: cell array of vectors of momentary evidence
        %
        % OUTPUT:
        % en: average
        % wt: number of valid frames included
        % t: time vector (in seconds)
        % S: array of settings
        %
        % OPTIONS:
        % 9/75 = 0.12 (sec)
        % 12/75 = 0.16 (sec)
        % 15/75 = 0.20 (sec)
        % 'truncate_st', 12
        % 'truncate_en', 15
        % 'align', -1 % -1: beginning; 0: middle; 1: end
        % 'n_bin_to_pool', 12
        % 'frame_rate', 75        
        
        S = varargin2S(varargin, {
            % 9/75 = 0.12 (sec)
            % 12/75 = 0.16 (sec)
            % 15/75 = 0.20 (sec)
            'truncate_st', 12 
            'truncate_en', 15 
            'align', -1 % -1: beginning; 0: middle; 1: end
            'n_bin_to_pool', 12
            'frame_rate', 75
            'cum', 'none' % 'sum'|'mean'|'none'
            });
        en0 = cellfun(@(v) v((S.truncate_st + 1):end), en0, ...
            'UniformOutput', false);
        en0 = cellfun(@(v) v(1:(end - S.truncate_en)), en0, ...
            'UniformOutput', false);
        
        switch S.cum
            case 'sum'
                en0 = cellfun(@(v) cumsum(v), en0, ...
                    'UniformOutput', false);
            case 'mean'
                en0 = cellfun(@(v) cumsum(v) ./ (1:length(v)), en0, ...
                    'UniformOutput', false);
            case 'none'
                % Do nothing.
        end
        
        switch S.align
            case -1
                en0 = cell2mat2(en0);
            case 0
                len0 = cellfun(@length, en0);
                max_len = max(len0);
                n_pad = round((max_len - len0) / 2);
                n_tr = size(en0, 1);
                en0 = arrayfun(@(ii) [nan(1, n_pad(ii)), en0{ii}], 1:n_tr, ...
                    'UniformOutput', false);                
                en0 = cell2mat2(en0);
            case 1
                en0 = cellfun(@flip, en0, 'UniformOutput', false);
                en0 = cell2mat2(en0);
        end
        
        %%
        n_tr = size(en0, 1);
        nt0 = size(en0, 2);
        nt = floor((nt0 - 1) / S.n_bin_to_pool) + 1;
        en = nan(n_tr, nt);
        wt = zeros(n_tr, nt);
        for it = 1:nt
            fr_st = (it - 1) * S.n_bin_to_pool + 1;
            fr_en = min(fr_st - 1 + S.n_bin_to_pool, nt0);
            en(:,it) = nanmean(en0(:,fr_st:fr_en), 2);
            wt(:,it) = sum(~isnan(en0(:,fr_st:fr_en)), 2);
        end
        wt = wt ./ S.n_bin_to_pool;
        
        %%
        if S.align == -1
            truncate = S.truncate_st;
        else
            truncate = S.truncate_en;
        end
        t = (truncate - S.n_bin_to_pool * 0.5 ...
            + S.n_bin_to_pool .* (1:nt)) / S.frame_rate;
    end
end    
end