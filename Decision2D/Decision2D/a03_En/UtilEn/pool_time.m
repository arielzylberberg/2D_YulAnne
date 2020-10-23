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
        'pad_with', 0 % nan
        });
    en0 = cellfun(@(v) v((S.truncate_st + 1):end), en0, ...
        'UniformOutput', false);
    en0 = cellfun(@(v) v(1:(end - S.truncate_en)), en0, ...
        'UniformOutput', false);
    switch S.align
        case -1
            en0 = cell2mat2(en0, S.pad_with);
        case 0
            len0 = cellfun(@length, en0);
            max_len = max(len0);
            n_pad = round((max_len - len0) / 2);
            n_tr = size(en0, 1);
            en0 = arrayfun(@(ii) ...
                [S.pad_with + zeros(1, n_pad(ii)), en0{ii}], ...
                1:n_tr, ...
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
    t = (S.truncate_st - S.n_bin_to_pool * 0.5 ...
        + S.n_bin_to_pool .* (1:nt)) / S.frame_rate;
end