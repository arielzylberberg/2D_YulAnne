classdef MainGLM < EnIxn.Common.DataFilterEn
%% Settings
properties
    lev_kind = 'raw'; % 'beta'; % 'beta'|'raw'
    lev_cum = []; % []: false; true
    thres_plot_prop = 0.9;
    
    res_props = {'est', 'ci', 'n_tr_in_fr', 'n_tr_total'};
end
%% Results
properties
    est % (fr, 1)
    ci % (fr, [lb, ub])
    n_tr_in_fr % (fr, 1)
    n_tr_total % scalar
end
%% Internal
properties
    W_now = [];
end
%% Init
methods
    function W = MainGLM(varargin)
        W.dif_rel_incl = 1;
        
        if nargin > 0
            W.init(varargin{:});
        end
    end
end
%% Batch
methods
    function batch(W0, varargin)
        S_batch = varargin2S(varargin, {
            'subj', Data.Consts.subjs_RT
            'n_dim_task', {1, 2}
            'dim_rel_W', {1, 2}
            'dif_rel_incl', {1, 1:2} % , 1:2}
            'truncate_first_msec', 0
            't0_kind', {'st', 'en'}
            });
        [Ss, n] = factorizeC(S_batch);
        
        for ii = 1:n
            S = Ss(ii);
            C = S2C(S);
            
            clf;
            S_batch = W0.plot_overlay(C{:});
            C_file = {
                'dfi', S_batch.dif_irr_incl
                };
            W = W0.W_now;
            t_str = W.get_title(C_file);
            title(t_str);
            file = W.get_file(C_file);
            savefigs(file);
                
%             W0.imgather_overlay(C{:});
        end
    end
    function imgather_overlay(W0, varargin)
        % gather st and en across subjects, 
        % given an n_dim_task and dim_rel_W
        
        S0 = varargin2S(varargin, {
            'lev_kind', 'beta'
            });
        
        subjs = Data.Consts.subjs_RT;
        n_subj = numel(subjs);

        t0_kinds = {'st', 'en'};
        n_t0_kind = numel(t0_kinds);
        
        clf;
        ax = subplotRCs(n_subj, n_t0_kind);
        
        %%
        for i_subj = 1:n_subj
            for i_t0_kind = 1:n_t0_kind
                subj = subjs{i_subj};
                t0_kind = t0_kinds{i_t0_kind};
                ax1 = ax(i_subj, i_t0_kind);
                
                S = varargin2S({
                    'subj', subj
                    't0_kind', t0_kind
                    'ax', ax1
                    }, S0);
                
                switch S0.lev_kind
                    case 'useixn'
                        S = varargin2S({
                            'comp_kind', 'dim_rel'
                            }, S);
                end
                        
                C = S2C(S);
                W0.plot_overlay(C{:});

                switch t0_kind
                    case 'en'
                        set(ax1, 'XDir', 'reverse');
                end
            end
        end
        
        %%
        for row = 1:size(ax, 1)
            sameAxes(ax(row, :), [], [], 'y');
        end
        
        %%
        W = feval(class(W0), varargin{:});
        file = W.get_file({
            'sbj', subjs
            't0', t0_kinds
            });
        savefigs(file, 'size', [400, 600]);
    end
    function S_batch = plot_overlay(W0, varargin)
        S0 = varargin2S(varargin, {
            'ax', gca
            'comp_kind', 'dif' % 'dif'|'dim_rel'
            });
        
        %%
        switch S0.comp_kind
            case 'dif'
                % compare dif_irr_incl keeping others same
                S_batch = varargin2S(S0, {
                    'dif_irr_incl', {1:2, 3:4, 5}
                    });                
                [Ss, n] = factorizeC(S_batch);
                
                colors = hsv2rev(n);
                W = feval(class(W0));
                W0.W_now = W;
                
                for ii = 1:n
                    S = Ss(ii);
                    C = varargin2C(S, S0);
                    
                    W = varargin2props(W, C);
                    if ~W.load_if_existing
                        W = feval(class(W0), C{:});
                        W.calculate;
                    end
                    W.plot('ax', S0.ax, 'color', colors(ii, :), C{:});
                    hold(S0.ax, 'on');
                end
                hold(S0.ax, 'off');
                
            case 'dim_rel' % compare dim_rel_W
                S_batch = varargin2S(S0, {
                    'dim_rel_W', {1, 2}
                    });
                [Ss, n] = factorizeC(S_batch);
                
                colors = hsv2rev(n);
                W = feval(class(W0));
                W0.W_now = W;
                
                for ii = 1:n
                    S = Ss(ii);
                    C = varargin2C(S, S0);
                    
                    W = varargin2props(W, C);
                    if ~W.load_if_existing
                        W = feval(class(W0), C{:});
                        W.calculate;
                    end
                    W.plot('ax', S0.ax, 'color', colors(ii, :), C{:});
                    hold(S0.ax, 'on');
                end
                hold(S0.ax, 'off');
        end
    end
end
%% Main
methods
    function main(W, varargin)
%         varargin2props(W, varargin);
%         if ~W.load_if_existing
%             if ~isempty(varargin)
%                 W.init(varargin{:});
%             end
%             W.calculate;
%         end
        W.calculate;
        W.plot(varargin{:});
    end
    function calculate(W)
        %% Load results if existing
        [loaded, file] = W.load_if_existing;
        if loaded
            return;
        end
        
        %%
        en = W.get_ens_mat;
        ch = W.Data.ch == 2;
        en_rel = en{W.dim_rel_W};
        n_tr_in_fr = sum(~isnan(en_rel))';
        ch_rel = ch(:, W.dim_rel_W);
        
        %%
        if ~isempty(W.lev_cum) && W.lev_cum
            switch W.t0_kind
                case 'st'
                    direction = 'forward';
                case 'en'
                    direction = 'reverse';
            end
            
            csum = cumsum(en_rel, 2, direction, 'omitnan');
            ccnt = cumsum(~isnan(en_rel), 2, direction, 'omitnan');
            en_rel = csum ./ ccnt;
        end
        
        se = [];
        ci = [];

        switch W.lev_kind
            case 'beta'
                roni = bml.stat.ind_cols(W.Data.cond(:,W.dim_rel_W), 0);
                n_fr = size(en{1}, 2);
                est = zeros(n_fr, 1);
                se = zeros(n_fr, 1);
%                 ci = zeros(n_fr, 2);
                
                for fr = 1:n_fr
                    x = standardize([en_rel(:, fr), roni]);
                    res = glmwrap(x, ch_rel, 'binomial');
                    est(fr) = res.b(2);
                    se(fr) = res.se(2);
%                     ci(fr,:) = est(fr) + [-1, 1] .* res.se(2);
                end
                
            case 'logit_for_ch'
                roni = bml.stat.ind_cols(W.Data.cond(:,W.dim_rel_W), 0);
                n_fr = size(en{1}, 2);
                est = zeros(n_fr, 1);
                se = zeros(n_fr, 1);
%                 ci = zeros(n_fr, 2);
                
                for fr = 1:n_fr
                    res = glmwrap([en_rel(:, fr), roni], ch_rel, 'binomial');
                    y = bml.stat.y_for_ch(en_rel(:,fr), ch_rel, res.b(2));
                    est(fr) = nanmean(y);
                    se(fr) = sqrt((res.se(2) .^ 2) + nansem(y).^2);
                    
%                     ci(fr,:) = est(fr) + [-1, 1] .* se;
                end
                
            case 'raw'
                n_fr = size(en{1}, 2);
                est = zeros(n_fr, 1);
                se = zeros(n_fr, 1);
%                 ci = zeros(n_fr, 2);
                
                [~, ~, dcond] = unique(W.Data.cond(:,W.dim_rel_W));
                en_rel1 = en_rel;
                ndcond = max(dcond);
                if ndcond > 1
                    for dcond1 = 1:ndcond
                        incl = dcond == dcond1;
                        mdcond = nanmean(en_rel(incl,:));
                        en_rel1(incl,:) = bsxfun(@minus, ...
                            en_rel1(incl,:), mdcond);
                    end
                end
                
                for fr = 1:n_fr
                    y = bml.stat.y_for_ch(en_rel1(:, fr), ch_rel);
                    est(fr) = nanmean(y);
                    se(fr) = nansem(y);
%                     ci(fr,:) = est(fr) + [-1, 1] .* nansem(y);
                end                
                
            case 'cumpred'
                % cumulative predictability
                [est, ci] = W.calculate_cumpred(en_rel, ch_rel, ...
                    'to_bootstrap', true, ...
                    'n_boot', 60, ...
                    't_max', 1.5);
                
            case 'useixn'
                % Modified from 'beta'
                n_fr = size(en{1}, 2);
                est = zeros(n_fr, 1);
                se = zeros(n_fr, 1);
%                 ci = zeros(n_fr, 2);

                roni0 = [
                    bml.stat.ind_cols(W.Data.cond(:,W.dim_rel_W), 0), ...
                    bml.stat.ind_cols(W.Data.cond(:,W.dim_irr_W)) ...
                    ];

                en_irr = en{W.dim_irr_W};
                ch_irr = ch(:, W.dim_irr_W);
                en_ch_irr = bsxfun(@times, en_irr, sign(ch_irr - 0.5));
                en_rel_ch_irr = en_rel .* en_ch_irr;
                
                for fr = 1:n_fr
                    x = standardize([en_rel_ch_irr(:,fr), ...
                         en_rel(:, fr), ...
                         en_irr(:, fr), ...
                         en_ch_irr(:,fr), ...
                         roni0]);

                    res = glmwrap(x, ch_rel, 'binomial');
                    est(fr) = res.b(2);
                    se(fr) = res.se(2);
%                     ci(fr,:) = est(fr) + [-1, 1] .* res.se(2);
                end
        end
        
%         if ~isempty(W.lev_cum) && W.lev_cum
% %             switch W.t0_kind
% %                 case 'st'
% %                     direction = 'forward';
% %                 case 'en'
% %                     direction = 'reverse';
% %             end
%             
%             csum = cumsum(est);
%             ccnt = cumsum(~isnan(est));
%             est = csum ./ ccnt;
%             se = sqrt(cumsum(se .^ 2) ./ ccnt);
%         end
        
        if isempty(ci)
            ci = [est - se, est + se];
        end        
        
        %%
        W.est = est;
        W.ci = ci;
        W.n_tr_in_fr = n_tr_in_fr;
        W.n_tr_total = W.Data.n_tr;
        
        L = struct;
        L = copyprops(L, W, 'props', W.res_props);
        L = copyFields(L, W.S0_file);
        mkdir2(fileparts(file));
        save(file, '-struct', 'L');
        fprintf('Saved results to %s.mat\n', file);
    end
    
    [est, ci] = calculate_cumpred(W, en_rel, ch_rel, varargin)
    
    function [loaded, file] = load_if_existing(W)
        file = W.get_file;
        does_exist = exist([file, '.mat'], 'file');
        if does_exist && W.skip_existing_mat
            L = load(file);
            fprintf('Loaded results from %s.mat\n', file);
            copyprops(W, L, 'props', W.res_props);
            loaded = true;
        else
            if does_exist
                fprintf('Results exist but not skipping!\n');
            else
                fprintf('Results don''t exist: %s.mat\n', file);
            end
            loaded = false;
        end
    end
    function plot(W, varargin)
        S = varargin2S(varargin, {
            'ax', gca
            'color', [0 0 0]
            });
        
        %%
        err = bsxfun(@minus, W.ci, W.est);
        fr_incl = (W.n_tr_in_fr / W.n_tr_total) >= W.thres_plot_prop;
        
        t = W.t(fr_incl) + W.truncate_first_msec / 1e3;
        est = W.est(fr_incl);
        err = err(fr_incl, :);
        
        errorbarShade(t, est, err, S.color, [], {}, {}, 'ax', S.ax);
        crossLine(S.ax, 'h', 0);
        bml.plot.beautify(S.ax);
        x_lim = xlim;
        xlim([0, x_lim(2)]);
        ylim auto;
    end
end
%% File
methods
    function v = get_file_fields0(W)
        v = union_general({
            'lev_kind', 'lev'
            'lev_cum', 'cum'
            }, W.get_file_fields0@EnIxn.Common.DataFilterEn, ...
            'stable', 'rows');
    end
end
end