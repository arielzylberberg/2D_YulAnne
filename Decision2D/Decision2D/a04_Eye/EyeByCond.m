classdef EyeByCond < matlab.mixin.Copyable
methods
    function main_eye_by_cond(EC)
        %%
        for y = {'vel'} % {'pos', 'vel', 'acc'}
            for summ = {'logit', 'slope', 'corr'}
                for i_subj = 1:3
                    C = varargin2C({
                        'y', y{1}
                        'summary', summ{1} % 'mean'|'slope'|'logit'|'corr'
                        't_smooth', 25
                        });
                    EC.eye_by_cond(i_subj, C{:});
                end
            end
        end
    end
    function eye_by_cond(EC, i_subj, varargin)
        S = varargin2S(varargin, {
            % position, velocity, acceleration
            'y', 'pos' % 'pos'|'vel'|'acc' 
            % 'summary'
            % - 'mean': mean
            % - 'slope': slope of linear regression
            % - 'logit': slope of logistic regression
            % - 'corr': moment-by-moment correlation with the other dim
            'summary', 'slope' % 'mean'|'slope'|'logit'|'corr'
            't_smooth', 100 % in ms
            });
        
        %%
        subj = Data.Consts.subjs_RT{i_subj};
        file_in = fullfile('../Data_2D/sTr', sprintf('RT_%s_eye.mat', subj));
        L = load(file_in);
        fprintf('Loaded %s\n', file_in);
        sTr = L.dat;
        n_tr = size(sTr, 1);
        
        %%        
        xypts = EC.truncate_w_RDK(sTr.eye_xypt, sTr.RT);
        
        %%
        cond = [sTr.condM, sTr.condC];
        ch = [sTr.subjM, sTr.subjC];
        
        %% Align at RDK onset & offset
        xypt = cell(1, 2);
        [xypt{1}, excl1] = EC.get_xypt(xypts, zeros(n_tr, 1), 1, 0.2, ...
            'val', S.y, 't_smooth', S.t_smooth);
        [xypt{2}, excl2] = EC.get_xypt(xypts, sTr.RT, -1, 0.2, ...
            'val', S.y, 't_smooth', S.t_smooth);
        excl = excl1 | excl2;
        n_align = numel(xypt);
        
        %% Flip according to choice
%         n_dim = 2;
%         for i_align = 1:n_align
%             for dim = 1:n_dim
%                 incl = ~isnan(ch(:,dim));
%                 xypt{i_align}(incl,:,dim) = xypt{i_align}(incl,:,dim) ...
%                     .* sign(ch(incl,dim) - 1.5);
%             end
%         end
        
        %% Plot
        n_row = 4;
        n_col = 4;
        clf;
        ax = subplotRCs(n_row, n_col);
        
        tasks = [
            'H', 'A'
            'V', 'A'
            ];        
        eye_dim_name = {'x', 'y'};
        cond_dim_name = {'M', 'C'};
        dir_align = {'normal', 'reverse'};
        align_label = {
            {'Time from', 'onset (s)'}, {'Time from', 'RT (s)'}};
        
        switch S.y
            case 'pos'
                y_lim = [-.2, .2];
            case 'vel'
                switch S.summary
                    case 'mean'
                        y_lim = [-1, 1];
                    case {'slope', 'logit'}
                        y_lim = [-0.1, 0.7];
                    case 'corr'
                        y_lim = [-.1, .1];
                end
            case 'acc'
                y_lim = [-5, 5];
        end
        
        n_dim = 2;
        for i_align = 1:n_align
            for n_dim_task = 1:n_dim
                for dim_eye = 1:n_dim
                    for dim_cond = 1:n_dim
                        tr_incl = (sTr.task == tasks(dim_eye, n_dim_task)) ...
                            & ~excl;
                        row1 = (dim_eye - 1) * n_dim + dim_cond;
                        col1 = (i_align - 1) * n_align + n_dim_task;
                        ax1 = ax(row1, col1);
                        axes(ax1);

                        switch S.summary
                            case 'mean'
                                cond1 = abs(cond(tr_incl, dim_cond));
                                ch1 = ch(tr_incl,dim_eye);
                            case {'slope', 'logit'}
                                if dim_cond == dim_eye
                                    cond1 = cond(tr_incl, dim_cond);
                                else
                                    cond1 = [ ...
                                        cond(tr_incl, dim_eye) .* ...
                                            abs(cond(tr_incl, dim_cond)), ...
                                        cond(tr_incl, dim_eye), ...
                                        abs(cond(tr_incl, dim_cond))];
                                end
                                ch1 = ch(tr_incl,dim_eye);
                            case 'corr'
                                dim_corr = n_dim + 1 - dim_eye;
                                cond1 = xypt{i_align}(tr_incl,:,dim_corr);
                                ch1 = ch(tr_incl, [dim_eye, dim_corr]);
                        end

                        xypt1 = xypt{i_align}(tr_incl,:,dim_eye);
                        EC.plot_eye_by_cond( ...
                            xypt1, ...
                            cond1, ch1, ...
                            'summary', S.summary);
                        bml.plot.beautify;

                        set(gca, 'XDir', dir_align{i_align});
                        ylim(y_lim);
                        
                        if row1 == 1
                            title(sprintf('%dD', n_dim_task));
                        end
                        if row1 < n_row
                            set(ax1, 'XTickLabel', '');
                        end
                        if row1 == n_row
                            xlabel(ax1, align_label{i_align});
                        end
                        if col1 == 1
                            switch S.summary
                                case 'mean'
                                    ylabel(ax1, ...
                                        sprintf('%s %s for ch\nby abs(%s)', ...
                                            S.y, ...
                                            eye_dim_name{dim_eye}, ...
                                            cond_dim_name{dim_cond}));
                                case 'slope'
                                    ylabel(ax1, ...
                                        sprintf('%s {\\beta}_{cond, %s}', ...
                                            S.y, ...
                                            cond_dim_name{dim_cond}));
                                case 'logit'
                                    ylabel(ax1, ...
                                        sprintf('%s {\\beta}_{ch, %s}', ...
                                            S.y, ...
                                            eye_dim_name{dim_eye}));
                                case 'corr'
                                    ylabel(ax1, ...
                                        sprintf('%s corr x and y', ...
                                            S.y));
                            end
                        else
                            set(ax1, 'YTickLabel', '');
                        end                        
                    end
                end
            end
        end
        
        %%
        bml.plot.position_subplots(ax, ...
            'margin_left', 0.15, ...
            'margin_right', 0.01, ...
            'margin_top', 0.05, ...
            'margin_bottom', 0.15, ...
            'btw_row', [0.02, 0.05, 0.02], ...
            'btw_col', [0.02, 0.05, 0.02]);
        
        %%
        pth_out = '../Data_2D/EyeByCond.eye_by_cond';
        S_file_out = varargin2S({
            'sbj', subj
            'coord', 'for_ch'
            'sep', 'by_acond'
            'y', S.y
            'tsm', S.t_smooth
            'summ', S.summary
            });
        file_out = fullfile(pth_out, ...
            bml.str.Serializer.convert(S_file_out));
        savefigs(file_out, 'size', [800, 600]);
    end
    function plot_eye_by_cond(~, xypt, cond, ch, varargin)
        S = varargin2S(varargin, {
            'summary', 'regr' % 'mean'|'regr'
            });
        
        %%
        switch S.summary
            case 'mean'
                xypt = bsxfun(@times, xypt, sign(ch - 1.5));
                
                [~,~,group] = unique(cond);
                n_cond = max(group);
                colors = bml.plot.hsv2(n_cond);

                for i_cond = 1:n_cond
                    incl = group == i_cond;

                    m = nanmean(xypt(incl,:));
                    e = nansem(xypt(incl,:));
                    t = (0:(size(m, 2) - 1)) ./ 1000;

                    errorbarShade(t, m, e, colors(i_cond, :));
                    hold on;
                end
            case 'slope'
                n_fr = size(xypt, 2);
                m = zeros(1, n_fr);
                e = zeros(1, n_fr);
                t = (0:(size(m, 2) - 1)) ./ 1000;
                
                xypt = bsxfun(@rdivide, ...
                    bsxfun(@minus, xypt, nanmean(xypt)), ...
                    nanstd(xypt));
                
                cond = (cond - nanmean(cond)) ./ nanstd(cond);
                
                for fr = 1:n_fr
                    [b, ~, res] = glmfit(cond, xypt(:,fr), 'normal');
                    m(fr) = b(2);
                    e(fr) = res.se(2);
                end
                errorbarShade(t, m, e, [0 0 0]);
                crossLine('h', 0, 'k:');
            case 'logit'
                n_fr = size(xypt, 2);
                m = zeros(1, n_fr);
                e = zeros(1, n_fr);
                t = (0:(size(m, 2) - 1)) ./ 1000;
                
                xypt = bsxfun(@rdivide, ...
                    bsxfun(@minus, xypt, nanmean(xypt)), ...
                    nanstd(xypt));
                
                for fr = 1:n_fr
                    [b, ~, res] = glmfit(xypt(:,fr), ch == 2, ...
                        'binomial');
                    m(fr) = b(2);
                    e(fr) = res.se(2);
                end
                errorbarShade(t, m, e, [0 0 0]);
                crossLine('h', 0, 'k:');          
            case 'corr'
                n_fr = size(xypt, 2);
                m = zeros(n_fr, 1);
                lb = zeros(n_fr, 1);
                ub = zeros(n_fr, 1);
                t = ((0:(n_fr - 1)) ./ 1000)';

                xypt = bsxfun(@minus, xypt, nanmean(xypt, 1));
                cond = bsxfun(@minus, cond, nanmean(cond, 1));
                
                xypt = bsxfun(@times, xypt, sign(ch(:,1) - 1.5));
                cond = bsxfun(@times, cond, sign(ch(:,2) - 1.5));
                
                for fr = 1:n_fr
                    [c, ~, lb1, ub1] = corrcoef( ...
                        xypt(:,fr), cond(:,fr));
                    m(fr) = c(1,2);
                    lb(fr) = lb1(1,2);
                    ub(fr) = ub1(1,2);
                end
                errorbarShade(t, m, bsxfun(@minus, [lb, ub], m), [0 0 0]);
                crossLine('h', 0, 'k:');          
        end
    end
    function xypt = truncate_w_RDK(~, xypt, rt)
        assert(iscell(xypt));
        assert(isvector(xypt));
        n_tr = length(xypt);
        for tr = 1:n_tr
            xypt1 = xypt{tr};
            incl = (xypt1(:,4) >= 0) & (xypt1(:,4) < rt(tr));
            xypt{tr} = xypt1(incl, :);
        end
    end
    function [xypt, excl] = get_xypt(~, ...
            xypt, t_align, max_dur, trunc_dur, ...
            varargin)
        S = varargin2S(varargin, {
            % position, velocity, or acceleration
            'val', 'pos' % 'pos'|'vel'|'acc'
            't_smooth', 100 % stdev of Gaussian kernel, in msec.
            });
        
        %%
        frame_rate = 1000;
        max_n_fr = max_dur * frame_rate;        
        trunc_fr = trunc_dur * frame_rate;
        
        n_tr = length(xypt);
        n_fr = max(cellfun(@(v) size(v, 1), xypt) - trunc_fr, 0);
        excl = false(n_tr, 1);
        
        %%
        for tr = 1:n_tr
            %%
            xypt1 = xypt{tr};
%             xypt1 = bsxfun(@minus, xypt1, xypt1(1,:));
            
            switch S.val
                case 'vel'
                    v1 = diff(smooth_gauss(xypt1(:,[1,2]), S.t_smooth), ...
                        1, 1) * 1e3;
                    xypt1 = [v1, ...
                        (xypt1(1:(end-1),[3,4]) ...
                            + (xypt1(2:end,[3,4]))) ./ 2];
                case 'acc'
                    v1 = diff(smooth_gauss(xypt1(:,[1,2]), S.t_smooth), ...
                        2, 1) * 1e6;
                    xypt1 = [v1, ...
                        xypt1(2:(end-1),[3,4])];
            end

            [~, fr_align] = min(abs(xypt1(:,4) - t_align(tr)));
            fr = fr_align + (0:sign(max_dur):max_n_fr);
            fr_incl = (fr >= 1) & (fr <= n_fr(tr));
            fr = fr(fr_incl);
            n_fr1 = numel(fr);
            
            xypt1 = [
                xypt1(fr, :)
                nan(max(0, abs(max_n_fr) + 1 - n_fr1), 4)
                ];
            if size(xypt1, 1) ~= abs(max_n_fr) + 1
                warning('xypt1 size = %d != max_n_fr + 1 = %d !', ...
                    size(xypt1, 1), abs(max_n_fr) + 1);
                keyboard;
            end
            
            fr_excl = (xypt1(:,1) < -10) | (xypt1(:,1) > 10) ...
                | (xypt1(:,2) < -10) | (xypt1(:,2) > 10);
            xypt1(fr_excl, :) = nan;            

%             if max_dur > 0
            switch S.val
                case 'pos'
                    xypt1 = bsxfun(@minus, xypt1, xypt1(1,:));
            end
%             else
%                 xypt1 = bsxfun(@minus, xypt1, xypt1(end,:));
%             end
            
            excl(tr) = any(fr_excl);
            xypt{tr} = xypt1;
        end
        
        %%
        xypt = permute(cat(3, xypt{:}), [3, 1, 2]);
    end
end
end