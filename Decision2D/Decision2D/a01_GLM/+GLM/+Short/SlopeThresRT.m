classdef SlopeThresRT
properties
    t = 0:0.01:5;
end
methods
    function batch(s)
        %%
        for thres_kind = {'combined', 'max'}
            for thres_ms = [0, 57, 100]
                s.main_slope_aft_thres_rt_dif(...
                    'thres_ms', thres_ms, ...
                    'thres_kind', thres_kind{1});
            end
        end
    end
    function main_slope_aft_thres_rt_dif(s, varargin)
        S = varargin2S(varargin, {
            'thres_ms', 100
            'thres_kind', 'combined' % 'max' or 'combined'
            });
        
        %%
        thres_ms = S.thres_ms; % 57;
        
        pth = '../Data_2D/GLM.Short.SlopeThresRT';
        
        subjs = Data.Consts.subjs_RT;
        n_subj = numel(subjs);
        
        ress = cell(n_subj, 1);
        bs = cell(n_subj, 1);
        es = cell(n_subj, 1);
        ps = cell(n_subj, 1);
        
        for i_subj = 1:n_subj
            file = fullfile('../Data_2D/sTr', ...
                sprintf('sh_%s', subjs{i_subj}));
            L = load(file);
            cond0 = [L.dat.condM, L.dat.condC];
            rt0 = L.dat.RT;
            ch0 = [L.dat.subjM, L.dat.subjC] == 2;

            %%
            fig_tag('indiv');
            [bs{i_subj}, es{i_subj}, ps{i_subj}, ress{i_subj}] = ...
                s.slope_aft_thres_rt_dif(cond0, rt0, ch0, ...
                    'thres', thres_ms / 1e3, ...
                    'thres_kind', S.thres_kind);
            
            subj = subjs{i_subj};
            file = fullfile(pth, sprintf('sbj=%s+thres=%d+plt=slope_thres_rt', ...
                subj, thres_ms));
            savefigs(file, 'size', [300, 600]);
        end
        
        %% Combined slope plot
        fig_tag('combined');
        clf;
        for i_subj = 1:n_subj
            s.plot_slope(bs{i_subj}, es{i_subj});
            hold on;
        end
        b = cell2mat(bs);
        e = cell2mat(es);
        xlim([0, max(b(:,1) + e(:,1))]);
        ylim([0, max(b(:,2) + e(:,2))]);
        
        file = fullfile(pth, sprintf('sbj={%s,x%d}+thres=%d+tkind=%s+plt=slope_combined', ...
                subjs{1}, n_subj, ...
                thres_ms, S.thres_kind));
        savefigs(file, 'size', [200, 200]);
        
        %% Save data
        file = fullfile(pth, ...
            sprintf('tbl=slope_thres_rt_combined+thres=%d+tkind=%s', ...
            thres_ms, S.thres_kind));
        
        ds = dataset;
        ds.subj = subjs(:);
        for i_subj = 1:n_subj
            res = ress{i_subj};
            ds.b_rel_hard_irr_easy{i_subj,1} = [res.b0(2,1), res.b0(3,2)];
            ds.b_rel_hard_irr_hard{i_subj,1} = res.b0(4,:);
            ds.e_rel_hard_irr_easy{i_subj,1} = [res.e0(2,1), res.e0(3,2)];
            ds.e_rel_hard_irr_hard{i_subj,1} = res.e0(4,:);
            ds.p{i_subj,1} = ps{i_subj};
        end
        for f = {
                'b_rel_hard_irr_easy'
                'e_rel_hard_irr_easy'
                'b_rel_hard_irr_hard'
                'e_rel_hard_irr_hard'
                'p'
                }'
            ds.(f{1}) = cell2mat(ds.(f{1}));
        end
        disp(ds);
        export(ds, 'file', [file, '.csv'], 'Delimiter', ',');
        
        save(file, 'bs', 'es', 'ps', 'ress');
        fprintf('Saved to %s.csv and .mat\n', file);
    end
    function [b, e, p, res] = ...
            slope_aft_thres_rt_dif(s, cond0, rt0, ch0, varargin)
        S = varargin2S(varargin, {
            'thres', 0.1
            'thres_kind', 'combined' % 'max' or 'combined'
            });
        
        %%
        n_dim = 2;
        conds = cell(1, n_dim);
        dcond = cell(1, n_dim);
        aconds = cell(1, n_dim);
        adcond = cell(1, n_dim);
        for dim = 1:n_dim
            [conds{dim},~,dcond{dim}] = unique(cond0(:,dim));
            [aconds{dim},~,adcond{dim}] = unique(abs(cond0(:,dim)));
        end
        
        prct = 0:1:100;
        
        dif_easy = 5;
        dif_hard = 1:2;
        
        difs = {
            {dif_easy, dif_easy}
            {dif_hard, dif_easy}
            {dif_easy, dif_hard}
            {dif_hard, dif_hard}
            };
        n_dif = numel(difs);
        rt = cell(1, n_dif);
        ch = cell(1, n_dif);
        incl = cell(1, n_dif);
        thres_prct = zeros(n_dif, 1);
        incl_thres = cell(n_dif, 1);
        cond = cell(1, n_dif);
        n_dim = 2;
        stats = cell(n_dif, n_dim);
        b0 = zeros(n_dif, n_dim);
        e0 = zeros(n_dif, n_dim);
        rt_incl = cell(1, n_dif);
        drt_incl = cell(1, n_dif);
        
        for ii = 1:n_dif
            dif1 = difs{ii};
            incl{ii} = ismember(adcond{1}, dif1{1}) ...
                     & ismember(adcond{2}, dif1{2});
            rt{ii} = prctile(rt0(incl{ii}), prct);
            if ii > 1
                thres_prct(ii) = find((rt{ii} - rt{1}) > S.thres, ...
                    1, 'first');
            end
        end
        
        switch S.thres_kind
            case 'max'
                thres_prct_common = max(thres_prct);
            case 'combined'
                thres_prct_common = thres_prct(end);
        end
        for ii = 2:n_dif
            incl_thres{ii} = incl{ii} & (rt0 > rt{ii}(thres_prct_common));
            ch{ii} = ch0(incl_thres{ii}, :);
            cond{ii} = cond0(incl_thres{ii}, :);
%             rt_incl{ii} = rt0(incl_thres{ii}, :);
%             drt_incl{ii} = rt_incl{ii} - rt_incl{1};

            for dim = 1:n_dim
                if isequal(difs{ii}{dim}, dif_easy)
                    continue;
                end
                
                o_dim = n_dim + 1 - dim;
                X = [cond{ii}(:,dim), ...
                    cond{ii}(:,o_dim), ...
                    abs(cond{ii}(:,o_dim))];
                [b1, ~, stats{ii,dim}] = glmfit( ...
                    X, ch{ii}(:,dim), ...
                    'binomial', 'link', 'probit');
                b0(ii,dim) = b1(2);
                e0(ii,dim) = stats{ii,dim}.se(2);
            end
        end
        
        clf;
        n_row = 4;
        subplot(n_row,1,1);
        for ii = 1:3
            ecdf(rt{ii});
            hold on;
        end
        hold off;
        bml.plot.beautify;
        xlabel('RT (s)');
        ylabel(sprintf('Cumulative\nProportion'));
        
        subplot(n_row,1,2);
        colors = {
            'k'
            bml.plot.color_lines('g')
            bml.plot.color_lines('b')
            bml.plot.color_lines('r')
            };
        drt = cell(1, n_dif);
        for ii = 2:n_dif
            drt{ii} = rt{ii} - rt{1};
            plot(prct, drt{ii}, '-', 'Color', colors{ii});
            hold on;
        end
        hold off;
        y_lim = ylim;
        ylim([0, y_lim(2)]);
        crossLine('h', S.thres, {'--', 0.5 + zeros(1,3)});
        crossLine('v', thres_prct_common, {':', 0 + zeros(1,3)});
        bml.plot.beautify;
        xlim([0, 100]);
        ylabel(sprintf('Minimum\nDecision Time'));
        xlabel('Percentile');
        ylim([0, max(bml.array.cell2vec(drt)) * 1.1]);
        
        %%
        xlabels = {'Motion Coherence (%)', 'Color Coherence (logit)'};
        x_plot_fac = [100, 1];
        for dim = 1:n_dim
            subplot(n_row,2, 4 + dim);
            
            for ii = 2:n_dif
                cond1 = cond{ii}(:,dim);
                if isequal(difs{ii}{dim}, dif_easy)
                    continue;
                end
                
                %%
                o_dim = n_dim + 1 - dim;
                if isequal(difs{ii}{o_dim}, dif_easy)
                    color = bml.plot.color_lines('b');
                    x_shift = -1;
                else
                    color = bml.plot.color_lines('r');
                    x_shift = +1;
                end
                
                %%
                ch1 = ch{ii}(:,dim);
                [conds1, ~, dcond1] = unique(cond1);
                p1 = accumarray(dcond1, ch1, [], @mean);
                e1 = accumarray(dcond1, ch1, [], @sem);

                x11 = linspace(conds1(1), conds1(end))';
                
                o_cond1 = cond{ii}(:,o_dim);
                reg = [mean(o_cond1), mean(abs(o_cond1))];

                x1 = [x11, bsxfun(@plus, reg, zeros(size(x11, 1), 2))];
                y1 = glmval(stats{ii,dim}.beta, x1, 'probit', stats{ii,dim});
                plot(x1(:,1) * x_plot_fac(dim), y1, 'Color', color);
                hold on;

                x_plot = (conds1 + x_shift * 0.02 * diff(conds1([end, 1]))) ...
                    * x_plot_fac(dim);
                errorbar_wo_tick(x_plot, p1, e1, [], {'Color', color});
                hold on;
                
                ylim([0, 1]);
                bml.plot.beautify;
                xlabel(xlabels{dim});
%                 bml.stat.glmplot(cond1, ch1, 'binomial');
            end
            hold off;
        end
        
        %% Slope plot
        subplot(n_row, 1, n_row);
        cla;
        
%         patch( ...
%             'XData', b(2,1) + e(2,1) * [-1, -1, +1, +1], ...
%             'YData', [x_lim, flip(x_lim)], ...
%             'EdgeColor', 'none', ...
%             'FaceColor', color_easy, ...
%             'FaceAlpha', 0.2);

        scale = [b0(2,1), b0(3,2)];
        b = bsxfun(@rdivide, b0, scale);
        e = bsxfun(@rdivide, e0, scale);
        
        s.plot_slope(b, e);
        
        %%
        m = sum(b(end,:));
        v = sum(e(end,:) .^ 2);
        p1 = normcdf(1, m, sqrt(v));
        disp('P(sum(b) < 1):');
        disp(p1);
        
        p2 = normcdf(2, m, sqrt(v));
        disp('P(sum(b) < 2):');
        disp(p2);

        p22 = min(p2, 1-p2) * 2;
        disp('P(sum(b) != 2):');
        disp(p22);
        
        p = [p1, p2, p22];

%         disp(thres_prct_max);
%         slope_ratio = bsxfun(@rdivide, b, b(end,:));
%         disp(slope_ratio);
%         time_slice1 = 1./[slope_ratio(3,1), slope_ratio(2,2)];
%         disp(time_slice1);
%         disp(sum(time_slice1));
        
        res = packStruct(b0, e0, b, e, p, S, incl, rt, stats, p1, p2, p22);
    end
    function plot_slope(s, b, e)
        color_easy = bml.plot.color_lines('b');
        color_hard = bml.plot.color_lines('r');
        
        x_lim = [0, max(b(:,1) + e(:,1)) * 1.1];
        y_lim = [0, max(b(:,2) + e(:,2)) * 1.1];
        
        crossLine('v', b(2,1), color_easy);
        hold on;
        
        crossLine('h', b(3,2), color_easy);
        hold on;
        
        pred_sum = 1; % S.thres / drt{4}(thres_prct_max);
        
        plot([0, pred_sum], [pred_sum, 0], ':', ...
            'LineWidth', 2, 'Color', color_hard);
        
        plot(b(4,1), b(4,2), 'o', ...
            'MarkerFaceColor', color_hard, ...
            'MarkerSize', 6, ...
            'MarkerEdgeColor', 'w');
        plot(b(4,1) + [0,0], b(4,2) + e(4,2) * [-1, 1], '-', ...
            'Color', color_hard)
        plot(b(4,1) + e(4,1) * [-1, 1], b(4,2) + [0,0], '-', ...
            'Color', color_hard)
        bml.plot.beautify;
        axis equal;
        xlim(x_lim);
        ylim(y_lim);
        xlabel(sprintf('Relative\nMotion Slope'));
        ylabel(sprintf('Relative\nColor Slope'));        
    end
end
%% Old demos
methods
    function demo(s)
        %%
        t = s.t;
        td = {
            gampdf_ms(t, 0.1, 0.05)
            gampdf_ms(t, 0.2, 0.1)
            };
        n = numel(td);
        tnd = gampdf_ms(t, 0.05, 0.02);
        
        n_row = 2;
        n_col = 2;
        
        p_rt = cell(1, n);
        thres = cell(1, n);
        rt = cell(1, n);
        
        ax = subplotRCs(n_row, n_col);
        
        for ii = 1:n
            [p_rt{ii}, thres{ii}, ~, rt{ii}] = s.get_p_rt(td{ii}, tnd, ax(:,ii));
        end
        
        fig_tag('qq');
        plot(rt{2} - rt{1}, 'o');
    end
    function [p_rt, thres, p_td_wi_thres, rt] = get_p_rt(s, p_td, p_tnd, ax)
        t = s.t;

        p_td_rt = GLM.Short.conv_t_jt(p_td, p_tnd);
        p_td_rt = p_td_rt / sum(p_td_rt(:));
        p_rt = sum(p_td_rt);
        ix = [
            find(cumsum(p_rt) >= 0.25, 1, 'first')
            find(cumsum(p_rt) <= 0.75, 1, 'last')
            ]';
        disp(ix);
        thres = t(ix);

        p_td_wi_thres = sum(p_td_rt(:, ix(1):ix(2)), 2);
        
%         axes(ax(1));
%         imagesc(t, t, p_td_rt);
%         xlabel('RT');
%         ylabel('Td');
%         axis square
%         h = crossLine('v', thres, 'w');
%         uistack(h, 'top');
%        
%         axes(ax(2));        
%         plot(t, p_td_wi_thres);
        
        rt = randsample(t, 100, true, p_rt);
        rt = sort(rt);
    end
    function demo_samp(s)
        %%
        n_dstr = 2;
        n_samp = 100;
        prm = {
            {0.1, 0.05}
            {0.3, 0.1}
            };
        rt = cell(1, n_dstr);
        td = cell(1, n_dstr);
        tnd = cell(1, n_dstr);
        ix = cell(1, n_dstr);
        
        for ii = 1:n_dstr
            td{ii} = gamrnd_ms(prm{ii}{:}, [1, n_samp]);
            tnd{ii} = gamrnd_ms(0.1, 0.05, [1, n_samp]);
            rt{ii} = td{ii} + tnd{ii};
            [rt{ii}, ix{ii}] = sort(rt{ii});
            td{ii} = td{ii}(ix{ii});
        end
        
        subplot(2,1,1);
        plot(rt{1}, td{1}, 'o');
        
        subplot(2,1,2);
        plot(td{2} - td{1}, 'ro');
        hold on;
        plot(rt{2} - rt{1}, 'bo');
        hold off;
    end
end
end
