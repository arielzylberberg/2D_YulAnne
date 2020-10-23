classdef DtbParticle ...
        < Fit.D2.Inh.Dtb ...
        & Fit.D2.Common.DataFilterEn
properties
    n_sim = 1e4;
    tr = []; % (tr, )
    en = []; % (tr, sim, fr, dim)
    drift = []; % (tr, sim, fr, dim)
    td_pdf = []; % (tr, sim, fr, dim, ch)
    y_unabs = []; % (tr, sim, fr, dim) % evidence of each unabsorbed particle
    p_unabs = []; % (tr, sim, fr, dim) % weight of each particle
end
methods
    function W = DtbParticle(varargin)
        W.truncate_first_msec = 0;
        W.truncate_last_msec = 0;
        
        if nargin > 0
            W.init(varargin{:});
        end
    end
    function init(W, varargin)
        W.init@Fit.D2.Common.DataFilterEn(varargin{:});
        
%         W.tr = repmat((1:W.get_n_tr)', [1, W.n_sim]);
%         
%         ens = W.get_ens_mat;
%         W.en = repmat(cat(4, ...
%             permute(ens{1}, [1 3 2]), ...
%             permute(ens{1}, [1 3 2])), [1, W.n_sim, 1]);
%         W.reset_calc;
    end
    function pred(W)
        W.reset_calc;
        W.simulate;
    end
    function reset_calc(W)
    end
    function simulate(W)
        
        
        for fr = 1:n_fr
        end
    end
end
%%
methods
    function demo_remove_fr(W)
        W = Fit.D2.Inh.DtbParticle( ...
            'subj', Data.Consts.subjs_RT{1}, ...
            'task', 'A', ...
            'dif_irr_incl', 'all', ... 1:3, ...
            'dif_rel_incl', 'all', ... 1:3, ...
            'rt_incl', [0 100]);
        
        %%
        fr_set{1} = [];
        fr_set{2} = 10 + (1:5);
        fr_set{3} = 40 + (1:15);
        fr_set{4} = union(fr_set{2}, fr_set{3});
        n_set = numel(fr_set);

        clear res_set
        for i_set = n_set:-1:1
            res_set{i_set} = W.demo('remove_fr', fr_set{i_set}, ...
                'rt_ch', 1:0.1:1.5, 'tr', 15);
        end
        
        %%
        res_set_array = [res_set{:}];
        p_ch = cat(3, res_set_array.p_ch);
        
        %%
        p_ch_prop = squeeze(p_ch(:,2,:) ./ sum(p_ch, 2));
        
        %%
        fig_tag('fr_removed');
        plot(res_set_array(1).S.rt_ch, p_ch_prop);
        ylim([0 1]);
        grid on;
        
        legend({'0', '1', '2', '3'})
    end
    function res = demo(W, varargin)
        S = varargin2S(varargin, {
            'remove_fr', []
            'rt_ch', 1:0.1:1.5
            'tr', 15
            });
        
        %% Load
        ens = W.get_ens_mat;

        %%
        en = ens{1};
        [~,~,cond] = unique(W.Data.ds.condM);

        en = en(S.tr,:);
        fr = S.remove_fr; % 50 + (1:8);
        n_fr = length(fr);
        en = [en(setdiff(1:end, fr)), zeros(1, n_fr)];
%         en = [en; [en(setdiff(1:end, fr)), zeros(1, n_fr)]];

        en(isnan(en)) = 0;

        %% Test
        % t_max = 3;
        refresh_rate = 75;
        % nk = t_max * refresh_rate;
        % n_cond = 9;
        % drift = bsxfun(@plus, linspace(1, -1, n_cond)', zeros(1, nk));
        drift = en * 5;
        nk = size(en, 2);
        t_max = (nk - 1) / refresh_rate;

        t = linspace(0, t_max, nk);
        sigma = ones(1, nk);
        Bup = ones(1, nk) * 1;
        Blo = -ones(1, nk) * 1;
        n_sim = 100000;

        tic;
        D = dtb.pred.particle_dtb(drift, t, Bup, Blo, [], [], [], sigma, n_sim);
        toc;

        %%
        fig_tag('p_abs');
        subplot(2,1,1);

        p_tnd = gampdf_ms(t(:), 0.2, 0.05, 1);

        n_cond = size(drift, 1);

        p_resp = conv_t(D.p_up', p_tnd);
        % p_resp = accumarray( ...
        %     [repmat((1:nk)', [n_cond, 1]), vVec(repmat(cond(:)', [nk, 1]))], ...
        %     p_resp(:), [], @nanmean);
        % p_resp_up = D.p_up';

        p_resp_all(:,1) = p_resp;

        subplot(3,1,1);
        plot(t', cumsum(p_resp));
        % plot(D.p_up');
        % plot(D.p_abs_up');
        grid on;
        ylim([0 1]);

        p_resp = conv_t(D.p_lo', p_tnd);
        % p_resp = accumarray( ...
        %     [repmat((1:nk)', [n_cond, 1]), vVec(repmat(cond(:)', [nk, 1]))], ...
        %     p_resp(:), [], @nanmean);
        % p_resp_lo = D.p_lo';

        p_resp_all(:,2) = p_resp;

        subplot(3,1,2);
        plot(t', cumsum(p_resp));
        % plot(D.p_lo');
        % plot(D.p_abs_lo');
        grid on;
        ylim([0 1]);

        % sameAxes(2,1);

        subplot(3,1,3);
        plot(t', cumsum(sum(p_resp_all,2)));
        grid on;
        ylim([0 1]);
        
        %%
        res.S = S;
        res.D = D;
        
        k_ch = round((S.rt_ch - W.truncate_first_msec / 1000) ...
            * W.refresh_rate) ...
            + 1;
        res.p_ch = p_resp_all(k_ch, :);

        %%
        % fig_tag('y');
        % plot(D.y_all(1:n_sim, :)');

        %%
        % fig_tag('wt');
        % plot(nansum(((D.wt_all < 1) & (D.wt_all > 0)) .* D.wt_all)' / (n_sim * n_cond));

        % fig_tag('unabs');
        % imagesc(D.unabs_all);
    end
end
end