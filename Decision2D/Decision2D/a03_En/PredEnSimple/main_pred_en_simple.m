init_path;
policy1 = {'TargNorm'};
pth_out = '../Data_2D/a03_En/main_pred_en_simple';

%% == XCorr
for policy1 = {
%         'TargNorm'
        'Serial', 'Parallel', 'ParAcqSerResp', 'SerMultSwitch', 'TargNorm'
        }
    
    %%
    policy = policy1{1};
    
    %% Simualte
    S = varargin2S({
        'policy', policy
        'conds0', repmat({
            {-0.002, -0.001, -0.0005, -0.00025, 0, 0.00025, 0.0005, 0.001, 0.002}
%             {-0.001, -0.0005, -0.00025, -0.000125, 0, 0.000125, 0.00025, 0.0005, 0.001}
            }, [1, 2])
        'n_tr_per_cond', 100
        });
    C = S2C(S);
    [sTr, S, Pred] = pred_en_simple(C{:});
    subj = S.subj;
    
    %% Ch-RT
    fig_tag(['ch_rt_', subj]);
    clf;
    
    sTr1 = sTr;
    sTr1.RT = sTr1.RT / 75;
    plot_ch_RT(sTr1);
    
%     ch = sTr.ch;
%     rt = sTr.RT;
%     cond = sTr.cond;
%     
%     n_row = 2;
%     n_col = 2;
%     n_dim = 2;
%     Pl = Plot2DSimple;
%     dim_names = {'Motion', 'Color'};
%     for dim_rel = 1:n_dim
%         dim_irr = setdiff(1:n_dim, dim_rel);
% 
%         subplotRC(n_row, n_col, 1, dim_rel);
%         Pl.plot_ch_by_irr_dif(ch(:,dim_rel), cond(:,dim_rel), ...
%             cond(:,dim_irr));
%         
%         title(dim_names{dim_rel});
%         if dim_rel == 1
%             ylabel('P(ch=2)');
%         end
%         bml.plot.beautify;
%     
%         subplotRC(n_row, n_col, 2, dim_rel);
%         Pl.plot_rt_by_irr_dif(rt / Pred.refresh_rate, cond(:,dim_rel), ...
%             cond(:,dim_irr));
%         
%         if dim_rel == 1
%             ylabel('RT (s)');
%         end
%         bml.plot.beautify;
%     end    
    file = fullfile(pth_out, bml.str.Serializer.convert(S));
    savefigs(file, 'size', [300, 200], 'resolution', 150);
    
    %% XCorr
%     clf;    
%     XC = XCorr;
%     XC.bootstrap(sTr, 'align', 'st');
% 
%     pth_out = '../Data_2D/a03_En/main_xcorr';
%     S_file = varargin2S({
%         'plt', 'xcorr'
%         'subj', S.subj
%         'policy', S.policy
%         });
%     file = bml.str.Serializer.convert(S_file);
%     savefigs(fullfile(pth_out, file));
    
    %% CumEnLogit

    %% === RevCorSimple
    Main = MainRevCorSimple;
    C0 = varargin2C({
        'subj', {subj}
        'n_dim_task', {2}
        'dim_rel', {1,2}
        'dif_rel', {1:2}
        'dif_irr', {1:2, 3:5}
        'align', {-1, 1}
        'n_bin_to_pool', {6}
        'en_fds', {{'simME', 'simCE'}}
        'lev', {'mean'}
        'cum', {'sum'} % 'mean'|'sum'|'none'
        });
    [Ss, S0, file] = Main.get_Ss(C0{:});
    tbl = Main.main(sTr, Ss);

    % save(file, 'S0', 'tbl');
    % fprintf('Saved to %s.mat\n', file);

    %% --- Plot RevCorSimple
    fig_tag(['en_', subj]);
    clf;

    Plot = MainPlotRevCorSimple;
    ax = subplotRCs(2,2);

    aligns = [-1, 1];
    n_align = numel(aligns);
    n_dim_task = 2;
    colors = hsv2rev(2);

    y_lim = [inf, -inf];
    
    for dim_rel = 1:2
        for i_align = 1:n_align
            align = aligns(i_align);

            for i_dif_irr = 1:numel(S0.dif_irr)
                dif_irr = S0.dif_irr{i_dif_irr};
                S1 = varargin2S({
                    'align', align
                    'subj', S0.subj{1}
                    'n_dim_task', n_dim_task
                    'dim_rel', dim_rel
                    'dif_rel', S0.dif_rel{1}
                    'dif_irr', dif_irr
                    }, S0);

                axes(ax(dim_rel, i_align));
                Plot.plot_align(tbl, S1, 'color', colors(i_dif_irr, :));
                hold on;
            end
            
            y_lim1 = bml.plot.beautify_lim;
            y_lim(1) = min([y_lim(1), y_lim1(1)]);
            y_lim(2) = max([y_lim(2), y_lim1(2)]);
            
%             ylim([0, 0.06]);
%             xlim([0, 1]);
            if dim_rel == 1
                set(gca, 'XTickLabel', '');
                xlabel('');
            end
            if i_align == 2
                set(gca, 'YTickLabel', '');
                ylabel('');
            end
        end
    end
    for dim_rel = 1:2
        for i_align = 1:n_align
            axes(ax(dim_rel, i_align));
            ylim(y_lim);
        end
    end
    

    bml.plot.position_subplots(ax);
    gltitle(ax, 'all', subj, 'shift', [0, 0], ...
        'title_args', {
            'FontSize', 12
            'Position', [0.5, 1.02, 0.5]
            });
    savefigs(file, 'size', [300, 200], 'resolution', 150);

end

%%
% fig_tag('revcor');
% Lev = Lev1D;
% clf;
% 
% aligns = [-1, 1];
% n_align = length(aligns);
% 
% for i_align = 1:n_align
%     align = aligns(i_align);
%     for dim = 1:n_dim
%         [ens0{dim}, wt, t, S] = Lev.pool_time(ens0{dim}, ...
%             'truncate_st', 0, ...
%             'truncate_en', 0, ...
%             'n_bin_to_pool', 6, ...
%             'align', align);
% 
%         %%
%         [slope, se_slope, bias, se_bias, res, S] = Lev.slope_by_time( ...
%             ens0{dim}, ch(:,dim) == 2, wt);
% 
%         %%
%         subplotRC(n_dim, n_align, dim, i_align);
%         errorbarShade(t, slope, se_slope);
%         if align == 1
%             set(gca, 'XDir', 'reverse');
%         end
%         xlim([0, 5]);
%         crossLine('h', 0, 'k--');
%     end
% end

