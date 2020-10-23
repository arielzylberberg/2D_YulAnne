
subjs = {'ID2', 'ID3'};
n_bin_to_pool = 1;
dif_rels = {1:2};
dif_irrs = {1:2, 3};
aligns = [-1, 1];
n_subj = 2;
n_dim = 2;

n_dif_irrs = numel(dif_irrs);
n_align = numel(aligns);

%%
C0 = varargin2C({
    'subj', subjs
    'n_dim_task', num2cell(1:n_dim)
    'dim_rel', num2cell(1:n_dim)
    'dif_rel', dif_rels
    'dif_irr', dif_irrs
    'align', num2cell(aligns)
    'n_bin_to_pool', n_bin_to_pool
    'en_fds', {{'en', 'en'}} 
%     'en_fds', {{'nnME', 'mCE'}} 
    % 'en_fds', {{'mME', 'mCE'}}
    't_RDK_dur', {0.6:0.12:1.2}
    });
pth = '../Data/an03_En/RevCorSimple';
[file, S0] = get_lev_file(C0, 'pth', pth);
L1 = load([file, '.mat']);
fprintf('Loaded %s.mat\n', file);
tbl = L1.tbl;

%% Plotting
Plot = MainPlotRevCorSimple;

for dim_rel = 1:2
    for dif_rel = dif_rels(:)'
        for subj = S0.subj(:)'

            colors = hsv2rev(numel(dif_irrs));

            fig_tag(sprintf( ...
                '%s - %s', subj{1}, bml.str.Serializer.convert(dif_rel{1})));
            clf;
            ax = subplotRCs(2,2);

            for n_dim_task = 1:2
                row = n_dim_task;
                
                for i_align = 1:n_align
                    col = i_align;
                    
                    align = aligns(i_align);

                    for i_dif_irr = 1:numel(dif_irrs)
                        dif_irr = dif_irrs{i_dif_irr};

                        S1 = varargin2S({
                            'align', align
                            'subj', subj{1}
                            'n_dim_task', n_dim_task
                            'dim_rel', dim_rel
                            'dif_rel', dif_rel{1}
                            'dif_irr', dif_irr
                            }, S0);

                        axes(ax(n_dim_task, i_align));
                        
                        if align == -1
                            time_shift = 0;
                        else
                            time_shift = 18/75;
                        end
                        
                        Plot.plot_align(tbl, S1, ...
                            'color', colors(i_dif_irr, :), ...
                            'time_shift', time_shift);
                        hold on;

                        xlim(sort([0, 0.6] * -align + time_shift));
                    end
            %         ylim([0, 0.06]);
            %         xlim([0, 1]);

                    if col == 1
                        if dim_rel == 1
                            ylabel(sprintf('motion %dD', n_dim_task));
                        else
                            ylabel(sprintf('color %dD', n_dim_task));
                        end
                    elseif col == 2
                        set(gca, 'YTickLabel', '');
                        ylabel('');
                    end
                    if row == 1
                        set(gca, 'XTickLabel', '');
                        xlabel('');
                    else
                        if align == 1
                            xlabel('time from offset (s)');
                        else
                            xlabel('time from onset (s)');
                        end
                    end
                end
                sameAxes(ax(row, :), [], [], 'y');
            end

            C1 = varargin2C({
                'subj', subj{1}
                'dim_rel', dim_rel
                'dif_rel', dif_rel{1}
                }, S0);
            file = get_lev_file(C1, 'pth', pth);
            savefigs(file);
        end
    end
end


% Plot = MainPlotRevCorSimple;
% for dif_incl1 = S0.dif_rel
%     dif_incl = dif_incl1{1};
% 
%     for subj = cell2mat(S0.subj)
%         fig_tag(sprintf('S%d', subj));
%         clf;
%         
%         S1 = varargin2S({
%             'dif_rel', {dif_incl}
%             'subj', {subj}
%             }, S0);
%         S1 = Plot.plot_by_dim_align(tbl, S1);
% 
%         nam = bml.str.Serializer.convert(varargin2S({
%             'plt', 'main_revcor_simple'
%             'sbj', subj
%             'y', tbl1.lev
%             'difch', dif_incl
%             'difcurve', S0.dif_irr
%             'pool', S0.n_bin_to_pool
%             'en_fds', S0.en_fds
%             }));
%         file = fullfile(pth, nam);
%         savefigs(file, 'size', [800, 400]);
%     end
% end

%%
% for ii = 1:4
%     subplotRC(2,4,2,ii);
%     ylim([0, 0.05]);
% end