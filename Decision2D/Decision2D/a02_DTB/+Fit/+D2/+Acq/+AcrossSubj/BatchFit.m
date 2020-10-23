classdef BatchFit < Fit.Common.AcrossSubj.BatchFit
    % Fit.D2.Acq.AcrossSubj.BatchFit
    %
    % 2015 YK wrote the initial version.
%% Batch fit facades
methods
    function [Ss, Fls] = batch_facade(B, facade_kind, ...
            S_batch, S_batch_opt)
        % See get_S_units_* for available facade kinds.
        if ~exist('S_batch', 'var'), S_batch = struct; end
        if ~exist('S_batch_opt', 'var'), S_batch_opt = struct; end
        
        S_units = B.(['get_S_units_' facade_kind])(S_batch);
        [Ss, Fls] = B.batch_fit(S_units, S_batch_opt);
    end
    
    function S_units = get_S_units_test1(B, S_batch)
        if ~exist('S_batch', 'var'), S_batch = struct; end
        S_batch = varargin2S(S_batch, {
            'subj', Data.Consts.subjs_short
            'Bound', {'BetaCdf'}
            });
        S_units = [
            B.get_S_units_fix_buffer_dur(varargin2S(S_batch, {
                'buffer_dur_msec', 2500
                'p_dim1_1st_100x', 0
            }))
            ];
    end
    function S_units = get_S_units_test(B, S_batch)
        if ~exist('S_batch', 'var'), S_batch = struct; end
        S_batch = varargin2S(S_batch, {
            'subj', Data.Consts.subjs_short
            });
        S_units = [
            B.get_S_units_free_buffer_dur(varargin2S(S_batch, {
                'buffer_dur_msec', [300 1000]
                'p_dim1_1st_100x', [0 100]
            }))
            B.get_S_units_fix_buffer_dur(varargin2S(S_batch, {
                'buffer_dur_msec', [300 1000]
                'p_dim1_1st_100x', [0 100]
            }))
            ];
    end
    function S_units = get_S_units_all(B, S_batch)
        if ~exist('S_batch', 'var'), S_batch = struct; end
        S_units = [
            B.get_S_units_free_buffer_dur(S_batch)
            B.get_S_units_fix_buffer_dur(S_batch)
            ];
    end
    function S_units = get_S_units_fix_buffer_dur(B, S_batch)
        if ~exist('S_batch', 'var'), S_batch = struct; end
        S_units = B.get_S_units(varargin2S(S_batch, varargin2S({
            'to_fix_buffer_dur', true
            'buffer_dur_msec', [120 200 300 500 750 1000 1500 2000]
            'to_fix_p_dim1_1st', true
            'p_dim1_1st_100x', [0, 25, 50, 75, 100]
            })));
    end        
    function S_units = get_S_units_free_buffer_dur(B, S_batch)
        if ~exist('S_batch', 'var'), S_batch = struct; end
        S_units = B.get_S_units(varargin2S(S_batch, varargin2S({
            'to_fix_buffer_dur', false
            'buffer_dur_msec', [120 200 300 500 750 1000 1500 2000]
            'to_fix_p_dim1_1st', false
            'p_dim1_1st_100x', [0, 25, 50, 75, 100]
            })));
    end    
end
%% Batch fit
methods
    function S_batch = get_S_batch(B, S_batch)
        if ~exist('S_batch', 'var'), S_batch = struct; end
        S_batch_default = varargin2S({
            'parad', {'sh'}
            'to_fix_buffer_dur', false
            'buffer_dur_msec', 120
            'to_fix_p_dim1_1st', false
            'p_dim1_1st_100x', 50
            }, B.get_S_batch@Fit.Common.AcrossSubj.BatchFit);
        S_batch = varargin2S(S_batch, S_batch_default);
        S_batch = copyFields(S_batch, varargin2S({
%             'SigmaSq', {'Const', 'Linear'}
            }));
    end
    function S_unit = get_S_unit(B, S_unit)
        if ~exist('S_unit', 'var'), S_unit = struct; end
        S_unit_default = varargin2S({
            'parad', 'sh'
            }, B.get_S_unit@Fit.Common.AcrossSubj.BatchFit);
        S_unit = varargin2S(S_unit, S_unit_default);
        S_unit = varargin2S(S_unit, {
%             'SigmaSq', 'Const'
            });
    end
    function S_name = get_S_unit_file(B, S_name)
        S_name = B.get_S_unit_file@Fit.Common.AcrossSubj.BatchFit(S_name);
        S_name = bml.struct.orderfields(S_name, ...
            {'parad', 'task', 'subj', 'fig', 'dimOnX'}, ... % 'SigmaSq', 
            'first');
    end
    function [Fl, S_unit] = batch_fit_unit(Batch, S_unit)
        if ~exist('S_unit', 'var'), S_unit = struct; end
        S_unit = Batch.get_S_unit(S_unit);
        C_unit = varargin2C(S_unit);
        
        W = Batch.Main;
        Fl = W.fit_unit(C_unit{:});
    end
    function files_fig = savefigs(Batch, S_unit, S_batch_opt)
        S_unit = Batch.get_S_unit(S_unit);
        S_batch_opt = Batch.get_batch_fit_opt(S_batch_opt);
        
        file_fit = Batch.get_batch_fit_unit_file(S_unit, ...
            varargin2S({
                'prefix', 'fit/'
                'postfix', '.mat'
            }));
        Plt = [];
        
        files_fig = {};
        
        for fig_batch = {
                'plotfuns', {}
                'rt', {'dimOnX', 1}
                'rt', {'dimOnX', 2}
                'ch', {'dimOnX', 1}
                'ch', {'dimOnX', 2}
                }'
            fig = fig_batch{1};
            S_fig = S_unit; % bml.struct.orderfields(S_unit, 'SigmaSq', 'last');
            S_fig = varargin2S({
                'fig', fig
                }, S_fig);
            S_fig = varargin2S(fig_batch{2}, S_fig);
%             S_fig = bml.struct.orderfields(S_fig, {'fig', 'dimOnX'}, 'last');
            
            S_fig_opt = varargin2S({'prefix', 'fig/'});
            file_fig = Batch.get_batch_fit_unit_file(S_fig, S_fig_opt);
            files_fig{end+1} = file_fig; %#ok<AGROW>
            
            if exist([file_fig '.fig'], 'file') ...
                    && S_batch_opt.skip_existing_figure
                fprintf('--- Skipping existing figure %s\n', ...
                    [file_fig '.fig']);
            else
                if isempty(Plt)
                    Fl = FitFlow.load_and_recover(file_fit);
                    Plt = eval(bml.pkg.get_class_rel(Fl.W, 'Plot'));
                    Plt.set_Fl(Fl);
                end
                
                clf;
                Plt.(fig)(S_fig);

                mkdir2(fileparts(file_fig));
                
                % size is set in Plt for each figure.
                savefigs(file_fig, 'size', []);
            end
        end
        files_fig = files_fig(:)';
    end
end
%% Plot
methods
    function [suffix, S] = plot_cost_across_expr_w_res(~, res, varargin)
        S = varargin2S(varargin, {
            'y',  'fval'
            'xlabel', 'Log likelihood'
            'Tds', {'Ser', 'Par'}
            'Td_names', {'Serial', 'Parallel'}
            });
        
        kinds = S.Tds;
        kind_names = S.Td_names;
        n_kind = numel(kinds);
        assert(n_kind == 2); % Necessary to plot just the difference
        
        for i_kind = n_kind:-1:1
            kind = kinds{i_kind};
            incl = strcmp(kind, res.Td);
            v{i_kind} = res.(S.y)(incl);
        end
        delta = v{2} - v{1};
        subjs = cellfun(@(c) c(2), res.subj(incl), 'UniformOutput', false);
            
        barh(delta, 'k');
        set(gca, 'YTickLabel', subjs, 'YDir', 'reverse');
            
        x_label = sprintf('%1$s_{%2$s} - %1$s_{%3$s}', ...
            S.xlabel, kind_names{1}, kind_names{2});
        xlabel(x_label);
        ylabel('Subject');
        box off;
        set(gca, 'TickDir', 'out');

        title(str_bridge(' vs ', kind_names{:}));
        
        suffix = S.y;
    end    
    %% Path
    function S = get_experiments_command(Batch, varargin)
        C = varargin2C(varargin, {
            'Td', {'Ser', 'Par'}
            'desc', {Batch.get_default_desc}
            });
        S = Batch.get_experiments_command@Fit.Common.AcrossSubj.BatchFit( ...
                C{:});        
    end
    function S = get_path_fit_command(Batch, varargin)
        C = varargin2C(varargin, {
            'Td', 'Ser'
            'desc', Batch.get_default_desc
            });
         S = Batch.get_path_fit_command@Fit.Common.AcrossSubj.BatchFit( ...
                 C{:});
    end
    function pre_desc = get_path_fit_pre_desc(Batch, varargin)
        S = varargin2S(varargin, Batch.get_path_fit_command);
        pre_desc = str_con('Td', S.Td);
    end
    function desc = get_default_desc(~)
        desc = 'w_miss';
    end
end
end