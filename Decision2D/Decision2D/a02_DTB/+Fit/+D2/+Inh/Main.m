classdef Main < Fit.D2.Bounded.Main
    % Fit.D2.Inh.Main
    %
    % 2015 YK wrote the initial version.
    
%% Demo - config
methods
    function Ss = demo_batch_fit(Main, varargin)
        %%
        Ss = Main.batch_fit_command(varargin{:});
        Ss = Ss(1);
        disp(Ss{1});
        
        %%
        Main.batch_fit_run(Ss);
        
        %%
%         jobs = Main.batch_fit_parallel_run(Ss(1));
    end
end
methods
    function [jobs, Ss] = batch_fit_parallel(Main, varargin)
        Ss = Main.batch_fit_command(varargin{:});
        [jobs, Ss] = Main.batch_fit_parallel_run(Ss);
        fprintf('Use Ss = Main.batch_fit_parallel_retrieve!');
    end
    function [Ss, W, Fl] = batch_fit(Main, varargin)
        Cs = Main.batch_fit_command(varargin{:});
        [Ss, W, Fl] = Main.batch_fit_run(Cs);
    end
    function [Ss, W, Fl] = batch_fit_run(Main, Ss)
        n = numel(Ss);
        for ii = 1:n
            fprintf('\n');
            fprintf('===== Beginning batch %d/%d\n', ii, n);
            C = S2C(Ss{ii});
            [W, Fl] = Main.demo_get_W_inh(C{:});
        end
    end
    function [jobs, Ss] = batch_fit_parallel_run(Main, Ss)
        n = numel(Ss);
        for ii = n:-1:1
            fprintf('\n');
            fprintf('===== Beginning batch %d/%d\n', ii, n);
            Ss{ii}.t_begin = now;
            C = S2C(Ss{ii});
            f = @(varargin) void(@() demo_get_W_inh(varargin{:}), Ss{ii});
            jobs{ii} = parfeval(f, ...
                1, Main, C{:});
        end
        jobs = [jobs{:}];
    end
    function [Ss, ix_to_read] = batch_fit_parallel_retrieve(Main, jobs, varargin)
        opt = varargin2S(varargin, {
            'retrieve_read', false
            });
        
        finished = strcmp('finished', {jobs.State});
        
        if opt.retrieve_read
            to_read = finished(:);
        else
            read = [jobs.Read];
            to_read = finished(:) & ~read(:);
        end
        ix_to_read = find(to_read(:));
        n_to_read = numel(ix_to_read);
        
        if n_to_read > 0
            for ii = n_to_read:-1:1
                ix = ix_to_read(ii);
                
                S = jobs(ix).fetchOutputs;
                S.Diary = jobs(ix).Diary;
                S.Error = jobs(ix).Error;
                
                S.is_error = Main.batch_fit_parallel_save_error(S);
                
                Main.batch_fit_parallel_save_diary(S);
                Main.batch_runPlotFcns(S);
                
                is_error(ii) = S.is_error;
                Ss{ii} = S;
            end
            
            fprintf('Retrieved (%d/%d):', numel(ix_to_read), numel(jobs));
            fprintf(' %d', ix_to_read);
            fprintf('\n');
            
            fprintf('Errored (%d/%d):', nnz(is_error), numel(jobs));
            fprintf(' %d', find(is_error));
            fprintf('\n');
        else
            fprintf('Nothing to retrieve!\n');
        end        
    end
    function batch_fit_parallel_save_diary(Main, S, kind)
        if ~exist('kind', 'var'), kind = 'fit'; end
        file = Main.get_file_full(S, kind, '.txt');
        
        if ~isfield(S, 'Diary') || ~ischar(S.Diary)
            fprintf('\n');
            fprintf('Diary is absent or not char: skipping %s..\n', ...
                file);
            return;
        end
        
        fid = fopen(file, 'w');
        fprintf(fid, '%s', S.Diary);
        fclose(fid);
        fprintf('\n');
        fprintf('Saved S.Diary to %s\n', file);
    end
    function is_error = batch_fit_parallel_save_error(Main, S, kind)
        if ~exist('kind', 'var'), kind = 'fit'; end
        file_err = Main.get_file_full(S, kind, '_error.mat');
        file_err_msg = Main.get_file_full(S, kind, '_error.txt');
        
        is_error = 0;
        
        if ~isfield(S, 'Error')
            fprintf('\n');
            fprintf('Error is not set: skipping %s..\n', ...
                file_err);
            is_error = nan;
            return;
        end
        if isempty(S.Error)
            % No error. Silently return.
            return;
        end
        
        is_error = 1;
        
        L = struct;
        L.Error = S.Error; %#ok<STRNU>
        save(file_err, '-struct', 'L');
        
        fid = fopen(file_err_msg, 'w');
        fprintf(fid, '%s', err_msg(S.Error));
        fclose(fid);
        fprintf('\n');
        fprintf('Saved Ss{%d}.Error to %s\n', ii, file_err_msg);
    end
    function batch_runPlotFcns(Main, S)
        ext_default = '.fig';
        file = Main.get_file_full(S, 'fit');
        file_fig = [file, ext_default];

        if S.to_avoid_replot && exist(file_fig, 'file')
            fprintf('\n');
            fprintf('----- Figure files already exist! Skipping %s\n', ...
                file_fig);
        else
            load([file '.mat'], 'Fl');
            fig_tag('runPlotFcns');
            Fl.runPlotFcns;
            savefigs(file, 'size', [1200, 1200]);
        end
    end
end
methods
    function W = Main(varargin)
        if nargin > 0
            W.init(varargin{:});
        end
    end
    function f = get_file_name_fit(~, S)
        % f = get_file_name_fit(~, S)
        
        f_fix = @(tf) iif(tf, 'fixed', true, 'free');
        f = str_con(S.batch, S.parad, S.subj, ...
            f_fix(S.fix_p_dim1_1st), 'p_dim1_1st', round(S.p_dim1_1st * 100), ...
            f_fix(S.fix_drift_fac), 'drift_fac', round(S.drift_fac * 100), ...
            f_fix(S.fix_sigmaSq_fac), 'sigmaSq_fac', round(S.sigmaSq_fac * 100), ...
            'bound', S.bound, ...
            'tnd', S.tnd);
        
        
    end
    function S = get_S_from_file_name(Main, file)
        [~, file] = fileparts(file);
        S = bml.str.Serializer.convert(file);
        S = bml.str.Serializer.field_recover(S, Main.file_fields);
        
    end
    function f = get_file_full(Main, S, kind, ext)
        % f = get_file_full(Main, S, kind='', ext='')
        %
        % kind : also becomes subdir.
        % ext : '.mat', etc.
        if ~exist('kind', 'var')
            kind = '';
        end
        if ~exist('ext', 'var')
            ext = '';
        end
        method_name = str_con('get_file_name', strrep(kind, '/', '_'));
        nam = Main.(method_name)(S);
        f = fullfile('Data', class(Main), kind, ... % pkg2dir(class2pkg(class(Main)))
            [nam, ext]);
    end
    function Ss = batch_fit_command(~, varargin)
        S = varargin2S(varargin, {
            'batch', 'GridDriftSigmaSq'
%                 'Indep'
%                 'ConstFano'
%                 'ConstSigmaSq'
%                 'ConstDrift'
%                 'GridDriftSigmaSq'
            ...
            'package', 'Inh'
            'dtb', 'Density' % 'DensityConstFanoSharedDriftFac'
            ...
            'subj', Data.Consts.subjs_RT
            'parad', {'RT'}
            ...
            ... % Factors - Common
            'kb_ratio', false % true
            ...
            'drift', {'Const'} % 'Const', 'Indiv'
            'bound', {'BetaCdf', 'Const'} % 'Const', 'BetaCdf'
            'sigmaSq', {'Const', 'Linear'}
            'tnd', {'halfnorm'} % 'halfnorm', 'gamma'
            ...
            ... % Factors - Indep
            ... % 'td', {'Ser', 'Par'}
            ...
            ... % Factors - Inh
            'p_dim1_1st', 0:0.25:1
            'drift_fac', 0:0.25:1
            'sigmaSq_fac', 0:0.25:2 % Try higher sigma as well
            ...
            'fix_p_dim1_1st', true
            'fix_drift_fac', true
            'fix_sigmaSq_fac', true
            ...
            'to_avoid_refit', true
            'to_avoid_replot', false
            ...
            'to_fit', true
            'to_clf', false
            'to_plot', true
            'to_save', true
            });

        %% Filter factors and determine file_name.
        if strcmp(S.batch, 'Indep')
            error('Under construction!');
            S.package = '';
%             Ss = varargin2S_all({ % make if needed
%                 {
%                 'file_name', 'IndepSer'
%                 'td', 'Ser'
%                 }
%                 {
%                 'file_name', 'IndepPar'
%                 'td', 'Par'
%                 }
%                 }, {
%                 'package', ''
%                 'dtb', 'IndepDim'
%                 });
%             Ss_all = [Ss_all; Ss(:)];
        else
            S.package = 'Inh';
            Ss = factorizeS(S, {
                'p_dim1_1st', 'drift_fac', 'sigmaSq_fac', ...
                'subj', 'parad', 'bound'});
            
            switch S.batch
                case 'GridDriftSigmaSq'
                    S.dtb = 'Density';
                    % No need to filter
                    
                case 'ConstFano'
%                     for ii = 1:n
%                         v = S.drift_fac(ii);
%                         Ss{ii} = varargin2C({
%                             'file_name', sprintf('InhConstFano_par%03.0f', v*100)
%                             'package', 'Inh'
%                             'dtb', 'DensityConstFanoSharedDriftFac'
%                             'drift_fac', max(v, 0.2)
%                             });
%                     end
                case 'ConstSigmaSq'
%                     if S.fix_drift_fac
%                         S = varargin2S(S, {
%                             'drift_fac', fix_vec % _union
%                             });
%                     else
%                         S = varargin2S(S, {
%                             'drift_fac', free_vec % _union
%                             });
%                     end
%                     if S.fix_p_dim1_1st
%                         S = varargin2S(S, {
%                             'p_dim1_1st', fix_vec % _union
%                             });
%                     else
%                         S = varargin2S(S, {
%                             'p_dim1_1st', free_vec % _union
%                             });
%                     end
%                     factors = factorizeS(S, {'drift_fac', 'p_dim1_1st'});
% 
%                     n = numel(factors);
%                     Ss = cell(n, 1);
% 
%                     for ii = 1:n       
%                         cf = factors(ii);
% 
%                         Ss{ii} = varargin2C({
%                             'file_name', ...
%                             sprintf('InhConstSigmaSq_par%03.0f_p_dim1_1st_%03.0f', ...
%                                 cf.drift_fac * 100, cf.p_dim1_1st * 100)
%                             'package', 'Inh'
%                             'dtb', 'DensityConstSigmaSqSharedDriftFac'
%                             'drift_fac', cf.drift_fac
%                             'p_dim1_1st', cf.p_dim1_1st
%                             });
%                     end
%                     Ss = [Ss; Ss(:)];
                case 'ConstDrift'
%                     if S.fix_sigmaSq_fac
%                         S = varargin2S(S, {
%                             'sigmaSq_fac', fix_vec_sigmaSq
%                             });
%                     else
%                         S = varargin2S(S, {
%                             'sigmaSq_fac', free_vec_sigmaSq
%                             });
%                     end
%                     if S.fix_p_dim1_1st
%                         S = varargin2S(S, {
%                             'p_dim1_1st', 1 % fix_vec % _union
%                             });
%                     else
%                         S = varargin2S(S, {
%                             'p_dim1_1st', 1 % free_vec % _union
%                             });
%                     end
%                     factors = factorizeS(S, {'sigmaSq_fac', 'p_dim1_1st'});
% 
%                     n = numel(factors);
%                     Ss = cell(n, 1);
% 
%                     for ii = 1:n       
%                         cf = factors(ii);
% 
%                         Ss{ii} = varargin2C({
%                             'file_name', ...
%                             sprintf('InhConstDrift_sigma%03.0f_p_dim1_1st_%03.0f', ...
%                                 cf.sigmaSq_fac * 100, cf.p_dim1_1st * 100)
%                             'package', 'Inh'
%                             'dtb', 'DensityConstDriftSharedSigmaSqFac'
%                             'sigmaSq_fac', cf.sigmaSq_fac
%                             'p_dim1_1st', cf.p_dim1_1st
%                             });
%                     end
            end
        end
        
        %% Convert to cell for inhomogeneous combination
        Ss = num2cell(Ss);
    end
end
%% Demo - run
methods
    function [W, Fl] = demo_get_W_inh(Main, varargin)
        S_default = Main.batch_fit_command;
        S_default = S_default{end};
        S = varargin2S(varargin, varargin2S(S_default, {
                'to_plot', true
                'UseParallel', 'always'
            }));
        
        fprintf('-----\n');
        tf_in_parallel = is_in_parallel();
        if tf_in_parallel
            disp('Already in parallel mode. skip plotting and UseParallel.');
            S.to_plot = false;
            S.UseParallel = 'never';
        else
            disp('Not in parallel mode. Keeping to_plot and UseParallel as is.');
        end
        fprintf('to_plot: %d; UseParallel: %s\n', S.to_plot, S.UseParallel);
        fprintf('-----\n');
        
        % Files
        file = Main.get_file_full(S, 'fit');
        mat_file = Main.get_file_full(S, 'fit', '.mat');
        if S.to_avoid_refit && exist(mat_file, 'file')
            fprintf('\n');
            fprintf('===== Skipped existing fit: %s\n', file);        
            W = [];
            Fl = [];
            return;
        end
        
        %% DEBUG - disabled temporarily
        W = Main.demo_get_W({
            'subj', S.subj, 'parad', S.parad});
        
        %% dtb
        Pkg = pkg2S(str_bridge('.', 'Fit.D2.RT.Bounded', S.package));        
        W.set_Dtb(Pkg.(['Dtb', S.dtb]));
        
        %% drift & bound
        if S.kb_ratio && ~strcmp(S.drift, 'Indiv')
            W.Dtb.set_KBRatios;
        end
        if strcmp(S.package, 'Inh')
            W.Dtb.set_Drifts(S.drift);
            W.Dtb.set_Bounds(S.bound);
        else
            for dim = 1:2
                W.Dtb.Dtbs{dim}.set_Drift(S.drift);
                W.Dtb.Dtbs{dim}.set_Bound(S.bound);
            end
        end
        W.Miss.customize_th_for_Data;
        
        %
        Fl = W.get_Fl;
        Fl.W0 = W;
        
        % Parameters
        if strcmp(S.package, 'Inh')
            W.Dtb.th0.p_dim1_1st = S.p_dim1_1st;
            if S.fix_p_dim1_1st
                W.Dtb.fix_('p_dim1_1st');
            end
            
            switch S.dtb
                case 'Density'
                    W.Dtb.set_th0_safe('drift_fac_together_dim1_2', S.drift_fac);
                    W.Dtb.set_th0_safe('drift_fac_together_dim2_1', S.drift_fac);
                    if S.fix_drift_fac
                        W.Dtb.fix_('drift_fac_together_dim1_2');
                        W.Dtb.fix_('drift_fac_together_dim2_1');
                    end
                    
                    W.Dtb.set_th0_safe('sigmaSq_fac_together_dim1_2', S.sigmaSq_fac);
                    W.Dtb.set_th0_safe('sigmaSq_fac_together_dim2_1', S.sigmaSq_fac);
                    if S.fix_sigmaSq_fac
                        W.Dtb.fix_('sigmaSq_fac_together_dim1_2');
                        W.Dtb.fix_('sigmaSq_fac_together_dim2_1');
                    end
            end
            
%                 W.Dtb.th0.drift_fac_together_dim1_2 = S.drift_fac;
%                 W.Dtb.th0.drift_fac_together_dim2_1 = S.drift_fac;
%                 if S.fix_drift_fac
%                     W.Dtb.fix_('drift_fac_together_dim1_2');
%                     W.Dtb.fix_('drift_fac_together_dim2_1');
%                 end
                
%             if isa(W.Dtb, ...
%                     'Fit.D2.Inh.DtbConstFanoSharedDriftFac')
%                 
%                 W.Dtb.th0.drift_fac_together_priority2 = max( ...
%                     W.Dtb.lb.drift_fac_together_priority2, ...
%                     S.drift_fac);
%                 
%                 if S.fix_drift_fac
%                     W.Dtb.fix_('drift_fac_together_priority2');
%                 end
%                 
%             elseif isa(W.Dtb, ...
%                     'Fit.D2.Inh.DtbConstSigmaSqSharedDriftFac')
%                 
%                 W.Dtb.th0.drift_fac_together_priority2 = S.drift_fac;
%                 if S.fix_drift_fac
%                     W.Dtb.fix_('drift_fac_together_priority2');
%                 end
%                 
%             elseif isa(W.Dtb, ...
%                     'Fit.D2.Inh.DtbConstDriftSharedSigmaSqFac')
%                 
%                 W.Dtb.th0.sigmaSq_fac_together_priority2 = max( ...
%                     W.Dtb.lb.sigmaSq_fac_together_priority2, ...
%                     S.sigmaSq_fac);                
%                 if S.fix_sigmaSq_fac
%                     W.Dtb.fix_('sigmaSq_fac_together_priority2');
%                 end
%             end
            
    %         W.Dtb.th0.Drift1__k = 37;

            if strcmp(S.drift, 'Indiv')
                W.Dtb.th0.Bound1__b = 1;
                W.Dtb.th0.Bound2__b = 0.7;
            elseif S.kb_ratio
                W.Dtb.th0.KBRatio1__k_b_ratio = 30;
                W.Dtb.th0.KBRatio2__k_b_ratio = 5;
            else
                W.Dtb.th0.Bound1__b = 0.8;
                W.Dtb.th0.Bound2__b = 0.7;
                W.Dtb.th0.Drift1__k = 40;
                W.Dtb.th0.Drift2__k = 3;
            end
        else
            W.Dtb.set_Td(S.td);
            
            if strcmp(S.drift, 'Indiv')
                W.Dtb.Dtb1.th0.Bound__b = 1;
                W.Dtb.Dtb2.th0.Bound__b = 0.7;
            elseif S.kb_ratio
                W.Dtb.Dtb1.th0.KBRatio__k_b_ratio = 30;
                W.Dtb.Dtb2.th0.KBRatio__k_b_ratio = 5;
            else
                W.Dtb.Dtb1.th0.Bound__b = 0.8;
                W.Dtb.Dtb2.th0.Bound__b = 0.7;
                W.Dtb.Dtb1.th0.Drift__k = 40;
                W.Dtb.Dtb2.th0.Drift__k = 3;
            end            
        end
                
        % Tnd
        W.Tnd.set_distrib(S.halfnorm);
        W.Tnd.ub.disper = -1;
        W.Tnd.th0.mu = 0.4;
        W.Tnd.ub.mu = 0.7;
        
        W.th = W.th0;
        
        %
        Fl.W0 = W.deep_copy;
        Fl.W = W.deep_copy;
        
        %% Add plotfun
        Fl.remove_plotfun_all;
        Fit.D2.Common.Plot.PlotFuns.add_plotfun(Fl);
        Fl.plot_opt.to_plot = S.to_plot;
        
        if strcmp(S.package, 'Inh')
            if ~strcmp(S.bound, 'Const')
                Fl.add_plotfun({
                    @(Fl) @(x,v,s) void(@() {
                        void(@() Fl.W.Dtb.Bound1.plot, 0)
                        }, 0)
                    @(Fl) @(x,v,s) void(@() {
                        void(@() Fl.W.Dtb.Bound2.plot, 0)
                        }, 0)
                    });
            end
        else
            if ~strcmp(S.bound, 'Const')
                Fl.add_plotfun({
                    @(Fl) @(x,v,s) void(@() {
                        void(@() Fl.W.Dtb.Dtb1.Bound.plot, 0)
                        }, 0)
                    @(Fl) @(x,v,s) void(@() {
                        void(@() Fl.W.Dtb.Dtb2.Bound.plot, 0)
                        }, 0)
                    });
            end
        end
        
        % Test run
        Fl.W.pred;
        
        if S.to_plot
            fig_tag('runPlotFcns');
            Fl.runPlotFcns;
        end
        
        % Test run
        tic;
        c = Fl.get_cost(Fl.th_vec);
        disp(c);
        toc;
        
        %% Plot RtDistribAll
        if S.to_plot
            fig_tag('RtDistribAll');
            if S.to_clf
                clf;
            end
            Pl = DtbPlot.PlotRtDistrib2DAll(Fl.W.Data.get_RT_pred_pdf);
            hAx = Pl.plot;
            for ii = 1:numel(hAx)
                axes(hAx(ii));
                hold('on');
            end

            %% Plot RT
            fig_tag('RT');
            if S.to_clf
                clf;
            end
            Pl = DtbPlot.PlotRt2D(Fl.W.Data.get_RT_pred_pdf);
            h = Pl.plot({}, {'dt', Fl.W.get_dt});
            h = [h{:}];
            if strcmp(S.package, 'Inh')
                color = 'r';
            else
                color = 'b';
            end
            set(h, 'Color', color);
            hold on;

            %% Plot Ch
            fig_tag('Ch');
            if S.to_clf
                clf;
            end
            Pl = DtbPlot.PlotCh2D(Fl.W.Data.get_RT_pred_pdf);
            h = Pl.plot;
            h = [h{:}];
            if strcmp(S.package, 'Inh')
                color = 'r';
            else
                color = 'b';
            end
            set(h, 'Color', color);
            hold on;
        end
        
        %% Fit
        if S.to_fit
            fprintf('===== Starting fit: %s\n', file);
            fprintf('S:\n');
            fprintf('-----\n');
            disp(S);

            fprintf('Fl.W0:\n');
            fprintf('-----\n');
            disp(Fl.W0);
            fprintf('-----\n\n');

            Fl.plot_opt.to_plot = S.to_plot;
            Fl.fit('opts', {'UseParallel', S.UseParallel});

            fprintf('----- Finished fitting.\n\n');
            
            %% Save
            if S.to_save
                res = Fl.res; %#ok<NASGU>

                mkdir2(fileparts(mat_file));
                save(mat_file, 'Fl', 'S', 'res');
                
                if S.to_plot
                    Main.batch_runPlotFcns(S);
                end
            end
        end           
    end
    function batch_compare_rt_pred(~, Fls)
        assert(isstruct(Fls));
        assert(all(vVec(cellfun(@(c) isa(c, 'FitFlow'), ...
            struct2cell(Fls)))));
        
        %%
        Fl_names = fieldnames(Fls)';
        base_name = 'IndepPar';
        
        %% Calculate
        for Fl_name = Fl_names(:)'
            Fl = Fls.(Fl_name{1});
            
            %%
            p = Fl.W.Data.get_RT_pred_pdf;
            sums_p.(Fl_name{1}) = sums(p, [1, 4, 5], true);
        end
        
        %% Plot
        sums_base = sums_p.(base_name);
        for Fl_name = Fl_names(:)'
            if strcmp(Fl_name{1}, base_name)
                continue; 
            end
            
            fig_tag(['compare_rt_pred_', Fl_name{1}]);
            imagesc((sums_p.(Fl_name{1}) - sums_base) ./ sums_base);
            colorbar;
        end
    end
end
end