classdef MainBatch < Fit.D2.Bounded.Main

properties (Dependent)
    dtb % Set according to sigmaSq
    drift % Set to the common one, or a cell array of both
    bound % Set to the common one, or a cell array of both
    sigmaSq % Set to the common one, or a cell array of both
    fano_max % Set to the common one, or a cell array of both
    to_slow_collapse % Set to the common one, or a cell array of both
    
    % Also required from Fit.D2.Common.Main:
    % drift_kind
    % bound_kind
    % sigmaSq_kind
    % tnd_distrib
    % n_tnd
end
properties
    dtb_ = 'DensityIndivJt'; % '';
    drift1 = 'Const';
    drift2 = 'Const';
    bound1 = 'BetaCdf'; % 'Const';
    bound2 = 'BetaCdf'; % 'Const';
    sigmaSq1 = 'LinearMinPreDrift'; % 'Const';
    sigmaSq2 = 'LinearMinPreDrift'; % 'Const';
    to_slow_collapse_ = '';
    tnd = 'gamma'; % 'halfnorm';
    
    p_dim1_1st = 0;
    
    drift_fac_1 = 0;
    sigmaSq_fac_1 = 0;
    
    drift_fac_2 = 0;
    sigmaSq_fac_2 = 0;
    
    fix_p_dim1_1st = true;
    
    fix_drift_fac_1 = true;
    fix_sigmaSq_fac_1 = true;
    
    fix_drift_fac_2 = true;
    fix_sigmaSq_fac_2 = true;
    
    fix_drift_bias_1 = false;
    fix_drift_bias_2 = false;
    
    fix_bias_irr_1 = false;
    fix_bias_abs_irr_1 = false;
    
    fix_bias_irr_2 = false;
    fix_bias_abs_irr_2 = false;
    
    fano_max_1 = 1;
    fano_max_2 = 1;
    
    fix_fano = true;
%     fix_fano_1 = true;
%     fix_fano_2 = true;
    
    kb_ratio = 0; % 1: use KBRatio; 2: use nonlcon
    
    to_crossval = false; % true;
    n_set_crossval = 2;
    crossval_op = 'Kfold';
    crossval_prct_holdout = 50;
    
    UseParallel = 'always';
    
    %% Batch
    parallel_mode = 'none'; % 'parfeval'; % 'none'|'parfeval'
    skip_plotting = true;
    
    jobs = {};
    n_submitted = 0;
    ix_submitted = [];    
    
    MaxIter = 1000;
    
    desc_ = ''; % Description
end
properties (Dependent)
    dtb_short
    to_slow_collapse_short
    desc % Assign automatically if desc_ is empty.
    
    % Summary variables for file name. [dim1, dim2]. Scalar if dim1 == dim2.
    drift_fac
    sigmaSq_fac
    fix_drift_fac
    fix_sigmaSq_fac
    fix_drift_bias
    fix_bias_irr
    fix_bias_abs_irr
end
%% Construction
methods
    function W = MainBatch(varargin)
        W.set_Tnd;
        W.add_deep_copy({'Fl'});
        
        if ~isempty(varargin)
            W.init(varargin{:});
        end
    end
    function [S_file, S0_file] = get_S_file(W, add_fields, remove_fields)
        if nargin < 2, add_fields = {}; end
        if nargin < 3, remove_fields = {}; end
        
        [S_file, S0_file] = W.get_S_file@bml.oop.PropFileName( ...
            add_fields, remove_fields);
        
        if S0_file.to_crossval
            if ~isfield(S_file, 'ist')
                S_file.ist = 0;
            end
            if ~isfield(S0_file, 'ist')
                S0_file.ist = 0;
            end
        end        
    end
    function fs = get_file_fields0(W)
        fs = union_general(W.get_file_fields0@Fit.D2.Bounded.Main, ...
            {
            'dtb_short',        'dtb'
            'kb_ratio',         'kb'
            ...
            'p_dim1_1st',       'p1'
            ...
            'drift_fac',        'd'
            'sigmaSq_fac',      's'
            'fano_max',         'fn'
            ...
%             'drift_fac_1',      'd1'
%             'drift_fac_2',      'd2'
%             'sigmaSq_fac_1',    's1'
%             'sigmaSq_fac_2',    's2'
%             'fano_max_1',       'fn1'
%             'fano_max_2',       'fn2'
            ...
            'fix_p_dim1_1st',   'pf'
            ...
            'fix_drift_fac',    'df'
            'fix_sigmaSq_fac',  'sf'
            'fix_fano',         'fnf'
            ...
%             'fix_drift_fac_1',  'd1f'
%             'fix_drift_fac_2',  'd2f'
%             'fix_sigmaSq_fac_1','s1f'
%             'fix_sigmaSq_fac_2','s2f'
%             'fix_fano_1',       'fn1f'
%             'fix_fano_2',       'fn2f'
            'to_slow_collapse_short', 'slc'
            ...
            'to_crossval',      'cv'
            ...
%             'desc',             'desc' % File name too long
            }, 'stable', 'rows');
        if W.to_crossval
            fs = [fs
                {
                'n_set_crossval', 'ncv'
                }];
            switch W.crossval_op
                case 'Holdout'
                    fs = [fs
                        {
                        'crossval_prct_holdout', 'pcv'
                        }];
            end
        end
    end
    function v = get_file_mult(~)
        v = {
            'p_dim1_1st', 100
            'drift_fac_1', 100
            'sigmaSq_fac_1', 100
            'drift_fac_2', 100
            'sigmaSq_fac_2', 100
            'fano_max_1', 100
            'fano_max_2', 100
            };
    end
end
methods
    function v = get.dtb_short(W)
        switch W.dtb
            case 'DensityIndivJt'
                v = 'DI';
            case 'DensitySlice'
                v = 'Sl';
            case 'DensitySliceFree'
                v = 'SF';
            case 'DensitySliceFix'
                v = 'SX';
            case 'DensitySliceInv'
                v = 'SI';
            case 'DensityEvScale'
                v = 'Sc';
            otherwise
                error('Not immplemented for %s\n', W.dtb);
        end
    end
    function v = get.desc(W)
        if ~isempty(W.desc_)
            v = W.desc_;
        else
            v = '';
            
            if W.fix_sigmaSq_fac_1 && W.fix_sigmaSq_fac_2
                if W.fix_drift_fac_1 && W.fix_drift_fac_2
                    if W.drift_fac_1 == 0 && W.drift_fac_2 == 0 ...
                            && W.sigmaSq_fac_1 == 0 && W.sigmaSq_fac_2 == 0
                        v = 'ser';

                    elseif W.drift_fac_1 == 1 && W.drift_fac_2 == 1 ...
                            && W.sigmaSq_fac_1 == 1 && W.sigmaSq_fac_2 == 1
                        v = 'par';
                    end
                end
                if strcmp(v, '') && ...
                        (W.drift_fac_1 ~= 1 || W.drift_fac_2 ~= 1 ...
                        || ~W.fix_drift_fac_1 || ~W.fix_drift_fac_2) ...
                        && W.sigmaSq_fac_1 == 1 && W.sigmaSq_fac_2 == 1
                    
                    % If drifts are not all fixed to 1 
                    % and the sigmas are all fixed to 1
                    v = 'ssq1';
                end
            end
        end
    end
end
%% Bias
methods
    function b = get_cond_bias(W)
        b = {W.Dtb.Drift1.get_cond_bias, W.Dtb.Drift2.get_cond_bias};
    end
end
%% Deleting files
methods
    function movefile_ser(W0, varargin)
        Ss = W0.cmd_ixn_irr_ser(varargin);
        W0.movefile_Ss(Ss);
    end
    function files = get_files_Ss(W0, Ss)
        n = numel(Ss);
        files = cell(n, 1);
        for ii = 1:n
            C = varargin2C(Ss(ii));
            W = feval(class(W0));
            bml.oop.varargin2props(W, C);
            
            files{ii} = [W.get_file '.mat'];
        end
    end
    function movefile_Ss(W0, Ss)
        files = W0.get_files_Ss(Ss);
        n = numel(files);
        
        fprintf('Files to move:\n');
        fprintf('%s\n', files{:});
        fprintf('Total %d files.\n', n);
        if inputYN_def('Do you want move the files to reserved', false)
            fprintf('Moving to:\n');
            for ii = 1:n
                file_src = files{ii};
                file_dst = strrep(file_src, 'Data_2D/', 'Data_reserved/');
                movefile2(file_src, file_dst);
                fprintf('%s\n', file_dst);
            end
            fprintf('Done.\n');
        end
    end
end
%% Facades - irr_ixn
methods
    function batch_all(W0, varargin)
        C = varargin2C(varargin, {
            });
        
        W0.batch_ixn_irr_ser(C{:});
        W0.batch_ixn_irr_par(C{:});
        W0.batch_ixn_irr_sigmaSq1(C{:});
        W0.batch_ixn_irr_simple_01_fano(C{:}); % time consuming
        
        W0.batch_ixn_irr_ser(C{:}, ...
            'bound', 'Const', 'sigmaSq', 'Const');
        W0.batch_ixn_irr_par(C{:}, ...
            'bound', 'Const', 'sigmaSq', 'Const');
        W0.batch_ixn_irr_sigmaSq1(C{:}, ...
            'bound', 'Const', 'sigmaSq', 'Const');
        W0.batch_ixn_irr_simple_01_fano(C{:}, ...
            'bound', 'Const', 'sigmaSq', 'Const'); % time consuming
    end
    function batch_sigmaSq1(W0, varargin)
        C = varargin2C(varargin, {
            });
        
        W0.batch_ixn_irr_sigmaSq1(C{:});
        W0.batch_ixn_irr_sigmaSq1(C{:}, ...
            'bound', 'Const', 'sigmaSq', 'Const');
    end
    function batch_simple(W0, varargin)
        C = varargin2C(varargin, {
            });
        
        W0.batch_ixn_irr_ser(C{:});
        W0.batch_ixn_irr_sigmaSq1(C{:});

        W0.batch_ixn_irr_ser(C{:}, ...
            'bound', 'Const', 'sigmaSq', 'Const');
        W0.batch_ixn_irr_sigmaSq1(C{:}, ...
            'bound', 'Const', 'sigmaSq', 'Const');
    end
    function batch_fano(W0, varargin)
        C = varargin2C(varargin, {
            });
        
        W0.batch_ixn_irr_simple_01_fano(C{:}); % time consuming
        W0.batch_ixn_irr_simple_01_fano(C{:}, ...
            'bound', 'Const', 'sigmaSq', 'Const'); % time consuming
    end
    
    function Ss = batch_ixn_irr_ser(W0, varargin)
        Ss = W0.cmd_ixn_irr_ser(varargin{:});
        Ss = W0.batch_Ss(Ss);
    end
    function [Ss, S_batch] = cmd_ixn_irr_ser(W0, varargin)
        % Same as the serial model, for sanity check.
        S_batch = varargin2S(varargin, {
            'subj', Data.Consts.subjs_RT
            'dtb', {'DensityIndivJt'}
            'bound', {'BetaCdf'}
            'sigmaSq', {'LinearMinPreDrift'}
            ...
            'p_dim1_1st', {0}
            'drift_fac_1', {0}
            'sigmaSq_fac_1', {0}
            'drift_fac_2', {0}
            'sigmaSq_fac_2', {0}
            ...
            'fix_p_dim1_1st', {true}
            'fix_drift_fac_1', {true}
            'fix_drift_bias_1', {false}
            'fix_sigmaSq_fac_1', {true}
            'fix_drift_fac_2', {true}
            'fix_drift_bias_2', {false}
            'fix_sigmaSq_fac_2', {true}
            ...
            'kb_ratio', {0}
            'parallel_mode', {'none'}
            });

        [Ss, S_batch] = W0.batch_cmd(S_batch);
    end

    function Ss = batch_ixn_irr_par(W0, varargin)
        Ss = W0.cmd_ixn_irr_par(varargin{:});
        Ss = W0.batch_Ss(Ss);
    end
    function [Ss, S_batch] = cmd_ixn_irr_par(W0, varargin)
        % Same as the serial model, for sanity check.
        S_batch = varargin2S(varargin, {
            'subj', Data.Consts.subjs_RT
            'dtb', {'DensityIndivJt'}
            'bound', {'BetaCdf'}
            'sigmaSq', {'LinearMinPreDrift'}
            ...
            'p_dim1_1st', {0} % if not 0 or 1, takes twice longer.
            'drift_fac_1', {1}
            'sigmaSq_fac_1', {1}
            'drift_fac_2', {1}
            'sigmaSq_fac_2', {1}
            ...
            'fix_p_dim1_1st', {true}
            'fix_drift_fac_1', {true}
            'fix_drift_bias_1', {false}
            'fix_sigmaSq_fac_1', {true}
            'fix_drift_fac_2', {true}
            'fix_drift_bias_2', {false}
            'fix_sigmaSq_fac_2', {true}
            ...
            'kb_ratio', {0}
            'parallel_mode', {'none'}
            });

        [Ss, S_batch] = W0.batch_cmd(S_batch);
    end
    
    function Ss = batch_ixn_irr_sigmaSq1(W0, varargin)
        Ss = W0.cmd_ixn_irr_sigmaSq1(varargin{:});
        Ss = W0.batch_Ss(Ss);
    end
    function [Ss, S_batch] = cmd_ixn_irr_sigmaSq1(W0, varargin)
        % Simple cases where sigmaSq_fac = 1 for both dim,
        % and drift_fac is either 0 or 1.
        S_batch = varargin2S(varargin, {
            'subj', Data.Consts.subjs_RT
            'dtb', {'DensityIndivJt'}
            'bound', {'BetaCdf'}
            'sigmaSq', {'LinearMinPreDrift'}
            ...
            'p_dim1_1st', {0.5} % {0, 0.5, 1}
            'drift_fac_1', {0, 1} % 1 = par; 0 = diffusion w/o drift
            'sigmaSq_fac_1', {1}
            'drift_fac_2', {0, 1}
            'sigmaSq_fac_2', {1}
            ...
            'fix_p_dim1_1st', {false}
            'fix_drift_fac_1', {true}
            'fix_drift_bias_1', {false}
            'fix_sigmaSq_fac_1', {true}
            'fix_drift_fac_2', {true}
            'fix_drift_bias_2', {false}
            'fix_sigmaSq_fac_2', {true}
            ...
            'kb_ratio', {0}
            'parallel_mode', {'none'}
            });

        [Ss, S_batch] = W0.batch_cmd(S_batch);
    end
    
    function [Ss, S_batch] = batch_ixn_irr_simple_01_fano_demo(W0, varargin)
        C = varargin2C(varargin, {
            'subj', Data.Consts.subjs_RT{1}
            'fano_max', 0.5
            'p_dim_1st', 0.5   
            'parallel_mode', 'none'
            });
        [Ss, S_batch] = W0.batch_ixn_irr_simple_01_fano(C{:});
    end
    
    function Ss = batch_ixn_irr_simple_01_fano(W0, varargin)
        Ss = W0.cmd_ixn_irr_simple_01_fano(varargin{:});
        Ss = W0.batch_Ss(Ss);
    end
    function [Ss, S_batch] = cmd_ixn_irr_simple_01_fano(W0, varargin)
        % Both drift_fac and sigmaSq_fac are let free,
        % from a condition closest to the serial,
        % with various fano_max.
        %
        % Similar to simple_01 but includes fano_max.
        
        S_batch = varargin2S(varargin, {
            'subj', Data.Consts.subjs_RT
            'dtb', {'DensityIndivJt'}
            'bound', {'BetaCdf'}
            'sigmaSq', {'LinearMinPreDrift'}
            ...
            'fano_max', {0.5, 1} % {0.5, 0.7, 0.9, 1}
            'p_dim1_1st', {0.5} % {0, 0.5, 1}
            'drift_fac_1', {0}
            'sigmaSq_fac_1', {0.16}
            'drift_fac_2', {0}
            'sigmaSq_fac_2', {0.16}
            ...
            'fix_p_dim1_1st', {false}
            'fix_drift_fac_1', {false}
            'fix_drift_bias_1', {false}
            'fix_sigmaSq_fac_1', {false}
            'fix_drift_fac_2', {false}
            'fix_drift_bias_2', {false}
            'fix_sigmaSq_fac_2', {false}
            ...
            'kb_ratio', {0}
            'parallel_mode', {'none'}
            });
        
        [Ss, S_batch] = W0.batch_cmd(S_batch);
    end
end
%% Batch
methods
    function [Ss, S_batch] = batch_cmd(W0, varargin)
        S_batch = varargin2S(varargin);
        Ss = bml.args.factorizeS(S_batch);
        Ss = W0.preprocess_Ss(Ss);
    end
    function [Ss, S_batch] = batch(W0, varargin)
        [Ss, S_batch] = W0.batch_cmd(varargin{:});
        Ss = W0.batch_Ss(Ss);
    end
    function Ss = preprocess_Ss(~, Ss0, varargin)
        opt = varargin2S(varargin, {
            'drift_fac_1_incl',   Ss0(1).drift_fac_1
            'sigmaSq_fac_1_incl', Ss0(1).sigmaSq_fac_1
            'drift_fac_2_incl',   Ss0(1).drift_fac_2
            'sigmaSq_fac_2_incl', Ss0(1).sigmaSq_fac_2
            'batch_incl', ':' % Set to ':' to use from_batch:to_batch
            'from_batch', 1
            'to_batch', 0
            });
        
        % If sigmaSq == 0, fix it - otherwise numerically unstable.
        is_sigmaSq_1_0 = [Ss0.sigmaSq_fac_1] == 0;
        is_sigmaSq_2_0 = [Ss0.sigmaSq_fac_2] == 0;
        
        for ii = find(is_sigmaSq_1_0)
            Ss0(ii).fix_sigmaSq_fac_1 = true;
        end
        for ii = find(is_sigmaSq_2_0)
            Ss0(ii).fix_sigmaSq_fac_2 = true;
        end
        
        incl = true(numel(Ss0), 1);
        
        % Only use SNR <= 1: by definition, 2nd process is noisier.
        is_SNR_higher_1 = vVec(nan2v( ...
            [Ss0.drift_fac_1] ./ [Ss0.sigmaSq_fac_1], 1) > 1);
        is_SNR_higher_2 = vVec(nan2v( ...
            [Ss0.drift_fac_2] ./ [Ss0.sigmaSq_fac_2], 1) > 1);

        incl = incl ...
            & (~is_SNR_higher_1) & (~is_SNR_higher_2);

        % When d1 = d2 = s1 = s2 = 1, p1 must be fixed, because it is
        % irrelevant.
        p1 = vVec([Ss0.p_dim1_1st]);
        d1 = vVec([Ss0.drift_fac_1]);
        d2 = vVec([Ss0.drift_fac_2]);
        s1 = vVec([Ss0.sigmaSq_fac_1]);
        s2 = vVec([Ss0.sigmaSq_fac_2]);
        
        p1f = vVec([Ss0.fix_p_dim1_1st]);
        d1f = vVec([Ss0.fix_drift_fac_1]);
        d2f = vVec([Ss0.fix_drift_fac_2]);
        s1f = vVec([Ss0.fix_sigmaSq_fac_1]);
        s2f = vVec([Ss0.fix_sigmaSq_fac_2]);
        
        to_excl = (d1 == 1) & (d2 == 1) & (s1 == 1) & (s2 == 1) ...
            & d1f & d2f & s1f & s2f ...
            & ~p1f;
        incl = incl & ~to_excl;
        
%         % When dim1 is always first, all params about dim 2 being first
%         % is irrelevant.
%         
%         is_dim1_1 = vVec([Ss0.p_dim1_1st] == 1) ...
%                  & vVec([Ss0.fix_p_dim1_1st]) ...
%                  & (~bsxEq([Ss0.drift_fac_1],   opt.drift_fac_1_incl) ...
%                   | ~bsxEq([Ss0.sigmaSq_fac_1], opt.sigmaSq_fac_1_incl));
%               
%         % Vice versa.
%         is_dim1_2 = vVec([Ss0.p_dim1_1st] == 0) ...
%                  & vVec([Ss0.fix_p_dim1_1st]) ...
%                  & (~bsxEq([Ss0.drift_fac_2],   opt.drift_fac_2_incl) ...
%                   | ~bsxEq([Ss0.sigmaSq_fac_2], opt.sigmaSq_fac_2_incl));
%         
%         incl = incl ...
%              & (~is_dim1_1) & (~is_dim1_2);
        
        Ss = Ss0(incl);
        n_batch = numel(Ss);
        
        if ischar(opt.batch_incl) && isequal(opt.batch_incl, ':')
            from_batch = bml.indsub.ix2py(opt.from_batch, n_batch);
            to_batch = bml.indsub.ix2py(opt.to_batch, n_batch);
            
            Ss = Ss(from_batch:to_batch);
        else
            batch_incl = bml.indsub.ix2py(opt.batch_incl, n_batch);
            Ss = Ss(batch_incl);
        end
    end
    function Ss = batch_Ss(W0, Ss)
        if isstruct(Ss)
            Ss = num2cell(Ss);
        end
        n = numel(Ss);
        
        W0.n_submitted = 0;
        pool = gcp;
        n_workers = pool.NumWorkers;
        W0.jobs = cell(n, 1);
        W0.ix_submitted = zeros(n, 1);
        showed_error = false(n, 1);
        
        fprintf('Given %d batch jobs\n', n);
        
        for ii = 1:n
            S = Ss{ii};
            C = varargin2C(S);
            
            switch S.parallel_mode % W0.parallel_mode
                case 'none'
                    W0.calculate(C{:});
                case 'parfeval'
%                     varargin2props(W0, C);
%                     if ~exist([W0.get_file '.mat'], 'file') ...
%                             || ~W0.skip_existing_mat
                        
                        W0.n_submitted = W0.n_submitted + 1;
                        W0.ix_submitted(W0.n_submitted) = ii;
                        
%                         bml.file.NullFile.save(W0.get_file);
%                         fprintf('Submitted %d/%d (submission %d) at %s : Saved a null file %s.mat to block duplicate processing.\n', ...
%                             ii, n, W0.n_submitted, datestr(now, 30), W0.get_file);
                        fprintf('Submitted %d/%d (submission %d) at %s\n', ...
                            ii, n, W0.n_submitted, datestr(now, 30));
                        
                        W = W0.deep_copy;
%                         W.calculate; % DEBUG
                        W0.jobs{W0.n_submitted} = ...
                            parfeval(@W.calculate, 2, C{:});
%                         W0.jobs{W0.n_submitted} = parfeval(@W.calculate_unit, 2);
                        
                        job = [W0.jobs{1:W0.n_submitted}];
                        
                        while (bml.parallel.n_running(job) >= n_workers) ...
                                || (ii == n && ...
                                    bml.parallel.n_running(job) > 0)
                        
                            WaitSecs(1);
                        end
                        
                        ix_incl = 1:W0.n_submitted;
                        err = {job.Error};
                        ix_new_err = hVec(find( ...
                            hVec(~cellfun(@isempty, err)) ...
                          & hVec(~showed_error(ix_incl))));
                         
                        for i_err = ix_new_err
                            ix_job_err = W0.ix_submitted(i_err);
                            
                            fprintf('Error processing job %d\n', ...
                                ix_job_err);
                            disp(Ss{ix_job_err});
                            
                            c_err = err{i_err};
                            disp(c_err);
                            disp(c_err.message);
                            for i_stack = numel(c_err.stack)
                                disp(c_err.stack(i_stack));
                                disp(c_err.stack(i_stack).file);
                                disp(c_err.stack(i_stack).name);
                                disp(c_err.stack(i_stack).line);
                            end
                            showed_error(i_err) = true;
                        end
%                     end
            end
            
            if mod(ii, 10) == 0, fprintf('.'); end
            if mod(ii, 100) == 0, fprintf('%d', ii); end
            if mod(ii, 500) == 0, fprintf('\n'); end
        end
        fprintf('Done at %s.\n', datestr(now, 30));
        
        if ~W0.skip_plotting
            W = W0.deep_copy;
            for ii = 1:n
                S = Ss{ii};
                C = varargin2C(S);
                varargin2props(W, C, true);
                W.plot_and_save;
            end
        end
    end
end
%% Unit
methods
    function [file, S_file] = main(W, varargin)
        bml.oop.varargin2props(W, varargin, true);
        file = W.get_file;
        S_file = W.get_S_file;
        
        if W.skip_existing_mat && ...
                exist([file, '.mat'], 'file') % && ...
                % ~bml.file.NullFile.isa([W.get_file, '.mat'])
            fprintf('Skipping existing fit %s\n', [file '.mat']);
        else
%             bml.file.NullFile.save(W.get_file);
%             fprintf('Saved a null file %s.mat to block duplicate processing.\n', ...
%                 W.get_file);
            W.calculate;
            if W.to_save_plot
                W.plot_and_save_all;
            end
        end
    end
    function calculate(W)        
        if W.to_crossval
            W.to_crossval = false;
            file_cv0 = [W.get_file '.mat'];
            W.to_crossval = true;
            file_cv1 = [W.get_file '.mat'];

            if exist(file_cv0, 'file') ...
                    && bml.file.NullFile.isa(file_cv0)
                fprintf('cval=0 file is still being processed! Skipping.\n');
                return;

            elseif ~exist(file_cv0, 'file')
                fprintf('cval=0 file absent and will be fit: %s\n', ...
                    file_cv0);

                W.to_crossval = false;
                W.main;
                W.to_crossval = true;
            end

            fprintf('Copied cval=0 file to cval=1:\n %s\n to %s\n', ...
                file_cv0, file_cv1);
            copyfile(file_cv0, file_cv1);
            L = load(W.get_file);

%                 if ~isfield(L.Fl.res, 'fval_test') || ...
%                         numel(L.Fl.res.fval_test) ~= W.n_set_crossval

            fprintf('Cross-validation began at %s\n', datestr(now, 30));
            t_st = tic;

            L.Fl.W.Data.loaded = false;
            L.Fl.W.load_data;
            [~,~,group] = unique(L.Fl.W.Data.ds(:, {'condM', 'condC'}));

            Fl = L.Fl;
            Fl.crossval( ...
                'n_set', W.n_set_crossval, ...
                'p_holdout', W.crossval_prct_holdout / 100, ...
                'op', W.crossval_op, ...
                'group', group, ...
                'files_ix', W.get_files_crossval_ix, ...
                'files_res', W.get_files_crossval_res, ...
                'fit_opts', {'MaxIter', W.MaxIter});

            W.Fl = Fl;
            W.save_mat;

            t_el = toc(t_st);
            fprintf('Cross-validation took %1.2f sec\n', t_el);
            fprintf('Saved cross-validated results at %s to %s\n', ...
                datestr(now, 30), W.get_file);

%                 else
%                     fprintf('Cross-validation done already. Skipping %s\n', ...
%                         W.get_file);
%                 end

        else
            % Fit with all data
            W.calculate_unit;
        end
    end
    function [file, S_file] = calculate_unit(W)
%         W.init;
        W.fit;
        W.save_mat;
        
        file = W.get_file;
        S_file = W.get_S_file;
    end
    function plot_and_save(W0)
        % Only uses loaded Fl and W
        if exist([W0.get_file '.mat'], 'file')
            if bml.file.NullFile.isa([W0.get_file '.mat'])
                fprintf('%s is a null file!\n', [W0.get_file '.mat']);
                return;
            end

            L = load([W0.get_file, '.mat']);
            W = L.Fl.W;
            try
                W.savefigs;
            catch err
                warning(err_msg(err));
            end
        else
            fprintf('%s not found!\n', [W0.get_file '.mat']);
            return;
        end
    end
    function init(W, varargin)
        W.init@Fit.D2.Bounded.Main(varargin{:});
        
%         %% Data
%         W.Data.set_path({'subj', W.subj, 'parad', W.parad}, ...
%             'A', 1);
%         W.Data.load_data;

        %% Dtb
        W.set_Dtb(W.dtb);
        W.Dtb.init(varargin{:});
        
        W.Dtb.set_Drift1(W.drift1);
        W.Dtb.set_Drift2(W.drift2);
        
        W.th0.Dtb__Drift1__k = 20;
        W.th0.Dtb__Drift2__k = 5;
        
        W.Dtb.set_Bound1(W.bound1);
        W.Dtb.set_Bound2(W.bound2);
        
        if isprop(W.Dtb, 'SigmaSq1')
            W.Dtb.set_SigmaSq1(W.sigmaSq1);
            W.Dtb.set_SigmaSq2(W.sigmaSq2);
        end
        
        W.Dtb.constrain_fano_unit(1, 2, W.fano_max_1);
        W.Dtb.constrain_fano_unit(2, 1, W.fano_max_2);
        
        %% p_dim1_1st
        W.Dtb.th0.p_dim1_1st = W.p_dim1_1st;
        if W.fix_p_dim1_1st
            W.Dtb.fix_('p_dim1_1st');
            
            if W.Dtb.th0.p_dim1_1st == 1
                W.fix_drift_fac_1 = true;
                W.fix_sigmaSq_fac_1 = true;
                
            elseif W.Dtb.th0.p_dim1_1st == 0
                W.fix_drift_fac_2 = true;
                W.fix_sigmaSq_fac_2 = true;
            end
        end

        %% drift_fac
        W.Dtb.th0.drift_fac_together_dim1_2 = W.drift_fac_1;
        W.Dtb.th0.drift_fac_together_dim2_1 = W.drift_fac_2;
%         W.Dtb.set_th0_safe('drift_fac_together_dim1_2', W.drift_fac_1);
%         W.Dtb.set_th0_safe('drift_fac_together_dim2_1', W.drift_fac_2);
        if W.fix_drift_fac_1
            W.Dtb.fix_to_th0_('drift_fac_together_dim1_2');
        else
            W.Dtb.th_lb.drift_fac_together_dim1_2 = 0;
            W.Dtb.th_ub.drift_fac_together_dim1_2 = 1;
        end
        if W.fix_drift_fac_2
            W.Dtb.fix_to_th0_('drift_fac_together_dim2_1');
        else
            W.Dtb.th_lb.drift_fac_together_dim2_1 = 0;
            W.Dtb.th_ub.drift_fac_together_dim2_1 = 1;
        end

        %% sigmaSq_fac
        if W.sigmaSq_fac_1 == 0
            if W.fix_sigmaSq_fac_1
                W.Dtb.th0.sigmaSq_fac_together_dim1_2 = W.sigmaSq_fac_1;
                W.Dtb.fix_to_th0_('sigmaSq_fac_together_dim1_2');            
            else
                W.Dtb.set_th0_safe('sigmaSq_fac_together_dim1_2', W.sigmaSq_fac_1);
                W.Dtb.th_lb.sigmaSq_fac_together_dim1_2 = 0.16;
                W.Dtb.th_ub.sigmaSq_fac_together_dim1_2 = 2;
            end
        else
            W.Dtb.set_th0_safe('sigmaSq_fac_together_dim1_2', W.sigmaSq_fac_1);
            if W.fix_sigmaSq_fac_1
                W.Dtb.fix_to_th0_('sigmaSq_fac_together_dim1_2');            
            else
                W.Dtb.th_lb.sigmaSq_fac_together_dim1_2 = 0.16;
                W.Dtb.th_ub.sigmaSq_fac_together_dim1_2 = 2;
            end
        end
        if W.sigmaSq_fac_2 == 0
            if W.fix_sigmaSq_fac_2
                W.Dtb.th0.sigmaSq_fac_together_dim2_1 = W.sigmaSq_fac_2;
                W.Dtb.fix_to_th0_('sigmaSq_fac_together_dim2_1');            
            else
                W.Dtb.set_th0_safe('sigmaSq_fac_together_dim2_1', W.sigmaSq_fac_2);
                W.Dtb.th_lb.sigmaSq_fac_together_dim2_1 = 0.16;
                W.Dtb.th_ub.sigmaSq_fac_together_dim2_1 = 2;
            end
        else
            W.Dtb.set_th0_safe('sigmaSq_fac_together_dim2_1', W.sigmaSq_fac_2);
            if W.fix_sigmaSq_fac_2
                W.Dtb.fix_to_th0_('sigmaSq_fac_together_dim2_1');            
            else
                W.Dtb.th_lb.sigmaSq_fac_together_dim2_1 = 0.16;
                W.Dtb.th_ub.sigmaSq_fac_together_dim2_1 = 2;
            end
        end
        
        %% KBRatio        
        switch W.kb_ratio
            case 1
                W.Dtb.set_KBRatios;

                W.Dtb.th0.KBRatio1__k_b_ratio = 30;
                W.Dtb.th0.KBRatio2__k_b_ratio = 5;
                
            case 2
                W.Dtb.set_KBRatio_nonlcon;
                
            otherwise
%                 error0('Not implemented!');
        end
        
        %% Bias
        if W.fix_drift_bias_1
            W.fix_to_th0_('Dtb__Drift1__bias');
        end
        if W.fix_drift_bias_2
            W.fix_to_th0_('Dtb__Drift2__bias');
        end
        
        %% Irr bias
        if isfield(W.th, 'Dtb__Drift1__k_irr')
            if W.fix_bias_irr_1
                W.fix_to_th0_('Dtb__Drift1__k_irr');
            end
            if W.fix_bias_irr_2
               W.fix_to_th0_('Dtb__Drift2__k_irr');
            end
            if W.fix_bias_abs_irr_1
                W.fix_to_th0_('Dtb__Drift1__k_abs_irr');
            end
            if W.fix_bias_abs_irr_2
                W.fix_to_th0_('Dtb__Drift2__k_abs_irr');
            end
        end
        
        %% Tnd
        W.Tnd.distrib = W.tnd;
        W.Tnd.n_Tnd = W.n_tnd;
        W.Tnd.init_params0;
        
        switch W.n_tnd
            case 3
                W.Tnd.th0.mu = 0.4;
                W.Tnd.th_ub.mu = 0.7;
                W.Tnd.th_ub.disper = -1;
            case 4
                for ch1 = 1:2
                    for ch2 = 1:2
                        W.Tnd.th_ub.(sprintf('disper_%d_%d', ch1, ch2)) = -1;
                    end
                end
        end
        
        %% Load simpler model
%         W.get_th0_from_simpler_model;

        %% Init children
        W.init_children(varargin{:});
    end
    function get_th0_from_simpler_model(W)
        if isa(W.Dtb, 'Fit.D2.Inh.DtbDensityIndivJt')
            %%
            S = W.get_S0_file;
            
            S.dtb = 'Density';
            S.sigmaSq = 'Const';
            S.fix_miss = true;
            S.to_crossval = [];
            
            W0 = feval(class(W));
            file = W0.get_file_from_S0(S);
            C = S2C(S);
            
            % Initializing W0 causes recursion.
%             W0 = feval(class(W), C{:});
%             file = W0.get_file({'cval', []});
            
            fprintf('Getting th0 from a simpler model %s\n', file);
            
            if ~exist([file '.mat'], 'file') || ...
                    bml.file.NullFile.isa([file '.mat'])
                
                prev_parallel_mode = W.parallel_mode;
                W.parallel_mode = 'none';
                
                S2 = S;
                S2.to_crossval = false;
                C2 = S2C(S2);
                W.batch(C2{:});
                W.parallel_mode = prev_parallel_mode;
                
                file =  W0.get_file({'cval', false});
                fprintf('Getting th0 from a simpler model %s\n', file);
            end
            L = load(file);
            
            th0 = L.Fl.res.th;
            W.th0 = copyFields(W.th0, th0);
            W.th = W.th0;
        end
    end
    function [Fl, res] = fit(W, varargin)
        if W.to_use_nested_fit
            W0 = fitflow.NestedFit(W, {
                setdiff(W.th_names, W.Dtb.th_names_prefixed);
                });
            [Fl, res] = W0.fit(varargin{:});
        else
            Fl = W.get_Fl;

            %% Test run
            tic;
            c = W.get_cost;
            disp(c);
            toc;

            %% Test plot
            Fl.runPlotFcns;

            %% Fit
            fprintf('=====\n');
            fprintf('Fitting %s began at %s\n', W.get_file, datestr(now, 30));
            t_st = tic;

            S = varargin2S(varargin, {
                'opts', {}
                });
            S.opts = varargin2C(S.opts, {
                'UseParallel', W.UseParallel
                'MaxIter', W.MaxIter
                });
            C = S2C(S);
            Fl.fit(C{:});

            fprintf('=====\n');
            fprintf('Fitting %s finished at %s\n', W.get_file, datestr(now, 30));
            toc(t_st);

            res = Fl.res;
        end
    end
    function save_mat(W0)
        W = W0.deep_copy;
        
        [L.S_file, L.S0_file] = W.get_S_file;
        L.res = W.Fl.res;
        L.Fl = W.Fl;
        
        file = W0.get_file;
        
        save(file, '-struct', 'L');
        fprintf('Saved to %s\n', file);
    end
    function W = load_mat(W0, file)
        if ~exist('file', 'var') || isempty(file)
            file = W0.get_file;
        end
        L = load(file);
        
        % Recover if not loaded properly
        if isfield(L, 'W')
            W = L.W;
        else
            W = L.Fl.W;
        end
        if ~isa(W, class(W0))
            W = feval(class(W0), L.S0_file);
        end
        W.get_Fl;
        W.Fl.res = L.res;
        W.Fl.res2W;
    end
    function batch_plot_files(W0, files)
        for ii = 1:numel(files)
            file = files{ii};
            W = W0.load_mat(file);
            W.plot_and_save_all;
        end
    end
    function savefigs(W, varargin)
        S = varargin2S(varargin, {
            'conds_oversample_factor', 10
            });
        
        Plt = Fit.D2.Inh.Plot(W.Fl);
        Plt.conds_oversample_factor = S.conds_oversample_factor;
        
        pred_done = false;
        
        file = [W.get_file({'plt', 'plotfuns'})];
        if exist([file, '.fig'], 'file') && W.skip_existing_fig
            fprintf('Skipping existing figure: %s\n', [file, '.fig']);
        else
            try
                if ~pred_done
                    W.pred;        
                    pred_done = true;
                end
                
                clf;
                Plt.plotfuns;
                savefigs(file, 'size', [1200 800]);
            catch err
                warning(err_msg(err));
            end
        end
        
        for dimOnX = 1:2
            for kind = {
                    'ch',        'ch'
                    'rt',        'rt'
                    'rt_stdev',  'rtsd'
                    'rt_skew',   'rtsk'
                    }'
                
                [kind_long, kind_short] = deal(kind{:});
                
                file = W.get_file({
                            'plt', kind_short, 'dX', dimOnX});
                if exist([file, '.fig'], 'file') && W.skip_existing_fig
                    fprintf('Skipping existing figure: %s\n', [file, '.fig']);
                else
                    try
                        if ~pred_done
                            W.pred;        
                            pred_done = true;
                        end
                        
                        clf;
                        Plt.(kind_long)({'dimOnX', dimOnX});
                        title(sprintf('%s-dimOnX=%d', ...
                            strrep(kind_long, '_', '-'), ...
                            dimOnX));
                        savefigs(file);
                    catch err
                        warning(err_msg(err));
                    end
                end
            end
        end
    end
end
%% Cross validation
methods
    function files = get_files_crossval_ix(W)
        S2s = bml.str.Serializer;
        n_set = W.n_set_crossval;
        
        files = cell(n_set, 1);
        Ss = W.get_Ss_file_crossval;
        
        for i_set = 1:n_set
            file = S2s.convert(varargin2S(Ss(i_set), {
                'subj', W.subj
                'parad', W.parad
                'task', W.task
                }));
            file = fullfile('Data_2D/crossval_ix', [file '.mat']);
            files{i_set} = file;
        end
    end
    function files = get_files_crossval_res(W)
        n_set = W.n_set_crossval;
        
        files = cell(n_set, 1);
        Ss = W.get_Ss_file_crossval;
        
        for i_set = 1:n_set
            file = [W.get_file(Ss(i_set)), '.mat'];
            files{i_set} = file;
        end
    end
    function Ss = get_Ss_file_crossval(W)
        n_set=  W.n_set_crossval;
        
        for i_set = n_set:-1:1
            switch W.crossval_op
                case 'Kfold'
                    Ss(i_set) = varargin2S({
                        'nst', W.n_set_crossval
                        'ist', i_set
                        });
                case 'Holdout'
                    Ss(i_set) = varargin2S({
                        'nst', W.n_set_crossval
                        'pcv', W.crossval_prct_holdout
                        'ist', i_set
                        });
                otherwise
                    error('Unsupported crossval_op=%s\n', W.crossval_op);
            end
        end
    end
end
%% Object properties
methods
    function set_Dtb(W, obj_or_name)
        if nargin < 2, obj_or_name = W.dtb; end
        W.Dtb = W.enforce_class('Fit.D2.Inh.Dtb', obj_or_name);
        W.set_sub_from_props({'Dtb'});
    end
end
%% Batch-related properties
methods
    function v = get.dtb(W)
        if isempty(W.dtb_)
%             if any(~strcmp(W.sigmaSq, 'Const'))
                v = 'DensityIndivJt';
%             else
%                 v = 'Density';
%             end
        else
            v = W.dtb_;
        end
    end
    function set.dtb(W, v)
        W.dtb_ = v;
    end
    
    function v = get.drift(W)
        if strcmp(W.drift1, W.drift2)
            v = W.drift1;
        else
            v = {W.drift1, W.drift2};
        end
    end
    function set.drift(W, v)
        if ischar(v)
            W.drift1 = v;
            W.drift2 = v;
        elseif iscell(v) && numel(v) == 2
            W.drift1 = v{1};
            W.drift2 = v{2};
        else
            error('Unknown input format!');
        end            
    end
    function v = get_drift_kind(W)
        v = W.drift;
    end
    function set_drift_kind(W, v)
        W.drift = v;
    end
    
    function v = get.bound(W)
        if strcmp(W.bound1, W.bound2)
            v = W.bound1;
        else
            v = {W.bound1, W.bound2};
        end
    end
    function set.bound(W, v)
        if ischar(v)
            W.bound1 = v;
            W.bound2 = v;
        elseif iscell(v) && numel(v) == 2
            W.bound1 = v{1};
            W.bound2 = v{2};
        else
            error('Unknown input format!');
        end
    end
    function v = get_bound_kind(W)
        v = W.bound;
    end
    function set_bound_kind(W, v)
        W.bound = v;
    end
    
    function set_td_kind(W, v)
        % Ignore. Doesn't apply to inh models.
    end
    function v = get_td_kind(W)
        v = '';
    end
    
    function v = get.sigmaSq(W)
        if strcmp(W.sigmaSq1, W.sigmaSq2)
            v = W.sigmaSq1;
        else
            v = {W.sigmaSq1, W.sigmaSq2};
        end
    end
    function set.sigmaSq(W, v)
        if ischar(v)
            W.sigmaSq1 = v;
            W.sigmaSq2 = v;
        elseif iscell(v) && numel(v) == 2
            W.sigmaSq1 = v{1};
            W.sigmaSq2 = v{2};
        else
            error('Unknown input format!');
        end 
    end
    function v = get_sigmaSq_kind(W)
        v = W.sigmaSq;
    end
    function set_sigmaSq_kind(W, v)
        W.sigmaSq = v;
    end
    
    function set.to_slow_collapse(W, v)
        assert(ischar(v) || ...
            iscell(v) && numel(v) == 2 && all(cellfun(@ischar, v)));
        W.to_slow_collapse_ = v;
    end
    function v = get.to_slow_collapse(W)
        v = W.to_slow_collapse_;
        if ischar(v)
            v = {v, v};
        end
    end
    function v = get.to_slow_collapse_short(W)
        v = W.to_slow_collapse_;
        if iscell(v) && isequal(v{1}, v{2})
            v = v{1};
        end
    end
    
    function v = get_tnd_distrib(W)
        v = W.tnd;
    end
    function set_tnd_distrib(W, v)
        W.tnd = v;
    end
    
    function set.fano_max(W, v)
        if isnumeric(v) && isscalar(v)
            W.fano_max_1 = v;
            W.fano_max_2 = v;
        elseif iscell(v) && numel(v) == 2
            W.fano_max_1 = v{1};
            W.fano_max_2 = v{2};
        else
            error('Unknown input format!');
        end
    end
    function v = get.fano_max(W)
        if isequal(W.fano_max_1, W.fano_max_2)
            v = W.fano_max_1;
        else
            v = {W.fano_max_1, W.fano_max_2};
        end
    end
    
    function v = get.drift_fac(W)
        if isequal(W.drift_fac_1, W.drift_fac_2)
            v = W.drift_fac_1;
        else
            v = {W.drift_fac_1, W.drift_fac_2};
        end
    end
    function set.drift_fac(W, v)
        if isnumeric(v) && isscalar(v)
            W.drift_fac_1 = v;
            W.drift_fac_2 = v;
        elseif iscell(v) && numel(v) == 2
            W.drift_fac_1 = v{1};
            W.drift_fac_2 = v{2};
        else
            error('Unknown input format!');
        end
    end
    
    function v = get.sigmaSq_fac(W)
        if isequal(W.sigmaSq_fac_1, W.sigmaSq_fac_2)
            v = W.sigmaSq_fac_1;
        else
            v = {W.sigmaSq_fac_1, W.sigmaSq_fac_2};
        end
    end
    function set.sigmaSq_fac(W, v)
        if isnumeric(v) && isscalar(v)
            W.sigmaSq_fac_1 = v;
            W.sigmaSq_fac_2 = v;
        elseif iscell(v) && numel(v) == 2
            W.sigmaSq_fac_1 = v{1};
            W.sigmaSq_fac_2 = v{2};
        else
            error('Unknown input format!');
        end
    end
    
    function v = get.fix_drift_fac(W)
        if isequal(W.fix_drift_fac_1, W.fix_drift_fac_2)
            v = W.fix_drift_fac_1;
        else
            v = {W.fix_drift_fac_1, W.fix_drift_fac_2};
        end
    end
    function set.fix_drift_fac(W, v)
        if isnumeric(v) && isscalar(v)
            W.fix_drift_fac_1 = v;
            W.fix_drift_fac_2 = v;
        elseif iscell(v) && numel(v) == 2
            W.fix_drift_fac_1 = v{1};
            W.fix_drift_fac_2 = v{2};
        else
            error('Unknown input format!');
        end
    end
    
    function v = get.fix_sigmaSq_fac(W)
        if isequal(W.fix_sigmaSq_fac_1, W.fix_sigmaSq_fac_2)
            v = W.fix_sigmaSq_fac_1;
        else
            v = {W.fix_sigmaSq_fac_1, W.fix_sigmaSq_fac_2};
        end
    end
    function set.fix_sigmaSq_fac(W, v)
        if isnumeric(v) && isscalar(v)
            W.fix_sigmaSq_fac_1 = v;
            W.fix_sigmaSq_fac_2 = v;
        elseif iscell(v) && numel(v) == 2
            W.fix_sigmaSq_fac_1 = v{1};
            W.fix_sigmaSq_fac_2 = v{2};
        else
            error('Unknown input format!');
        end
    end
    
    function v = get.fix_drift_bias(W)
        if isequal(W.fix_drift_bias_1, W.fix_drift_bias_2)
            v = W.fix_drift_bias_1;
        else
            v = {W.fix_drift_bias_1, W.fix_drift_bias_2};
        end
    end
    function set.fix_drift_bias(W, v)
        if isnumeric(v) && isscalar(v)
            W.fix_drift_bias_1 = v;
            W.fix_drift_bias_2 = v;
        elseif iscell(v) && numel(v) == 2
            W.fix_drift_bias_1 = v{1};
            W.fix_drift_bias_2 = v{2};
        else
            error('Unknown input format!');
        end
    end
    
    function v = get.fix_bias_irr(W)
        if isequal(W.fix_bias_irr_1, W.fix_bias_irr_2)
            v = W.fix_bias_irr_1;
        else
            v = {W.fix_bias_irr_1, W.fix_bias_irr_2};
        end
    end
    function set.fix_bias_irr(W, v)
        if isnumeric(v) && isscalar(v)
            W.fix_bias_irr_1 = v;
            W.fix_bias_irr_2 = v;
        elseif iscell(v) && numel(v) == 2
            W.fix_bias_irr_1 = v{1};
            W.fix_bias_irr_2 = v{2};
        else
            error('Unknown input format!');
        end
    end
    
    function v = get.fix_bias_abs_irr(W)
        if isequal(W.fix_bias_abs_irr_1, W.fix_bias_abs_irr_2)
            v = W.fix_bias_abs_irr_1;
        else
            v = {W.fix_bias_abs_irr_1, W.fix_bias_abs_irr_2};
        end
    end
    function set.fix_bias_abs_irr(W, v)
        if isnumeric(v) && isscalar(v)
            W.fix_bias_abs_irr_1 = v;
            W.fix_bias_abs_irr_2 = v;
        elseif iscell(v) && numel(v) == 2
            W.fix_bias_abs_irr_1 = v{1};
            W.fix_bias_abs_irr_2 = v{2};
        else
            error('Unknown input format!');
        end
    end
end
%% Demo plot
methods
    function axs = plot_given_th_scenario(W, varargin)
        S = varargin2S(varargin, {
            'axs', []
            'th', {
                'drift', 1
                'diffusion', 1
                }
            'kinds', {'ch', 'rt'}
            'leave', 'hardest'
            'plot_param', true
            });
        th = cell2dataset(S.th');
        n = size(th, 1);
        
        x = double(th(:,1));
        y = double(th(:,2));
        x_name = th.Properties.VarNames{1};
        y_name = th.Properties.VarNames{2};
        
        n_row = 1;
        n_kind = numel(S.kinds);
        
        if S.plot_param
            n_col = n_kind + 1;
        else
            n_col = n_kind;
        end
          
        %% Preplot to make sure all are cached
        fig = gcf;
        fig2 = fig_tag('temp');
        for ii = 1:n            
            for jj = 1:n_kind
                kind = S.kinds{jj};
                
                clf;
                W.plot_given_th( ...
                    'kind', kind, ...
                    x_name, x(ii), ...
                    y_name, y(ii));
            end
        end
        delete(fig2);
        
        %%
        figure(fig);
        axs = S.axs;
        if isempty(axs)
            axs = subplotRCs(n_row, n_col);
        end
        n_kind = numel(S.kinds);
        
        if S.plot_param
            ax1 = axs(1,1);
            gradLine(x, y, ...
                'color', @hsv2, ...
                'edge_args', {
                    'LineWidth', 1
                }, ...
                'marker_args', {
                    'Marker', 'o'
                }, ...
                'ax', ax1);
            xlabel(ax1, x_name);
            ylabel(ax1, y_name);
            bml.plot.beautify(ax1);
            axis(ax1, 'equal');
        end
        
        colors = hsv2(n);
        for ii = 1:n            
            for jj = 1:n_kind
                kind = S.kinds{jj};
                color = colors(ii,:);
                
                ax1 = axs(1, jj+S.plot_param);
                ax1 = W.plot_given_th( ...
                    'ax', ax1, ...
                    'kind', kind, ...
                    x_name, x(ii), ...
                    y_name, y(ii), ...
                    'color', color, ...
                    'leave', S.leave);
                
                axs(1, jj+S.plot_param) = ax1;
            end
        end
    end
    function [ax, hs] = plot_given_th(W, varargin)
        S = varargin2S(varargin, {
            'ax', gca
            'kind', 'rt' % 'rt'|'ch'
            'drift', 1
            'diffusion', 1
            'color', [0 0 0]
            'dimOnX', []
            'leave', 'hardest' % 'hardest' | 'extreme' (easiest and hardest)
            });
        ax = S.ax;
        
        if isempty(S.dimOnX)
            switch S.kind
                case 'ch'
                    S.dimOnX = 2;
                case 'rt'
                    S.dimOnX = 2;
            end
        end
        
        k1 = W.th.Dtb__Drift1__k;
        k2 = W.th.Dtb__Drift2__k;
        
        S2s = bml.str.Serializer;
        S_file = varargin2S({
            'drift', S.drift
            'diffusion', S.diffusion
            'dX', S.dimOnX'
            'k1', k1
            'k2', k2
            'plt', S.kind
            });
        file_mat = fullfile('Data', class(W), ...
            S2s.convert(rmfield(S_file, 'plt')));
        file_fig = fullfile('Data', class(W), ...
            S2s.convert(S_file));

        is_fig_present = exist([file_fig, '.fig'], 'file');
        is_mat_present = exist([file_mat, '.mat'], 'file');

        if ~is_mat_present || ~is_fig_present
            W.th.Dtb__Drift1__bias = 0;
            W.th.Dtb__Drift2__bias = 0;
            
            W.th.Dtb__drift_fac_together_dim2_1 = S.drift;
            W.th.Dtb__sigmaSq_fac_together_dim2_1 = S.diffusion;
            W.pred;
            
            if ~is_mat_present
                save(file_mat, 'W', 'S', 'S_file');
                fprintf('Saved to %s.mat\n', file_mat);
            end
        end
        
        if is_fig_present
            [ax, hs] = openfig_to_axes([file_fig, '.fig'], ax);
            hs = hs.src;
        else
            W.(['plot_', S.kind])('dimOnX', S.dimOnX);
            savefigs(file_fig);
            
            hs = figure2struct(ax);
        end        
        
        delete(hs.marker);
        switch S.leave
            case 'hardest'
                switch S.kind
                    case 'ch'
                        to_delete = 1:4; % 1:5
                    case 'rt'
                        to_delete = 1:8; % 1:10
                end
                set(hs.nonsegment, 'Color', S.color);
                
            case 'extreme'
                switch S.kind
                    case 'ch'
                        to_delete = 2:4; % 1:5
                    case 'rt'
                        to_delete = 3:8; % 1:10
                end
                
            case 'three'
                switch S.kind
                    case 'ch'
                        to_delete = [2,4]; % 1:5
                    case 'rt'
                        to_delete = [3:4, 7:8]; % 1:10
                end
        end
        delete(hs.nonsegment(to_delete));
        hs.nonsegment(to_delete) = [];
    end
end
end