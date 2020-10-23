classdef FindOutlierRuns < Fit.D2.Common.CommonWorkspace    
%% Setting
properties
    % to_excl_outlier_runs: preset controlling all other settings.
    % 'n'|0: no exclusion
    % 't'|1: excl based on # trials only (thres_n_tr_in_run=150)
    % 'd': ntr + thres_method=FDR_run + shuf_method=trial + FDR=0.01
%     to_excl_outlier_runs = 'ntr'; % defined in CommonWorkspace
    
    FDR = 0.01; % False discovery rate
    
    % 'thres_method'
    % : 'FWE_expr' : FWE across runs within experiment
    % : 'FDR_run' : FDR per individual run
    thres_method = 'FDR_run';
    
    % 'shuf_method'
    % : 'trial' : shuffle trials separately
    % : 'cond' : shuffle condition between runs
    shuf_method = 'trial';
    
    n_shuf = 1e3;
    thres_n_tr_in_run = 150;
end
%% Results
properties
    ix_run_to_excl = [];
end
%% Main
methods
    function W = FindOutlierRuns(varargin)
        if nargin > 0
            W.init(varargin{:});
        end
    end
    function batch(W0, varargin)
        S_batch = varargin2S(varargin, {
            'subj', Data.Consts.subjs_RT
            'parad', 'RT'
            'task', {'A', 'H', 'V'}
            });
        [Ss, n] = bml.args.factorizeS(S_batch);
        for ii = 1:n
            S = Ss(ii);
            C = S2C(S);
            W = feval(class(W0), C{:});
            W.main;
        end
    end
    function main(W)        
        W.ix_run_to_excl = [];
        W.exclude_outliers(W.to_excl_outlier_runs);
        W.save_mat;
    end
    function exclude_outliers(W, op)
        if nargin < 2
            op = W.to_excl_outlier_runs;
        end
        switch op
            case {'n', 0}
                % Do nothing
            case {'t', 1}
                W.exclude_by_ntr;
            case 'tr'
                W.exclude_by_ntr;
                W.exclude_by_rt;
            case 'd'
                W.exclude_outliers('t');
                
                W.FDR = 0.01;
                W.thres_method = 'FDR_run';
                W.shuf_method = 'trial';
            otherwise
                error('Illegal to_excl_outlier_runs=%s', ...
                    W.to_excl_outlier_runs);
        end
    end
    function exclude_by_ntr(W)
        clf;
        ix_run_few_tr = W.thres_by_n_tr_in_run;
        file = W.get_file({'plt', 'n_tr_in_run'});
        savefigs(file);
        
        W.ix_run_to_excl = union(W.ix_run_to_excl(:), ix_run_few_tr(:));
    end
    function exclude_by_rt(W)
        clf;
        ix_run_outlier_rt = W.find_outlier_run;
        file = W.get_file({'plt', 'log_p_run'});
        savefigs(file);
        
        W.ix_run_to_excl = union(W.ix_run_to_excl(:), ix_run_outlier_rt(:));
    end
    function save_mat(W)
        L = struct;
        L.W = W;
        L = bml.oop.copyprops(L, W, 'props', {
            'ix_run_to_excl'
            }); %#ok<NASGU>
        file = W.get_file({}, {'eor'});
        mkdir2(fileparts(file));
        save(file, '-struct', 'L');
        fprintf('Saved to %s\n', file);
    end
    function [ix_run_to_excl, L] = load_mat(W)
        file = [W.get_file({}, {'eor'}), '.mat'];
        
        if exist(file, 'file')
            L = load(file);
            ix_run_to_excl = L.ix_run_to_excl;
            fprintf('Loaded ix_run_to_excl (%d entries) from %s\n', ...
                numel(ix_run_to_excl), file);
        else
            ix_run_to_excl = [];
            L = struct;
            L.ix_run_to_excl = [];
            fprintf(['Returning an empty ix_run_to_excl ' ...
                'because no flie is found at %s\n'], file);
        end
    end
end
%% Stats
methods
    function ix_run_outlier0 = find_outlier_run(W, varargin)
        S = varargin2S(varargin, ...
            copyFields(struct, W, {
                'FDR'
                'thres_method'
                'shuf_method'
                'n_shuf'
                }));

        %% Get shuffled p
        run_orig = unique(W.Data.ds.i_all_Run);        
        rng('shuffle');
        
        n_shuf = S.n_shuf;
        for i_shuf = n_shuf:-1:1
            C = varargin2C({
                'to_shuffle', i_shuf > 1
                }, S);
            p(:,:,i_shuf) = W.p_RT_per_cond(C{:});
        end
        
        %% Threshold
        log_p = log(p);
        log_p_run0 = squeeze(nanmean(log_p, 2)); % (run, shuf)
        
        n_run = size(p, 1);
        
        switch S.thres_method
            case 'FWE_expr'
                %%
                [log_p_run, ix_run] = sort(log_p_run0);
                p_FDR = nanmean(bsxfun(@ge, ...
                    log_p_run(:,1), ...
                    log_p_run(:,2:end)), 2);
                n_run_outlier = find(p_FDR >= S.FDR, 1, 'first') - 1;
                ix_run_outlier = ix_run(1:n_run_outlier, 1); 
                
                disp(p_FDR);
                
                plot(run_orig, log_p_run);

            case 'FDR_run'
                %%
                ix_shuf0 = repmat(1:n_shuf, [n_run, 1]);
                ix_run0 = repmat((1:n_run), [1, n_shuf]);

                log_p_run = log_p_run0(:);
                ix_shuf = ix_shuf0(:);
                ix_run = ix_run0(:);
                tbl0 = table(log_p_run, ix_shuf, ix_run);
                tbl0 = sortrows(tbl0, 'log_p_run');
                tbl0.FDR = cumsum(tbl0.ix_shuf > 1) ...
                    ./ nnz(tbl0.ix_shuf > 1);

                ix_cutoff = find(tbl0.FDR < S.FDR, 1, 'last');

                tbl = tbl0(1:ix_cutoff, :);
                ix_run_outlier = tbl.ix_run(tbl.ix_shuf == 1);
                log_p_run_cutoff = tbl.log_p_run(end);
                n_run_outlier = length(ix_run_outlier);
                
                disp(log_p_run_cutoff);
                disp(n_run_outlier);
                disp(ix_run_outlier);

                plot(run_orig, log_p_run0(:,1), 'o-');
                crossLine('h', log_p_run_cutoff);
        end
        
        %% Display
        fprintf('# Outlier runs: %d/%d\n', n_run_outlier, n_run);
        fprintf('Outlier runs:');
        
        ix_run_outlier0 = run_orig(ix_run_outlier);
        fprintf(' %d', ix_run_outlier0);
        fprintf('\n');
    end
    function ix_run_few_tr = thres_by_n_tr_in_run(W, thres_n_tr)
        if nargin < 2
            thres_n_tr = W.thres_n_tr_in_run;
        end
        
        filt = W.Data.get_dat_filt_numeric;
        ix_run = W.Data.ds.i_all_Run;
        
        n_tr_in_run = accumarray(ix_run, 1, [], @nnz);
        
        ix_run_few_tr = vVec(find(n_tr_in_run < thres_n_tr));
        n_tr_excl = n_tr_in_run(ix_run_few_tr);
        ds = dataset(ix_run_few_tr, n_tr_excl);
        fprintf('Runs to exclude due to too few trials (< %d tr, %d runs):', ...
            thres_n_tr, numel(ix_run_few_tr));
        disp(ds);
        fprintf('\n');
        
        plot(1:max(ix_run), n_tr_in_run, 'ro-');
        crossLine('h', thres_n_tr);
        ylabel('# Trial in run');
        xlabel('Run');
        text(0, thres_n_tr, 'Threshold # trial', ...
            'VerticalAlignment', 'top');

        % Filter data
        incl_n_tr = find(n_tr_in_run >= thres_n_tr);
        filt_n_tr = ismember(ix_run, incl_n_tr);
        
        W.Data.set_filt_spec(filt(filt_n_tr));
        W.Data.filt_ds;
    end
    function p = p_RT_per_cond(W, varargin)
        % p = p_RT_per_cond(W, varargin)
        %
        % cond, ix_run, rt(tr, 1)
        % : column vectors.
        %
        % p(run, cond) 
        % : p-value of RTs in (run, cond) having the same median as RTs 
        %   from the same condition in other runs.
        
        S = varargin2S(varargin, {
            'to_shuffle', false
            'to_plot', false
            'shuf_method', 'trial'
            'thres_n_tr_in_run', 150
            });
        
        %%
        ds = W.Data.ds;
        [~,~,cond] = unique(abs(ds.cond), 'rows');
        rt = ds.RT;
        [~,~,ix_run] = unique(ds.i_all_Run);
                
        n_run = max(ix_run);
        n_cond = max(cond);
        
        %%
        if S.to_shuffle
            switch S.shuf_method
                case 'trial'
                    for i_cond = 1:n_cond
                        incl = find(cond == i_cond);
                        ix_run(incl) = ...
                            ix_run(incl(randperm(length(incl))));
                    end
                    
                case 'cond'
                    for i_cond = 1:n_cond
                        ix_run1 = randperm(n_run);
                        
                        ix_run0 = ix_run;
                        for i_run = 1:n_run
                            incl = (cond == i_cond) & (ix_run0 == i_run);
                            ix_run(incl) = ix_run1(i_run);
                        end
                    end
            end
        end
        
        %%
        clear p
        for i_run = n_run:-1:1
            for i_cond = 1:n_cond
                incl_cond = cond == i_cond;
                incl_run = i_run == ix_run;
                
                incl1 = incl_cond & incl_run;
                incl_rest = incl_cond; % & ~incl_run;
                
                if any(incl1) && any(incl_rest)
                    rt1 = rt(incl1);
                    rt_rest = rt(incl_rest);
                    p(i_run, i_cond) = ranksum(rt1, rt_rest);
                else
                    p(i_run, i_cond) = nan;
                end
            end
        end
        
        %%
        if S.to_plot
            ecdf(p(:));
            axis square
            set(gca, 'TickDir', 'out', 'Box', 'off');
            grid on;
            crossLine('NE', 0);
        end
        
        %%
%         log_p = log(p);
%         log_p_run = nansum(log_p, 2);
    end
end
end
