classdef CompareModels < FitData
properties (SetAccess = protected)
    % Given
    %   S_filt.(f1) = {v1, v2}
    %   S_filt.(f2) = {v3, v4}
    % chooses (ds.(f1) = v1 or v2) and (ds.(f2) = v3 or v4) ...
    S_filt = struct;
end
properties
    % path_fit : a folder within 'Data' whos *.mat files are
    %            named with field1=val1+... format (Serializer format)
    %            and contain 'res' struct with variables of gof_names.
    path_fit = 'Fit.D2.Inh.MainBatch';
    
    % gof_names : goodness of fits to load from 'res' struct.
    gof_names = {'fval', 'bic', 'aic', 'aic_c'};
    
    criteria_file = {
        'allof', {
            'tnd=gamma'
            'ntnd=4'
            'cv=0'
            }
        };
end
properties (Dependent)
    files_fit = {};
end
%% Facades - Comparing
methods
    function [res, res0, S] = compare_fano_vs_ser(Comp, varargin)
        S = varargin2S(varargin, {
            'min', 'fval'
            'subjs', hVec(unique(Comp.ds0.subj))
            'max_fano', 50
            });
        
        %%
        res_ser = Comp.find_ser;
        res_fano = Comp.find_fano;

        res_ser.fn1 = 0;
        res_ser.fn2 = 0;
%         res_ser.is_ser = true;
%         res_fano.is_ser = false;
        
        res0 = [
            res_ser
            res_fano
            ];
        
        res0.cval(isnan(res0.cval)) = 0;
        
        incl_fano = (res0.fn1 <= S.max_fano) & (res0.fn2 <= S.max_fano);
        res0 = res0(incl_fano, :);
        
        %%
        res = [];
        for subj = S.subjs
            res1 = bml.ds.find(res0, {'sbj', subj{1}, 'cv', 0});
            [~, ix] = min(res1.(S.min));
            res = [res; res1(ix,:)]; %#ok<AGROW>
        end
    end
    function plot_fval_vs_drift_fac(Comp, varargin)
        [~, res0, S] = Comp.compare_fano_vs_ser(varargin{:});
        
        %%
        n_subj = numel(S.subjs);
        for i_subj = 1:n_subj
            subj = S.subjs{i_subj};
            subplot(n_subj, 1, i_subj);
            cla;
            
            res1 = bml.ds.find(res0, {'sbj', subj, 'cv', 0});
%             res1 = bml.ds.find(res0, {'sbj', subj{1}});
            
            
            plot(res1.fn1, res1.fval, 'o'); 
%             plot(res1.th_Dtb__drift_fac_together_dim2_1, res1.fval, 'o'); 
%             plot(res1.th_Dtb__drift_fac_together_dim1_2, res1.fval, 'o'); 

%             hold on;
%             input(':', 's');
        end
    end
    function [res, ix] = find_ser(Comp)
        [res, ix] = bml.ds.find(Comp.ds0, {
            'p1', 0
            'd1', 0
            's1', 0
            'd2', 0
            's2', 0
            ...
            'pf', 1
            'd1f', 1
            's1f', 1
            'd2f', 1
            's2f', 1
            ...
            'dtb', 'DnIvJt'
            });
    end
    function res = find_fano(Comp)
        res = [];
        for fano = [50 75 90 95 100]
            res = [
                res
                bml.ds.find(Comp.ds0, {
                    'fn1', fano
                    'cv', 0
                    })
                ]; %#ok<AGROW>
        end
    end
end
%% Initialization
methods
    function Comp = CompareModels(varargin)
        Comp.dat_filt_spec = ':';
        if nargin > 0
            Comp.init(varargin{:});
        end
    end
    function init(Comp, varargin)
        bml.oop.varargin2props(Comp, varargin, true);
        
        file = Comp.get_path;
        
        if exist(file, 'file')
            Comp.load_data;
        else
            Comp.reload;
        end
    end
    function reload(Comp)
        Comp.calc_ds;
        Comp.load_fit;        
        Comp.loaded = true;
    end
end
%% File list
methods
    function calc_ds(Comp)
        if isempty(Comp.ds0)
            Comp.files_fit = 'auto';
        end
        
        fs = Comp.ds0.Properties.VarNames;
        
        for f = fs(:)'
            v0 = Comp.ds0.(f{1});
            
            if all(cellfun(@(v) isnumeric(v) || isempty(v), v0))
                Comp.ds0.(f{1}) = cell2mat2(v0);
            end
        end
    end
    function v = get.files_fit(Comp)
        if ~isempty(Comp.ds0) && isdscolumn(Comp.ds0, 'file_fit')
            v = Comp.ds0.file_fit;
        else
            v = {};
        end
    end
    function set.files_fit(Comp, v)
        if ischar(v)
            if strcmp(v, 'auto')
                path_fit = fullfile('Data', Comp.path_fit, '*.mat');
            else
                path_fit = v;
            end
            fprintf('Getting fits from %s ..\n', path_fit);
        else
            assert(iscell(v));
            assert(all(cellfun(@ischar, v)));
            
            path_fit = v;
        end
        
        S2s = bml.str.Serializer;
        [Comp.ds0, files] = S2s.ls2ds(path_fit, ...
            varargin2C(Comp.criteria_file));
        Comp.ds0.file_fit = files;
    end
    function ds = find0(Comp, varargin)
        ds = bml.ds.find(Comp.ds0, varargin);
    end
    function succ = load_fit(Comp)
        n = size(Comp.ds0, 1);
        res = struct;
        th = struct;
        files = Comp.ds0.file_fit;
        gof_names = Comp.gof_names(:)'; % goodness-of-fit
        
        succ = false(n, 1);
        
        fprintf('Loading fits from %d files\n', n);
        for ii = 1:n
            try
                L = load(files{ii}, 'res');
                for gof = gof_names
                    res.(gof{1})(ii, 1) = L.res.(gof{1});
                end
                
            catch % err
%                 warning(err_msg(err));
                fprintf('x');
                
                for gof = gof_names
                    res.(gof{1})(ii, 1) = nan;
                end
                continue;
            end
            
            th0 = bml.struct.prefix_fields(L.res.th, 'th_');
            for f = fieldnames(th0)'
                th.(f{1}){ii, 1} = th0.(f{1});
            end
            
            succ(ii) = true;
            
            if mod(ii, 10) == 0, fprintf('.'); end
            if mod(ii, 100) == 0, fprintf('%d\n', ii); end
        end
        fprintf('Done.\n');
        fprintf('Successfully loaded %d/%d results.\n', ...
            nnz(succ), n);
        
        for gof = gof_names
            Comp.ds0.(gof{1}) = res.(gof{1});
        end
        
        for f = fieldnames(th)'
            if length(th.(f{1})) < n
                th.(f{1}){n,1} = nan;
            end
            th.(f{1}) = cell2mat2(th.(f{1}));
            
            Comp.ds0.(f{1}) = th.(f{1});
        end
    end
end
%% Querrying
methods
    function [z, x, y, ix, w] = min_slice2D(Comp, x_name, y_name, z_name, ...
            w_name)
        x0 = Comp.ds.(x_name);
        y0 = Comp.ds.(y_name);
        z0 = Comp.ds.(z_name);
        
        [x, ~, xi] = uniquenan(x0);
        [y, ~, yi] = uniquenan(y0);
        
        nx = numel(x);
        ny = numel(y);
        
        z = nan(nx, ny);
        ix = nan(nx, ny);
        
        for ii = 1:nx
            for jj = 1:ny
                incl0 = (xi == ii) & (yi == jj);
                [z(ii, jj), ix(ii, jj)] = min(z0(incl0));
                
                incl = find(incl0);
                ix(ii, jj) = incl(ix(ii, jj));
            end
        end
        
        if exist('w_name', 'var')
            w0 = Comp.ds.(w_name);
            w = w0(ix);
        end
    end
    function [z, x, ix, w] = min_slice1D(Comp, x_name, z_name, w_name)
        x0 = Comp.ds.(x_name);
        z0 = Comp.ds.(z_name);
        
        [x, ~, xi] = uniquenan(x0);
        
        nx = numel(x);
        
        z = nan(nx, 1);
        ix = nan(nx, 1);
        
        for ii = 1:nx
            incl0 = (xi == ii);
            [z(ii, 1), ix(ii, 1)] = min(z0(incl0));

            incl = find(incl0);
            ix(ii, 1) = incl(ix(ii, 1));
        end
        
        if exist('w_name', 'var')
            w0 = Comp.ds.(w_name);
            w = w0(ix);
        end
    end
end
%% Filtering
methods
    function filt = get_dat_filt(Comp)
        filt = Comp.get_dat_filt@FitData;
        for f = fieldnames(Comp.S_filt)'
            v = hVec(Comp.S_filt.(f{1}));
            if ischar(v)
                v = {v};
            end
            
            filt = filt & bml.matrix.bsxIsequal( ...
                Comp.ds0.(f{1}), v);
        end
    end
    function set_S_filt_field(Comp, f, v)
        assert(ischar(f));
        assert(iscell(v));
        Comp.S_filt.(f) = v;
    end
end
%% Plotting - for fixed parameters
methods
    function [h, w, x, y] = imagesc(Comp, x_name, y_name, z_name, w_name)
        if ~exist('w_name', 'var')
            w_name = z_name;
        end
        [~, x, y, ~, w] = Comp.min_slice2D(x_name, y_name, z_name, w_name);
        
        h = imagesc(x, y, w');
        axis xy;
        
        xlabel(x_name);
        ylabel(y_name);
        title(sprintf('%s at min(%s)', w_name, z_name));
    end
    function [h, w, x] = plot(Comp, x_name, z_name, w_name)
        if ~exist('w_name', 'var')
            w_name = z_name;
        end
        [~, x, ~, w] = Comp.min_slice1D(x_name, z_name, w_name);
        
        h = plot(x, w');
        
        xlabel(x_name);
        ylabel(z_name);
        title(sprintf('%s at min(%s)', w_name, z_name));
    end
end
%% Plotting - for free parameters
properties
    column = varargin2S({
        'drtfac1', 'Dtb__drift_fac_together_dim1_2'
        'drtfac2', 'Dtb__drift_fac_together_dim2_1'
        'ssqfac1', 'Dtb__sigmaSq_fac_together_dim1_2'
        'ssqfac2', 'Dtb__sigmaSq_fac_together_dim2_1'
        });
end
methods
    function plot_st_en(Comp, varargin)
        S = varargin2S(varargin, {
            'filt', {}
            'th1', 'drtfac1'
            'th2', 'drtfac2'
            });
        S.filt = varargin2S(S.filt, {
            'sbj', Data.Consts.subjs_RT{1}
            });
        
        res = Comp.ds0;
        
        Pl = Fig.PlotStEn;
        Pl.plot(res, Comp.column.(S.th1), Comp.column.(S.th2), ...
            'filt', S.filt);
    end
end
%% Minimum entries for each subject
methods
    function ds = get_min_rows_batch(Comp, S_batch, z_name)
        if ~exist('S_batch', 'var')
            S_batch = struct;
        end
        S_batch = varargin2S(S_batch, {
                'sbj', Data.Consts.subjs_RT
                'cv', {1}
                });
        
        [Ss, n] = bml.args.factorizeS(S_batch);
        if ~exist('z_name', 'var')
            z_name = 'fval';
        end
        
        ds = dataset;
        for ii = 1:n
            S = Ss(ii);
            Comp.S_filt = S;
            Comp.filt_ds;
            rows = Comp.get_min_rows(z_name);
            
            ds = [ds; rows]; %#ok<AGROW>
        end
        
        %% Save
        S_batch.z_name = z_name;
        file = [Comp.get_file(S_batch) '.csv'];
        mkdir2(fileparts(file));
        export(ds, 'File', file, 'delimiter', ',');
        fprintf('Saved to %s\n', file);
    end
    function ds = get_min_rows(Comp, z_name)
        if ~exist('z_name', 'var')
            z_name = 'bic';
        end
        
        z_min = min(Comp.ds.(z_name));
        ix = Comp.ds.(z_name) == z_min;
        ds = Comp.ds(ix, :);
    end
end
%% Saving
methods
    function save_ds0(Comp)
        file = [Comp.get_file(varargin2S({
            'src', 'ds0'
            })) '.csv'];
        export(Comp.ds0, 'File', file, 'delimiter', ',');
        fprintf('Saved ds0 to %s\n', file);
    end
    function save_ds(Comp)
        file = [Comp.get_file(varargin2S({
            'src', 'ds'
            })) '.csv'];
        export(Comp.ds, 'File', file, 'delimiter', ',');
        fprintf('Saved ds to %s\n', file);
    end
    function file = get_file(Comp, S)
        if ~exist('S', 'var')
            S = varargin2S({'type', 'default'});
        end
        S2s = bml.str.Serializer;
        nam = S2s.convert(S);
        file = fullfile('Data', class(Comp), nam);
    end
    function v = get_path(Comp)
        pth = Comp.get_path@FitData;
        
        if isempty(pth)
            Comp.path = fullfile('Data', class(Comp), Comp.path_fit, ...
                'Comp');
        end
        v = Comp.path;
    end
end
end