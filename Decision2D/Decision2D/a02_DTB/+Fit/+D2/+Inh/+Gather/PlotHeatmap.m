classdef PlotHeatmap < FitWorkspace
properties
    subj = Data.Consts.subjs_RT{1};
    parad = 'RT';
    
    Loc = Fit.D2.Inh.Locator;

    slice_var = 'p_dim1_1st';
    x_var = 'drift_fac';
    y_var = 'sigmaSq_fac';
    
    rep = varargin2S({
        'p_dim1_1st', (0:25:100) / 100
        'drift_fac', (0:25:100) / 100
        'sigmaSq_fac', (0:25:200) / 100
        });
    
    z_kind = 'fval'; % 'fval', 'aic', 'bic', 'aic_c'
    
    ds_res = dataset;
end
methods
    function Pl = PlotHeatmap(varargin)
        if nargin > 0
            Pl.init(varargin{:});
        end
    end
    function plot(Pl)
        slice_val = Pl.get_slice_val;
        n_slice = Pl.get_n_slice;
        
        for i_slice = 1:n_slice
            subplot(1,n_slice, i_slice);
            Pl.plot_slice(slice_val(i_slice));
        end
    end
    function varargout = plot_SNR_1_drift_fac_0(Pl)
        Pl.slice_var = 'SNR';
        Pl.y_var = 'drift_fac';
        Pl.x_var = 'p_dim1_1st';
        [varargout{1:nargout}] = Pl.plot_line(1, 0);
    end
    function varargout = plot_SNR_1(Pl)
        Pl.slice_var = 'SNR';
        Pl.x_var = 'p_dim1_1st';
        Pl.y_var = 'drift_fac';
        [varargout{1:nargout}] = Pl.plot_slice(1);
    end
    function highlight(Pl)
    end
    function permute(Pl, order)
        % order: [slice, x, y]
        assert(isnumeric(order));
        assert(isvector(order));
        assert(length(order) == 3);
        
        vars0 = fieldnames(Pl.rep);
        vars = vars0(order);
        Pl.slice_var = vars{1};
        Pl.x_var = vars{2};
        Pl.y_var = vars{3};
    end
end
methods
    function find_min(Pl)
        ds = Pl.ds_res;

        incl = find(ds.SNR <= 1);
        [~, ix] = min(ds.z(incl));
        ix = incl(ix);

%         [~, ix] = min(ds.z);

        disp(ds(1,:));
        disp(ds(ix,:));
    end
    function [ds_better, incl_better, ix_best_d0s0] = ...
            find_better_than_drift_0_sigmaSq_0(Pl)
        ds = Pl.ds_res;
        incl_d0s0 = find((ds.drift_fac == 0) & (ds.sigmaSq_fac == 0));
        [z0, ix_best_d0s0] = min(ds.z(incl_d0s0));
        ix_best_d0s0 = incl_d0s0(ix_best_d0s0);
        
        incl_SNR = ds.SNR <= 1;
        incl_better = incl_SNR & ds.z <= z0;
        ds_better = ds(incl_better,:);
        
        disp(ds(ix_best_d0s0, :));
        disp(ds_better);
    end
    function [z, ds] = get_z(Pl)
        ds = Pl.ds_res;
        [~, ~, ix_1st] = unique(ds.p_dim1_1st);
        [~, ~, ix_drift] = unique(ds.drift_fac);
        [~, ~, ix_sigma] = unique(ds.sigmaSq_fac);
        z = accumarray([ix_1st, ix_drift, ix_sigma], ds.z);
    end
end
methods (Hidden)
    function init(Pl, varargin)
        varargin2props(Pl, varargin);
        Pl.Loc = Fit.D2.Inh.Locator( ...
            'subj', Pl.subj, ...
            'parad', Pl.parad);
        Pl.load_z;
    end
    function plot_line(Pl, slice_val, y_val)
        [z, x] = Pl.get_line(slice_val, y_val);
        plot(x, z);
        xlabel(strrep(Pl.x_var, '_', '-'));
        ylabel(Pl.z_kind);
        set(gca, 'XTick', x);
        title(sprintf('%s = %g\n%s = %g', ...
            strrep(Pl.slice_var, '_', '-'), slice_val, ...
            strrep(Pl.y_var, '_', '-'), y_val));
    end
    function [h, z, x, y] = plot_slice(Pl, slice_val)
        [z, x, y] = Pl.get_slice(slice_val);
        h = contourf(x, y, z');
        axis xy;
        set(gca, 'XTick', x, 'YTick', y);
        xlabel(strrep(Pl.x_var, '_', '-'));
        ylabel(strrep(Pl.y_var, '_', '-'));
        title(sprintf('Color: %s\n%s = %g', ...
            Pl.z_kind, ...
            strrep(Pl.slice_var, '_', '-'), ...
            slice_val));
    end
    function [z, x_val] = get_line(Pl, slice_val, y_val)
        incl = (Pl.ds_res.(Pl.slice_var) == slice_val) ...
             & (Pl.ds_res.(Pl.y_var) == y_val);
        ds = Pl.ds_res(incl, :); 
         
        [x_val, i_x] = unique(ds.(Pl.x_var));
        n_x = max(i_x);
        z = ds.z;
        
        z = accumarray(i_x, z, [n_x, 1], @nanmean);
    end
    function [z, x_val, y_val] = get_slice(Pl, slice_val)
        % [z, x_val, y_val] = get_cost_slice(Pl, slice_var, slice_val, x_var, y_var)
        %
        % z(x,y) : a matrix of value with slice_var = slice_val, 
        %             x_var = x_val(x), y_var = y_val(y).
        
        incl = Pl.ds_res.(Pl.slice_var) == slice_val;
        ds = Pl.ds_res(incl, :);
        
        [x_val, ~, i_x] = unique(ds.(Pl.x_var));
        [y_val, ~, i_y] = unique(ds.(Pl.y_var));
        z = ds.z;
        n_xy = [max(i_x), max(i_y)];
        z = accumarray([i_x, i_y], z, n_xy, @nanmean);
    end
    function load_z(Pl)
        rep_all = Pl.get_rep_all;
        Pl.ds_res = struct2dataset(rep_all);
        
        n = length(Pl.ds_res);
        for ii = 1:n
            Pl.ds_res.z(ii,1) = Pl.load_z_unit(rep_all(ii));
        end
        
        Pl.load_z_postprocess;
    end
    function load_z_postprocess(Pl)
        Pl.ds_res.SNR = Pl.ds_res.drift_fac ./ Pl.ds_res.sigmaSq_fac;
        Pl.ds_res.SNR((Pl.ds_res.drift_fac == 0) ...
                    & (Pl.ds_res.sigmaSq_fac == 0)) = 1;        
    end
    function [z, L] = load_z_unit(Pl, rep)
        file = Pl.get_file_unit(rep);
        L = load(file, 'res');
        L.file = file;
        z = L.res.(Pl.z_kind);
    end
    function file = get_file_unit(Pl, rep)
        C = varargin2C(rep);
        file = Pl.Loc.get_file_full(C{:});
    end
    function rep_all = get_rep_all(Pl)
        rep_all = factorizeS(Pl.rep);
    end
    function v = get_slice_val(Pl)
        v = Pl.rep.(Pl.slice_var);
    end
    function v = get_n_slice(Pl)
        v = length(Pl.get_slice_val);
    end
end
end