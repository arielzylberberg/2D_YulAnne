classdef EnCosBases < Fit.Common.EnCosBases & Fit.D2.Common.CommonWorkspace
    % Fit.D2.RT.En.EnCosBases
    %
    % 2015 YK wrote the initial version
properties (Access = protected)
    width_per_cycle_sec = 0.2;
    to_truncate_first_sec = [0.2, 0];
    to_truncate_last_sec = [0.2, 0];
end
methods
    function set_Data(W, obj_or_name)
        if nargin < 2, obj_or_name = Fit.D2.Common.DataChRtPdfEn; end
        obj_or_name = ...
            W.enforce_class('Fit.D2.Common.DataChRtPdfEn', ...
                obj_or_name);
        W.set_Data@Fit.D2.Common.CommonWorkspace(obj_or_name);
        W.adapt_Data;
    end
    function adapt_Data(W)
        n_dim = W.Data.get_n_dim;
        for dim = 1:n_dim
            ts_mat = W.Data.Ens{dim}.get_ts_mat( ...
                'truncate_first_sec', W.to_truncate_first_sec(dim), ...
                'truncate_last_sec',  W.to_truncate_last_sec(dim));
            
            W.Bases{dim} = zEn.CosBases( ...
                'dat', ts_mat, ...
                'wavelength_in_bin', W.get_wavelength_in_bin);
            W.Bases{dim}.get_wt;
        end
    end
end
%% Get/Set
methods
    function set_width_per_cycle_sec(W, v)
        W.width_per_cycle_sec = v;
    end
    function v = get_wavelength_in_sec(W)
        v = W.width_per_cycle_sec;
    end
    
    function v = get_wavelength_in_bin(W)
        v = W.get_wavelength_in_sec / W.Data.get_dt;
    end
    
    function set_to_truncate_first_sec(W, v)
        W.to_truncate_first_sec = v;
    end
    function v = get_to_truncate_first_sec(W)
        v = W.to_truncate_first_sec;
    end

    function set_to_truncate_last_sec(W, v)
        W.to_truncate_last_sec = v;
    end
    function v = get_to_truncate_last_sec(W)
        v = W.to_truncate_last_sec;
    end
end
%% Demo
methods
    function demo(W)
        %%
        W = Fit.D2.RT.En.EnCosBases();
        W.Data.set_path;
        
        %%
        W.Data.load_data;
        W.adapt_Data;
        
        %%
        n_dim = W.Data.get_n_dim;
        for dim = 1:n_dim
            Bases = W.Bases{dim};
            En = W.Data.Ens{dim};

            %%
            subplotRC(n_dim, 1, dim, 1);
            Bases.plot_recon_dat('tr', 400);
        end
        
        %%
    end
end
end