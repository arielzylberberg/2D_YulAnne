classdef TndSt < Fit.Common.Tnd
    % Fit.D2.RT.BoundedCondEn.TndSt
    %
    % 2015 YK wrote the initial version.
properties (SetAccess = protected)
    Tnd1
    Tnd2
end
properties (Dependent)
    Tnds
end
methods
    function W = TndSt
        W.add_deep_copy({'Tnd1', 'Tnd2'});
        
        W.remove_params_all;
        
        W.set_Data;
        W.set_Tnd1;
        W.set_Tnd2;
    end
    function pred(W)
        % Do not update RT_pred_pdf. Use get_pdf_tnd instead.
    end
    function pdf_tnd = get_pdf_tnd(W, ix_dim)
        % pdf_tnd = get_pdf_tnd(W, ix_dim = 1:2)
        % pdf_tnd : nt x n_dim
        if ~exist('i_dim', 'var'), ix_dim = 1:2; end
        
        n_dim = 2;
        pdf_tnd = zeros(W.nt, n_dim);
        
        for i_dim = 1:n_dim
            pdf_tnd(:,i_dim) = W.Tnds{i_dim}.get_pdf_tnd;
        end
        
        pdf_tnd = pdf_tnd(:, ix_dim);
    end
    %% Get/Set
    function set_Data(W, obj_or_name)
        if nargin < 2, obj_or_name = ''; end
        obj_or_name = ...
            W.enforce_class('Fit.D2.Common.DataChRtPdfEn', ...
                obj_or_name);
        W.set_Data@Fit.Common.Tnd(obj_or_name);
    end
    function Tnds = get.Tnds(W)
        Tnds = {W.Tnd1, W.Tnd2};
    end
    function set.Tnds(W, Tnds)
        assert(iscell(Tnds));
        assert(numel(Tnds) == 2);
        W.set_Tnd1(Tnds{1});
        W.set_Tnd2(Tnds{2});
    end
    function set_Tnd1(W, obj_or_name)
        if nargin < 2, obj_or_name = ''; end
        W.Tnd1 = W.enforce_class('Fit.Common.Tnd', obj_or_name);
        W.set_sub_from_props({'Tnd1'});
    end
    function set_Tnd2(W, obj_or_name)
        if nargin < 2, obj_or_name = ''; end
        W.Tnd2 = W.enforce_class('Fit.Common.Tnd', obj_or_name);
        W.set_sub_from_props({'Tnd2'});
    end
end
end