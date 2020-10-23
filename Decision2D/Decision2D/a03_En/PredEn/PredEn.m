classdef PredEn < FitWorkspace
%%
properties
    Fls = {[], []};
end
properties (Dependent)
    Ws
end
properties 
    n_dim = 2;
end
%% Params
properties
    p_dim1_1st = 1;
end
%% Init
methods
    function PE = PredEn(varargin)
        if nargin > 0
            PE.init(varargin{:});
        end
    end
    function init(PE, Fls, varargin)
        PE.Fls = Fls;
        varargin2props(PE, varargin);
    end    
    function Ws = get.Ws(PE)
        Ws = cell(1, PE.n_dim);
        for dim_rel = 1:PE.n_dim
            Fl1 = PE.Fls{dim_rel};
            if ~isempty(Fl1)
                W1 = Fl1.W;
            else
                W1 = [];
            end
            Ws{dim_rel} = W1;
        end
    end
    function set.Ws(PE, Ws)
        assert(iscell(Ws));
        assert(numel(Ws) == PE.n_dim);
        for dim_rel = 1:PE.n_dim
            Fl1 = PE.Fls{dim_rel};
            W1 = Ws{dim};
            if ~isempty(Fl1)
                Fl1.W = W1;
            else
                Fl1 = W1.get_Fl;
            end
            PE.Fls{dim_rel} = Fl1;
        end
    end
end
%% Pred
methods
    function [rt, ch] = pred_dim(PE, dim_rel)
        %% Sample RT & choice from RT_pred_pdf_tr;
        W = PE.Ws{dim_rel};
        p0 = W.Data.RT_pred_pdf_tr;
        p0 = permute(p0, [2, 1, 3]); % [tr, rt, ch]
        siz = size(p0);
        n_tr = siz(1);
        n_bins = siz(2) * siz(3);
        p = reshape(p0, [n_tr, n_bins]);        
        rt = zeros(n_tr, 1);
        ch = zeros(n_tr, 1);
        
        for tr = 1:n_tr
            rt_ch = randsample(n_bins, 1, true, p(tr, :));
            [rt(tr), ch(tr)] = ind2sub(siz(2:3), rt_ch);
        end
    end
end
end