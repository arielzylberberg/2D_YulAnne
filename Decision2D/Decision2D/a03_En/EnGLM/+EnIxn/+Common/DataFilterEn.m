classdef DataFilterEn < Fit.D2.Common.DataFilterEn
properties
    fr_st = 1;
    fr_en_ = [];
    dfr = 1;
end
properties (Dependent)
    t_st % linked to fr_st
    t_en % linked to fr_en
    t_st_en % [t_st, t_en]. Used in batch.
    
    % fr_en:
    % If isempty(fr_en_), set automatically from truncate_last_msec
    % Note that fr_en is *excluded* from fr_incl,
    % i.e., fr_incl = fr_st:dfr:(fr_en - 1),
    % so that when I set t_st = 0, t_en = 0.5 in one condition and
    % t_st = 0.5 and t_en = 1 in another condition,
    % the two are guaranteed not to have an overlap.
    fr_en 
    fr_st_en % [fr_st, fr_en]
    fr_incl % fr_st:fr_en
    t_incl % [t_st, t_en]
end
%% User interface
methods
    function W = DataFilterEn(varargin)
        W.dif_rel_incl = 'all'; % 1:3;
        W.dif_irr_incl = 'all'; % 1:3;
        W.accu_rel_incl = [0 1];
        W.accu_irr_incl = [0 1];    
        
        % t_plot_max:
        % Defaults to longer than longest 10 percentile RT across subj
        W.t_plot_max = 1.2; 
        
        W.truncate_first_msec = 150; % 200; % Previously -inf.
        W.truncate_last_msec = 0; % 100; % Previously -inf which included tapering end.

        if nargin > 0
            W.init(varargin{:});
        end
    end
end
%% Time filtering
methods
    function v = get.fr_incl(W)
        v = W.fr_st:W.dfr:(W.fr_en - 1);
    end
    function set.fr_incl(W, v)
        if isempty(v)
            W.fr_st = [];
            W.fr_en = [];
            W.dfr = [];
        else
            W.fr_st = v(1);
            W.fr_en = (v(end) + 1);
        
            if numel(v) >= 2
                W.dfr = v(2) - v(1);
            end
        end
    end
    function v = get.t_incl(W)
        v = W.fr_incl / W.refresh_rate;
        switch W.t0_kind
            case 'st'
                v = v + W.truncate_first_msec / 1e3;
            case 'en'
                v = v + W.truncate_last_msec / 1e3;
        end
    end
    function set.t_st(W, v)
        W.fr_st = round(v * W.refresh_rate) + 1;
    end
    function v = get.t_st(W)
        v = (W.fr_st - 1) / W.refresh_rate;
    end
    function set.t_en(W, v)
        W.fr_en = round(v * W.refresh_rate) + 1;
    end
    function v = get.t_en(W)
        v = (W.fr_en - 1) / W.refresh_rate;
    end
    function v = get.t_st_en(W)
        v = [W.t_st, W.t_en];
    end
    function set.t_st_en(W, v)
        if isempty(v)
            % Do nothing
            return; 
        else
            assert(numel(v) == 2);
            W.t_st = v(1);
            W.t_en = v(2);
        end
    end
    function v = get.fr_st_en(W)
        v = [W.fr_st, W.fr_en];
    end
    function set.fr_st_en(W, v)
        if isempty(v)
            % Do nothing
            return;
        else
            assert(numel(v) == 2);
            W.fr_st = v(1);
            W.fr_en = v(2);
        end
    end
    function set.fr_en(W, v)
        W.fr_en_ = v;
    end
    function v = get.fr_en(W)
        if isempty(W.fr_en_)
            v = floor((W.rt_incl_ms(1) ...
                - W.truncate_first_msec ...
                - W.truncate_last_msec ...
                ) / 1000 * W.refresh_rate);
        else
            v = W.fr_en_;
        end
    end
end
end