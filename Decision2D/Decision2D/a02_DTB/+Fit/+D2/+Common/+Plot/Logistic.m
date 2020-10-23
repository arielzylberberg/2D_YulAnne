classdef Logistic < Fit.D2.Common.Plot.Adaptor
properties
    n_ch = []; % (cond_rel, ch_rel, cond_irr)    
    res = {}; % (cond_irr)
end
properties (Dependent)
%     x % (cond_irr)
%     y % (cond_irr)
%     lb % (cond_irr)
%     ub % (cond_irr)
end
methods
    function Plt = Logistic(varargin)
        Plt.pdf_kind = 'RT_data_pdf';
        
        if nargin > 0
            Plt.init(varargin{:});
            Plt.plot;
        end
    end
    function [h, he, res] = plot(Plt, plot_args, tick_args)
        if ~exist('plot_args', 'var'), plot_args = {}; end
        if ~exist('tick_args', 'var'), tick_args = {}; end
        
        x = Plt.get_x;
        y = Plt.get_y;
        [lb, ub] = Plt.get_bnd;
        le = bsxfun(@minus, lb, y);
        ue = bsxfun(@minus, ub, y);
        
        [h, he] = errorbar_wo_tick(x, y, le, ue, ...
            plot_args, tick_args);
        
        res = packStruct(x, y, le, ue, plot_args, tick_args);
        
        Plt.ylabel;
    end
    function ylabel(~)
        ylabel('Slope');
    end
end
%% init
methods
    function init(Plt, inp, varargin)
        varargin2props(Plt, varargin);
        if exist('inp', 'var') && ~isempty(inp)
            Plt.import_data(inp);
        end
        Plt.calc_n_ch;
        Plt.fit;
    end
    function n_ch = calc_n_ch(Plt, p)
        if ~exist('p', 'var')
            p = Plt.p; %
        end
        
        if Plt.dimOnX == 1
            n_ch = sums(p, [1,5], true); % (cond_rel, cond_irr, ch_rel)
            n_ch = permute(n_ch, [1 3 2]); % (cond_rel, ch_rel, cond_irr)
        else
            n_ch = sums(p, [1,4], true); % (cond_irr, cond_rel, ch_rel)
            n_ch = permute(n_ch, [2 3 1]); % (cond_rel, ch_rel, cond_irr)
        end
        
        Plt.n_ch = n_ch;
    end
    function fit(Plt)
        n_ch0 = Plt.n_ch;
        n_conds_irr = size(n_ch0, 3);
        res = cell(1, n_conds_irr);
        conds_rel = Plt.get_conds_rel;
        
        for ii = 1:n_conds_irr
            n_ch = n_ch0(:,:,ii);
            n_tot = sum(n_ch, 2);
            n_ch2 = n_ch(:,2);
            
            res{ii} = glmwrap(conds_rel, [n_ch2, n_tot], 'binomial');
        end
        Plt.res = res;
    end
end
%% Get
methods
    function x = get_x(Plt)
        x = Plt.get_conds_rel;
    end
    function y = get_y(Plt)
        res = Plt.res;
        n_cond_irr = numel(res);
        
        y = zeros(n_cond_irr, 1);
        for ii = 1:n_cond_irr
            y(ii) = res{ii}.b(2);
        end
    end
    function [lb, ub] = get_bnd(Plt)
        res = Plt.res;
        n_cond_irr = numel(res);
        
        lb = zeros(n_cond_irr, 1);
        ub = zeros(n_cond_irr, 1);
        for ii = 1:n_cond_irr
            lb(ii) = res{ii}.b(2) - res{ii}.se(2);
            ub(ii) = res{ii}.b(2) + res{ii}.se(2);
        end
    end
    function d_cond_rel = get_d_cond_rel(Plt)
        d_cond_rel = Plt.Data.get_dCond;
        d_cond_rel = d_cond_rel(:,Plt.dimOnX);
    end
    function d_cond_irr = get_d_cond_irr(Plt)
        d_cond_irr = Plt.Data.get_dCond;
        d_cond_irr = d_cond_irr(:,Plt.dimSep);
    end
    function conds_rel = get_conds_rel(Plt)
        conds_rel = Plt.Data.get_conds;
        conds_rel = conds_rel{Plt.dimOnX};
    end
    function ch_rel = get_ch_rel(Plt)
        ch_rel = Plt.Data.get_ch;
        ch_rel = ch_rel(:,Plt.dimOnX);
    end
end
end