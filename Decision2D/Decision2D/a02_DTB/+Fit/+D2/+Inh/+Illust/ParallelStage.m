classdef ParallelStage < FitWorkspace
properties
%     Fl = FitFlow;
    
    % cond_plot(dim)
    % : A joint condition to draw. Defaults to an intermediate level.
    cond_plot = [7 6]; 

    dim_on_xy = 1;
    
    dim_plot = 1; % Transiently shared between methods.
    t_plot_max = 1;
    p_td_max = []; % Scales Td. May set externally, too.
    
    t_of_interest = 0.5;
    ch_of_interest = 2;
    
    UTd = [];
    
    W
end
properties (Dependent)
    Dtb
    bound
    unabs_together % (t, y) of the cond_plot(dim_plot)
    td_together % (t, ch) of the cond_plot(dim_plot)
    td_together_first % (t, ch) of the cond_plot(dim_plot)
    t_plot
    t_incl
    y_plot
    y_incl
    dim_on_xz
    
    cond12_plot
end
methods
    function Il = demo(Il0, varargin)
        %%
        init_path;
        if ~exist('Il0', 'var')
            Il0 = Fit.D2.Inh.Illust.ParallelStage;
        end
        Il = feval(class(Il0));
        
        C = varargin2C(varargin, {
            'subj', 'DX'
            'model', 'InhPar'
            'bound_kind', 'Const'
            });
        Il.W = feval('Fit.D2.Inh.MainBatch', C{:}); % L.W;
        
        %%
        Il.W.th.Dtb__drift_fac_together_dim1_2 = 0.5;
        Il.W.th.Dtb__drift_fac_together_dim2_1 = 0.5;
        Il.W.th.Dtb__sigmaSq_fac_together_dim1_2 = 1;
        Il.W.th.Dtb__sigmaSq_fac_together_dim2_1 = 1;
        
        %%
        Il.W.get_Fl;
        Il.W.pred;
        
        %%
        clf;
        Il.plot;
    end
    function Il = demo_w_load(Il0)
        %%
        L = Il0.demo_load;
        
        Il = feval(class(Il0));
        
        C = S2C(L.S0_file);
        Il.W = feval('Fit.D2.Inh.MainBatch', C{:}); % L.W;
        
        Il.W.get_Fl;
        Il.W.Fl.res = L.res;
        Il.W.Fl.res2W;
        
        %%
        Il.plot;
    end
    function L = demo_load(~)
        % bnd=Const
%         L = load('Data_2D/Fit.D2.RT.Inh.MainBatch/old/subj=MA+parad=RT+dtb=DensitySigmaSq+drift=Const+bnd=Const+sigSq=Linear+tnd=halfnorm+kb=0+p1st=0+fx_p=1+d1=0+fx_d1=1+fx_db1=1+s1=0+fx_s1=1+d2=0+fx_d2=1+fx_db2=1+s2=0+fx_s2=1+fx_ms=1.mat');
%         L = load('Data_2D/Fit.D2.RT.Inh.MainBatch/old/subj=MA+parad=RT+dtb=DensitySigmaSq+drift=Const+bnd=Const+sigSq=Linear+tnd=halfnorm+kb=0+p1st=0+fx_p=1+d1=0+fx_d1=1+fx_db1=0+s1=0+fx_s1=1+d2=0+fx_d2=1+fx_db2=0+s2=0+fx_s2=1+fx_ms=1.mat');
%         L = load('Data_2D/Fit.D2.RT.Inh.MainBatch/old/subj=DX+parad=RT+dtb=DensitySigmaSq+drift=Const+bnd=Const+sigSq=Linear+tnd=halfnorm+kb=0+p1st=0+fx_p=1+d1=0+fx_d1=1+fx_db1=0+s1=0+fx_s1=1+d2=0+fx_d2=1+fx_db2=0+s2=0+fx_s2=1+fx_ms=1.mat');
%         L = load('Data_2D/Fit.D2.RT.Inh.MainBatch/old/subj=VL+parad=RT+dtb=Density+drift=Const+bnd=Const+sigSq=Const+tnd=halfnorm+kb=0+p1st=0+fx_p=1+d1=0+fx_d1=1+fx_db1=0+s1=0+fx_s1=1+d2=0+fx_d2=1+fx_db2=0+s2=0+fx_s2=1+fx_ms=1.mat');

%         L = load('Data_2D/Fit.D2.RT.Inh.MainBatch/subj=VL+parad=RT+dtb=Density+drift=Const+bnd=Const+sigSq=Const+tnd=halfnorm+kb=0+p1st=50+d1=100+d2=100+s1=100+s2=100+fn1=1+fn2=1+fx_p=1+fx_d1=0+fx_d2=0+fx_db1=1+fx_db2=1+fx_s1=0+fx_s2=0+fx_ms=1.mat');

        L = load('/Users/yulkang/Dropbox/CodeNData_2D/ExtRepos/ShadlenLab/Decision2D/Data_2D/Fit.D2.Inh.MainBatch/sbj=DX+prd=RT+tsk=A+dtk=2+dmr=1+trm=201+eor=t+dft=S+bnd=A2+ssq=LMPrD+tnd=g+ntnd=4+msf=0+dtb=Sc+kb=0+p1=1+d=0^5+s=0^5+fn=1+pf=1+df=1+sf=1+fnf=0+cv=0+fsqs=0+fbst=0.mat');
%         L = load('Data_2D/Fit.D2.Inh.MainBatch/sbj=DX+prd=RT+tsk=A+nd_tsk=2+dim_r=1+dif_r=[2,3,4,5]+dif_i=[2,3,4,5]+acc_i=[0,1]+tr_min=201+nre=0+dft=C+bnd=O+ssq=LMPrD+tnd=g+ntnd=4+msf=0+dtb=DI+kb=0+p1=50+d1=0+d2=0+s1=100+s2=100+fn1=100+fn2=100+pf=0+d1f=1+d2f=1+s1f=1+s2f=1+fn1f=1+fn2f=1+cv=0.mat');

        % bnd=Betacdf
%         L = load('Data_2D/Fit.D2.RT.Inh.MainBatch/subj=VL+parad=RT+dtb=DensityIndivJt+drift=Const+bnd=BetaCdf+sigSq=LinearPreDrift+tnd=halfnorm+kb=0+p1st=50+d1=0+d2=0+s1=100+s2=16+fn1=1+fn2=1+fx_p=1+fx_d1=0+fx_d2=0+fx_db1=0+fx_db2=0+fx_s1=0+fx_s2=0+fx_ms=0');
%         L = load('Data_2D/Fit.D2.RT.Inh.MainBatch/subj=DX+parad=RT+dtb=Density+drift=Const+bnd=BetaCdf+sigSq=Const+tnd=halfnorm+kb=1+p1st=0+fx_p=0+d1=0+fx_d1=0+s1=0+fx_s1=0+d2=0+fx_d2=0+s2=0+fx_s2=0.mat');
    end
end
methods
    function Il = ParallelStage(varargin)
        if nargin > 0
            Il.init(varargin{:});
        end
    end
    function init(Il, varargin)
        bml.oop.varargin2props(Il, varargin, true);
        
%         if exist('Fl', 'var') && ~isempty(Fl)
%             Il.Fl = Fl;
%             Il.W.pred;
%             Il.set_UTd;
%         end
    end
    function plot(Il)
        %%
        clf;
        Il.dim_on_xy = 1;
        
        %%
        Il.dim_plot = Il.dim_on_xy;
        Il.set_UTd;
        Il.UTd.plane = 'xy';
        Il.UTd.origin = [0 0 2];
        Il.plot_dim;
        hold on;
        
        %%
        Il.dim_plot = Il.dim_on_xz;
        Il.set_UTd;
        Il.UTd.plane = 'xz';
        Il.UTd.origin = [0, -1, 0];
        Il.plot_dim;
        hold on;

        %%
        if ~isempty(Il.t_of_interest) && ~isempty(Il.ch_of_interest)
            Il.dim_plot = Il.dim_on_xy;
            Il.UTd.plane = 'xz';
            Il.UTd.origin = [0, -1, 0];
            Il.UTd.mark_td_ch( ...
                Il.td_together, ...
                Il.t_of_interest, Il.ch_of_interest, 'r', ...
                'y_max', Il.p_td_max);
            Il.UTd.area_unabs(Il.unabs_together, Il.t_of_interest, 'r');
        end
        hold off;
        
        %% Axis labels for reference
        axis on
        xlabel('x')
        ylabel('y')
        zlabel('z')
        
        %%
        axis equal;
        view([37, 28]);
        
    end
    function plot_dim(Il)
        Il.area_td_together;
        hold on;
        
%         Il.area_td_together_first;
%         hold on;
%         
        Il.imagesc_unabs_together;
        hold on;
        
        axis off; % DEBUG
    end
    function imagesc_unabs_together(Il)
        Il.UTd.imagesc_unabs(Il.unabs_together);
    end
    function varargout = area_td_together(Il)
        [varargout{1:nargout}] = ...
            Il.area_td(Il.td_together, 'k', ...
            'z_offset', -0.001, ...
            'alpha_area', 0.25);
    end
    function varargout = area_td_together_first(Il)
        [varargout{1:nargout}] = ...
            Il.area_td(Il.td_together_first, lines(1), ...
            'z_offset', 0, ...
            'alpha_area', 0.75);
    end
    function varargout = area_td(Il, td, color, varargin)
        [varargout{1:nargout}] = Il.UTd.area_td(td, color, ...
            'y_max', Il.p_td_max, varargin{:});
    end
end
%% Get/Set
methods
%     function v = get.W(Il)
%         v = Il.Fl.W;
%     end
    function v = get_Fl(Il)
        if ~isempty(Il.W)
            v = Il.W.get_Fl;
        else
            v = [];
        end
    end
    function v = get.Dtb(Il)
        v = Il.W.Dtb;
    end
    function v = get.bound(Il)
        Bound = Il.Dtb.Bounds{Il.dim_plot};
        v = Bound.get_bound_t_ch;
    end
    function v = get.unabs_together(Il)
        v = Il.Dtb.unabs_together{Il.dim_plot} ...
            (Il.t_incl, Il.y_incl, Il.cond12_plot);
    end
    function v = get.td_together(Il)
        for ch = 2:-1:1
            v(:,ch) = Il.Dtb.td_together{Il.dim_plot} ...
                (Il.t_incl, Il.cond12_plot, ch);
        end
    end
    function v = get.td_together_first(Il)
        for ch = 2:-1:1
            v(:,ch) = Il.Dtb.td_together_first{Il.dim_plot} ...
                (Il.t_incl, Il.cond12_plot, ch);
        end
    end
    function v = get.p_td_max(Il)
        if isempty(Il.p_td_max)
            v = max(Il.td_together(:));
%             v = max([Il.td_together(:); Il.td_together_first(:)]);
        end
    end
    function y_plot = get.y_plot(Il)
        y_all = Il.Dtb.y;        
        y_plot = y_all(Il.y_incl);
        y_plot = y_plot ./ max(abs(y_plot(:)));
    end
    function y_incl = get.y_incl(Il)
        y_all = Il.Dtb.y;
        bound = Il.bound;
        max_y = [min(bound(:,1)), max(bound(:,2))];
        max_y_ix = [find(y_all > max_y(1), 1, 'first'), ...
                 find(y_all < max_y(2), 1, 'last')];
        y_incl = max_y_ix(1):max_y_ix(2);
    end
    function t_plot = get.t_plot(Il)
        t_all = Il.Dtb.t;
        t_plot = t_all(Il.t_incl);
        t_plot = t_plot ./ max(t_plot(:)) .* 3;
    end
    function t_incl = get.t_incl(Il)
        t_all = Il.Dtb.t;
        t_incl = t_all <= Il.t_plot_max;
    end
    function v = get.dim_on_xz(Il)
        v = 3 - Il.dim_on_xy;
    end
    function set_UTd(Il)
        Il.UTd = Fit.D2.Inh.Illust.UnabsWithTd3D(Il);
        bml.oop.copyprops(Il.UTd, Il, 'hide_error', true);
    end
    function v = get.cond12_plot(Il)
        v = sub2ind(Il.W.Data.nConds, Il.cond_plot(1), Il.cond_plot(2));
    end
end
end