classdef Factorized < FitWorkspace
properties
%     Fl = FitFlow;
    
    % cond_plot(dim)
    % : A joint condition to draw. Defaults to an intermediate level.
    cond_plot = [2 5]; 

    dim_on_xy = 1;
    
    dim_plot = 1; % Transiently shared between methods.
    t_plot_max = 3;
    p_td_max_ = []; % Scales Td. May set externally, too.
    
    t_of_interest = 1;
    t_of_interest2 = 2;
    y_of_interest = -0.2;
    ch_of_interest = 1;
    ch_of_interest2 = 1;
    
    UTd = [];
    
    W
end
properties (Dependent)
    Dtb
    bound
    unabs_together % (t, y) of the cond_plot(dim_plot)
    unabs_alone % (t, y) of the cond_plot(dim_plot)
    td_together % (t, ch) of the cond_plot(dim_plot)
    td_together_first % (t, ch) of the cond_plot(dim_plot)
    td_alone % (t, ch)
    t_plot
    t_incl
    y_plot
    y_incl
    dim_on_xz
    p_td_max
    
    cond12_plot
end
methods
    function Il = demo(Il0)
        %%
        init_path;
        Il0 = Fit.D2.Inh.Illust.Factorized;
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
        origin_xy = [0 0 -2];
        Il.UTd.origin = origin_xy;
        Il.imagesc_unabs_together;
        hold on;
        
        %%
        Il.dim_plot = Il.dim_on_xz;
        Il.set_UTd;
        Il.UTd.plane = 'xz';
        origin_xz = [0, 2, 0];
        Il.UTd.origin = origin_xz;
        Il.area_td_together;
        hold on;
        
        Il.imagesc_unabs_together;
        colormap(parula(1024));
        hold on;

        %%
        if ~isempty(Il.t_of_interest) && ~isempty(Il.ch_of_interest)
%             Il.dim_plot = Il.dim_on_xz;
%             Il.UTd.plane = 'xz';
%             Il.UTd.origin = origin_xz;
%             Il.UTd.mark_td_ch( ...
%                 Il.td_together, ...
%                 Il.t_of_interest, Il.ch_of_interest, 'r', ...
%                 'y_max', Il.p_td_max, ...
%                 'opt_plot', {
%                     'LineWidth', 2
%                     });
%             hold on;
                
            %%
            Il.dim_plot = Il.dim_on_xy;
            Il.UTd.plane = 'xy';
            Il.UTd.origin = origin_xy;
%             Il.UTd.mark_td_unabs( ...
%                 Il.t_of_interest, 'w', ...
%                 'y_max', Il.p_td_max, ...
%                 'opt_plot', {
%                     'LineWidth', 2
%                     'LineStyle', ':'
%                     });
%             hold on;
        
            %%
            Il.UTd.area_unabs(Il.unabs_together, Il.t_of_interest, 'w', ...
                'alpha_area', 0.5, ...
                'opt_plot', {
                    'LineWidth', 1
                    });
            hold on;
        end
        
        %% Unabs of the 2nd phase
        Il.dim_plot = Il.dim_on_xy;
        Il.set_UTd;
        Il.UTd.plane = 'xy';
        origin_xy2 = [4.5 0 -2.2];
        Il.UTd.origin = origin_xy2;
        Il.imagesc_unabs_alone;
        hold on;

        %% Unabs area of the 2nd phase
        [h_face, h_line] = Il.UTd.area_unabs(Il.unabs_alone, Il.t_of_interest, 'w', ...
            'alpha_area', 0.5, ...
            'opt_plot', {
                'LineWidth', 1
                });
        hold on;

        %% Td of the 2nd phase
        Il.area_td_alone;
        hold on;
        
        %% Dashed lines between box and 2nd stage
        xyz1 = origin_xy2 + [0, -1, 0];
        xyz2 = origin_xy2 + [3, 1, 0];
        xyz2(3) = -1;
        
        [h_line, h_face] = bml.plot.box3d( ...
            xyz1, ...
            xyz2, {
            'Color', 0.7 + [0 0 0]
            'LineWidth', 1
            'LineStyle', '--'
            }, {
            'FaceAlpha', 0
            });
        
        delete(h_line([1, 2, 9, 10, 4, 5, 7, 8]));
        
        %% Box
        xyz1 = origin_xy2 + [0, -1, 0];
        xyz1(3) = -1;
        xyz2 = origin_xy2 + [3, 1, 0];
        xyz2(3) = 1;
        
        origin_box = [origin_xy2(1), 0, 0];
        [h_line, h_face] = bml.plot.box3d( ...
            xyz1, ...
            xyz2, {
            'Color', 0 + [0 0 0]
            'LineWidth', 1
            }, {
            'FaceAlpha', 0
            });
        set(h_line([2, 10, 11]), 'Color', 0.8 + [0 0 0]);
        
        %% Trajectory - common settings
        marker_size = 4;        
        
        %% Trajectory - 1st stage - xz
        Il.dim_plot = Il.dim_on_xz;
        drift_vec = Il.Dtb.Drifts{Il.dim_plot}.get_drift_vec;
        drift = drift_vec(Il.cond_plot(Il.dim_plot)) ...
            .* Il.Dtb.drift_fac_together(Il.dim_plot, 3 - Il.dim_plot);
        bound = Il.Dtb.Bounds{Il.dim_plot}.get_bound_t_ch;
        t_incl = Il.Dtb.t <= Il.t_of_interest;
        t = Il.Dtb.t(t_incl);
        bound = bound(t_incl,:);
        y0 = 0;
        y_end = bound(end, Il.ch_of_interest);
        
        rng(2);
        traj = bml.diffusion.get_traj_unabsorbed( ...
            t, drift, bound, y0, y_end, Il.Dtb.y);
        
        origin = origin_xz;
        x = origin(1) + t;
        y = origin(2) + zeros(size(t));
        z = origin(3) + traj;
        
        h = plot3(x, y, z, 'k-');
        h0 = plot3(x(1), y(1), z(1), 'ro', ...
            'MarkerFaceColor', 'r', ...
            'MarkerEdgeColor', 'none', ...
            'MarkerSize', marker_size);

        color = 'c';
        h1 = plot3(x(end), y(end), z(end), 'o', ...
            'Color', color, ...
            'MarkerFaceColor', color, ...
            'MarkerEdgeColor', 'none', ...
            'MarkerSize', marker_size);

        traj_xz = traj;
        
        %% Trajectory - 1st stage - xy
        Il.dim_plot = Il.dim_on_xy;
        drift_vec = Il.Dtb.Drifts{Il.dim_plot}.get_drift_vec;
        drift = drift_vec(Il.cond_plot(Il.dim_plot)) ...
            .* Il.Dtb.drift_fac_together(Il.dim_plot, 3 - Il.dim_plot);
        bound = Il.Dtb.Bounds{Il.dim_plot}.get_bound_t_ch;
        t_incl = Il.Dtb.t <= Il.t_of_interest;
        t = Il.Dtb.t(t_incl);
        bound = bound(t_incl,:);
        y0 = 0;
        y_end = Il.y_of_interest;
        
        rng(3);
        traj = bml.diffusion.get_traj_unabsorbed( ...
            t, drift, bound, y0, y_end, Il.Dtb.y);
        
        origin = origin_xy;
        x = origin(1) + t;
        y = origin(2) + traj;
        z = origin(3) + zeros(size(t));
        
        h = plot3(x, y, z, 'r-');
        h0 = plot3(x(1), y(1), z(1), 'ro', ...
            'MarkerFaceColor', 'r', ...
            'MarkerEdgeColor', 'none', ...
            'MarkerSize', marker_size);
        
        color = 'c';
        h1 = plot3(x(end), y(end), z(end), 'o', ...
            'Color', color, ...
            'MarkerFaceColor', color, ...
            'MarkerSize', marker_size);

        traj_xy = traj;
        
        %% Trajectory - box
        x = origin_box(1) + t;
        y = origin_box(2) + traj_xy;
        z = origin_box(3) + traj_xz;
        h = plot3(x, y, z, 'r-');
        h = plot3(x, y, z, 'k--');
        h0 = plot3(x(1), y(1), z(1), 'ro', ...
            'MarkerFaceColor', 'r', ...
            'MarkerEdgeColor', 'none', ...
            'MarkerSize', marker_size);
        h1 = plot3(x(end), y(end), z(end), 'co', ...
            'MarkerFaceColor', 'c', ...
            'MarkerEdgeColor', 'none', ...
            'MarkerSize', marker_size);
        
        traj_box_xyz = [x(:), y(:), z(:)];
        
        %% Trajectory - xy 2nd
        Il.dim_plot = Il.dim_on_xy;
        drift_vec = Il.Dtb.Drifts{Il.dim_plot}.get_drift_vec;
        drift = drift_vec(Il.cond_plot(Il.dim_plot));
        bound = Il.Dtb.Bounds{Il.dim_plot}.get_bound_t_ch;
        t_incl = (Il.t_of_interest <= Il.Dtb.t) ...
               & (Il.Dtb.t <= Il.t_of_interest2);
        t = Il.Dtb.t(t_incl);
        bound = bound(t_incl,:);
        y0 = Il.y_of_interest;
        y_end = bound(end, Il.ch_of_interest2);
        
        rng(3);
        traj = bml.diffusion.get_traj_unabsorbed( ...
            t, drift, bound, y0, y_end, Il.Dtb.y);
        
        origin = origin_xy2;
        x = origin(1) + t;
        y = origin(2) + traj;
        z = origin(3) + zeros(size(t));
        
        color_xy2 = 'c';
        h = plot3(x, y, z, '-', 'Color', color_xy2);
        h0 = plot3(x(1), y(1), z(1), 'co', ...
            'MarkerFaceColor', 'c', ...
            'MarkerEdgeColor', 'none', ...
            'MarkerSize', marker_size);
        h1 = plot3(x(end), y(end), z(end), 'wo', ...
            'MarkerFaceColor', 'w', ...
            'MarkerEdgeColor', 'k', ...
            'LineWidth', 0.5, ...
            'MarkerSize', marker_size + 1);
        
        traj_xy2 = traj;
        traj_xy2_xyz = [x(:), y(:), z(:)];
        
        %% Dashed line connecting two trajectories
        line_box_xy2 = [traj_box_xyz(end,:); traj_xy2_xyz(1,:)];
        plot3(line_box_xy2(:,1), line_box_xy2(:,2), line_box_xy2(:,3), ...
            'c--');

        %% Axis labels for reference
        axis on
        xlabel('x')
        ylabel('y')
        zlabel('z')
        
        %% Viewing angle
        axis on;
        axis equal;
        view([25, 15]);
%         view([35, 35]);
%         view([25, 37]);
        camproj('perspective');
        camdolly(0, 0, 0.6, 'fixtarget', 'camera');
        
        %% Print
        axis off;
        file = fullfile('../Data', class(Il), 'plt=all');
        savefigs(file, 'ext', {'.png', '.tif', '.fig'});
    end
    function plot_dim(Il)
        Il.area_td_together;
        hold on;
        
%         Il.area_td_together_first;
%         hold on;
%         
        Il.imagesc_unabs_together;
        hold on;
        
%         axis off; % DEBUG
    end
    function imagesc_unabs_together(Il)
        p = Il.unabs_together;
        p = p ./ prctile(p(:), 99);
        Il.UTd.imagesc_unabs(p);
    end
    function imagesc_unabs_alone(Il)
        p = Il.unabs_alone;
        p = p ./ prctile(p(:), 99);
        Il.UTd.imagesc_unabs(p);
    end
    function varargout = area_td_together(Il)
        [varargout{1:nargout}] = ...
            Il.area_td(Il.td_together, 0.75 + [0 0 0], ...
            'z_offset', -0.001, ...
            'alpha_area', 1);
    end
    function varargout = area_td_alone(Il)
        [varargout{1:nargout}] = ...
            Il.area_td(Il.td_alone, 0.75 + [0 0 0], ...
            'z_offset', -0.001, ...
            'alpha_area', 1);
    end
    function varargout = area_td_together_first(Il)
        [varargout{1:nargout}] = ...
            Il.area_td(Il.td_together_first, lines(1), ...
            'z_offset', 0, ...
            'alpha_area', 0.75);
    end
    function varargout = area_td(Il, td, color, varargin)
        [varargout{1:nargout}] = Il.UTd.area_td(td, color, ...
            varargin{:});
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
    function v = get.unabs_alone(Il)
        v = Il.Dtb.unabs_alone{Il.dim_plot} ...
            (Il.t_incl, Il.y_incl, Il.ch_of_interest, Il.cond12_plot);
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
    function v = get.td_alone(Il)
        v = permute(Il.Dtb.td_diff_ch{Il.dim_plot}( ...
            Il.t_incl, Il.ch_of_interest, :, Il.cond12_plot), ...
            [1, 3, 2, 4]);
    end
    function v = get.p_td_max(Il)
        if isempty(Il.p_td_max_)
            v = max(Il.td_together(:));
        else
            v = Il.p_td_max_;
%             v = max([Il.td_together(:); Il.td_together_first(:)]);
        end
    end
    function set.p_td_max(Il, v)
        Il.p_td_max_ = v;
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