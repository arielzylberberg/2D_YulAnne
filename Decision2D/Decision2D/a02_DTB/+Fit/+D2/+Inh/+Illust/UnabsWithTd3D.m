classdef UnabsWithTd3D < FitWorkspace
properties
    t_plot = [];
    y_plot = [];
    
    plane = 'xy';
    origin = [0 0 0];
end
properties (Dependent)
    Il
end
methods
    function UTd = UnabsWithTd3D(varargin)
        if nargin > 0
            UTd.init(varargin{:});
        end
    end
    function init(UTd, Il, varargin)
        bml.oop.varargin2props(UTd, varargin);
        
        if exist('Il', 'var') && ~isempty(Il)
            UTd.Il = Il;
        end
    end
    function imagesc_unabs(UTd, unabs)
        % unabs(t, y)
        y_plot = UTd.y_plot;
        t_plot = UTd.t_plot;
        
        bml.plot.imagesc3(t_plot, y_plot, unabs, ...
            'plane', UTd.plane, 'origin', UTd.origin);
        
        colormap(parula(256));
        bml.plot.beautify;
    end
    function mark_td_ch(UTd, p_td0, t_of_interest, ch_of_interest, color, varargin)
        % mark_td_ch(UTd, p_td0, t_of_interest, ch_of_interest, color, varargin)        
        %
        % 'y_max', max(p_td0(:))
        
        S = varargin2S(varargin, {
            'y_max', max(p_td0(:))
            });
        
        p_td0 = p_td0 ./ S.y_max .* 0.4;
        p_td = p_td0(:, ch_of_interest) .* sign(ch_of_interest - 1.5);
        [~, ix] = min(abs(UTd.t_plot - t_of_interest));
        
        y_max = UTd.y_plot([1, end]);
        
        bml.plot.plot_on_plane( ...
            t_of_interest + [0 0], ...
            [0, p_td(ix)], ...
            'base', y_max(ch_of_interest), ...
            'origin', UTd.origin, ...
            'plane', UTd.plane, ...
            'Color', color, ...
            varargin{:});
    end
    function mark_td_unabs(UTd, t_of_interest, color, varargin)
        % mark_td_unabs(UTd, t_of_interest, color, varargin)
        
        y_max = UTd.y_plot([1, end]);
        
        bml.plot.plot_on_plane( ...
            t_of_interest + [0 0], ...
            y_max, ...
            'origin', UTd.origin, ...
            'plane', UTd.plane, ...
            'Color', color, ...
            varargin{:});
    end
    function [h_face, h_line] = area_unabs(UTd, unabs, t_of_interest, color, varargin)
        S = varargin2S(varargin, {
            'y_max', 0.3
            'z_offset', 0 % Positive is closer to the viewer at -x, -y, +z.
            'alpha_area', 0.25
            'plane', []
            'opt_plot', {}
            });
        S.opt_plot = varargin2C(S.opt_plot);
                
        if isempty(S.plane)
            switch UTd.plane
                case 'xy'
                    S.plane = 'yz';
                otherwise
                    error('Unsupported UTd.plane: %s\n', UTd.plane);
            end
        end     
        
        % Shift time
        origin = UTd.origin + [t_of_interest, 0, 0];
        
        [~, ix] = min(abs(UTd.t_plot - t_of_interest));
        p = unabs(ix,:);
        p = nan0(p ./ max(p)) .* S.y_max;

        [h_face, h_line] = bml.plot.area3(UTd.y_plot, p, ...
            'z_offset', S.z_offset, ...
            'FaceAlpha', S.alpha_area, ...
            'Color', color, ...
            'plane', S.plane, ...
            'origin', origin, ...
            'opt_plot', S.opt_plot);
    end
    function area_td(UTd, p_td0, color, varargin)
        S = varargin2S(varargin, {
            'y_max', max(p_td0(:));
            'z_offset', 0 % Positive is closer to the viewer at -x, -y, +z.
            'alpha_area', 0.25
            });
        
        base0 = UTd.y_plot([1 end]);
        p_td0 = p_td0 ./ S.y_max;
        fac0 = [-1, 1] .* diff(base0) .* 0.2;
        
        for ch = 1:2
            base = base0(ch);
            fac = fac0(ch);
            p_td = p_td0(:,ch) * fac;
            
            bml.plot.area3(UTd.t_plot, p_td, ...
                'z_offset', S.z_offset, ...
                'FaceAlpha', S.alpha_area, ...
                'Color', color, ...
                'plane', UTd.plane, ...
                'origin', UTd.origin, ...
                'base', base);
        end
    end
    function ylim(UTd, v)
        switch UTd.plane
            case 'xy'
                ylim(v);
                
            case 'xz'
                zlim(v);
        end
    end
end
%% Get/Set
methods
    function set.Il(UTd, Il)
        UTd.t_plot = Il.t_plot;
        UTd.y_plot = Il.y_plot;
    end
end
end