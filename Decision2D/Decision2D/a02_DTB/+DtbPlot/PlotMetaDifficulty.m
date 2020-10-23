classdef PlotMetaDifficulty < DtbPlot.PlotMeta
properties
    componentClass = 'PlotCh2D';
    componentArgs = {};
    difLevel = 0; % Equal to 'end'
end
methods
    function Pl = PlotMetaDifficulty(varargin)
        Pl = Pl@DtbPlot.PlotMeta(varargin{:});
        Pl.plotArgs = {'Marker', 'o'};
    end
    function [y, x] = get_xy(Pl)
        xy = cell(Pl.nPdfs,1);
        for iPdf = 1:Pl.nPdfs
            cPl = DtbPlot.(Pl.componentClass)(Pl.pdfs{iPdf}, ...
                Pl.componentArgs{:});
            for iDim = 1:2
                cPl.set_dimOnX(iDim);
                cy = cPl.get_y;
                
                if ischar(Pl.difLevel) && strcmp(Pl.difLevel, 'end')
                    ix = size(cy,1);
                elseif Pl.difLevel <= 0
                    ix = size(cy,1) - Pl.difLevel;
                else
                    ix = Pl.difLevel;
                end     
                xy{iPdf, iDim} = cy(ix,:);
            end
        end
        x = cell2mat2(xy(:,1)); % iPdf x dif
        y = cell2mat2(xy(:,2)); % iPdf x dif
        
        if Pl.dimOnX == 1
            x = x'; % dif x iPdf
            y = y'; % dif x iPdf
        end        
    end
    function y = get_y(Pl)
        y = get_xy(Pl);
    end
    function x = get_x(Pl)
        [~, x] = get_xy(Pl);
    end
%     function varargout = plot(Pl, varargin)
%         [varargout{1:nargout}] = plot@DtbPlot.PlotMeta(Pl, varargin{:});
%         
% %         xlabel('Motion Lapse');
% %         ylabel('Color Lapse');
%     end
end
end