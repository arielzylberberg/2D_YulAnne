classdef SimulateXyct < matlab.mixin.Copyable
properties
    RDK
end
methods
    function init(SE, L_RDK)
        if nargin < 2 || isempty(L_RDK)
            L_RDK = SE.load_RDK;
        end
        RDK = L_RDK.RDKCol;
        RDK.Scr = L_RDK.Scr;
        SE.RDK = RDK;
    end
    function L_RDK = load_RDK(SE)
        L_RDK = load('RDKCol.mat');        
    end
    function xyct = simulate_xyct(SE, condM, condC, seedM, seedC, varargin)
        S = varargin2S({
            't_max', 5
            'refresh_rate', 75
            });
        n_fr = S.refresh_rate * S.t_max;
        
        RDK = SE.RDK;
%         RDK.Scr.cFr = 1;
        RDK.init(condM, condC, {0, full(seedM), full(seedC)});
        RDK.initLogTrial(false, false);
        RDK.initLogEntries;
        RDK.xyct = [];

        RDK.visible = true;
        for fr = 1:n_fr
%             RDK.Scr.cFr = fr;
            RDK.update('befDraw');
        end

        RDK.closeLog;
        xyct = RDK.xyct;        
        
%         RDK.plotTraj;
    end
    function max_dif = compare_xyct(SE, xyct1, xyct2)
        n_dot = 4;
        xycts = {xyct1, xyct2};
        n_fr_min = inf;
        for ii = 1:2
            xyct = xycts{ii};
            n_fr = size(xyct, 1) / n_dot;
            
            if n_fr < n_fr_min
                n_fr_min = n_fr;
            end
        end
        for ii = 1:2
            xyct = xycts{ii};
            n_fr = size(xyct, 1) / n_dot;
            xyct = reshape(xyct, [n_fr, n_dot, 4]);
            
            xyct = xyct(1:n_fr_min, :, :);
            xycts{ii} = xyct;
        end
        max_dif = max(max(abs(xycts{1} - xycts{2}), [], 1), [], 3);
    end
end
end