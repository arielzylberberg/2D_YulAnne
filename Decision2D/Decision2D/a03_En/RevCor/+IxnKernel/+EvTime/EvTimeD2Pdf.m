classdef EvTimeD2Pdf ...
        < IxnKernel.EvTime.EvTimeD2
    
%% Init
methods
    function Ev = EvTimeD2Pdf(varargin)
        Ev.Evs = { ...
            IxnKernel.EvTime.EvTimeD1Pdf, ...
            IxnKernel.EvTime.EvTimeD1Pdf
            };
        if nargin > 0
            Ev.init(varargin{:});
        end
    end
end
end