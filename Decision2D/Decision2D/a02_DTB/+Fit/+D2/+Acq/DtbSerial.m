classdef DtbSerial < Fit.D2.Short.Acq.Dtb & Fit.D2.Inh.DtbDensitySerial
    % Fit.D2.Acq.DtbSerial
    %
    % Fixes drift and sigmaSq to zero.
    %
    % 2016 YK wrote the initial version.
    
%% Init
methods
    function add_params0(W)
        W.add_params0@Fit.D2.Short.Acq.Dtb;
        W.add_params0@Fit.D2.Inh.DtbDensitySerial;
    end
end
end