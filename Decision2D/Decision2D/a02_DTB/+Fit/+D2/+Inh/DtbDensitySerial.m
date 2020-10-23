classdef DtbDensitySerial < Fit.D2.Inh.DtbDensity & Fit.D2.Inh.DtbSerial
    % Fit.D2.Inh.DtbDensitySerial
    %
    % sigmaSq = 0 is handled in calc_dtb, so this class is just to inherit
    % add_params0 from DtbSerial and the rest from DtbDensity.
    %
    % 2016 YK wrote the initial version.
end