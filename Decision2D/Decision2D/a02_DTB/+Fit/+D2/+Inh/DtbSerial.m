classdef DtbSerial < Fit.D2.Inh.Dtb
    % Fit.D2.Inh.DtbSerial
    %
    % Fixes drift_fac and sigmaSq_fac to zero.
    %
    % 2016 YK wrote the initial verison.
methods
    function add_params0(W)
        W.add_params0@Fit.D2.Inh.Dtb;
        for prefix = {'drift', 'sigmaSq'}
            for postfix = {'1_2', '2_1'}
                f = [prefix{1}, '_fac_together_dim', postfix{1}];

                W.th0.(f) = 0;
                W.fix_to_th0_(f);
            end
        end
    end
end
end