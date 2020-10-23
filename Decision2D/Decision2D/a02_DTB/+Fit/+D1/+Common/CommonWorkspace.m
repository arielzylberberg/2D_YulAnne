classdef CommonWorkspace ...
        < Fit.Common.CommonWorkspace
    %
    % 2015 YK wrote the initial version.
methods
    function set_Data(W, obj_or_name)
        if nargin < 2, obj_or_name = ''; end
        obj_or_name = W.enforce_class('Fit.D1.Common.DataChRtPdf', ...
            obj_or_name);
        W.set_Data@Fit.Common.CommonWorkspace(obj_or_name);
    end
end    
end