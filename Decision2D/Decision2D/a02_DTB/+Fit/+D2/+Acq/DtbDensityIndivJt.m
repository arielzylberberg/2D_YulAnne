classdef DtbDensityIndivJt < Fit.D2.Inh.DtbDensityIndivJt
    % Fit.D2.Acq.DtbDensityIndivJt
    %
    % 2016 YK wrote the initial version.
%% User interface
methods
    function W = DtbDensityIndivJt(varargin)
        varargin2fields(W, varargin);
        W.add_params0;
    end
    function set_buffer_dur_sec_fix(W, v)
        W.th.buffer_dur_sec = v;
        W.fix_to_th_('buffer_dur_sec');
    end
end
%% Init
methods
    function add_params0(W)
        W.add_params0@Fit.D2.Inh.DtbDensity;
        W.add_params({
            {'buffer_dur_sec', ...
                0.2, 0.1, 1}
            });
        
%         for postfix = {'1_2', '2_1'}
%             for prefix = {'drift', 'sigmaSq'}
%                 name = [prefix{1} '_fac_together_dim', postfix{1}];
%                 
%                 % Serial model - fix both drift and sigmaSq to minimum
%                 % while together and deprioritized.
%                 W.th.(name) = W.lb.(name);
%                 W.fix_to_th_(name);
%             end
%         end
    end
end
%% Internal
methods
    function pred(W)
        % Adds unabs at the end of Td_pdf
        %
        % [td_pdf, unabs] = get_Td_pdf(W)
        
        buffer_dur_sec = W.th.buffer_dur_sec;
        t_ens = repmat({buffer_dur_sec}, [1, 2]);
        
        W.t_sign_rule_together_sec = t_ens;
        W.t_sign_rule_alone_sec = t_ens;
        
        %% Predict td_pdf using max_t = buffer_dur_sec        
        % Sign rule is applied via calc_dtb
        W.pred@Fit.D2.Inh.DtbDensityIndivJt;
    end
%     function [td_pdf, unabs] = calc_dtb(W, drift_cond_t, t, bound_t_ch, y, ...
%             varargin)
%         % td_pdf(t, cond, ch) = p(t, ch | cond)
%         % unabs(t, y, cond)  = p(t, y | cond)
%         
%         C = varargin2C(varargin, {
%             'apply_sign_rule', true
%             't_sign_rule_sec', W.buffer_dur_sec
%             });
%         
%         [td_pdf, unabs] = ...
%             dtb.pred.calc_dtb( ...
%                 drift_cond_t, t, bound_t_ch, y, ...
%                 C{:});
%             % TODO: Spectral just to match y with spectral.
%             %       May still use analytic if y's are matched.
%     end
end
end