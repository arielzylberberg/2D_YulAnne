classdef Tnd < Fit.Common.Tnd & PartialSave
    % Doesn't modify init_bef_fit, pred, or get_cost.
    % Not called from FitFlow directly.
    %
    % 2015 YK wrote the initial version.
properties
    n_Tnd = 2; % 1 or 2
    to_use_diff = false;
    
    WTnd = Fit.Common.Tnd;
end
properties (Transient)
    pdf_tnd = [];
    RT_obs = [];
    RT_pred = [];
end
methods
function W = Tnd(varargin)
%     W.reset_on_save('WTnd', Fit.Common.Tnd);
%     W.empty_on_save({'pdf_tnd', 'RT_pred', 'RT_obs'});
    
    W.init_params0(varargin{:});
end
function init_params0(W, varargin)
    % FitWsTnd for calculation
    varargin2fields(W, varargin);
    W.WTnd = Fit.Common.Tnd('t', W.t, 'distrib', W.distrib);
    W.remove_constraints_all;
    switch W.n_Tnd
        case 1
            W.add_params0('mu');
            W.add_params0('disper')
        case 2
            if W.to_use_diff
                % Differs between 1D and 2D
                W.add_params0('mu');
                W.add_params0('disper')
                W.add_params({
                    {'mu_2m1', 0, -0.2, 0.2}
                    });
                W.add_constraints({
                    {'A', {'mu', 'mu_2m1'}, {[-1, -0.5], 0.1}} % mu - 0.5mu_UmD >= 0.1
                    {'A', {'mu', 'mu_2m1'}, {[-1,  0.5], 0.1}} % mu + 0.5mu_UmD >= 0.1
                    });
            else
                W.add_params0('mu', '1');
                W.add_params0('mu', '2');
                W.add_params0('disper', '1');
                W.add_params0('disper', '2');
            end
    end
end
function pdf_tnd = get_pdf_tnd(W)
    % pdf_tnd = get_pdf_tnd(W)
    % pdf_tnd: nt x 2 matrix.
    mus = W.get_mus;
    sds = W.get_sds;
    WTnd = Fit.Common.Tnd('t', W.t, 'distrib', W.distrib);
    
    pdf_tnd = zeros(length(W.t), 2);
    for ii = 1:2 % Differs between 1D and 2D
        WTnd.mu = mus(ii);
        WTnd.set_sd(sds(ii));
        pdf_tnd(:,ii) = WTnd.get_pdf_tnd; % Differs between 1D and 2D
    end
    W.pdf_tnd = pdf_tnd;
end
function pdf_RT = conv_Td_w_tnd(W, pdf_Td, pdf_tnd)
    if nargin < 3 || isempty(pdf_tnd)
        pdf_tnd = W.get_pdf_tnd;
    end
    
    nt = size(pdf_tnd, 1);
    assert(nt == size(pdf_Td, 1));
    
    DIM_CH = 2;
    n_ch = size(pdf_tnd, DIM_CH);
    
    pdf_RT = zeros(size(pdf_Td));
    for i_ch = 1:n_ch
        pdf_RT(:, :, i_ch) = conv_t(pdf_Td(:, :, i_ch), pdf_tnd(:, i_ch));
    end
    
    W.RT_pred = pdf_RT;
end
function pdf_RT = Td2RT(W, pdf_Td)
    % pdf_RT = Td2RT(W, pdf_Td)
    pdf_RT = W.conv_Td_w_tnd(pdf_Td);
end
function mus = get_mus(W)
    % Differs between 1D and 2D
    switch W.n_Tnd
        case 1
            mus = repmat(W.get_('mu'), [1 2]);
        case 2
            if W.to_use_diff
                mu0 = W.get_('mu');
                mu_2m1 = W.get_('mu_2m1');

                mus(1,1) = mu0 - mu_2m1 / 2;
                mus(1,2) = mu0 + mu_2m1 / 2;
            else
                mus(1,1) = W.get_('mu_1');
                mus(1,2) = W.get_('mu_2');
            end
    end
end
function set_mus(W, v)
    assert(~W.to_use_diff);
    
    v = rep2fit(v, [1, W.n_Tnd]);
    W.th.mu_1 = v(1);
    W.th.mu_2 = v(2);
end
function sds = get_sds(W)
    % Differs between 1D and 2D
    if W.n_Tnd == 2 && W.to_use_diff
        disper = W.get_('disper');
        sds = W.get_sd(W.get_mus, disper);
    else
        disper = [W.get_('disper_1'), W.get_('disper_2')];
        sds = W.get_sd(W.get_mus, disper);
    end
end
function set_sds(W, v)
    if W.n_Tnd == 2 && W.to_use_diff
        error('to_use_diff=1 is unsupported!');
    else
        v = rep2fit(v, [1, W.n_Tnd]);
        disper = W.calc_disper(v);
        W.th.disper_1 = disper(1);
        W.th.disper_2 = disper(2);
    end
end
function [h, x, y] = plot(W, varargin)
    x = W.t;
    y = W.get_pdf_tnd;
    for ii = 2:-1:1
%         subplotRC(2,1,ii,1);  % Differs between 1D and 2D
        h(ii) = plot(x, y(:,ii), varargin{:});
        hold on;
    end
    hold off;
end
end
methods (Static)
function WTnd = demo
    WTnd = Fit.D1.Bounded.Tnd;
    WTnd.plot;
end
end
end