classdef TndSim < Fit.D2.Common.Tnd
    % 2016 YK wrote the initial version.
properties
    mus_sim = [];
    sds_sim = [];
    pdf_tnd_sim = [];
    ratio_thres = 1.2;
end
methods
    function [tf_passed, Fl, res] = test_fit_sim(W)
        %%
        W.simulate;
        Fl = W.get_Fl;
        Fl.fit;
        
        %%
        mus_fit = Fl.W.get_mus;
        sds_fit = Fl.W.get_sds;
        
        mus_dif = mus_fit - W.mus_sim;
        sds_dif = sds_fit - W.sds_sim;
        
        mus_ratio = mus_fit ./ W.mus_sim;
        sds_ratio = sds_fit ./ W.sds_sim;
        
        disp('mus_fit:');
        disp(mus_fit);
        disp('sds_fit:');
        disp(sds_fit);
        
        disp('mus_dif:');
        disp(mus_dif);
        disp('sds_dif:');
        disp(sds_dif);
        
        %%
        max_abs_log_ratio = max(abs([
            vVec(log(mus_ratio))
            vVec(log(sds_ratio))
            ]));
        tf_passed = max_abs_log_ratio <= log(W.ratio_thres);
        
        %%
        mus_dif_thres = min(diff(sort(W.mus_sim(:))));
        sds_dif_thres = min(diff(sort(W.sds_sim(:))));
        
        tf_passed = tf_passed ...
            & all(abs(mus_dif(:)) < mus_dif_thres) ...
            & all(abs(sds_dif(:)) < sds_dif_thres);
        
        res = packStruct( ...
            mus_ratio, sds_ratio, ...
            mus_dif, sds_dif, ...
            mus_dif_thres, sds_dif_thres, ...
            max_abs_log_ratio);
        
        %%
        if nargout == 0
            assert(tf_passed);
        end
    end
    function simulate(W)
        mus = [0.2, 0.3; 0.25, 0.4];
        sds = mus ./ 2;
        
        W.pdf_tnd_sim = W.get_pdf_tnd_with_mu_sd(mus, sds);
        W.mus_sim = mus;
        W.sds_sim = sds;
    end
    function pred(W)
        % Do nothing
    end
    function [c, c_sep] = calc_cost(W)
        [c, c_sep] = nll_bin( ...
            W.get_pdf_tnd, W.pdf_tnd_sim);
    end
end
end