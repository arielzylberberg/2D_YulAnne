function [td_pdf, unabs] = apply_sign_rule(td_pdf, unabs, y, varargin)
    % td_pdf(t, cond, ch) = p(t, ch | cond)
    % unabs(t, y, cond) = p(y | t, cond)
    % y(ix) : a vector of evidence level.
    S = varargin2S(varargin, {
        't_sec', []
        't_en_sec', []
        't_en_bin', []
        });
    if isempty(S.t_en_bin)
        if isempty(S.t_en_sec)
            S.t_en_bin = size(td_pdf, 1);
        else
            [~, S.t_en_bin] = min(abs(S.t_sec - S.t_en_sec));
        end
    end
    t_en = S.t_en_bin;

    y_neg = y < 0;
    y_pos = y > 0;
    y_0 = y == 0;
    tf_y_0_exists = any(y_0);

    n_drift = size(td_pdf, 2);
    for drift = 1:n_drift
        td_pdf(t_en, drift, 1) = td_pdf(t_en, drift, 1) + ...
            sum(unabs(t_en, y_neg, drift));        
        td_pdf(t_en, drift, 2) = td_pdf(t_en, drift, 2) + ...
            sum(unabs(t_en, y_pos, drift));

        if tf_y_0_exists
            % split in half
            for ch = 1:2
                td_pdf(t_en, drift, ch) = td_pdf(t_en, drift, ch) ...
                    + sum(unabs(t_en, y_0, drift)) / 2;
            end
        end
    end
    td_pdf((t_en + 1):end, :,:) = 0;
    unabs(t_en:end, :, :) = 0; % At t_en, all unabs is added to td_pdf.
end