function p_a_given_s = get_p_a_given_s(p_acq_given_a_s)
% p_a_given_s = get_p_a_given_s(p_acq_given_a_s)
%
% INPUT:
% p_acq_given_a_s(a+1, s)
%
% OUTPUT:
% p_a_given_s(a+1, s)
%
% See Yul Kang 2018, PhD Thesis, Chapter 6.

n_itv = size(p_acq_given_a_s, 1);
assert(size(p_acq_given_a_s, 2) == n_itv, ...
    'p_acq_given_a_s must be a square matrix!');

p_a_given_s = zeros(n_itv, n_itv);
p_a_given_s(1,1) = 1; % Initial condition

for s = 2:n_itv
    for a = 0:(s-1)
        if s == 1
            p_wo_acq = 0;
        else
            p_wo_acq = (1 - p_acq_given_a_s(a+1, s-1)) ...
                    .* p_a_given_s(a+1, s-1);
        end
        if s == 1 || a == 0
            p_w_acq = 0;
        else
            p_w_acq = p_acq_given_a_s(a, s-1) ...
                   .* p_a_given_s(a, s-1);
        end
        
        p_a_given_s(a+1, s) = p_w_acq + p_wo_acq;
    end
end