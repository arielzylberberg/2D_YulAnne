function p = conv_t_jt(p_row, p_col)
nt = length(p_col);
p = bsxfun(@times, p_row(:), p_col(:)');
for it = 1:nt
    p(it,:) = shift_pad(p(it,:), it - 1, 0);
end
end