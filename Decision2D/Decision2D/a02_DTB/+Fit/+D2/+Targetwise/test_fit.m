    % tests Fit.D2.RT.Targetwise
    %
    % 2015 YK wrote the initial version.

% Dtb = Fit.D1.Bounded.Dtb;
% Data = Fit.D1.Common.DataChRtPdf.demo;
% Data.load_data;
% Dtb.set_Data(Data);
Fl = Fit.D2.Targetwise.Main.fit_aft_create;
res = Fl.res;

disp(res.th);
disp(res.se);

%% fval
assert(isequal_within(res.fval, -17878.6, 10));

%% th_k
assert(isequal_within(res.th.Dtb__Drift__Drift1__k, 42.6, 1));

%% se_k
try
    assert(isequal_within(res.se.Dtb__Drift__Drift1__k, 2.3558, 0.5));
catch err
    warning(err_msg(err));
end

%% th_miss
assert(isequal_within(res.th.Miss__miss, 0.0034, 5e-3));

%% se_miss
try
    assert(isequal_within(res.se.Miss__miss, 0.0394, 0.02));
catch err
    warning(err_msg(err));
end
