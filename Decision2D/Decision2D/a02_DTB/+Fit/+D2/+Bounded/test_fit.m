    % test Fit.D2.RT...
    %
    % 2015 YK wrote the initial version.

% Dtb = Fit.D1.Bounded.Dtb;
% Data = Fit.D1.Common.DataChRtPdf.demo;
% Data.load_data;
% Dtb.set_Data(Data);
Fl = Fit.D2.Bounded.Main.fit_aft_create;
res = Fl.res;

disp(res.fval);
disp(res.th);
disp(res.se);

%% fval
assert(isequal_within(res.fval, -15558, 1e3)); % for FinDiffRelStep = 1e-3 %  -15375, 1e3)); % after increasing FinDiffRel % -15497, 1e3)); % 10));

%% th_k
assert(isequal_within(res.th.Dtb__Dtb1__Drift__k, 43.43, 1));
assert(isequal_within(res.th.Dtb__Dtb2__Drift__k,  4.08, 0.5));

%% se_k
% try
%     assert(isequal_within(res.se.Dtb__Dtb1__Drift__k, 1.2021, 0.3));
% catch err
%     warning(err_msg(err));
% end
% try
%     assert(isequal_within(res.se.Dtb__Dtb2__Drift__k, 0.2853, 0.05));
% catch err
%     warning(err_msg(err));
% end

%% th_miss
assert(isequal_within(res.th.Miss__miss, 1.5232e-09, 1e-2));

%% se_miss
% try
%     assert(isequal_within(res.se.Miss__miss, 0.0012, 0.001));
% catch err
%     warning(err_msg(err));
% end
