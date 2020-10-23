    % test Fit.D2.RT...
    %
    % 2015 YK wrote the initial version.

% Dtb = Fit.D1.Bounded.Dtb;
% Data = Fit.D1.Common.DataChRtPdf.demo;
% Data.load_data;
% Dtb.set_Data(Data);
Fl = Fit.D2.Bounded.Main.fit_aft_create('Td', 'Par');

res = Fl.res;

disp(res.th);
disp(res.se);

%% fval
assert(isequal_within(res.fval, -15391.5, 1e3)); % 10));

%% th_k
assert(isequal_within(res.th.Dtb__Dtb1__Drift__k, 36.2723, 1));
assert(isequal_within(res.th.Dtb__Dtb2__Drift__k,  3.9371, 0.5));

%% se_k
% try
%     assert(isequal_within(res.se.Dtb__Dtb1__Drift__k, 138.7732, 10));
% catch err
%     warning(err_msg(err));
% end
% try
%     assert(isequal_within(res.se.Dtb__Dtb2__Drift__k, 22.7759, 1));
% catch err
%     warning(err_msg(err));
% end

%% th_miss
assert(isequal_within(res.th.Miss__miss, 8.9037e-09, 1e-3));

%% se_miss
% try
%     assert(isequal_within(res.se.Miss__miss, 0.3023, 0.1));
% catch err
%     warning(err_msg(err));
% end
