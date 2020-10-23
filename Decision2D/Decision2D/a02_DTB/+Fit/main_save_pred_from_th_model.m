init_path;

%% Load models
L_comp = load('../Data/Fit.main_compare_dtb_all/main_compare_dtb_all.mat');

ds_file = L_comp.ds_file;
n_model = size(L_comp.mdl_disp_names, 1);
n_model1 = n_model + 1;
mdl_disp_names = [
    L_comp.mdl_disp_names
    {'data', 'Data'}];
models = mdl_disp_names(:,1);
subjs = L_comp.subjs;
n_subj = numel(subjs);

data = cell(n_subj, 1);
pred = cell(n_subj, n_model1);
cond_ch_incl_train = cell(n_subj, 1);
cond_ch_incl_valid = cell(n_subj, 1);
S0_file = cell(n_subj, n_model1);
S_file = cell(n_subj, n_model1);

%%
model_th = 'min_sub';
[~, ix_th] = ismember(model_th, models);
ths = cell(1, n_subj);

for i_subj = 1:n_subj
    file1 = ds_file.(model_th){i_subj};
    L1 = load(file1, 'res');
    ths{i_subj} = L1.res.th;
end

th_src = [
    {'Dtb__log10_sigmaSq_st_1'              }
    {'Dtb__Drift1__k'                       }
    {'Dtb__Drift1__bias'                    }
    {'Dtb__Drift1__bias_irr_1'              }
    {'Dtb__Drift1__bias_irr_2'              }
    {'Dtb__Drift1__bias_irr_3'              }
    {'Dtb__Drift1__bias_irr_4'              }
    {'Dtb__Drift1__bias_irr_5'              }
    {'Dtb__Drift1__bias_irr_6'              }
    {'Dtb__Drift1__bias_irr_7'              }
    {'Dtb__Drift1__bias_irr_8'              }
    {'Dtb__Drift1__bias_irr_9'              }
    {'Dtb__Bound1__b'                       }
    {'Dtb__Bound1__bias'                    }
    {'Dtb__Bound1__b_logbsum'               }
    {'Dtb__Bound1__b_logitmean'             }
    {'Dtb__Bound1__b_asym'                  }
    {'Dtb__SigmaSq1__log10_sigmaSq_max_cond'}
    {'Dtb__log10_sigmaSq_st_2'              }    
    {'Dtb__Drift2__k'                       }
    {'Dtb__Drift2__bias'                    }
    {'Dtb__Drift2__bias_irr_1'              }
    {'Dtb__Drift2__bias_irr_2'              }
    {'Dtb__Drift2__bias_irr_3'              }
    {'Dtb__Drift2__bias_irr_4'              }
    {'Dtb__Drift2__bias_irr_5'              }
    {'Dtb__Drift2__bias_irr_6'              }
    {'Dtb__Drift2__bias_irr_7'              }
    {'Dtb__Drift2__bias_irr_8'              }
    {'Dtb__Drift2__bias_irr_9'              }
    {'Dtb__Bound2__b'                       }
    {'Dtb__Bound2__bias'                    }
    {'Dtb__Bound2__b_logbsum'               }
    {'Dtb__Bound2__b_logitmean'             }
    {'Dtb__Bound2__b_asym'                  }
    {'Dtb__SigmaSq2__log10_sigmaSq_max_cond'}
    {'Tnd__mu_1_1'                          }
    {'Tnd__disper_1_1'                      }
    {'Tnd__mu_1_2'                          }
    {'Tnd__disper_1_2'                      }
    {'Tnd__mu_2_1'                          }
    {'Tnd__disper_2_1'                      }
    {'Tnd__mu_2_2'                          }
    {'Tnd__disper_2_2'                      }
    {'Miss__logit_miss'                     }
    ];
th_dst = [
    {'Dtb__Dtb1__log10_sigmaSq_st'               }
    {'Dtb__Dtb1__Drift__k'                       }
    {'Dtb__Dtb1__Drift__bias'                    }
    {'Dtb__Dtb1__Drift__bias_irr_1'              }
    {'Dtb__Dtb1__Drift__bias_irr_2'              }
    {'Dtb__Dtb1__Drift__bias_irr_3'              }
    {'Dtb__Dtb1__Drift__bias_irr_4'              }
    {'Dtb__Dtb1__Drift__bias_irr_5'              }
    {'Dtb__Dtb1__Drift__bias_irr_6'              }
    {'Dtb__Dtb1__Drift__bias_irr_7'              }
    {'Dtb__Dtb1__Drift__bias_irr_8'              }
    {'Dtb__Dtb1__Drift__bias_irr_9'              }
    {'Dtb__Dtb1__Bound__b'                       }
    {'Dtb__Dtb1__Bound__bias'                    }
    {'Dtb__Dtb1__Bound__b_logbsum'               }
    {'Dtb__Dtb1__Bound__b_logitmean'             }
    {'Dtb__Dtb1__Bound__b_asym'                  }
    {'Dtb__Dtb1__SigmaSq__log10_sigmaSq_max_cond'}
    {'Dtb__Dtb2__log10_sigmaSq_st'               }
    {'Dtb__Dtb2__Drift__k'                       }
    {'Dtb__Dtb2__Drift__bias'                    }
    {'Dtb__Dtb2__Drift__bias_irr_1'              }
    {'Dtb__Dtb2__Drift__bias_irr_2'              }
    {'Dtb__Dtb2__Drift__bias_irr_3'              }
    {'Dtb__Dtb2__Drift__bias_irr_4'              }
    {'Dtb__Dtb2__Drift__bias_irr_5'              }
    {'Dtb__Dtb2__Drift__bias_irr_6'              }
    {'Dtb__Dtb2__Drift__bias_irr_7'              }
    {'Dtb__Dtb2__Drift__bias_irr_8'              }
    {'Dtb__Dtb2__Drift__bias_irr_9'              }
    {'Dtb__Dtb2__Bound__b'                       }
    {'Dtb__Dtb2__Bound__bias'                    }
    {'Dtb__Dtb2__Bound__b_logbsum'               }
    {'Dtb__Dtb2__Bound__b_logitmean'             }
    {'Dtb__Dtb2__Bound__b_asym'                  }
    {'Dtb__Dtb2__SigmaSq__log10_sigmaSq_max_cond'}
    {'Tnd__mu_1_1'                               }
    {'Tnd__disper_1_1'                           }
    {'Tnd__mu_1_2'                               }
    {'Tnd__disper_1_2'                           }
    {'Tnd__mu_2_1'                               }
    {'Tnd__disper_2_1'                           }
    {'Tnd__mu_2_2'                               }
    {'Tnd__disper_2_2'                           }
    {'Miss__logit_miss'                          }
    ];
%%
Ls = cell(n_subj, n_model);
for i_subj = 1:n_subj
    
    for i_model = 1:n_model1
        if i_model == n_model1 % use data
            L = Ls{i_subj, 1};
            W = L.Fl.W;
            t = W.t(:);
            pred1 = W.Data.RT_data_pdf;
            pred1 = bsxfun(@rdivide, pred1, sums(pred1, [1, 4, 5]));
        else
            model = models{i_model};
            file1 = ds_file.(model){i_subj};
            fprintf('Loading S%d/%d, model %d/%d: %s\n', ...
                i_subj, n_subj, i_model, n_model, file1);
            L1 = load(file1);
            th = L1.Fl.res.th;
            
            th0 = ths{i_subj};
            for i_th = 1:numel(th_src)
                th_src1 = th_src{i_th};
                th_dst1 = th_dst{i_th};
                if isfield(th, th_src1)
                    th.(th_src1) = th0.(th_src1);
                elseif isfield(th, th_dst1)
                    th.(th_dst1) = th0.(th_src1);
                else
                    warning('%s lacks th.(%s) or (%s)', ...
                        model, th_src1, th_dst1);
                end
            end
            L1.Fl.res.th = th;

            W = L1.Fl.W;
            W.th = th;
            W.get_cost;
%             L1.Fl.res2W;
            Ls{i_subj, i_model} = L1;

            t = W.t(:);
%             W.pred;
            pred1 = W.Data.RT_pred_pdf;
        end
        data1 = W.Data.RT_data_pdf;        
        data{i_subj} = data1;
        pred{i_subj, i_model} = pred1;
        cond_ch_incl_train{i_subj} = W.get_cond_ch_to_include_train;
        cond_ch_incl_valid{i_subj} = W.get_cond_ch_to_include_train;
        S0_file{i_subj, i_model} = W.S0_file;
        S_file{i_subj, i_model} = W.S_file;
    end
end

%%
file = [
    '../Data/Fit.RewardRate.main_reward_rate/pred_data_by_model_w_th_', ...
    model_th];
mkdir2(fileparts(file));
save(file, 'models', 'mdl_disp_names', ...
    'ds_file', 'data', 'pred', 'subjs', ...
    'cond_ch_incl_train', 'cond_ch_incl_valid', ...
    'S0_file', 'S_file');
fprintf('Saved pred and data to %s\n', file);
