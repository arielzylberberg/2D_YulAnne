classdef gNet < handle
    
    properties(SetAccess = public)
        inputSeq;           % input sequence
        respSeq;            % response sequence (specifies when response is recorded)
        
        tau                 % netinput integration parameter
        dt                  % euler integration step size
        alpha;              % learning rate
        
        gain;               % gain parameter of goal layer
        mag_noise           % noise in input activation
        
        W_input;            % network weights from input to goal layer
        W_selfexcite;       % self excitation weights of goal units
        W_inhib             % lateral inhibition weights between goal units
        
        curr_act;           % current goal unit activity
        curr_net;           % current goal unit net input
        curr_noise;              % noise in net activation
        
        act_log;            % logs activity of goal units
        net_log;            % logs net input of goal units  
        input_log;          % logs in external input
        MSE_log;            % logs MSE
        MSE_respLog         % logs MSE at response time
        gain_respLog        % logs learned gain parameter for each trial
        
        runODE              % run ODE?
        % plot variables
        plot_run             % plot network while running?
        AS_Handle            % activation state handle
        VF_Handle            % vector field handle
        S_Handle             % surface handle
        R_Handle             % response acquisition handle
        unit_Handle          % graphics handle for units

    end
    
    methods
        
        % constructor
        function this = gNet(varargin)
            if(isempty(varargin))
                this.gain = 1;
            end
            
            if(isa(varargin{1},'gNet'))
                copy = varargin{1};
                this.inputSeq       = copy.inputSeq;
                this.respSeq       = copy.respSeq;
                this.tau        = copy.tau;
                this.gain         = copy.gain;
                this.W_input          = copy.W_input;
                this.W_selfexcite        = copy.W_selfexcite;
                this.W_inhib        = copy.W_inhib;
                this.curr_act    = copy.curr_act;
                this.curr_net    = copy.curr_net;
                this.act_log    = copy.act_log;
                this.net_log    = copy.net_log;
                this.MSE_log          = copy.MSE_log;
                this.plot_run = copy.plot_run;
                this.AS_Handle = copy.AS_Handle;
                this.VF_Handle = copy.VF_Handle;
                this.S_Handle = copy.S_Handle;
                this.S_Handle = copy.S_Handle;
                this.mag_noise = copy.mag_noise;
                this.curr_noise = copy.curr_noise;
                this.input_log = copy.input_log;
                this.MSE_respLog = copy.MSE_respLog;
                this.gain_respLog = copy.gain_respLog;
                this.runODE = copy.runODE;
                this.dt = copy.dt;
            else
               this.gain = varargin{1};  
            end
            
            if(length(varargin)>=2)
               this.tau = varargin{2};  
            else
               this.tau = 0.3; 
            end
            
            if(length(varargin)>=3)
               this.alpha = varargin{3};  
            else
               this.alpha = -1; 
            end
            
            if(length(varargin)>=4)
               this.mag_noise = varargin{4};  
            else
               this.mag_noise = 0;
            end
            
            if(length(varargin)>=5)
               this.plot_run = varargin{5};  
            else
               this.plot_run = 1;
            end
            
            if(length(varargin)>=8)
               this.W_input = varargin{6};  
               this.W_selfexcite = varargin{7}; 
               this.W_inhib = varargin{8}; 
            else
               this.W_input = 1; 
               this.W_selfexcite = 1;
               this.W_inhib = 1;
            end
            
            this.dt = 0.01;
            this.runODE = 0;

        end
        
        % configure the net
        function configure(this, varargin)
            
            % evaluate input arguments
            if(length(varargin)==1)
                if(isa(varargin{1},'gNet'))
                    netObj = varargin{1};
                    this.inputSeq       = netObj.inputSeq;
                    this.respSeq       = netObj.respSeq;
                    this.tau        = netObj.tau;
                    this.gain         = netObj.gain;
                    this.W_input          = netObj.W_input;
                    this.W_selfexcite        = netObj.W_selfexcite;
                    this.W_inhib        = netObj.W_inhib;
                    this.curr_act    = copy.curr_act;
                    this.curr_net    = copy.curr_net;
                    this.act_log    = netObj.act_log;
                    this.net_log    = netObj.net_log;
                    this.MSE_log          = netObj.MSE_log;
                    this.plot_run = netObj.plot_run;
                    this.AS_Handle = netObj.AS_Handle;
                    this.VF_Handle = netObj.VF_Handle;
                    this.S_Handle = netObj.S_Handle;
                    this.mag_noise = netObj.mag_noise;
                    this.curr_noise = netObj.curr_noise;
                    this.input_log = netObj.input_log;
                    this.MSE_respLog = netObj.MSE_respLog;
                    this.gain_respLog = netObj.gain_respLog;
                end
            else
                if(length(varargin)>=3)
                   this.W_input = varargin{1};
                   this.W_selfexcite = varargin{2};
                   this.W_inhib = varargin{3};
                else
                   if(isempty(this.W_input) || isempty(this.W_selfexcite) || isempty(this.W_inhib))
                       error('Weights need to be specified in order to configure network.');
                   end  
                end
            end
            
            % initialize net input & activation for goal units
            this.curr_act = [0 0];
            this.curr_net = [0 0];
        end
        
        % setODE
        function setODE(this, ODEswitch, varargin)
           if(ODEswitch)
               this.runODE = 1;
           end
           if(~isempty(varargin))
              this.dt = varargin{1}; 
           end
        end
        % train a batch
        function [MSE, goal_act, goal_net] = trainSequence(this, varargin)
            
            % parse input
            if(~isempty(varargin))
                inputSequence = varargin{1};
            else
                inputSequence = this.inputSeq;
            end
            
            if(length(varargin) >=2)
                respSequence = varargin{1};
            else
                respSequence = this.respSeq;
            end
            
            % check if input & response sequences are of the same length
            if(~isempty(respSequence))
                if(length(inputSequence) ~= length(respSequence))
                    error('Task data has to have same number of rows as input data.');
                end
            end
            
            goal_act = zeros(2,length(inputSequence));
            goal_net = zeros(2,length(inputSequence));
            MSE = zeros(2,length(inputSequence));
            this.MSE_respLog = [];
            this.gain_respLog = [];

            % initialize net input & activation for goal units
            this.curr_act = [0 0];
            this.curr_net = [0 0];
            this.gain = 3; % to remove
            for t = 1:length(inputSequence);
                % get current state variables
                
                [goal_act(:,t), goal_net(:,t), MSE(:,t)] = this.runTimeStep(inputSequence(:,t));
                
                this.act_log = goal_act;
                this.net_log = goal_net;
                this.MSE_log = MSE;
                
                if(respSequence(t) == 2)
                   [delta_gain] = this.trainTrial(inputSequence, respSequence, t);
                   this.MSE_respLog = [this.MSE_respLog MSE(:,t)];
                   this.gain_respLog = [this.gain_respLog this.gain+sum(delta_gain)]; % for now, log the common gain
                end
            end
            
        end
        
        function [MSE, goal_act, goal_net] = trainSequenceODE(this, varargin)
            
            % parse input
            if(~isempty(varargin))
                inputSequence = varargin{1};
            else
                inputSequence = this.inputSeq;
            end
            
            if(length(varargin) >=2)
                respSequence = varargin{1};
            else
                respSequence = this.respSeq;
            end
            
            % check if input & response sequences are of the same length
            if(~isempty(respSequence))
                if(length(inputSequence) ~= length(respSequence))
                    error('Task data has to have same number of rows as input data.');
                end
            end
            
            g_set = [0.1:0.1:5];
            MSE_sum = zeros(1,length(g_set));
            for i = 1:length(g_set)
                goal_act = zeros(2,length(inputSequence));
                goal_net = zeros(2,length(inputSequence));
                MSE = zeros(2,length(inputSequence));
                this.MSE_respLog = [];
                this.gain_respLog = [];

                % initialize net input & activation for goal units
                this.curr_act = [0 0];
                this.curr_net = [0 0];
                this.gain = g_set(i); % to remove
                for t = 1:length(inputSequence);
                    % get current state variables

                    [goal_act(:,t), goal_net(:,t), MSE(:,t)] = this.runTimeStepODE(inputSequence(:,t));

                    this.act_log = goal_act;
                    this.net_log = goal_net;
                    this.MSE_log = MSE;

                    if(respSequence(t) == 2)
                       this.MSE_respLog = [this.MSE_respLog MSE(:,t)];
                       this.gain_respLog = [this.gain_respLog this.gain]; % for now, log the common gain
                    end
                end
                MSE_sum(i) = sum(sum(this.MSE_respLog,1));
            end
            this.gain = g_set(MSE_sum == min(MSE_sum));
            MSE = min(MSE_sum);
            
        end
        
        % train a trial
        function [mod_gain] = trainTrial(this, inputSequence, respSequence, t)
               
               % get index of response start
               respStarts = find(respSequence == 1);
               respStarts = respStarts(respStarts < t);
               trainStart = max(respStarts);
               
               delta_gain = zeros(2,(t-trainStart)+1);
               deltaErr = zeros(2,(t-trainStart)+1);
               % backpropagate errors until beginning of response
               % acquisition (respSequence == 1)
               for i = 0:1:(t-trainStart)
                   
                   if(i == 0)
                       % special case for the output layer at time t
                       
                       % gain adjustment at output layer
                       delta_gain(1,i+1) = - this.alpha * (this.act_log(1,t) - inputSequence(1,t)) ...
                                                        * this.net_log(1,t) * this.act_log(1,t) * (1 - this.act_log(1,t));
                       delta_gain(2,i+1) = - this.alpha * (this.act_log(2,t) - inputSequence(2,t)) ...
                                                        * this.net_log(2,t) * this.act_log(2,t) * (1 - this.act_log(2,t));
                       % error delta for output layer                             
                       deltaErr(1,i+1) = (this.act_log(1,t) - inputSequence(1,t)) * this.gain ...
                                                        * this.act_log(1,t) * (1 - this.act_log(1,t));
                       deltaErr(2,i+1) = (this.act_log(2,t) - inputSequence(2,t)) * this.gain ...
                                                        * this.act_log(2,t) * (1 - this.act_log(2,t));                             
                   else
                       % treat each previous time point as hidden layer
                       
                       % gain adjustment at hidden layer
                       delta_gain(1,i+1) = - this.alpha * (deltaErr(1,i) * this.tau * this.W_selfexcite(1) + deltaErr(2,i) * this.tau * this.W_inhib(2)) ...
                                                        * this.net_log(1,t-i) * this.act_log(1,t-i) * (1 - this.act_log(1,t-i));
                       delta_gain(2,i+1) = - this.alpha * (deltaErr(1,i) * this.tau * this.W_inhib(1) + deltaErr(2,i) * this.tau * this.W_selfexcite(2)) ...
                                                        * this.net_log(2,t-i) * this.act_log(2,t-i) * (1 - this.act_log(2,t-i));
                       % error delta for hidden layer
                       deltaErr(1,i+1) = (deltaErr(1,i) * this.tau * this.W_selfexcite(1) + deltaErr(2,i) * this.tau * this.W_inhib(2)) ...
                                                        * this.gain * this.act_log(1,t-i) * (1 - this.act_log(1,t-i));
                       deltaErr(2,i+1) = (deltaErr(1,i) * this.tau * this.W_inhib(1) + deltaErr(2,i) * this.tau * this.W_selfexcite(2)) ...
                                                        * this.gain * this.act_log(2,t-i) * (1 - this.act_log(2,t-i));
                   end
                   
               end
               
               % finally add gain
               mod_gain = sum(delta_gain,2);
               this.gain = this.gain + sum(mod_gain);
               
        end

        % run through a data set
        function [goal_act MSE] = runSequence(this, varargin)
            if(~isempty(varargin))
                inputSequence = varargin{1};
            else
                inputSequence = this.inputSeq;
            end
            
            if(length(varargin) >=2)
                respSequence = varargin{1};
            else
                respSequence = this.respSeq;
            end
            
            % check if input & response sequences are of the same length
            if(~isempty(respSequence))
                if(length(inputSequence) ~= length(respSequence))
                    error('Task data has to have same number of rows as input data.');
                end
            end
            
            goal_act = zeros(2,length(inputSequence));
            goal_net = zeros(2,length(inputSequence));
            MSE = zeros(2,length(inputSequence));

            % initialize net input & activation for goal units
            this.curr_act = [0 0];
            this.curr_net = [0 0];
            
            if(this.plot_run)
               filename = 'testnew51.gif';
               figure(1);
            end
            
            for t = 1:length(inputSequence);
                % get current state variables
                if(this.runODE)
                    [goal_act(:,t), goal_net(:,t), MSE(:,t)] = this.runTimeStepODE(inputSequence(:,t));
                else 
                    [goal_act(:,t), goal_net(:,t), MSE(:,t)] = this.runTimeStep(inputSequence(:,t));
                end
                
                % plot
                if(this.plot_run)
                   this.plotNet(inputSequence(:,t), respSequence(t)); 
                   frame = getframe(1);
                   im = frame2im(frame);
                   [imind,cm] = rgb2ind(im,256);
                   if t == 1;
                        imwrite(imind,cm,filename,'gif', 'Loopcount',inf);
                   else
                        imwrite(imind,cm,filename,'gif','WriteMode','append', 'DelayTime', 0.05);
                   end
                end
            end
            
            this.act_log = goal_act;
            this.net_log = goal_net;
            this.MSE_log = MSE;
            
        end
        
        % run a trial
        function [goal_act, goal_net, MSE] = runTimeStep(this, input)
            
            % calculate new net input
            this.curr_noise = [normrnd(0,this.mag_noise*sqrt(1)) normrnd(0,this.mag_noise*sqrt(1))];
   
            goal_net(1) = this.tau*(this.W_input(1)*input(1) + this.W_selfexcite(1)*this.curr_act(1) + this.W_inhib(1)*this.curr_act(2) + this.curr_noise(1)) + (1-this.tau)*this.curr_net(1);
            goal_net(2) = this.tau*(this.W_input(2)*input(2) + this.W_inhib(2)*this.curr_act(1) + this.W_selfexcite(2)*this.curr_act(2) + this.curr_noise(2)) + (1-this.tau)*this.curr_net(2);
            
%            disp(['goal net: ' num2str(goal_net)]);
            goal_act(1) = 1./(1+exp(-this.gain*goal_net(1)));
            goal_act(2) = 1./(1+exp(-this.gain*goal_net(2)));
            
            this.curr_net = goal_net;
            this.curr_act = goal_act;
            
            MSE = [0.5*(goal_act(1)-input(1))^2, 0.5*(goal_act(2)-input(2))^2];
            
        end
        
        function [goal_act, goal_net, MSE] = runTimeStepODE(this, input)
            
            % noise component
            this.curr_noise = [normrnd(0,this.mag_noise*sqrt(1)) normrnd(0,this.mag_noise*sqrt(1))];
            
            % current net input
            goal_net(1) = this.W_input(1)*input(1) + this.W_selfexcite(1)*this.curr_act(1) + this.W_inhib(1)*this.curr_act(2) + this.curr_noise(1);
            goal_net(2) = this.W_input(2)*input(2) + this.W_inhib(2)*this.curr_act(1) + this.W_selfexcite(2)*this.curr_act(2) + this.curr_noise(2);
            
            % euler integration
            goal_act(1) = this.curr_act(1) + this.dt*(-this.curr_act(1) + 1./(1+exp(-this.gain*goal_net(1))));
            goal_act(2) = this.curr_act(2) + this.dt*(-this.curr_act(2) + 1./(1+exp(-this.gain*goal_net(2))));
            
            this.curr_net = goal_net;
            this.curr_act = goal_act;
            
            MSE = [0.5*(goal_act(1)-input(1))^2, 0.5*(goal_act(2)-input(2))^2];
        end
        
        function setSequence(this, inputSequence, responseSequence)
           this.inputSeq = inputSequence;

           if(length(responseSequence) ~= length(inputSequence))
               error('Input sequence and response sequence need to be of the same length');
           else
                this.respSeq = responseSequence;
           end

        end
        
        function plotNet_CogSci(this, input, activationTrajectory)
            
            % calculate vector field
            grid = 0.00:0.1:1;
            act_grid = combvec(grid,grid);
            actNew_grid = zeros(size(act_grid));
            net_grid = zeros(size(act_grid));
            netNew_grid = zeros(size(act_grid));
            
            if(~this.runODE)
                
                % calculate vector field
                grid = 0.00:0.1:1;
                act_grid = combvec(grid,grid);
                actNew_grid = zeros(size(act_grid));
                net_grid = zeros(size(act_grid));
                netNew_grid = zeros(size(act_grid));
                
                net_grid(1,:) = -(log(1./(act_grid(1,:))-1))/this.gain;
                net_grid(2,:) = -(log(1./(act_grid(2,:))-1))/this.gain;

                netNew_grid(1,:) = this.tau.*(this.W_input(1)*input(1) + this.W_selfexcite(1).*act_grid(1,:) + this.W_inhib(1).*act_grid(2,:)) + (1-this.tau) .* net_grid(1,:);
                netNew_grid(2,:) = this.tau.*(this.W_input(2)*input(2) + this.W_inhib(2).*act_grid(1,:) + this.W_selfexcite(2).*act_grid(2,:)) + (1-this.tau) .* net_grid(2,:);

                actNew_grid(1,:) = 1./(1+exp(-this.gain*netNew_grid(1,:)));
                actNew_grid(2,:) = 1./(1+exp(-this.gain*netNew_grid(2,:)));
                act_vec = actNew_grid - act_grid;
                
                % calculate surface
                normvec = zeros(1,size(act_grid,2));

                for i = 1:size(act_vec,2)
                   normvec(i) = norm(act_vec(:,i));
                end

                Z = reshape(normvec,length(grid),length(grid));

                % if no figure available, create new plot
                if(isempty(this.AS_Handle) || isempty(this.VF_Handle) || isempty(this.S_Handle))
                    hFig = figure(1);
                    set(hFig, 'Position', [200 200 200 200])
                    clf;
                    set(gcf,'Color',[1 1 1]);
                    
                    hold on;
                    lower_limit = 0.00;
                    upper_limit = 1;
                    xlim([lower_limit upper_limit]);
                    ylim([lower_limit upper_limit]);
                    axis equal;

                    % plot vector field
                    this.VF_Handle = quiver(act_grid(1,:),act_grid(2,:),act_vec(1,:),act_vec(2,:),'Color',[0 0 0 ]);

                    % plot surface
                    [~,this.S_Handle] = contour(grid,grid,Z');

                    hold off;
                    xlabel('act_1 ','FontSize',20, 'FontWeight','bold');
                    ylabel('act_2 ','FontSize',20, 'FontWeight','bold');
                    set(gca, 'XTick', [0 1]);
                    set(gca, 'YTick', [0 1]);

                    set(gca, 'FontSize', 16);
                                     
                    % plot activation trajectory
                    line(activationTrajectory(1,:),activationTrajectory(2,:),'color',[0 0 0]/256, 'LineWidth',1);
                    for t = 1:size(activationTrajectory, 2)
                        if(t == 1)
                            color = [192 0 1]/256;
                        elseif(t == size(activationTrajectory, 2))
                            color = [47 85 151]/256;
                        else
                            color = [0 0 0]/256;
                        end
                        line(activationTrajectory(1,t),activationTrajectory(2,t),'color', color,'marker','.','markersize',25);
                    end
                    
   

                else                
                    disp('Figure already exists');
                end

                
            else
                this.plotNet(input, 0);
            end
            
        end
        
        function plotNet(this, input, respAcquisition)
            
            % calculate vector field
            grid = 0.00:0.1:1;
            act_grid = combvec(grid,grid);
            actNew_grid = zeros(size(act_grid));
            net_grid = zeros(size(act_grid));
            netNew_grid = zeros(size(act_grid));
            
            net_grid(1,:) = -(log(1./(act_grid(1,:))-1))/this.gain;
            net_grid(2,:) = -(log(1./(act_grid(2,:))-1))/this.gain;

            netNew_grid(1,:) = this.tau.*(this.W_input(1)*input(1) + this.W_selfexcite(1).*act_grid(1,:) + this.W_inhib(1).*act_grid(2,:)) + (1-this.tau) .* net_grid(1,:);
            netNew_grid(2,:) = this.tau.*(this.W_input(2)*input(2) + this.W_inhib(2).*act_grid(1,:) + this.W_selfexcite(2).*act_grid(2,:)) + (1-this.tau) .* net_grid(2,:);

            actNew_grid(1,:) = 1./(1+exp(-this.gain*netNew_grid(1,:)));
            actNew_grid(2,:) = 1./(1+exp(-this.gain*netNew_grid(2,:)));
            act_vec = actNew_grid - act_grid;
            
            % calculate surface
            normvec = zeros(1,size(act_grid,2));

            for i = 1:size(act_vec,2)
               normvec(i) = norm(act_vec(:,i));
            end

            Z = reshape(normvec,length(grid),length(grid));

            % if no figure available, create new plot
            if(isempty(this.AS_Handle) || isempty(this.VF_Handle) || isempty(this.S_Handle))
                hFig = figure(1);
                set(hFig, 'Position', [200 200 800 400])
                clf;
                subplot(1,2,1);
                set(gcf,'Color',[1 1 1]);
                % plot activation state
                this.AS_Handle = line(this.curr_act(1),this.curr_act(2),'color',[96 181 204]/256,'marker','.','markersize',50);
                hold on;
                lower_limit = 0.01;
                upper_limit = 0.99;
                xlim([lower_limit upper_limit]);
                ylim([lower_limit upper_limit]);
                axis equal;
                
                % plot vector field
                this.VF_Handle = quiver(act_grid(1,:),act_grid(2,:),act_vec(1,:),act_vec(2,:),'Color',[1 1 1]);
                
                % plot surface
                [~,this.S_Handle] = contour(grid,grid,Z');
                
                % indicate response
                this.R_Handle = patch([lower_limit lower_limit upper_limit upper_limit], ...
                     [lower_limit upper_limit upper_limit lower_limit], ...
                     [0 0 0]);

                hold off;
                xlabel('activation a_A ','FontSize',20);
                ylabel('activation a_B ','FontSize',20);
                title('attractor dynamics ','FontSize',20,'FontWeight','bold');
                set(gca, 'FontSize', 16);
                this.plotNetAct(hFig);
                
                % added for black figures

                set(gcf, 'Color', 'k');
                set(gca, 'Color', 'k');
                set(gca, 'xColor', 'w');
                set(gca, 'yColor', 'w');

                % preserve background color when saving figure
                fig = gcf;
                fig.InvertHardcopy = 'off';
            else                
                % update plot
                set(this.AS_Handle, 'XData', this.curr_act(1), 'YData', this.curr_act(2)); 
                set(this.VF_Handle, 'XData', act_grid(1,:), 'YData', act_grid(2,:), 'UData', act_vec(1,:), 'VData', act_vec(2,:)); 
                set(this.S_Handle, 'ZData',Z'); drawnow
                set(this.unit_Handle(1),'FaceColor',([96 181 204]/256)*this.curr_act(1)); 
                set(this.unit_Handle(4),'FaceColor',([96 181 204]/256)*this.curr_act(2));
                set(this.unit_Handle(3),'FaceColor',([96 181 204]/256)*input(1));
                set(this.unit_Handle(2),'FaceColor',([96 181 204]/256)*input(2));
            end
            
            % update response acquisition frame
%             if(respAcquisition == 2)
%                 set(this.R_Handle,'FaceAlpha',0.5); drawnow
%             else
%                 set(this.R_Handle,'FaceAlpha',0); drawnow
%             end 
            set(this.R_Handle,'FaceAlpha',0); drawnow
            
            % pause to make smooth plot
            pause(0.01);
            
        end
        
        function plotNetAct(this, figure1)
            
        % Create ellipse
        this.unit_Handle(1) = annotation(figure1,'ellipse',[0.5985 0.576176470588237 0.089 0.17],...
            'FaceColor',[0 0 0], 'EdgeColor', [1 1 1]);

        % Create ellipse
        this.unit_Handle(2) =annotation(figure1,'ellipse',...
            [0.754055555555556 0.222450980392161 0.089 0.17], 'EdgeColor', [1 1 1]);

        % Create arrow
        annotation(figure1,'arrow',[0.642222222222222 0.642222222222222],...
            [0.397692810457516 0.572984749455338], 'Color', [1 1 1]);

        % Create arrow
        annotation(figure1,'arrow',[0.801111111111111 0.801111111111111],...
            [0.3951220043573 0.570413943355121], 'Color', [1 1 1]);

        % Create arrow
        annotation(figure1,'arrow',[0.682222222222222 0.761111111111111],...
            [0.612200435729848 0.612200435729848], 'Color', [1 1 1]);

        % Create arrow
        annotation(figure1,'arrow',[0.754444444444444 0.686666666666667],...
            [0.690631808278867 0.690631808278867], 'Color', [1 1 1]);

        % Create arrow
        annotation(figure1,'arrow',[0.563333333333333 0.604444444444444],...
            [0.675381263616558 0.712418300653595], 'Color', [1 1 1]);

        % Create arrow
        annotation(figure1,'arrow',[0.88 0.841111111111111],...
            [0.647058823529412 0.686274509803922], 'Color', [1 1 1]);

        % Create line
        annotation(figure1,'line',[0.564444444444444 0.6],...
            [0.676559912854031 0.636165577342048], 'Color', [1 1 1]);

        % Create line
        annotation(figure1,'line',[0.881111111111111 0.842222222222222],...
            [0.644880174291939 0.62962962962963], 'Color', [1 1 1]);

        % Create ellipse
        this.unit_Handle(3) =annotation(figure1,'ellipse',[0.5985 0.227200435729851 0.089 0.17], 'EdgeColor', [1 1 1], 'FaceColor',[0 0 0]);

        % Create ellipse
        this.unit_Handle(4) =annotation(figure1,'ellipse',[0.75475 0.573676470588236 0.089 0.17],...
            'FaceColor',[0 0 0], 'EdgeColor', [1 1 1]);
        
        % Create textbox
        annotation(figure1,'textbox',...
            [0.624444444444444 0.625272331154687 0.03 0.0926819172113295],...
            'String',{'a'},...
            'FontSize',24,...
            'FitBoxToText','off',...
            'Color', [1 1 1], ...
            'EdgeColor','none');

        % Create textbox
        annotation(figure1,'textbox',...
            [0.781111111111111 0.624880174291943 0.03 0.0926819172113295],...
            'String',{'a'},...
            'FontSize',24,...
            'Color', [1 1 1], ...
            'FitBoxToText','on',...
            'EdgeColor','none');

        % Create textbox
        annotation(figure1,'textbox',...
            [0.624444444444444 0.267581699346409 0.03 0.0926819172113295],...
            'String',{'I'},...
            'FontSize',24,...
            'Color', [1 1 1], ...
            'FitBoxToText','off',...
            'EdgeColor','none');

        % Create textbox
        annotation(figure1,'textbox',...
            [0.785555555555556 0.269760348583882 0.03 0.0926819172113295],...
            'String',{'I'},...
            'FontSize',24,...
            'Color', [1 1 1], ...
            'FitBoxToText','off',...
            'EdgeColor','none');

        % Create textbox
        annotation(figure1,'textbox',...
            [0.641111111111111 0.600915032679741 0.03 0.0926819172113295],...
            'String',{'A'},...
            'FontSize',20,...
            'Color', [1 1 1], ...
            'FitBoxToText','off',...
            'EdgeColor','none');

        % Create textbox
        annotation(figure1,'textbox',...
            [0.636666666666667 0.243224400871463 0.03 0.0926819172113295],...
            'String',{'A'},...
            'FontSize',20,...
            'Color', [1 1 1], ...
            'FitBoxToText','off',...
            'EdgeColor','none');

        % Create textbox
        annotation(figure1,'textbox',...
            [0.797777777777778 0.596165577342052 0.03 0.0926819172113295],...
            'String',{'B'},...
            'FontSize',20,...
            'Color', [1 1 1], ...
            'FitBoxToText','on',...
            'EdgeColor','none');

        % Create textbox
        annotation(figure1,'textbox',...
            [0.794444444444444 0.238474945533774 0.03 0.0926819172113295],...
            'String',{'B'},...
            'FontSize',20,...
            'Color', [1 1 1], ...
            'FitBoxToText','on',...
            'EdgeColor','none');

            % Create textbox
        annotation(figure1,'textbox',...
            [0.645277777777778 0.446581196581198 0.033611111111111 0.0741709401709409],...
            'String',{'+'},...
            'Color', [1 1 1], ...
            'FontSize',20,...
            'FitBoxToText','off',...
            'EdgeColor','none');

        % Create textbox
        annotation(figure1,'textbox',...
            [0.770833333333334 0.445811965811968 0.033611111111111 0.0741709401709409],...
            'String',{'+'},...
            'Color', [1 1 1], ...
            'FontSize',20,...
            'FitBoxToText','off',...
            'EdgeColor','none');

        % Create textbox
        annotation(figure1,'textbox',...
            [0.863055555555556 0.644529914529918 0.033611111111111 0.0741709401709409],...
            'String',{'+'},...
            'FontSize',20,...
            'Color', [1 1 1], ...
            'FitBoxToText','off',...
            'EdgeColor','none');

        % Create textbox
        annotation(figure1,'textbox',...
            [0.561944444444445 0.676581196581198 0.033611111111111 0.0741709401709409],...
            'String',{'+'},...
            'FontSize',20,...
            'Color', [1 1 1], ...
            'FitBoxToText','off',...
            'EdgeColor','none');

        % Create textbox
        annotation(figure1,'textbox',...
            [0.710833333333334 0.552649572649576 0.033611111111111 0.0741709401709409],...
            'String',{'-'},...
            'FontSize',20,...
            'Color', [1 1 1], ...
            'FitBoxToText','off',...
            'EdgeColor','none');

        % Create textbox
        annotation(figure1,'textbox',...
            [0.711944444444445 0.680854700854704 0.033611111111111 0.0741709401709409],...
            'String',{'-'},...
            'Color', [1 1 1], ...
            'FontSize',20,...
            'FitBoxToText','off',...
            'EdgeColor','none');
        end
        
        function plotDynamics(this, gains, timeSteps)
            
            % build input sequence & run simulation for different gains
            trajectory = zeros(length(gains),2,timeSteps+1);
            
            for i = 1:length(gains)
                
                this.gain = gains(i);
                [inputSequence, respSequence] = gNet.buildSwitchSequence(2,timeSteps,1,timeSteps);
                [goal_act ~] = this.runSequence(inputSequence, respSequence);
                trajectory(i,:,:) = goal_act(:,timeSteps:(timeSteps*2));
            
            end
            
            figure(1);
            clf;
            hold on;
            
            % color map for lines
            colorDataL = [linspace(0,128,length(gains)); linspace(0,128,length(gains)); linspace(0,128,length(gains))]'/255;
            
            % plot vector field
            input = [0 1];
            this.gain = gains(1);
            grid = 0.00:0.1:1;
            
            act_grid = combvec(grid,grid);
            actNew_grid = zeros(size(act_grid));
            net_grid = zeros(size(act_grid));
            netNew_grid = zeros(size(act_grid));
            
            net_grid(1,:) = -(log(1./(act_grid(1,:))-1))/this.gain;
            net_grid(2,:) = -(log(1./(act_grid(2,:))-1))/this.gain;

            netNew_grid(1,:) = this.tau.*(this.W_input(1)*input(1) + this.W_selfexcite(1).*act_grid(1,:) + this.W_inhib(1).*act_grid(2,:)) + (1-this.tau) .* net_grid(1,:);
            netNew_grid(2,:) = this.tau.*(this.W_input(2)*input(2) + this.W_inhib(2).*act_grid(1,:) + this.W_selfexcite(2).*act_grid(2,:)) + (1-this.tau) .* net_grid(2,:);

            actNew_grid(1,:) = 1./(1+exp(-this.gain*netNew_grid(1,:)));
            actNew_grid(2,:) = 1./(1+exp(-this.gain*netNew_grid(2,:)));
            act_vec = actNew_grid - act_grid;
            quiver(act_grid(1,:),act_grid(2,:),act_vec(1,:),act_vec(2,:), 'Color',colorDataL(1,:));
            
            h = zeros(length(gains));
            for i = 1:length(gains)
                h(i) = plot(squeeze(trajectory(i,1,:)),squeeze(trajectory(i,2,:)),'-k','LineWidth',3,'Color',colorDataL(i,:)); drawnow;
                scatter(squeeze(trajectory(i,1,:)),squeeze(trajectory(i,2,:)));
                colorData = [linspace(255,0,100); linspace(0,0,100); linspace(0,255,100)]/255;
                map = colorData';
                c = colormap(map);
                %disp(round(((squeeze(trajectory(i,1,:))-squeeze(trajectory(i,2,:)))/2+0.5)*100));
                c = c(round(((squeeze(trajectory(i,1,:))-squeeze(trajectory(i,2,:)))/2+0.5)*100),:);
                scatter(squeeze(trajectory(i,1,:)),squeeze(trajectory(i,2,:)),100,c,'filled');
            end
            
            lower_limit = 0;
            upper_limit = 1;
            xlim([lower_limit upper_limit]);
            ylim([lower_limit upper_limit]);
            axis equal;
            
            % add descriptions
            fontsize = 20;
            text(trajectory(1,1,1)-0.08, trajectory(1,2,1), 'A','Color',map(end,:),'FontSize',fontsize,'FontWeight','bold');
            text(trajectory(1,1,end), trajectory(1,2,end)-0.08, 'B','Color',map(1,:),'FontSize',fontsize,'FontWeight','bold');
            leg = legend([h(1) h(2)],{strcat('g = ',num2str(gains(1))), strcat('g = ',num2str(gains(2)))},'Location','southwest');
            xlabel('activation of goal unit a ','FontSize',fontsize);
            ylabel('activation of goal unit b ','FontSize',fontsize);
            title('gaol unit dynamics ','FontSize',fontsize,'FontWeight','bold');
            set(leg, 'FontSize', fontsize);
            set(gca, 'FontSize', fontsize*0.8);
            hold off;
            
            % get final state for different inputs
        end
        
        function plotActivityProfile(this, gains, timeSteps)
            
            colors = [253 120 21; ... % orange
                          31 104 172; ... % blue
                          44 155 37; ... % green
                          0     0   0  ; ... % black
                        142 142 142; ... % grey 
                        255 255 255] / 255; % white 
            
            goalActProfile = zeros(length(gains),timeSteps);
            
            for i = 1:length(gains)
                this.gain = gains(i);
                [inputSequence, respSequence] = gNet.buildSwitchSequence(2,timeSteps,1,timeSteps);
                [goalAct] = this.runSequence(inputSequence, respSequence);
                goalAct = goalAct(2,:);
                goalActProfile(i,:) = goalAct((timeSteps+1):(timeSteps*2));
            end
            
            colorDataL = [linspace(0,128,length(gains)); linspace(0,128,length(gains)); linspace(0,128,length(gains))]'/255;
            
            for i = 1:length(gains)
                h(i) = plot(1:timeSteps,goalActProfile(i,:),'-k','LineWidth',5,'Color',colors(i,:)); drawnow;
                hold on;
            end
            fontsize=20;
            set(gca, 'FontSize', fontsize*0.8);
            leg = legend([h(1) h(2)],{strcat('Low Gain (g=',num2str(gains(1)), ')'), strcat('High Gain (g=',num2str(gains(2)), ')')},'Location','southeast');
            xlabel('Time Step ','FontSize',fontsize);
            ylabel('Control Signal Intensity ','FontSize',fontsize);
            set(leg, 'FontSize', fontsize);
            xlim([1,timeSteps]);
            ylim([0,1]);
            hold off;
            
            % added for black figures

            set(gcf, 'Color', 'k');
            set(gca, 'Color', 'k');
            set(leg, 'Color', 'k');
            set(leg, 'TextColor', 'w');
            set(gca, 'xColor', 'w');
            set(gca, 'yColor', 'w');

            % preserve background color when saving figure
            fig = gcf;
            fig.InvertHardcopy = 'off';


        end
        
        
        function plotMSEprofile(this, gains, timeSteps)
            
            MSEprofile = zeros(length(gains),timeSteps);
            
            for i = 1:length(gains)
                this.gain = gains(i);
                [inputSequence, respSequence] = gNet.buildSwitchSequence(2,timeSteps,1,timeSteps);
                [~, MSE] = this.runSequence(inputSequence, respSequence);
                MSE = sum(MSE,1);
                MSEprofile(i,:) = MSE((timeSteps+1):(timeSteps*2));
            end
            
            colorDataL = [linspace(0,128,length(gains)); linspace(0,128,length(gains)); linspace(0,128,length(gains))]'/255;
            
            for i = 1:length(gains)
                h(i) = plot(1:timeSteps,MSEprofile(i,:),'-k','LineWidth',5,'Color',colorDataL(i,:)); drawnow;
                hold on;
            end
            fontsize=20;
            leg = legend([h(1) h(2)],{strcat('g = ',num2str(gains(1))), strcat('g = ',num2str(gains(2)))},'Location','northeast');
            xlabel('time step ','FontSize',fontsize);
            ylabel('mean squared error ','FontSize',fontsize);
            xlim([1,timeSteps]);
            ylim([0,1]);
            title('error function ','FontSize',fontsize,'FontWeight','bold');
            hold off;
            set(leg, 'FontSize', fontsize);
            set(gca, 'FontSize', fontsize*0.8);
        end
        
        function plotGainRespDeadlineBlack(this, deadlines, timeStepsPerTrial, nTrials, gain_init)
            
            optimalG = zeros(1,length(deadlines));
            optimalMaxAct = zeros(1,length(deadlines));
            for i = 1:length(deadlines)
                responseStart = 1;
                responseStop = deadlines(i);
                [inputSequence respSequence] = gNet.buildSwitchSequence(nTrials, timeStepsPerTrial, responseStart, responseStop);
                this.gain = gain_init;
                this.setSequence(inputSequence, respSequence);
                if(this.runODE)
                    this.trainSequenceODE();
                else
                    this.trainSequence();
                end
                
                [act] = this.runSequence();
                
                optimalG(i) = this.gain;
                optimalMaxAct(i) = max(max(act));
            end
            
            %%
            fig = figure(1);
            set(fig, 'Position', [100, 100, 400, 300]);
            fontsize=20;
            plot(deadlines,optimalG,'.-w','LineWidth',2,'MarkerSize',40);
            set(gca, 'FontSize', fontsize*0.8);
            xlabel('Time of Response','FontSize',fontsize);
            ylabel('Optimal Gain','FontSize',fontsize);
            xlim([deadlines(1)-1,deadlines(end)]);
            ylim([0,max(optimalG)*1.1]);
%             title('response deadline affects optimal gain','FontSize',fontsize,'FontWeight','bold');
            hold off;
           
            
            % added for black figures

            set(gcf, 'Color', 'k');
            set(gca, 'Color', 'k');
            set(gca, 'xColor', 'w');
            set(gca, 'yColor', 'w');

            % preserve background color when saving figure
            fig = gcf;
            fig.InvertHardcopy = 'off';
            
            %%
            fig = figure(1);
            set(fig, 'Position', [100, 100, 400, 300]);
            fontsize=20;
            plot(deadlines,optimalMaxAct,'.-w','LineWidth',2,'MarkerSize',40);
            set(gca, 'FontSize', fontsize*0.8);
            xlabel('Time of Response','FontSize',fontsize);
            ylabel({'Optimal Maximal', 'Control Intensity'},'FontSize',fontsize);
            xlim([deadlines(1)-1,deadlines(end)]);
            ylim([0,max(optimalMaxAct)*1.1]);
%             title('response deadline affects optimal gain','FontSize',fontsize,'FontWeight','bold');
            hold off;
           
            
            % added for black figures

            set(gcf, 'Color', 'k');
            set(gca, 'Color', 'k');
            set(gca, 'xColor', 'w');
            set(gca, 'yColor', 'w');

            % preserve background color when saving figure
            fig = gcf;
            fig.InvertHardcopy = 'off';
            %%
        end
        
        
        function plotGainRespDeadline(this, deadlines, timeStepsPerTrial, nTrials, gain_init)
            
            optimalG = zeros(1,length(deadlines));
            for i = 1:length(deadlines)
                responseStart = 1;
                responseStop = deadlines(i);
                [inputSequence respSequence] = gNet.buildSwitchSequence(nTrials, timeStepsPerTrial, responseStart, responseStop);
                this.gain = gain_init;
                this.setSequence(inputSequence, respSequence);
                if(this.runODE)
                    this.trainSequenceODE();
                else
                    this.trainSequence();
                end
                
                optimalG(i) = this.gain;
            end
            
            fontsize=20;
            plot(deadlines,optimalG,'.-k','LineWidth',2,'MarkerSize',40);
            xlabel('time of response','FontSize',fontsize);
            ylabel('learned optimal gain','FontSize',fontsize);
            xlim([deadlines(1)-1,deadlines(end)]);
            ylim([0,max(optimalG)*1.1]);
            title('response deadline affects optimal gain','FontSize',fontsize,'FontWeight','bold');
            hold off;
            set(gca, 'FontSize', fontsize*0.8);
        end
        
        function plotGainFlexibilityBlack(this, switchRates, timeStepsPerTrial, responseStop, nTrials, gain_init)
            
            optimalG = zeros(1,length(switchRates));
            optimalMaxAct = zeros(1,length(switchRates));
            for i = 1:length(switchRates)
                disp(['tested ' num2str(i) '/' num2str(length(switchRates))]);
                responseStart = 1;
                switchFreq = switchRates(i);
                [inputSequence respSequence] = gNet.buildSwitchSequence(nTrials, timeStepsPerTrial, responseStart, responseStop, switchFreq);
                this.gain = gain_init;
                this.setSequence(inputSequence, respSequence);
                if(this.runODE)
                    this.trainSequenceODE();
                else
                    this.trainSequence();
                end
                
                [act] = this.runSequence();
                
                optimalG(i) = this.gain;
                optimalMaxAct(i) = max(max(act));
            end
            
            save('switchRatePlot','optimalG', 'switchRates');
            %%
            fig = figure(1);
            set(fig, 'Position', [100, 100, 400, 300]);
            
            fontsize=20;
            plot(switchRates,optimalG,'.-w','LineWidth',2,'MarkerSize',40);
            set(gca, 'FontSize', fontsize*0.8);
            xlabel('Ratio of Task Switches','FontSize',fontsize);
            ylabel('Optimal Gain','FontSize',fontsize);
            xlim([0,1]);
            ylim([0,max(optimalG)*1.1]);
            hold off;
            
            % added for black figures

            set(gcf, 'Color', 'k');
            set(gca, 'Color', 'k');
            set(gca, 'xColor', 'w');
            set(gca, 'yColor', 'w');

            % preserve background color when saving figure
            fig = gcf;
            fig.InvertHardcopy = 'off';
            
            %%
            
            fig = figure(1);
            set(fig, 'Position', [100, 100, 400, 300]);
            
            fontsize=20;
            plot(switchRates,optimalMaxAct,'.-w','LineWidth',2,'MarkerSize',40);
            set(gca, 'FontSize', fontsize*0.8);
            xlabel('Ratio of Task Switches','FontSize',fontsize);
            ylabel({'Optimal Maximal', 'Control Intensity'},'FontSize',fontsize);
            xlim([0,1]);
            ylim([0,max(optimalMaxAct)*1.1]);
            hold off;
            
            % added for black figures

            set(gcf, 'Color', 'k');
            set(gca, 'Color', 'k');
            set(gca, 'xColor', 'w');
            set(gca, 'yColor', 'w');

            % preserve background color when saving figure
            fig = gcf;
            fig.InvertHardcopy = 'off';
            
        end
        
        function plotGainFlexibility(this, switchRates, timeStepsPerTrial, responseStop, nTrials, gain_init)
            
            optimalG = zeros(1,length(switchRates));
            for i = 1:length(switchRates)
                responseStart = 1;
                switchFreq = switchRates(i);
                [inputSequence respSequence] = gNet.buildSwitchSequence(nTrials, timeStepsPerTrial, responseStart, responseStop, switchFreq);
                this.gain = gain_init;
                this.setSequence(inputSequence, respSequence);
                if(this.runODE)
                    this.trainSequenceODE();
                else
                    this.trainSequence();
                end
                
                optimalG(i) = this.gain;
            end
            
            save('switchRatePlot','optimalG', 'switchRates');
            fontsize=20;
            plot(switchRates,optimalG,'.-k','LineWidth',2,'MarkerSize',40);
            xlabel('ratio of task switches','FontSize',fontsize);
            ylabel('learned optimal gain','FontSize',fontsize);
            xlim([0,1]);
            ylim([0,max(optimalG)*1.1]);
            title('switch frequency effect','FontSize',fontsize,'FontWeight','bold');
            hold off;
            set(gca, 'FontSize', fontsize*0.8);
        end
        
        function plotErrorLandscape(this, responseDeadline, timeStepsPerTrial, gains)
            
            % create testing set
            responseStart = 1;
            [inputSequence respSequence] = gNet.buildSwitchSequence(2, timeStepsPerTrial, responseStart, responseDeadline);
            this.setSequence(inputSequence, respSequence);
            MSE = zeros(1,length(gains));
            
            for i = 1:length(gains)
                this.gain = gains(i);
                [goalActs MSElog] = this.runSequence();
            
%             this.gain = 3;
%             goal_act = zeros(2,length(inputSequence));
%             goal_net = zeros(2,length(inputSequence));
%             MSEc = zeros(2,length(inputSequence));
% 
%             % initialize net input & activation for goal units
%             this.curr_act = [0 0];
%             this.curr_net = [0 0];
%             
%             for t = 1:length(inputSequence);
%                 
%                 if(t == 21)
%                     this.gain = gains(i);
%                 end;
%                 
%                 % get current state variables
%                 [goal_act(:,t), goal_net(:,t), MSEc(:,t)] = this.runTimeStep(inputSequence(:,t));
%                 
%                 
%                 % plot
%                 if(this.plot_run)
%                    this.plotNet(inputSequence(:,t), respSequence(t)); 
%                 end
%             end
%             
%             this.act_log = goal_act;
%             this.net_log = goal_net;
%             this.MSE_log = MSEc;
%                 MSElog = MSEc;
                
                % get MSE after switch trial
                MSE(i) = sum(MSElog(:,timeStepsPerTrial + responseDeadline),1);
            end
            
            fontsize=20;
            plot(gains,MSE,'-k','LineWidth',2,'MarkerSize',40);
            xlabel('current gain','FontSize',fontsize);
            ylabel('MSE at response in switch trial','FontSize',fontsize);
            xlim([gains(1),gains(end)]);
            ylim([0,max(MSE)*1.1]);
            title('Error Landscape','FontSize',fontsize,'FontWeight','bold');
            hold off;
            set(gca, 'FontSize', fontsize*0.8);
        end
        
         function [switchRT, repetitionRT, switchER, repetitionER] = simulateTaskSwitchingExperiment(this, nTrials, timeStepsPerTrial, responseStart, responseStop, ddmp)
            
            switchRT = [];
            switchER = [];
            repetitionRT = [];
            repetitionER = [];
             
            [inputSeq_new, respSeq_new, relevantTask, taskTransition, miniBlockTrial, miniBlockNumber] = gNet.buildTaskSwitchExperiment(nTrials, timeStepsPerTrial, responseStart, responseStop, ddmp);
            
            this.setSequence(inputSeq_new, respSeq_new);
            [goalAct, ~] = this.runSequence();
            
            relevantTrials = find(~isnan(taskTransition));
            
            for trialIdx = 1:length(relevantTrials) 
                trial = relevantTrials(trialIdx);
                
                transition = taskTransition(trial);
                currentTask = relevantTask(trial);
                currentGoalAct = goalAct(:, trial+1);
                
                [RT, ER] = this.simulateDDM(ddmp, currentGoalAct, currentTask);
                
                if(transition == 0)
                    repetitionRT = [repetitionRT RT];
                    repetitionER = [repetitionER ER];
                else
                    switchRT = [switchRT RT];
                    switchER = [switchER ER];
                end
                
            end
            
            
         end
        
         function [RT, ER] = simulateDDM(this, ddmp, currentGoalAct, currentTask) 
             
             relevantAct = currentGoalAct(currentTask);
             irrelevantAct = currentGoalAct(mod(currentTask,2)+1);
             
%              effectiveDrift = ddmp.drift * (relevantAct - irrelevantAct);
             effectiveDrift = ddmp.drift * relevantAct;
             
             e = ddmp.bias;
             RT = 0;
             ER = 0;
             
             while (abs(e) < ddmp.z)
                 e = e + effectiveDrift * ddmp.dt + sqrt(ddmp.dt) * randn * ddmp.c;
                 RT = RT + ddmp.dt;
             end
             if(sign(e) < 0)
                 ER = 1;
             end
             
             RT= RT + ddmp.T0;
         end
        
    end
    
    
    methods(Static)
        function [inputSeq, respSeq] = buildSwitchSequence(nTrials, timeStepsPerTrial, responseStart, responseStop, varargin)
           
            if(mod(nTrials,2) == 1)
                error('Number of trials needs to be even.');
            end
            
            onTrial = repmat(1,1,timeStepsPerTrial);
            %onTrial(responseStop:end) = 0; % remove stimulus after response
            offTrial = repmat(0,1,timeStepsPerTrial);
            
            % additional parameter: switch frequency
            if(isempty(varargin))
                extX = repmat([onTrial offTrial],1,nTrials/2); 
                extY = repmat([offTrial onTrial],1,nTrials/2);

                inputSeq = [extX;extY];
            else
                if(mod(nTrials,100) == 1)
                    error('Number of trials needs to be a multiple of one hundred.');
                end
                
                total_extX = [];
                total_extY = [];
                for i = 1:(nTrials/100)
                    switchFreq = varargin{1};
                    switchIdx = randperm(100);
                    switchIdx = switchIdx(1:(round(100*switchFreq)));
                    transitions = zeros(1,100);
                    transitions(switchIdx) = 1;
                    
                    extX = []; % ignore first trial
                    extY = [];
                    task = 1;
                    
                    for j = 1: 100
                        if(transitions(j) == 1)
                            task = mod(task+1,2);
                        end
                        
                        if(task == 1)
                            extX = [extX onTrial];
                            extY = [extY offTrial];
                        else
                            extX = [extX offTrial];
                            extY = [extY onTrial];
                        end
                        
                    end
                    
                    total_extX = [total_extX extX];
                    total_extY = [total_extY extY]; 
                end
                inputSeq = [total_extX; total_extY];
            end
            respTrial = repmat(0,1,timeStepsPerTrial);
            % TODO: for now ignore responseStart
            respTrial(responseStart) = 1;
            respTrial(responseStop) = 2;
            %respTrial(responseStop+1) = 1; % to delete
            respSeq = repmat(respTrial,1,nTrials);
            %respSeq(1) = 1; % to delete
        end
        
        function [inputSeq_new, respSeq_new, relevantTask, taskTransition, miniBlockTrial, miniBlockNumber] = buildTaskSwitchExperiment(nTrials, timeStepsPerTrial, responseStart, responseStop, varargin)

            [inputSeq, respSeq] = gNet.buildSwitchSequence(nTrials, timeStepsPerTrial, responseStart, responseStop);
            miniBlockTrial = repmat(1:timeStepsPerTrial, 1, nTrials);
            numMiniBlocks = nTrials;
            miniBlockNumber = repmat(1:numMiniBlocks, timeStepsPerTrial, 1);
            miniBlockNumber = transpose(miniBlockNumber(:));

            miniBlockOrder = randperm(numMiniBlocks);

            inputSeq_new = nan(size(inputSeq));
            respSeq_new = nan(size(respSeq));
            taskTransition = nan(1, size(respSeq, 2));

            for miniBlock = 1:numMiniBlocks
                for trial = 1:timeStepsPerTrial

                    index_new = (miniBlock-1) * timeStepsPerTrial + trial;
                    index_old = find(miniBlockTrial == trial & miniBlockOrder(miniBlockNumber) == miniBlock);
                    inputSeq_new(:, index_new) = inputSeq(:, index_old);
                    respSeq_new(:, index_new) = respSeq(:, index_old);

                    if(miniBlock > 1)
                        if(trial == 1 & inputSeq_new(:, index_new) == inputSeq_new(:, index_new-1))
                            taskTransition(index_new) = 0;
                        elseif(trial == 1 & inputSeq_new(:, index_new) ~= inputSeq_new(:, index_new-1))
                            taskTransition(index_new) = 1;
                        end
                    end
                end
            end

            relevantTask = nan(size(miniBlockTrial));
            relevantTask(inputSeq_new(1,:) == 1) = 1;
            relevantTask(inputSeq_new(1,:) == 0) = 2;
        end
        
    end
    
end

