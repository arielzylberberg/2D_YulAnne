clear
close all
clc
clf

for subj=1:2
    
    if subj==1
        load 'dataAll_ID2.mat'
    else
        load 'dataAll_ID3.mat'
        
    end
    %% MODEL OPTIMIZATION
    for w=1:2
        
        mysubplot(2,2,subj,w)
        if w==1
            D.s_this_coh=dataAll.ColorCohCol;
            D.s_other_coh=dataAll.MotionCohDir;
            D.this_choice= dataAll.ChoiceColor;
            D.other_choice= dataAll.ChoiceMotion==0;
            
            D.this_correct= dataAll.AccuracyColor;
            D.other_correct= dataAll.AccuracyMotion;
            t0 = 0.5;
        else
            D.s_other_coh=dataAll.ColorCohCol;
            D.s_this_coh=dataAll.MotionCohDir;
            D.this_choice= dataAll.ChoiceMotion==0;
            D.other_choice= dataAll.AccuracyColor;
            
            D.other_correct= dataAll.AccuracyColor;
            D.this_correct= dataAll.AccuracyMotion;
            t0 = 0.9;
        end
        D.session=dataAll.Session;
        D.dur=dataAll.StimulusDuration;
        D.u_this_coh=abs(D.s_this_coh);
        D.u_other_coh=abs(D.s_other_coh);
        D.I_low_this= (D.u_this_coh < 0.1);
        D.I_low_other= (D.u_other_coh < 0.1);
        D.I_very_low_this= (D.u_this_coh < 0.05);
        D.I_very_low_other= (D.u_other_coh < 0.05);
        
        
        %%
        
        
        clear fval otheta
        
        D.other_choice= dataAll.ChoiceMotion==0;
        
        DT=struct2table(D);
        DT=DT(DT.other_correct==1,:);
        D=table2struct(DT,'ToScalar',true);
        
        for k=1:10
            k
            
            %initial and lower bad upper
            T{1}.gamma1=1;      T{2}.gamma1=0.01;     T{3}.gamma1=0.5;
            T{1}.mu=0.9;        T{2}.mu=0;            T{3}.mu=1.5;
            T{1}.sigma1=0.4;    T{2}.sigma1=0.05;     T{3}.sigma1=3;
            T{1}.alpha1=0.2;    T{2}.alpha1=-10;      T{3}.alpha1=10;
            %T{1}.alpha1=0;      T{2}.alpha1=0;        T{3}.alpha1=0;
            
            if k==1
                [theta,theta_lo,theta_hi,S]= opt_pack(T);
            else
                [theta,theta_lo,theta_hi,S]= opt_pack(T,inf);
            end
            
            [otheta(k,:),fval(k,1),exitflag,output] = fminsearchbnd(@(theta) objfun(theta, D,S), theta, theta_lo, theta_hi);
            %[theta(k,:),fval(k),exitflag,output] = fmincon(@(fitParams) objfun(fitParams, D), fitParams, [],[],[],[],lb, ub);
            [otheta fval-min(fval)]
        end
        
        i=argmin(fval);
        opt_theta=otheta(i,:);
        % run model with optimised parameters
        [dev B stats yPred] = objfun(opt_theta,D,S);
        stats.p
        B'
        
        Q=opt_unpack(opt_theta,S)
        v2struct(Q);
        
        % Plots results
        
        x =  linspace(0,1.2,1000);
        
        y= B(2) + B(3) * (x.^gamma1);
        dy=B(4)*normpdf(x,  mu,sigma1).*normcdf(alpha1*(x-mu)/sigma1);
        
        plot(x, y, 'k-'); hold on %
        plot(x, dy, 'r-');
        plot(x, y+dy, 'b-');
        
        xlabel('Stimulus duration (sec)');
        if w==1
            ylabel('Color sensitivity');
        else
            ylabel('Motion sensitivity');
            
        end
        
        set(gca, 'YLim', [-20 40], 'YTick', -20:10:30);
        %legend({'b2 + b3 *(t^a_1)','b4*normpdf(t,t0,sigma))'}, 'Interpreter', 'None', 'Location', 'NorthOutside');
        grid on
               title(sprintf('Subj%i p=%f',subj+1,stats.p(4)))
 shg
        
        
        
        %%
        v2struct(D)
        u=unique(dur);
        clear R
        for k=1:length(u)
            for j=1:2
                s=dur==u(k) & I_low_other==(j-1);
                x=s_this_coh;
                [R(j,k,:), dev, stats] = glmfit(s_this_coh(s),this_choice(s),'binomial','logit');
                se(j,k)=stats.se(2);
            end
        end
        
        errorbar(u,squeeze(R(1,:,2)),se(1,:),'ko')
        hold on
        errorbar(u,squeeze(R(2,:,2)),se(2,:),'bo')
        
        shg
        
        s=I_low_other;
        W=bindata(dur,other_correct)
    end
end
    %%
    
    clear R se
    v2struct(D)
    u=unique(dur);
    for k=1:length(u)
        s=[dur;dur]==u(k);
        
        Y=[this_choice; other_choice];
        I_this=[this_choice==this_choice; 0*other_choice];
        I_other=[0*this_choice; other_choice==other_choice];
        
        X=[[s_this_coh ;s_this_coh].*I_this I_this  [s_other_coh;s_other_coh].*I_other  -[u_other_coh ;u_this_coh]];
        %  X=[[s_this_coh ;s_this_coh].*I_this I_this  [s_other_coh;s_other_coh].*I_other  [other_correct;this_correct].*[s_this_coh ;s_this_coh]];
        %  X=[[s_this_coh ;s_this_coh].*I_this I_this  [s_other_coh;s_other_coh].*I_other   [other_correct;this_correct]];
        
        [R(k,:), dev, stats] = glmfit(X(s,:),Y(s),'binomial','logit');
        se(k)=stats.se(5);
    end
    %errorbar(u,squeeze(R(1,:,2)),se(1,:),'ko')
    hold on
    %errorbar(u,R(:,5),se,'bo')
    
    shg
    


        mysubplot(2,2,1,1)
        title('Subj ID2')
  mysubplot(2,2,1,1)
        title('Subj ID2') 
        mysubplot(2,2,1,1)
        title('Subj ID2')
          mysubplot(2,2,1,1)
        title('Subj ID2')


    %%
    figure(2)
    clf
    u=unique(D.dur);
    v2struct(D)
    
    for k=1:length(u)
        s=D.dur==u(k) & I_low_other & I_low_this;
        %  s=D.dur==u(k) & I_very_low_other & I_very_low_this;
        
        [m,CHI2,pchi(k)]=crosstab(D.this_correct(s), D.other_correct(s));
        
        [H,P,STATS] = fishertest(m);
        
        c(k)     = STATS.OddsRatio;
        cl(k,:)  = STATS.ConfidenceInterval;
    end
    
    errbar(u,c,c-cl(:,1)',cl(:,2)'-c,'k-')
    hold on
    plot(u,c,'ko-')
    box off
    shg
    
    
    
