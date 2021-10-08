function [ Low, High0,  High1]=likelihoods(results, method)

% load data
name='../groupdata';
load (name);
data = groupdata.subdata(groupdata.i);
  
f2= @(x,data) MB_exhaustive_complexity_llik(x,data); 
f3= @(x,data) MB_optimum_exhaustive_complexity_llik(x,data); 


% init likelihoods
Low=[]; High0=[]; High1=[]; 
X=results.x;
%loop over subjects
for s = 1:length(data)
    subdata=data(s);
    x= X(s,:); % params(s)
    [LL_low LL_high0 LL_high1 ] = f2(x,subdata);
    [LL_low_opt LL_high0_opt LL_high1_opt] = f3(x,subdata);
    if strcmp(method ,'method2')
    Low(end+1)=LL_low; 
    High0(end+1)= LL_high0 ; 
    High1(end+1)=LL_high1;
    else
    Low(end+1)=LL_low-LL_low_opt ; 
    High0(end+1)= LL_high0- LL_high0_opt ; 
    High1(end+1)=LL_high1-LL_high1_opt ;        
    end
end
end


function  [LL_low LL_high_top LL_high_med ] = MB_exhaustive_complexity_llik(x,subdata)

% Loglikelihood function for Experiment 1

% parameters
b_low = x(1);           % softmax inverse temperature
b_high0 = x(2);
b_high1 = x(3);
lr = x(4);          % learning rate

% initialization
% Q(s,a): state-action value function for Q-learning for different stages
Qmf_terminal = zeros(3,1);

LL_low=0;  LL_high_med=0; LL_high_top=0;
count_low=0; count_high=0; 

% loop through trials
for t = 1:length(subdata.choice0)
   
    if subdata.missed(t) == 1
        continue
    end
    
    %% likelihoods
    if subdata.high_effort(t)==1 % high effort trial
        count_high=count_high+1;
        % level 0
        
        Qmb_middle = zeros(3,2);
        for state = 1:3
            Qmb_middle(state,:) = subdata.Tm_middle(subdata.middle_stims{2}(state,:),:)*Qmf_terminal;   % find model-based values at stage 1
        end
                
        Qmb_top = subdata.Tm_top(subdata.stims0(t,:),:)*max(Qmb_middle,[],2);                           % find model-based values at stage 0
        
        Q_top = Qmb_top' ; 
        action = subdata.choice0(t)==subdata.stims0(t,:);
        LL_high_top = LL_high_top + b_high0*Q_top(action)-logsumexp(b_high0*Q_top);

        % level 1
        stims1 = subdata.stims1(t,1:2);
        
        
    else % low effort trial
        count_low=count_low+1;
        stims1 = subdata.stims1(t,:);
        
        
    end
    
    % level 1
    Qmb_middle = subdata.Tm_middle(stims1,:)*Qmf_terminal;                     % find model-based values at stage 0
    Q_middle = Qmb_middle;
    action = subdata.choice1(t)==stims1;
    if subdata.high_effort(t)==1
        LL_high_med = LL_high_med + b_high1*Q_middle(action)-logsumexp(b_high1*Q_middle);
       
   
    else
        LL_low = LL_low + b_low*Q_middle(action)-logsumexp(b_low*Q_middle);
        
    end
    
    %% updating
    dtQ = zeros(3,1);
    
    %terminal level
    dtQ(3) = subdata.points(t) - Qmf_terminal(subdata.state2(t));
    Qmf_terminal(subdata.state2(t)) = Qmf_terminal(subdata.state2(t)) + lr*dtQ(3);

    
end
LL_low=LL_low/count_low;
LL_high_top = LL_high_top/count_high;
LL_high_med = LL_high_med/count_high;
end


function  [LL_low LL_high0 LL_high1 ] = MB_optimum_exhaustive_complexity_llik(x,subdata)

% parameters
%b_low = x(1);           % softmax inverse temperature
% b_high0 = x(2);
% b_high1 = x(3);
lr = x(4);

% initialization
% Q(s,a): state-action value function for Q-learning for different stages

LL_low=0;  LL_high0=0; LL_high1=0;
count_low=0; count_high=0;


% loop through trials
for t = 1:length(subdata.choice0)
  
    if subdata.missed(t) == 1
        continue
    end
    Qmf_terminal=subdata.rews(t, :)';
    %% likelihoods
    if subdata.high_effort(t)==1 % high effort trial
        count_high=count_high+1;
        % level 0
        
        Qmb_middle = zeros(3,2);
        for state = 1:3
            Qmb_middle(state,:) = subdata.Tm_middle(subdata.middle_stims{2}(state,:),:)*Qmf_terminal;   % find model-based values at stage 1
        end
                
        Qmb_top = subdata.Tm_top(subdata.stims0(t,:),:)*max(Qmb_middle,[],2);                           % find model-based values at stage 0
        
        Q_top = Qmb_top' ; 
        action = find(Q_top==max(Q_top),1);         % choose 1 or 2 
        choice0=subdata.stims0(t,action); % 1,2 or 3
        LL_high0= LL_high0 + Q_top(action)-logsumexp(Q_top);

        % level 1
        state1=find(subdata.Tm_top(choice0,:)==1); % 1, 2 or 3
        stims1=subdata.middle_stims{2}(state1,:);
        
    else % low effort trial
        count_low=count_low+1;
        stims1 = subdata.stims1(t,:);
        
        
    end
   
    % level 1
    Qmb_middle = subdata.Tm_middle(stims1,:)*Qmf_terminal;                     % find model-based values at stage 0
    Q_middle = Qmb_middle;
    action = find(Q_middle==max(Q_middle),1);         % choose 1 or 2 
    choice1=stims1(action) ;% 1,2,3,4,5 or 6
    state2=find(subdata.Tm_middle(choice1,:)==1); % 1,2 or 3
  
    if subdata.high_effort(t)==1
        LL_high1 = LL_high1 + Q_middle(action)-logsumexp(Q_middle);
    else
        LL_low = LL_low + Q_middle(action)-logsumexp(Q_middle); 
    end
    
 
    
end
LL_low=LL_low/count_low;
LL_high0 = LL_high0/count_high;
LL_high1 = LL_high1/count_high;
end



