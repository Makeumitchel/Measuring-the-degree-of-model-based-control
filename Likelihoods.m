function [Low, High0,  High1]=Likelihoods(model, condition, X)
addpath('C:\Users\Matthieu Pâques\Documents\Nottingham University\Research_project\kool_2018\mfit');
% load data
name=append('../groupdata_',condition);
load (name);
data = groupdata.subdata;
if strcmp(model,'MBMF')
    f = @(x,data) MBMF_exhaustive_complexity_llik(x,data); 
end
if strcmp(model,'MB')
    f = @(x,data) MB_exhaustive_complexity_llik(x,data); 
end
if strcmp(model,'MB_optimum')
    f = @(x,data) MB_optimum_exhaustive_complexity_llik(x,data); 
end
% init likelihoods and rewards
Low=[]; High0=[]; High1=[];

%loop over subjects
for s = 1:length(data)
    %disp(['Subject ',num2str(s)]);
    subdata=data(s);
    x= X(s,:); % params(s)
    [LL_low LL_high0 LL_high1 ] = f(x,subdata);
     Low(end+1)=LL_low; 
     High0(end+1)= LL_high0 ; 
     High1(end+1)=LL_high1;
     
end
end

function  [LL_low LL_high_top LL_high_med ] = MBMF_exhaustive_complexity_llik(x,subdata)

% Loglikelihood function for Experiment 1

% parameters
b_low = x(1);           % softmax inverse temperature
b_high0 = x(2);
b_high1 = x(3);
lr = x(4);          % learning rate
lambda = x(5);      % eligibility trace decay
w_low = x(6);           % mixing weight
w_high0 = x(7);           % mixing weight
w_high1 = x(8);           % mixing weight

% initialization
% Q(s,a): state-action value function for Q-learning for different stages
Qmf_top = zeros(1,3);
Qmf_middle = zeros(6,1);
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
        
        Q_top = w_high0*Qmb_top' + (1-w_high0)*Qmf_top(subdata.stims0(t,:));                            % mix TD and model value
        action = subdata.choice0(t)==subdata.stims0(t,:);
        
        LL_high_top = LL_high_top + b_high0*Q_top(action)-logsumexp(b_high0*Q_top);
        
        % level 1
        stims1 = subdata.stims1(t,1:2);
        w = w_high1;
        b = b_high1;
        
    else % low effort trial
        count_low=count_low+1;
        stims1 = subdata.stims1(t,:);
        w = w_low;
        b = b_low; 
    end
    
    % level 1
    Qmb_middle = subdata.Tm_middle(stims1,:)*Qmf_terminal;                     % find model-based values at stage 0
    Q_middle = w*Qmb_middle + (1-w)*Qmf_middle(stims1);                        % mix TD and model value
    action = subdata.choice1(t)==stims1;
    if subdata.high_effort(t)==1 % high effort trial
        LL_high_med = LL_high_med + b*Q_middle(action)-logsumexp(b*Q_middle);
    else
        LL_low = LL_low + b*Q_middle(action)-logsumexp(b*Q_middle);
    end
    %% updating
    
    dtQ = zeros(3,1);
    
    if subdata.high_effort(t)==1
        % top level
        dtQ(1) = Qmf_middle(subdata.choice1(t)) - Qmf_top(subdata.choice0(t));
        Qmf_top(subdata.choice0(t)) = Qmf_top(subdata.choice0(t)) + lr*dtQ(1);
    end
    
    %middle level
    dtQ(2) = Qmf_terminal(subdata.state2(t)) - Qmf_middle(subdata.choice1(t));
    Qmf_middle(subdata.choice1(t)) = Qmf_middle(subdata.choice1(t)) + lr*dtQ(2);
    if subdata.high_effort(t)==1
        Qmf_top(subdata.choice0(t)) = Qmf_top(subdata.choice0(t)) + lambda*lr*dtQ(2);
    end
    
    %terminal level
    dtQ(3) = subdata.points(t) - Qmf_terminal(subdata.state2(t));
    Qmf_terminal(subdata.state2(t)) = Qmf_terminal(subdata.state2(t)) + lr*dtQ(3);
    Qmf_middle(subdata.choice1(t)) = Qmf_middle(subdata.choice1(t)) + lambda*lr*dtQ(3);
    if subdata.high_effort(t)==1
        Qmf_top(subdata.choice0(t)) = Qmf_top(subdata.choice0(t)) + (lambda^2)*lr*dtQ(3);
    end
    
end
LL_low=LL_low/count_low;
LL_high_med = LL_high_med/count_high;
LL_high_top = LL_high_top/count_high;
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

function  [LL_low LL_high0 LL_high1 ] = MB_machine_exhaustive_complexity_llik(x,subdata)

% Loglikelihood function for Experiment 1

% parameters
b_low = x(1);           % softmax inverse temperature
b_high0 = x(2);
b_high1 = x(3);
lr = x(4);          % learning rate

% initialization
% Q(s,a): state-action value function for Q-learning for different stages
Qmf_terminal = zeros(3,1);

LL_low=0;  LL_high0=0; LL_high1=0;
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
        ps = exp(b_high0*Q_top)/sum(exp(b_high0*Q_top));            %compute choice probabilities for each action
        action = find(rand<cumsum(ps),1);         % choose 1 or 2 
        choice0=subdata.stims0(t,action); % 1,2 or 3
        LL_high0= LL_high0 + b_high0*Q_top(action)-logsumexp(b_high0*Q_top);

        % level 1
        state1=find(subdata.Tm_top(choice0,:)==1); % 1, 2 or 3
        stims1=subdata.middle_stims{2}(state1,:);
        b=b_high1;
    else % low effort trial
        count_low=count_low+1;
        stims1 = subdata.stims1(t,:);
        b=b_low;
        
    end
   
    % level 1
    Qmb_middle = subdata.Tm_middle(stims1,:)*Qmf_terminal;                     % find model-based values at stage 0
    Q_middle = Qmb_middle;
    ps = exp(b*Q_middle)/sum(exp(b*Q_middle));            %compute choice probabilities for each action
    action = find(rand<cumsum(ps),1);         % choose 1 or 2 
    choice1=stims1(action) ;% 1,2,3,4,5 or 6
    state2=find(subdata.Tm_middle(choice1,:)==1); % 1,2 or 3
  
    if subdata.high_effort(t)==1
        LL_high1 = LL_high1 + b*Q_middle(action)-logsumexp(b*Q_middle);
    else
        LL_low = LL_low + b*Q_middle(action)-logsumexp(b*Q_middle); 
    end
    
    %% updating
    dtQ = zeros(3,1);
    reward=subdata.rews(t, state2);
    %terminal level
    dtQ(3) = reward - Qmf_terminal(state2);
    Qmf_terminal(state2) = Qmf_terminal(state2) + lr*dtQ(3);

    
end
LL_low=LL_low/count_low;
LL_high0 = LL_high0/count_high;
LL_high1 = LL_high1/count_high;
end

function  [LL_low LL_high0 LL_high1 ] = MB_optimum_exhaustive_complexity_llik(x,subdata)
% parameters
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
    
    %% likelihoods
    Qmf_terminal=subdata.rews(t, :)';
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
    action = find(Q_middle==max(Q_middle),1);         % choose 1 or 2 or 3
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

function  [LL_low LL_high0 LL_high1 ] = MB_optimum2_exhaustive_complexity_llik(x,subdata)

% parameters
b_low = x(1);           % softmax inverse temperature
b_high0 =x(2);
b_high1 =x(3);
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
        LL_high0= LL_high0 + b_high0*Q_top(action)-logsumexp(b_high0*Q_top);

        % level 1
        state1=find(subdata.Tm_top(choice0,:)==1); % 1, 2 or 3
        stims1=subdata.middle_stims{2}(state1,:);
        b=b_high1;
    else % low effort trial
        count_low=count_low+1;
        stims1 = subdata.stims1(t,:);
        b=b_low;
        
    end
   
    % level 1
    Qmb_middle = subdata.Tm_middle(stims1,:)*Qmf_terminal;                     % find model-based values at stage 0
    Q_middle = Qmb_middle;
    action = find(Q_middle==max(Q_middle),1);         % choose 1 or 2 
    choice1=stims1(action) ;% 1,2,3,4,5 or 6
    state2=find(subdata.Tm_middle(choice1,:)==1); % 1,2 or 3
  
    if subdata.high_effort(t)==1
        LL_high1 = LL_high1 + b*Q_middle(action)-logsumexp(b*Q_middle);
    else
        LL_low = LL_low + b*Q_middle(action)-logsumexp(b*Q_middle); 
    end
    
 
    
end
LL_low=LL_low/count_low;
LL_high0 = LL_high0/count_high;
LL_high1 = LL_high1/count_high;
end


function  [LL_low LL_high_top LL_high_med ] = Random_complexity_llik(x, subdata)

% Loglikelihood function for Experiment 1: Random agent
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
        LL_high_top = LL_high_top + log(1/2);

        % level 1
        LL_high_med = LL_high_med + log(1/2);

    else % low effort trial
        % level 1
        count_low=count_low+1;
        LL_low = LL_low +log(1/3);
    end 
end
LL_low=LL_low/count_low;
LL_high_med = LL_high_med/count_high;
LL_high_top = LL_high_top/count_high;
end

