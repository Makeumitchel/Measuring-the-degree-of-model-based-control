function  output =  agent(parameters, rews, high_effort) 
%% Output 
N = size(rews,1);
output.state0=-ones(N,1);
output.stims0=-ones(N,2);
output.choice0=-ones(N,1);
output.state1=-ones(N,1);
output.stims1=-ones(N,3);
output.choice1=-ones(N,1);
output.state2=-ones(N,1);
output.reward=-ones(N,1);


%% parameters
b_low = parameters(1);       % softmax inverse temperature
b_high0 = parameters(2);
b_high1 = parameters(3);
lr = parameters(4);          % learning rate
lambda = parameters(5);      % eligibility trace decay
w_low = parameters(6);       % mixing weight
w_high0 = parameters(7);           
w_high1 = parameters(8);          
fr = parameters(9);            % forgetting rate
mu=0.5;                        % forgetting regression value
%% initialization
% Q(s,a): state-action value function for Q-learning for different stages
Qmf_top = zeros(1,3);
Qmf_middle = zeros(6,1);
Qmf_terminal = zeros(3,1);
% Transition matrices
Tm_top = [1 0 0; 0 1 0; 0 0 1];
Tm_middle = [1 0 0; 0 1 0; 0 0 1; 1 0 0; 0 1 0; 0 0 1];   
rocket_order = [1 2 3 4 5 6];
middle_stims{1} = [rocket_order(1) rocket_order(2) rocket_order(3); rocket_order(4) rocket_order(5) rocket_order(6)];
middle_stims{2} = [rocket_order(1) rocket_order(5); rocket_order(4) rocket_order(3) ; rocket_order(2) rocket_order(6)];
  

%% loop through trials
for t = 1:N
 
    %% Decisions
    if high_effort(t)==1 % high effort trial
        % level 0
        state0=1;
        stims0 = datasample(1:3,2,'Replace',false); % 2 stations beyond [1,2,3]
        Qmb_middle = zeros(3,2);
        for state = 1:3
            Qmb_middle(state,:) = Tm_middle(middle_stims{2}(state,:),:)*Qmf_terminal ;  % find model-based values at stage 1
        end     
        
        Qmb_top = Tm_top(stims0,:)*max(Qmb_middle,[],2);  % find model-based values at stage 0
        Q_top = w_high0*Qmb_top' + (1-w_high0)*Qmf_top(stims0) ;   % mix TD and model value
        ps = exp(b_high0*Q_top)/sum(exp(b_high0*Q_top));            %compute choice probabilities for each action
        action = find(rand<cumsum(ps),1);         % choose 1 or 2 
        choice0=stims0(action); % 1,2 or 3
     
        % level 1
        state1=find(Tm_top(choice0,:)) ;    
        stims1 = middle_stims{2}(state1,:) ;
        w = w_high1;
        b= b_high1;
        
    else % low effort trial
        state0 = -1;
        choice0 = -1;
        stims0 =[-1,-1];
        state1 = ceil(rand*2); %1 or 2 
        stims1 = middle_stims{1}(state1,:) ;
        w = w_low;
        b = b_low;
        
    end
    
    % level 1
    Qmb_middle = Tm_middle(stims1,:)*Qmf_terminal;                     % find model-based values at stage 0
    Q_middle = w*Qmb_middle + (1-w)*Qmf_middle(stims1);                        % mix TD and model value
    ps = exp(b*Q_middle)/sum(exp(b*Q_middle));            %compute choice probabilities for each action
    action = find(rand<cumsum(ps),1); % 1 or 2 if high; 1,2 or 3 if low
    choice1=stims1(action); % 1,2,3,4,5,6
    
    % level 2
    state2=find(Tm_middle(choice1,:)) ; % 1,2 or 3
    reward=rews(t, state2);
    
    %% Learning (updating Q)
    
    dtQ = zeros(3,1);
    
    if high_effort(t)==1
        % top level
        dtQ(1) = Qmf_middle(choice1) - Qmf_top(choice0);
        Qmf_top(choice0) = Qmf_top(choice0) + lr*dtQ(1);
    end
    
    %middle level
    dtQ(2) = Qmf_terminal(state2) - Qmf_middle(choice1);
    Qmf_middle(choice1) = Qmf_middle(choice1) + lr*dtQ(2);
    if high_effort(t)==1
        Qmf_top(choice0) = Qmf_top(choice0) + lambda*lr*dtQ(2);
    end
    
    %terminal level
    dtQ(3) = reward - Qmf_terminal(state2);
    Qmf_terminal(state2) = Qmf_terminal(state2) + lr*dtQ(3);
    Qmf_middle(choice1) = Qmf_middle(choice1) + lambda*lr*dtQ(3);
    if high_effort(t)==1
        Qmf_top(choice0) = Qmf_top(choice0) + (lambda^2)*lr*dtQ(3);
    end
    
    
       %% Forgetting (updating Q)
    
    % top level 
    K=ones(1,3);
    if high_effort(t)==1
        K(1,choice0)=0;
    end   
    Qmf_top = Qmf_top + fr*(mu-K.*Qmf_top);
    % middle level 
    K=ones(6,1);
    K(choice1,1)=0;
    Qmf_middle = Qmf_middle + fr*(mu-K.*Qmf_middle);
    % terminal level 
    K=ones(3,1);
    K(state2,1)=0;
    Qmf_terminal = Qmf_terminal + fr*(mu-K.*Qmf_terminal);
    
    
    %% Save Output
    output.state0(t)=state0;
    output.stims0(t,:)=stims0;
    output.choice0(t,:)=choice0;
    output.state1(t,:)=state1;
    output.stims1(t,1:length(stims1) )=stims1;
    output.choice1(t)=choice1;
    output.state2(t)=state2;
    output.reward(t)=reward;
    
end

end
