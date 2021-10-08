function data = wrapper(parameters, nsubjects, ntrials)
% Add path of the mfit library (see Gerschman et al., 2019)
addpath('../mfit');

trials=cell(ntrials*nsubjects, 25);
t=1;
for subject=1:nsubjects
    display(['subject ',num2str(subject)]);
    [rews, high_effort] = generate_rews(ntrials);
    rews=rews/9; % Normalize the reward
    output = agent(parameters, rews, high_effort); 
   
    for trial = 1:ntrials
        t=t+1;
        
        state0=output.state0(trial);
        stims0=output.stims0(trial,:);
        choice0=output.choice0(trial,:);
        state1=output.state1(trial,:);
        stims1=output.stims1(trial,:);
        choice1=output.choice1(trial);
        state2=output.state2(trial);
        reward=output.reward(trial);


        trials(t,1)={append('subject',num2str(subject))};
        trials(t,2)={ high_effort(trial)}; %high_effort
        
        trials(t,3)={state0}; %state0
        trials(t,4)={stims0(1)};% stim_0_1 
        trials(t,5)={stims0(2)}; %stim_0_2
        trials(t,6)={1000*(high_effort(trial)==1)-1*(high_effort(trial)==0)}; %rt0 
        trials(t,7)={choice0} ; %choice0 
        trials(t,8)={1}; % response0 
        
        trials(t,9)={state1} ; %state1 
        trials(t,10)={stims1(1)}; %stim_1_1 
        trials(t,11)={stims1(2)}; %stim_1_2 
        trials(t,12)={stims1(3)};  %stim_1_3 
        trials(t,13)={1000}; % rt1 
        trials(t,14)={choice1}; % choice1 
        trials(t,15)={1};  %response1 (key)
        
        trials(t,16)={1000}; % rt2
        trials(t,17)={reward}; % points 
        trials(t,18)={state2}; % state2 
        trials(t,19)={sum(output.reward(1:trial,1))}; % score 
        trials(t,20)={0}; %practice 
        trials(t,21)={rews(trial,1)}; % rews1 
        trials(t,22)={rews(trial,2)}; %rews2 
        trials(t,23)={rews(trial,3)}; % rews3 
        trials(t,24)={trial}; % trial_number 
        trials(t,25)={trial}; %time_elapsed 
    end
        
  
end

data=trials(2:end,:);

end





