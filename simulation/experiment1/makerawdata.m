function groupdata = makerawdata(data, nsubjects)

% This function takes the raw data and turns it into a matlab structure
% with the behavior and trial information for each participant in a
% separate field.
scaling_factor=1; % we already rescaled 1/9*rewards

for subject = 1:nsubjects
    id = append('subject',num2str(subject)); 
    trials = cell2mat(data(strcmp(data(:,1),{id}),2:25));
    trials(trials(:,19)==1,:) = []; % throwout practice
    
    subdata.id = id;
    subdata.gender = 0;
    subdata.age = 0;
    subdata.N = size(trials,1);
    
    subdata.high_effort = trials(:,1);
    subdata.state0 = trials(:,2);
    subdata.stim_0_1 = trials(:,3);
    subdata.stim_0_2 = trials(:,4);
    subdata.stims0 = trials(:,3:4);
    subdata.rt0 = trials(:,5);
    subdata.choice0 = trials(:,6);
    subdata.response0 = trials(:,7);
    subdata.state1 = trials(:,8);
    subdata.stim_1_1 = trials(:,9);
    subdata.stim_1_2 = trials(:,10);
    subdata.stim_1_3 = trials(:,11);
    subdata.stims1 = trials(:,9:11);
    subdata.rt1 = trials(:,12);
    subdata.choice1 = trials(:,13);
    subdata.response1 = trials(:,14);
    subdata.rt2 = trials(:,15);
    subdata.points = trials(:,16)*scaling_factor;
    subdata.state2 = trials(:,17);
    subdata.score = trials(:,18);
    subdata.practice = trials(:,19);
    subdata.rews1 = trials(:,20)*scaling_factor;
    subdata.rews2 = trials(:,21)*scaling_factor;
    subdata.rews3 = trials(:,22)*scaling_factor;
    subdata.rews = trials(:,20:22)*scaling_factor;
    subdata.trial_number = trials(:,23);
    subdata.time_elapsed = trials(:,24);
    subdata.rocket_order = [1 2 3 4 5 6] ;
    
   
    subdata.middle_stims{1} = [subdata.rocket_order(1) subdata.rocket_order(2) subdata.rocket_order(3);
        subdata.rocket_order(4) subdata.rocket_order(5) subdata.rocket_order(6)];
    subdata.middle_stims{2} = [subdata.rocket_order(1) subdata.rocket_order(5); 
        subdata.rocket_order(4) subdata.rocket_order(3);
        subdata.rocket_order(2) subdata.rocket_order(6)];
  
        
     
     subdata.Tm_top = [1 0 0; 0 1 0; 0 0 1];
     subdata.Tm_middle = [1 0 0; 0 1 0; 0 0 1; 1 0 0; 0 1 0; 0 0 1] ;
    
    
    subdata.tasktime = 1;
    subdata.instructiontime = 1;
    subdata.totaltime = 1;
    
    subdata.missed = subdata.rt2==-1;
    
    groupdata.subdata(subject) = subdata;
    
end
groupdata.i=1:nsubjects % field "i" necessary to work with Kool et al. code
end

