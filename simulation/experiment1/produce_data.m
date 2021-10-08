function  groupdata = produce_data(parameters, nsubjects, ntrials)
% parameters = [b_low, b_high0, b_high1, lr, lambda, w_low, w_high0, w_high1, fr]
% nsubjects: number of subjects
% ntrials: number of trials
%% Models
% if MB simple (2 free parameters) :  
% b_low=b_high0=b_high1; w_low=w_high0=w_high1=1; lambda=0; fr=0 

% if MBMF simple (6 free parameters): 
% b_low=b_high0=b_high1; fr=0  

% if MB exhaustive (4 free parameters):  
% w_low=w_high0=w_high1=1; lambda=0; fr=0 

% if MBMF exhaustive (8 free parameters):  
% fr=0 

% if MB exhaustive forget (5 free parameters):  
% w_low=w_high0=w_high1=1; lambda=0 

% if MBMF exhaustive forget (9 free parameters)
%%

data = wrapper(parameters, nsubjects, ntrials);
groupdata = makerawdata(data, nsubjects);
end

