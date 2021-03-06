function [rewards, high_effort] = generate_rews(ntrials)

bounds = [0 9];
sd = 2;
high_effort= randi([0,1], ntrials,1); % randomly chose high or low effort trial

terminal_states = 3;
bounds = sort(bounds);
rewards = zeros(ntrials,terminal_states);
rewards(1,:) = round(rand(1,3)*(bounds(2)-bounds(1))+bounds(1));

for t = 2:ntrials
    
        for s = 1:terminal_states
            d = round(normrnd(0,sd));
            rewards(t,s) = rewards(t-1,s)+d;
            rewards(t,s) = min(rewards(t,s),max(bounds(2)*2 - rewards(t,s), bounds(1)));
            rewards(t,s) = max(rewards(t,s),min(bounds(1)*2 - rewards(t,s), bounds(2)));
        end
end