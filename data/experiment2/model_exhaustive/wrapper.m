function results = wrapper(method)
addpath('../mfit'); % add mfit path to current directory
nstarts = 2; %number of random parameter initializations 
load ../groupdata
data = groupdata.subdata(groupdata.i);

% run optimization
if strcmp(method,'method1')
    f = @(x,data) MBMF_exhaustive_probes_complexity_llik(x,data);
    model='MBMF';
else
    f = @(x,data) MB_exhaustive_probes_complexity_llik(x,data);
    model='MB';
end


results = mfit_optimize(f,set_params(model),data,nstarts);

% likelihoods
[ Low, High0,  High1]=likelihoods(results, method)

if strcmp(method ,'method1')
    results.metrics = [results.x(:,6),results.x(:,7), results.x(:,8)];
else
    results.metrics = [Low, High0 ,  High1];
end

disp(['----- ', method, ' ------']);
disp(['mean metrics: Low = ', num2str(mean(results.metrics(:,1))), '; High0 = ', num2str(mean(results.metrics(:,2))),  '; High1 = ', num2str(mean(results.metrics(:,3)))]);
end