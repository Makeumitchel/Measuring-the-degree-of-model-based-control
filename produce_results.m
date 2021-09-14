model='MBMF';

for i=1:30
    display(['trial =', num2str(i)]);
    condition=num2str(i);
    for n=3:3
        results = wrapper(model, condition)
        name=append('results_',model,'_',condition,'_', num2str(n));
        save(name,'results');
    end
end