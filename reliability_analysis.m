%% Reliability analysis (using correlation test)

repet=2;% nb of repetition of the fitted procedure
R_LL=[];  P_LL=[];  
R_M=[];  P_M=[]; 
R_W=[]; P_W=[];


for i=1:30
    display(['trial =', num2str(i)]);
    condition=num2str(i);
    %name=append('results_', condition);
    LL_low=zeros(98,repet); LL_high0=zeros(98,repet); LL_high1 =zeros(98,repet);
    LL_opt_low=zeros(98,repet); LL_opt_high0=zeros(98,repet); LL_opt_high1 =zeros(98,repet);
    M_low=zeros(98,repet); M_high0=zeros(98,repet); M_high1 =zeros(98,repet);
    W_low=zeros(98,repet); W_high0=zeros(98,repet); W_high1 =zeros(98,repet);
    for n=1:repet
        index=append(condition, '_', num2str(n));
        % MB method
        name=append('results_','MB','_', index);
        load(name);
        X=results.x;
       [ll_low , ll_high0, ll_high1]=Likelihoods('MB',condition, X);
        [ll_low_opt , ll_high0_opt, ll_high1_opt]=Likelihoods('MB_optimum',condition, X);
         LL_low(:,n)= ll_low; 
        LL_high0(:,n)=ll_high0; 
        LL_high1(:,n)= ll_high1 ;
        LL_opt_low(:,n)= ll_low_opt; 
        LL_opt_high0(:,n)=ll_high0_opt; 
        LL_opt_high1(:,n)= ll_high1_opt ;
        M_low(:,n)= ll_low-ll_low_opt ;
        M_high0(:,n)=(ll_high0-ll_high0_opt);
        M_high1(:,n)=(ll_high1-ll_high1_opt);
        % MBMF method
        name=append('results_','MBMF','_', index);
        load(name);
        X=results.x;
        b_low=X(:,1);  b_high0=X(:,2); b_high1=X(:,3);
        w_low=X(:,6);  w_high0=X(:,7); w_high1=X(:,8);
        W_low(1:98,n)= w_low.*b_low ;
        W_high0(1:98,n)=w_high0.*b_high0;
        W_high1(1:98,n)=w_high1.*b_high1;
        
    end
    [R_low,P_low] = corrcoef(LL_low);
    [R_high0,P_high0] = corrcoef(LL_high0);
    [R_high1,P_high1 ] = corrcoef(LL_high1);
    R_LL(end+1,:)=[mean(R_low(1, 2:end) ) , mean(R_high0(1, 2:end)), mean(R_high1(1, 2:end))];
    P_LL(end+1,:)=[mean(P_low(1, 2:end)) , mean(P_high0(1, 2:end)), mean(P_high1(1, 2:end))];
    [row,col] = find(isnan(R_LL(end,:)));
    R_LL(end,col)=1;
    [row,col] = find(isnan(P_LL(end,:)));
    P_LL(end,col)=0;
    
    
    [R_low,P_low] = corrcoef(M_low);
    [R_high0,P_high0] = corrcoef(M_high0);
    [R_high1,P_high1 ] = corrcoef(M_high1);
    R_M(end+1,:)=[mean(R_low(1, 2:end)) , mean(R_high0(1, 2:end)), mean(R_high1(1, 2:end))];
    P_M(end+1,:)=[mean(P_low(1, 2:end)) , mean(P_high0(1,2:end)), mean(P_high1(1, 2:end))];
    [row,col] = find(isnan(R_M(end,:)));
    R_M(end,col)=1;
    [row,col] = find(isnan(P_M(end,:)));
    P_M(end,col)=0;
    
    [R_low,P_low] = corrcoef(W_low);
    [R_high0,P_high0] = corrcoef(W_high0);
    [R_high1,P_high1 ] = corrcoef(W_high1);
    R_W(end+1,:)=[mean(R_low(1, 2:end)) , mean(R_high0(1, 2:end)), mean(R_high1(1, 2:end))];
    P_W(end+1,:)=[mean(P_low(1, 2:end)) , mean(P_high0(1,2:end)), mean(P_high1(1, 2:end))];
    [row,col] = find(isnan(R_W(end,:)));
    R_W(end,col)=1;
    [row,col] = find(isnan(P_W(end,:)));
    P_W(end,col)=0;
    
    

end



% correlation
display(['======== correlation between LL MB across 6 simulations ===========']);

display(['LL low: r=', num2str(mean( R_LL(:,1) )),' ;p=', num2str(mean( P_LL(:,1) ))  ]);
display(['LL high0: r=', num2str(mean( R_LL(:,2) )),' ;p=', num2str(mean( P_LL(:,2) )) ]);
display(['LL high1: r=', num2str(mean( R_LL(:,3) )),' ;p=', num2str(mean( P_LL(:,3) )) ]);

display(['========== correlation between Metrics 4 across 6 simulations ==========']);
display(['M low: r=', num2str(mean( R_M(:,1) )),' ;p=', num2str(mean( P_M(:,1) ))  ]);
display(['M high0: r=', num2str(mean( R_M(:,2) )),' ;p=', num2str(mean( P_M(:,2) )) ]);
display(['M high1: r=', num2str(mean( R_M(:,3) )),' ;p=', num2str(mean( P_M(:,3) )) ]);

display(['========== correlation between (b*w) fitted across 6 simulations ==========']);
display(['W low: r=', num2str(mean( R_W(:,1) )),' ;p=', num2str(mean( P_W(:,1) ))  ]);
display(['W high0: r=', num2str(mean( R_W(:,2) )),' ;p=', num2str(mean( P_W(:,2) )) ]);
display(['W high1: r=', num2str(mean( R_W(:,3) )),' ;p=', num2str(mean( P_W(:,3) )) ]);
