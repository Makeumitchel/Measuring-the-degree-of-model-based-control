%% Reliability analysis (using correlation test)

repet=2;% nb of repetition of the fitted procedure
P_M=[]; 
P_W=[];
P_LL=[];  


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
        W_low(:,n)= w_low.*b_low ;
        W_high0(:,n)=w_high0.*b_high0;
        W_high1(:,n)=w_high1.*b_high1;
        
    end
    [h_low,p_low] = ttest(LL_low(:,1), LL_low(:,2) );
    [h_high0,p_high0] = ttest(LL_high0(:,1), LL_high0(:,2));
    [h_high1,p_high1 ] = ttest(LL_high1(:,1) ,LL_high1(:,2) );  
    P_LL(end+1,:)=[ p_low, p_high0, p_high1];
    
    [h_low,p_low] = ttest(M_low(:,1),M_low(:,2) );
    [h_high0,p_high0] = ttest(M_high0(:,1), M_high0(:,2));
    [h_high1,p_high1 ] = ttest(M_high1(:,1), M_high1(:,2));
    P_M(end+1,:)=[ p_low, p_high0, p_high1];
   

    [h_low,p_low] = ttest(W_low(:,1), W_low(:,2));
    [h_high0,p_high0] = ttest(W_high0(:,1), W_high0(:,2));
    [h_high1,p_high1 ] = ttest(W_high1(:,1), W_high1(:,2));
    P_W(end+1,:)=[ p_low, p_high0, p_high1];
   
    

end



% correlation
display(['======== correlation between LL MB across 6 simulations ===========']);

display(['LL low: p=', num2str(mean( P_LL(:,1) ))  ]);
display(['LL high0: p=',  num2str(mean( P_LL(:,2) )) ]);
display(['LL high1: p=', num2str(mean( P_LL(:,3) )) ]);

display(['========== correlation between Metrics 4 across 6 simulations ==========']);
display(['M low: p=',  num2str(mean( P_M(:,1) ))  ]);
display(['M high0: p=', num2str(mean( P_M(:,2) )) ]);
display(['M high1: p=',  num2str(mean( P_M(:,3) )) ]);

display(['========== correlation between (b*w) fitted across 6 simulations ==========']);
display(['W low: p=',  num2str(mean( P_W(:,1) ))  ]);
display(['W high0: p=',  num2str(mean( P_W(:,2) )) ]);
display(['W high1: p=',  num2str(mean( P_W(:,3) )) ]);
