clear;clc
datalist{1}='BBC';
datalist{2}='MSRC_v1'; 
datalist{8}='Mfeat';
datalist{7}='100Leaves'; 
datalist{5}='yaleA_3view';
datalist{6}='3sources';  
datalist{4}='bbcsport_2view';  
datalist{3}='NGs';
% datalist{7}='bbcsport_2view';   
% datalist{8}='BBC';
% datalist{9}='WebKB_2views';
% datalist{10}='Caltech101-7';
% datalist{1}='NGs';
% datalist{2}='yaleA_3view';
% datalist{3}='bbcsport_2view';
% datalist{4}='NGs';
addpath('./functions')
%%
for dataset_i = 1:4
    eval(['load ./dataset/' datalist{dataset_i}]);
    disp(['run on ' datalist{dataset_i}])
    maxIter=100;
    kind = length(unique(Y));   %聚类簇数
    resultdir = 'resM/';


    % 数据转置 每个视图的数据是m x n的，m是维度，n是样本数
    for i =1:size(X,2)
        if issparse(X{i})
            X{i}=full(X{i});
        end
        X{i}=X{i}';
        X{i} = NormalizeFea(X{i},0);
    end
     % 设置保存路径
    filepath = strcat(resultdir,char(datalist{dataset_i}),'_MSresult.csv');
    fid = fopen(filepath, 'w');
    if fid == -1
        error('无法打开或创建目标文件');
    end
    fprintf(fid, 'alpha,beta,lambda,gamma,ACCmean,ACCstd,NMImean,NMIstd,F,Fstd,AR,ARstd\n');%% 写入表头
    % 加载参数
    ALPHA = [0.001 0.005 0.01 0.05 0.1 0.5 1 5 10 100 1000];  
    BETA =  [0.001 0.005 0.01 0.05 0.1 0.5 1 5 10 100 1000];
    GAMA =  [0.001 0.005 0.01 0.05 0.1 0.5 1 5 10 100 1000];
    LAMBDA =  [0.001 0.005 0.01 0.05 0.1 0.5 1 5 10 100 1000];
%     ALPHA = [0.001  0.01  0.1  1  10 100 1000];  
%     BETA =  [0.001  0.01  0.1  1  10 100 1000];
%     GAMA =  [0.001  0.01  0.1  1  10 100 1000];
%     LAMBDA =  [0.001  0.01  0.1 1 10 100 1000];
    K_t = [100:100:1000];
    Epilision = [1, 1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10,1e-11,1e-12,1e-13,1e-14,1e-15,1e-16];
% alpha=5;
%     ALPHA= [0.001];
%     BETA = [0.001];
%     GAMA = [0.001];
%     LAMBDA = [0.001 ];
    epilision = 1e-12;
    K = 100;
%     Accmean=zeros(10,1);
%     NMImean=zeros(10,1);
%     ARmean = zeros(10,1);
%     Fmean = zeros(10,1);
    Accmean=[];
    NMImean=[];
    ARmean = [];
    Fmean = [];
%     eval(['load ./result/100Leaves_Nbestparameter.mat']);
%     alpha = parameter_best(1); beta=parameter_best(2); gamma=parameter_best(3);
%     lambda=parameter_best(4); e=1e-12;K=100;
    %%
% 找alpha beta gamma lambda
for i = 1:length(ALPHA)
    alpha = ALPHA(i); beta = BETA(1); gamma = GAMA(1); lambda = LAMBDA(1);
    [P,H,Z,S,obj] = myLMSC(X,alpha,beta,gamma,lambda,epilision,maxIter,K);
%     Z=abs(Z)+abs(Z');
Z = (S+S')/2;
    for d = 1:10
            [L] = SpectralClustering(Z,kind);
            L=L';
            acc = Accuracy(L',double(Y));
            [A,nmi,avgent] = compute_nmi(Y,L');
            [f,p,r] = compute_f(Y,L');
            [AR,RI,MI,HI]=RandIndex(Y,L'); 
%             disp(['ACC  ',num2str(acc)]);
%             disp(['NMI  ',num2str(nmi)]);
%             disp(['F  ',num2str(f)]);
%             disp(['AR  ',num2str(AR)]);
%             fprintf('\n');
            Accmean(d)=acc;
            NMImean(d)=nmi;
            Fmean(d) = f;
            ARmean(d) = AR;
    end
    ACC = mean(Accmean);
    ACCstd = std(Accmean);
    NMI = mean(NMImean);
    NMIstd = std(NMImean);
    F = mean(Fmean);
    Fstd = std(Fmean);
    AR = mean(ARmean);
    ARstd = std(ARmean);
    result_a(i,:) = [alpha, ACC, NMI, F, AR];
end
[~, idx] = max(result_a(:,2));
bestalpha = ALPHA(idx);

    for j = 1:length(BETA)
        alpha=bestalpha; beta = BETA(j); gamma = GAMA(1); lambda = LAMBDA(1);
            [P,H,Z,S,obj] = myLMSC(X,alpha,beta,gamma,lambda,epilision,maxIter,K);
%             Z=abs(Z)+abs(Z');
            Z = (S+S')/2;
            for d = 1:10
                [L] = SpectralClustering(Z,kind);
                L=L';
                acc = Accuracy(L',double(Y));
                [A,nmi,avgent] = compute_nmi(Y,L');
                [f,p,r] = compute_f(Y,L');
                [AR,RI,MI,HI]=RandIndex(Y,L'); 
                Accmean(d)=acc;
                NMImean(d)=nmi;
                Fmean(d) = f;
                ARmean(d) = AR;
            end
            ACC = mean(Accmean);
            ACCstd = std(Accmean);
            NMI = mean(NMImean);
            NMIstd = std(NMImean);
            F = mean(Fmean);
            Fstd = std(Fmean);
            AR = mean(ARmean);
            ARstd = std(ARmean);
            result_b(j,:) = [beta, ACC, NMI, F, AR];
    end
    [~, idx] = max(result_b(:,2));
    bestbeta = BETA(idx);
            for i_gamma = 1:length(GAMA)
               alpha=bestalpha; beta=bestbeta; gamma = GAMA(i_gamma);lambda = LAMBDA(1);

               [P,H,Z,S,obj] = myLMSC(X,alpha,beta,gamma,lambda,epilision,maxIter,K);
%                Z=abs(Z)+abs(Z');
                Z = (S+S')/2;
               for d = 1:10
                    [L] = SpectralClustering(Z,kind);
                    L=L';
                    acc = Accuracy(L',double(Y));
                    [A,nmi,avgent] = compute_nmi(Y,L');
                    [f,p,r] = compute_f(Y,L');
                    [AR,RI,MI,HI]=RandIndex(Y,L'); 
                    Accmean(d)=acc;
                    NMImean(d)=nmi;
                    Fmean(d) = f;
                    ARmean(d) = AR;
               end
                ACC = mean(Accmean);
                ACCstd = std(Accmean);
                NMI = mean(NMImean);
                NMIstd = std(NMImean);
                F = mean(Fmean);
                Fstd = std(Fmean);
                AR = mean(ARmean);
                ARstd = std(ARmean);
                result_g(i_gamma,:) = [gamma, ACC, NMI, F, AR];
            end
            [~, idx] = max(result_g(:,2));
            bestgamma = GAMA(idx);
            %%
                for j_lambda = 1:length(LAMBDA)
                   alpha=bestalpha; beta=bestbeta; gamma = bestgamma; lambda = LAMBDA(j_lambda);
                    
                    [P,H,Z,S,obj] = myLMSC(X,alpha,beta,gamma,lambda,epilision,maxIter,K);
%                     Z=abs(Z)+abs(Z');
                    Z = (S+S')/2;
                    for d = 1:10
                        [L] = SpectralClustering(Z,kind);
                        L=L';
                        acc = Accuracy(L',double(Y));
                        [A,nmi,avgent] = compute_nmi(Y,L');
                        [f,p,r] = compute_f(Y,L');
                        [AR,RI,MI,HI]=RandIndex(Y,L'); 
                        Accmean(d)=acc;
                        NMImean(d)=nmi;
                        Fmean(d) = f;
                        ARmean(d) = AR;
                    end
                    % 保存数据
                    ACC = mean(Accmean);
                    ACCstd = std(Accmean);
                    NMI = mean(NMImean);
                    NMIstd = std(NMImean);
                    F = mean(Fmean);
                    Fstd = std(Fmean);
                    AR = mean(ARmean);
                    ARstd = std(ARmean);
                    result_l(j_lambda,:) = [lambda, ACC, ACCstd, NMI, NMIstd, F, Fstd, AR, ARstd];
                end

                [~, idx] = max(result_l(:,2));
                bestlambda = LAMBDA(idx);
                fprintf(fid, '%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n', bestalpha,bestbeta,bestgamma,bestlambda,result_l(idx,2),result_l(idx,3),result_l(idx,4),result_l(idx,5),result_l(idx,6),result_l(idx,7),result_l(idx,8),result_l(idx,9));
    
    %%
              for K_i = 1:length(K_t)
%                     bestalpha=5;bestbeta=1;bestgamma=5;bestlambda=0.01;
                    alpha=bestalpha; beta=bestbeta; gamma = bestgamma; lambda = bestlambda; K = K_t(K_i);
                    [P,H,Z,S,obj] = myLMSC(X,alpha,beta,gamma,lambda,epilision,maxIter,K);
%                     Z=abs(Z)+abs(Z');
                    Z = (S+S')/2;
                    for d = 1:10
                        [L] = SpectralClustering(Z,kind);
                        L=L';
                        acc = Accuracy(L',double(Y));
                        [A,nmi,avgent] = compute_nmi(Y,L');
                        [f,p,r] = compute_f(Y,L');
                        [AR,RI,MI,HI]=RandIndex(Y,L'); 
                        Accmean(d)=acc;
                        NMImean(d)=nmi;
                        Fmean(d) = f;
                        ARmean(d) = AR;
                    end
                    % 保存数据
                    ACC = mean(Accmean);
                    ACCstd = std(Accmean);
                    NMI = mean(NMImean);
                    NMIstd = std(NMImean);
                    F = mean(Fmean);
                    Fstd = std(Fmean);
                    AR = mean(ARmean);
                    ARstd = std(ARmean);
                    result_k(K_i,:) = [K, ACC, ACCstd, NMI, NMIstd, F, Fstd, AR, ARstd];
              end
                [~, idx] = max(result_k(:,2));
                bestK = K_t(idx);
               
                save(['./resM/' ,datalist{dataset_i},'_',num2str(bestK),'_MSresult.mat'],'result_k');
   %%    

    for e_j = 1:length(Epilision)
        
        alpha=bestalpha; beta=bestbeta; gamma = bestgamma; lambda = bestlambda; K = bestK; epilision=Epilision(e_j);

                    [P,H,Z,S,obj] = myLMSC(X,alpha,beta,gamma,lambda,epilision,maxIter,K);
%                     Z=abs(Z)+abs(Z');
                    Z = (S+S')/2;
                    for d = 1:10
                        [L] = SpectralClustering(Z,kind);
                        L=L';
                        acc = Accuracy(L',double(Y));
                        [A,nmi,avgent] = compute_nmi(Y,L');
                        [f,p,r] = compute_f(Y,L');
                        [AR,RI,MI,HI]=RandIndex(Y,L'); 
                        Accmean(d)=acc;
                        NMImean(d)=nmi;
                        Fmean(d) = f;
                        ARmean(d) = AR;
                    end
                    % 保存数据
                    ACC = mean(Accmean);
                    ACCstd = std(Accmean);
                    NMI = mean(NMImean);
                    NMIstd = std(NMImean);
                    F = mean(Fmean);
                    Fstd = std(Fmean);
                    AR = mean(ARmean);
                    ARstd = std(ARmean);
                    result_e(e_j,:) = [epilision, ACC, ACCstd, NMI, NMIstd, F, Fstd, AR, ARstd];
             
    end
                [~, idx] = max(result_e(:,2));
                beste = Epilision(idx);
                save(['./resM/' ,datalist{dataset_i},'_',num2str(beste),'_MSresult.mat'],'result_e');
%%


end