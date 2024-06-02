clear;clc
 
datalist{1}='100Leaves'; 
datalist{7}='BBC';
datalist{2}='Mfeat';
datalist{4}='NGs';
datalist{5}='3sources';  
datalist{6}='MSRC_v1';    
datalist{3}='bbcsport_2view';   
datalist{8}='yaleA_3view';
datalist{9}='WebKB_2views';
datalist{10}='Caltech101-7';
datalist{11}='Reuters';
addpath('./functions')

%% 
%-------------------
% 参数调整范围?0.001 0.01 0.1 1 10
% ALPHA = [0.001 0.005 0.01 0.05 0.1 0.5 1 5 10];
ALPHA = [0.0005 0.001 0.005 0.01 0.05 0.1 0.5 1 5 10];  
BETA =  [0.0005 0.001 0.005 0.01 0.05 0.1 0.5 1 5 10];
GAMA =  [0.0005 0.001 0.005 0.01 0.05 0.1 0.5 1 5 10];
LAMBDA =  [0.0005 0.001 0.005 0.01 0.05 0.1 0.5 1 5 10];
Epilision = [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10,1e-11,1e-12,1e-13,1e-14,1e-15,1e-16];
alpha = 0.001; % 0.001 0.01 1000
% beta=10;
beta = 0.01; % 0.01
gama=0.01;     %  0.01 0.001 0.001
lambda=0.1; % 0.001 10 0.1
epilision = 1e-3; % 1e-3 -12 1e-3
K=100;
%%
%----------------
for dataset_i = 2:2
    eval(['load ./dataset/' datalist{dataset_i}]);
    fprintf('训练数据集 %s \n', datalist{dataset_i});
%     加载参数
    eval(['load ./result/para_' datalist{dataset_i}])
%     eval('load ./result/100Leaves_Nbestparameter')
%     alpha=parameter_best(1);beta=parameter_best(2);gamma=parameter_best(3);lambda=parameter_best(4);
%     K=100;epilision=1e-12;
    maxIter=20;
    kind = length(unique(Y));   
    % 稀疏转换为全连接
    for i =1:length(X)
        if issparse(X{i})
            X{i}=full(X{i});
        end
        X{i}=X{i}';
        X{i} = NormalizeFea(X{i},0);
    end
    maxACC = 0;
for data_t = 1:1
    Accmean=zeros(10,1);
    NMImean=zeros(10,1);
    ARmean = zeros(10,1);
    Fmean = zeros(10,1);
    result = [];  
    t=1
    tic
%     [P,H,Z,S,obj,iter] = myLMSC(X,alpha,beta,gama,lambda,epilision,maxIter,K);
    [P,H,Z,S,obj,iter, allacc] = plotLMSC(X,alpha,beta,gama,lambda,epilision,maxIter,K,kind,Y);
time(t,:) = toc;    
%% 绘制双栏图
    allacc = allacc*100;
    figure()
%     yyaxis left
%%
    plot(obj,'^-','LineWidth',1,'Color','red');
    xlabel('iterations');
    ylabel('Objective Value');
%     xlabel('迭代次数');
%     ylabel('目标函数值');
%     yyaxis right
%     plot(allacc,'*-','LineWidth',1,'Color','red');
%     ylabel('ACC(%)');
    set(gca,'Box','off')
%     set(gca,'YLim',[0 100]);
%     set(gca,'YTick',[0 10 20 30 40 50 60 70 80 90 100]);
%     set(gca,'YTickLabel',[0 10 20 30 40 50 60 70 80 90 100]);

%% 聚类
    Z=abs(Z)+abs(Z');
%     [L,C] = run_kmeans(Z,kind);  
for iters = 1:1
    [L] = SpectralClustering(Z,kind);
    L=L';
    acc = Accuracy(L',double(Y));
    [A,nmi,avgent] = compute_nmi(Y,L');
    [f,p,r] = compute_f(Y,L');
    [AR,RI,MI,HI]=RandIndex(Y,L'); 
    %
    Accmean(iters)=acc;
    NMImean(iters)=nmi;
    Fmean(iters) = f;
    ARmean(iters) = AR;
    disp(['e:',num2str(epilision)]);
    disp(['K:',num2str(K)]);
    disp(['alpha',num2str(alpha), ' beta',num2str(beta),' gama',num2str(gama),' lambda',num2str(lambda)]);
    disp(['ACC  ',num2str(acc)]);
    disp(['NMI  ',num2str(nmi)]);
    disp(['F  ',num2str(f)]);
    disp(['AR  ',num2str(AR)]);
    fprintf('\n');
   
    %找最优参数
%     if acc>=result_best(1,:)
%         result_best(1,:)=acc;result(2,:)=nmi;result(3,:)=f;result(4,:)=AR;
%         parameter_best=[alpha;beta;gama;lambda];
%     end


end

% save(['result/' datalist{dataset_i} '_Nbestparameter'],'result_best','parameter_best')
% ACC = mean(Accmean);
% ACCstd = std(Accmean);
% NMI = mean(NMImean);
% NMIstd = std(NMImean);
% F = mean(Fmean);
% Fstd = std(Fmean);
% AR = mean(ARmean);
% ARstd = std(ARmean);
% if ACC > maxACC
%     maxACC = ACC;
%     result(t,:) = [[ACC, ACCstd], [NMI, NMIstd], [F, Fstd], [AR, ARstd]];
%     save(['results/' datalist{dataset_i} '_N5bestparameter'],'result')
%    
% end
% 
 end
     t = t+1;

end
% result(t,:) = [acc,nmi,f,AR];

%%   


%%
% result=[Accmean NMImean Fmean ARmean];
% res = zeros(2,4);
% res(1,1)=ACC;res(2,1)=ACCstd;
% res(1,2)=NMI;res(2,2)=NMIstd;
% res(1,3)=F;res(2,3)=Fstd;
% res(1,4)=AR;res(2,4)=ARstd;
% save(['result/' datalist{dataset_i} '_bestresult'],'result','res')
% save(['res/' datalist{dataset_i} '_KRtime'],'time')
%%
% figure()
% Z = reshape(result(:,1),9,10);
% X = repmat([1:9]',1,10);
% Y = repmat([1:10],9,1);
% surf(X,Y,Z);
% xlabel('alpha');
% ylabel('K');
% zlabel('ACC');
% set(gca,'xticklabel',{'0.001','0.005','0.01','0.05','0.1','0.5','1','5','10'});
% set(gca,'yticklabel',{'20','40','60','80','100','120','140','160','180','200'});
% grid on
%%
figure()
imagesc(Z)
% lim=caxis
% cmap = parula(64);
% cmap(64,:) = [1,0,0];
% colormap(cmap)
%% 绘制时间图
% clear;
% clc;
% figure('Position',[200,200,1000,500])
% Y = [10.28,57.958,3.735,58.238,1.571,13.1,37.2162808,17.819,55.86346657,54.307,29.556;
%     45.332,189.444,26.406,31.465,3.322,56.594,70.5188046,30.325,68.91854529,75.46,50.833;
%     14.636,53.094,40.612,30.287,0.725,1.157,39.7805978,17.865,21.1122168,23.309,35.189;
%     4.345,33.42,1.619,2.678,1.081,2.792,15.288023,9.19,28.60386769,17.96,5.679;
%     14.616,3.297,55.995,3.882,1.396,1.68,31.5980834,11.956,49.47514194,2.629,4.711;
%     152.53,415.034,84.25,181.177,18.94,8.379,690.6842602,1055.659,476.6268644,12.7,198.915;
%     984.367,718.591,554.887,313.723,29.488,21.976,1376.676344,84.284,137.4260324,9.514,217.649;
%     3.645,9.119,0.64,4.081,1.684,0.272,25.0612365,7.457,35.12052169,1.052,1.8;
%     ];
% Y = log(Y);
% X = 1:8;
% % color
% c1 = jet(11);
% 
% h = bar(X,Y,1);
% for i = 1:11
%     h(i).FaceColor=c1(i,:);
% end
% 
% 
% set(gca,'Box','off', ...
%     'XGrid','off','YGrid','on', ...
%     'Xticklabel',{'BBCsport','BBC','NGs','3-source','MSRC-v1','100leaves','Mfeat','YaleA'})
% hYlabel = ylabel('log_{2}Time'); hXlabel = xlabel('Dateset');
% hLegend = legend('DiMSC','LMSC','MLRSSC','FMR','NESE','GMC','SFMC','FESRL','EOMSC-CA','FastMICE','Ours');
% set(hLegend,'NumColumns',3);