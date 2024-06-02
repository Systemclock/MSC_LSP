function [P,H,Z,S,obj,iter, Allacc] = myLMSC(X,alpha,beta,gama,lamda,epilsion,maxIter,K, kind, Y)
% alpha beta gama lamda 超参数
% output 
% P 
% H 
% Z 
% S 
V = size(X,2);    
N = size(X{1},2); 
for i=1:V
    D{i}=size(X{i},1);             
end
SD=0;
M=[];
for i=1:V
    SD = SD+D{i};                    
    M = [M;X{i}];                  
end
% rand('twister',5489);
% initial
% H = getPCA(M',K);
P = zeros(SD,K);
H = rand(K,N);   % H Z两种初始化方式
% Z = zeros(N,N);
Z = rand(N,N);
S = zeros(N,N);
obj = [];
allacc = [];
iter=1;err=1;
% 使用高斯函数初始化S 后面两项不起作用时，不跟新S



while (err>0.00001 && iter<=maxIter)
 
   for i=1:N
       for j=1:N    
             temp(i,j) = exp(-1*gama*(norm(H(:,i)-H(:,j),2)^2)/(lamda+1e-11));   
       end
       theta(i) = lamda *(1-log(sum(temp(i,:))));     
   end
   for i=1:N
       for j=1:N
             S(i,j) = exp((theta(i) - gama*norm(H(:,i)-H(:,j),2)^2)/(lamda+1e-11));
       end
       S(i,:) = S(i,:)./sum(S(i,:));
   end
   S=(S+S')./2;
   %
   LapMatrix = diag(sum(S,2))-S;  %L
   A = eye(N)+alpha *(eye(N) - Z - Z' + Z*Z')+gama*LapMatrix;
   C = P'* M;
   H = C/A;
   %% fixed H,P,S update Z
   
%    iter_Z=1;
%    while iter_Z<maxIter
%        Q = diag(0.5./sqrt(sum(Z.*Z,2)+epilsion));
%        Z = real(alpha * inv(alpha* H'* H+beta*Q)*H'*H);
%        iter_Z = iter_Z+1;
%    end
   Q = diag(0.5./sqrt(sum(Z.*Z,2)+epilsion));
   Z = real(alpha * inv(alpha* H'* H+beta*Q)*H'*H);
   
   %% fixed Z,H,S update P
   temp_M = M*H';
   [svd_U,~,svd_V] = svd(temp_M,'econ');
   P = svd_U*svd_V';
   
 
   temp_formulation1 = norm(M-P*H,'fro')^2;
 
   temp_formulation2 = alpha*norm(H-H*Z,'fro')^2;

   tmp1 = zeros(N,1);
   for i=1:N
       tmp1(i)=norm(Z(i,:),2);
   end
   temp_formulation3=beta*norm(tmp1,1);

   LapMatrix = diag(sum(S,2))-S;
   temp_formulation4 = gama*trace(H*LapMatrix*H');
   
   tmp2 = zeros(N,1);
   for i=1:N
       tmp2(i) = sum(S(:,i).*log(S(:,i)));
   end
   temp_formulation5=lamda*sum(tmp2);
   obj(iter) = temp_formulation1+temp_formulation2+temp_formulation3+temp_formulation4+temp_formulation5;%鎬诲??
   if iter>2
        err = abs(obj(iter-1)-obj(iter));
   end
   Z1 = abs(Z)+abs(Z');
   [L] = SpectralClustering(Z1,kind);
   L=L';
   Allacc(iter) = Accuracy(L',double(Y));
   iter = iter+1;
   

end

