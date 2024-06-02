function [P,H,Z,S,obj,iter] = myLMSC_gama(X,alpha,beta,gama,epilsion,maxIter,K)
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
obj = zeros(maxIter,1);
iter=1;err=1;
% 使用高斯函数初始化S 后面两项不起作用时，不跟新S
for i=1:N
     for j=1:N
         diff = M(:,i)-M(:,j);
         S(i,j) = exp(- diff'*diff * 21.5);
     end
       S(i,:) = S(i,:)./sum(S(i,:));
end
S=(S+S')./2;

while (err>0.00001 && iter<maxIter)
 
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
   
  
   obj(iter) = temp_formulation1+temp_formulation2+temp_formulation3+temp_formulation4;%鎬诲??
   if iter>2
        err = abs(obj(iter-1)-obj(iter));
    end
   iter = iter+1;
   

end

