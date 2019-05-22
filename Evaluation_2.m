%Speaker-dependent ASR (multiple speakers) evaluation
load('mfcc_all.mat','mfcc1')%the first person is jackson
train = cell(1,140,10);
test = cell(1,60,10);
mfcc = reshape(mfcc1,[200,10]);

for i = 1:10
    v = randperm(200);
    for j = 1:200
        if j <= 140 
            train{1,j,i} = mfcc{v(j),i};
        else
            test{1,j-140,i} = mfcc{v(j),i};
        end
    end
end

%compute initial gaussian parameter
init = struct;
init.weight = cell(1,10);
init.mu = cell(1,10);
init.sigma = cell(1,10);
state_num = [7 6 5 6 6 6 7 8 5 6];
stat = cell(1,10);
Data = cell(1,10);
for i = 1:10
   data = partition(train,i-1);
   Data{i} = data;
   weight = cell(1,state_num(i));
   mu = cell(1,state_num(i));
   sigma = cell(1,state_num(i));
   stat{1,i} = zeros(1,state_num(i));
   for j = 1:state_num(i)
       [mu{1,j},sigma{1,j},weight{1,j}] = mixgauss_init(3,data{j,1},'diag','rnd');
       stat{1,i}(1,j) = size(data{j,1},2);
   end
   init.weight{1,i} = weight;
   init.mu{1,i} = mu;
   init.sigma{1,i} = sigma;
end

%train GMM model 
pr = {[1 0 0 0 0 0 0],...
         [1 0 0 0 0 0],...
         [1 0 0 0 0],...
         [1 0 0 0 0 0],...
         [1 0 0 0 0 0],...
         [1 0 0 0 0 0],...
         [1 0 0 0 0 0 0],...
         [1 0 0 0 0 0 0 0],...
         [1 0 0 0 0],...
         [1 0 0 0 0 0],...
         };
num = [7 6 5 6 6 6 7 8 5 6];
data1 = reshape(train, [140,10]);
Mu = cell(10,1);
Sigma = cell(10,1);
Weight = cell(10,1);
for k = 1:10
    mu = zeros(13,num(k),3);
    sigma = zeros(13,13,num(k),3);
    weight = zeros(num(k),3);
    for i = 1:num(k)%state number
        weight(i,:) = init.weight{k}{i};
        for j = 1:3
            mu(:,i,j) = init.mu{k}{i}(:,j);
            sigma(:,:,i,j) = init.sigma{k}{i}(:,:,j);
        end
    end
    Mu{k} = mu;
    Sigma{k} = sigma;
    Weight{k} = weight;
end

transmat = cell(10,1);
for i = 1:10
    transmat{i} = diag(1/2*ones(1,num(i)),0) + diag(1/2*ones(1,num(i)-1),1);
    transmat{i}(num(i),num(i)) = transmat{i}(num(i),num(i))*2;
end

[LL, prior_new, transmat_new, mu_new, Sigma_new, mixmat_new] = deal(cell(10,1));
for k = 1:10
    digit = cell(140,1);
    for i = 1:140
        digit{i,1} = data1{i,k};
    end

    [LL{k}, prior_new{k}, transmat_new{k}, mu_new{k}, Sigma_new{k}, mixmat_new{k}] = ...
        mhmm_em(digit,pr{k},transmat{k},Mu{k},Sigma{k},Weight{k});
end

%evaluate GMM
loglik = zeros(60,10,10);
error = cell(60,10,10);
for j = 1:10
    for g = 1:60
        digit1 = test{1,g,j};
    
        for k = 1:10
            [loglik(g,j,k), error{g,j,k}] = mhmm_logprob(digit1, prior_new{k}, transmat_new{k}, ...
                mu_new{k}, Sigma_new{k}, mixmat_new{k});
        end
    end
end

correct = zeros(1,10);
for i = 1:10
    for j = 1:60
        [u,p] = max(loglik(j,i,:));
        if i==p
            correct(i) = correct(i)+1;
        end
    end
end

fprintf('accuracy is %f\n', correct/60);