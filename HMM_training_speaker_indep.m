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
M = load('mfcc_all.mat', 'mfcc1');
data = reshape(M.mfcc1,[200,10]);
load('init_gauss.mat');
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
    digit = cell(200,1);
    for i = 1:200
        digit{i,1} = data{i,k};
    end

    [LL{k}, prior_new{k}, transmat_new{k}, mu_new{k}, Sigma_new{k}, mixmat_new{k}] = ...
        mhmm_em(digit,pr{k},transmat{k},Mu{k},Sigma{k},Weight{k});
end

figure('name','log likelyhood corresponding to iteration number','color','white')
set(gcf,'outerposition',get(0,'screensize'));
title({'log likelyhood corresponding to iteration number'},'interpreter','latex')
hold on
for k = 1:10
    plot(1:10,LL{k});
end
xlabel('iteration number','interpreter','latex')
ylabel('log likelyhood/($$dB$$)','interpreter','latex')
legend({'digit 0','digit 1','digit 2','digit 3','digit 4','digit 5','digit 6',...
    'digit 7','digit 8','digit 9'},'interpreter','latex')
set(gca,'TickLabelInterpreter','latex')%use latex to generate label
%transmat_new{k}, mu_new{k}, Sigma_new{k}, mixmat_new{k}
save('loglikelyhood.mat','LL')
save('prior.mat','prior_new')
save('transmation.mat','transmat_new')
save('mu.mat','mu_new')
save('sigma.mat','Sigma_new')
save('mixmatrix.mat','mixmat_new')



