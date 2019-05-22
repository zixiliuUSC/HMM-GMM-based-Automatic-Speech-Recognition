load('all_states_seperate.mat','Data')
load('mfcc_all.mat','mfcc1')
load('mixmatrix.mat','mixmat_new')
load('mu.mat','mu_new')
load('prior.mat','prior_new')
load('sigma.mat','Sigma_new')
load('transmation.mat','transmat_new')
%load('sigma.mat')
path = cell(10,200);
for i = 1:10
    for j = 1:4
        for k = 1:50
            [B, B2] = mixgauss_prob(mfcc1{j,k,i}, mu_new{i}, Sigma_new{i}, mixmat_new{i});
            path{i,(j-1)*50+k} = viterbi_path(prior_new{i}, transmat_new{i}, B);
        end
    end
end
%Plot_GM(mixmat_new{7}(3,:),reshape(mu_new{7}(:,3,:),[13,3]),reshape(Sigma_new{7}(:,:,3,:),[13,13,3]))

figure('name','Histogram of the 2nd MFCC, for the 3rd state of the corresponding HMM in digit 6','color','white')
set(gcf,'outerposition',get(0,'screensize'));
stat = Data{7}{3}(2,:);%digit 6, state 3, the 2nd MFCC collection. 
histogram(stat,-10:0.5:11)
title({'Histogram of 2nd MFCC,3rd state of digit 6'},'interpreter','latex')
xlabel('MFCC value','interpreter','latex')
ylabel('frequency ','interpreter','latex')
legend({'digit 0','digit 1','digit 2','digit 3','digit 4','digit 5','digit 6',...
    'digit 7','digit 8','digit 9'},'interpreter','latex')
set(gca,'TickLabelInterpreter','latex')%use latex to generate label

%compute the marginal distribution of 2nd MFCC in digit 6 state 3. 
sigma_d6s3 = Sigma_new{7}(:,:,3,:);
sigma_d6s3 = reshape(sigma_d6s3,[13,13,3]);
sigma_d6s3([1,2],:,:) = sigma_d6s3([2,1],:,:);
v = sigma_d6s3(1,1,:);
sigma_d6s3(1,1,:) = sigma_d6s3(1,2,:);sigma_d6s3(1,2,:) = v;
v = sigma_d6s3(2,1,:);
sigma_d6s3(2,1,:) = sigma_d6s3(2,2,:);sigma_d6s3(2,2,:) = v;
mu_d6s3 = mu_new{7}(:,3,:);
mu_d6s3 = reshape(mu_d6s3,[13,3]);
mu_d6s3([1,2],:) = mu_d6s3([2,1],:);
mixmat_d6s3 = mixmat_new{7}(3,:);





