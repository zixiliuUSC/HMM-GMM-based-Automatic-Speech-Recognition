%{
called function:
partition(mfcc,digit): partition each digit's frames according to the HMM state number
of the digit equally. Return a cell with size {}_#state*1. Each celler in
it contains a 13*#frames matrix. 
mixgauss_init(M, data, cov_type, method): the data input should be one
celler of the output of partition, i.e. a matrix of 13*#frames. And return
initials of guassian function of that state. 
%}
M = load('mfcc_all.mat', 'mfcc1');
init = struct;
init.weight = cell(1,10);
init.mu = cell(1,10);
init.sigma = cell(1,10);
state_num = [7 6 5 6 6 6 7 8 5 6];
stat = cell(1,10);
Data = cell(1,10);
for i = 1:10
   data = partition(M.mfcc1,i-1); 
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
save('init_gauss.mat','init')
save('statistic_speaker_indep_setting.mat','stat')
save('all_states_seperate.mat','Data')
fprintf('Report 7 numbers for the speaker-independent setting of digit six.\n')
Data{7}%Report 6 numbers for the speaker-independent setting of digit six. 
fprintf('Report 3 numbers for the speaker-independent setting of digit six.\n')
Data{4}%Report 3 numbers for the speaker-independent setting of digit six. 
