function data = partition(mfcc,digit)
%{
Each recording for the digit d is uniformly partitioned into Nd (almost)
equal parts where each part has about Nf =Nd MFCC vectors (frames).
%}
num = [7 6 5 6 6 6 7 8 5 6];
data = cell(num(digit+1),1);
[name, dupl, dig] = size(mfcc);
l = num(digit+1);
for k = 1:name
    for i = 1:dupl
        f = size(mfcc{k, i, digit+1},2);
        rec = mfcc{k, i, digit+1};
        for j = 1:l-1
            temp = rec(:,floor((j-1)/l*f)+1:floor(j/l*f));
            data{j,1} = [data{j,1},temp];
        end
        temp = rec(:,floor((l-1)/l*f)+1:end);
        data{num(digit+1),1} = [data{num(digit+1),1},temp];
    end
end
end