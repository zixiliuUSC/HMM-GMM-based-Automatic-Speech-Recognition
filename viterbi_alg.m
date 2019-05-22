function [prob, I, score1] = viterbi_alg(a, b, p, output)
%{
header: hw5@EE519, answer for Problem 1 (a)
Copyright: Zixi Liu, USCID: 2219403275, Email: zixiliu@usc.edu
Disctiption: Function viterbi_alg(a, b, p, output) is used to implement
viterbi algorithm. 
a: a is state transition matrix, size: # of states * # of states
b: b is output probability matrix, size: # of states * # of possible outputs
p: p is initial state probability, size: # of states * 1
output: observation sequence, size: column vector, # of observation * 1
prob: prob is probability of most probable sequence, in log, i.e. prob =
log(P_max[o, S|lambda])
I: I is the most probable state sequence. 
score: score is the probability that the model generate this observation
sequence, i.e., P(o|/lambda) in decimal. 
%}
A = log(a); B = log(b); pai = log(p); o = output; 
delta = zeros(size(A,1), size(o,1));
delta(:,1) = pai + B(:,o(1));
psi = zeros(size(delta,1),size(o,1));
score = exp(delta);
for t = 2:size(o)
   for j = 1:size(A, 1)%state order
      [delta(j,t), psi(j,t)] = max(delta(:,t-1) + A(:,j));%max[/delta_t-1(i)+a_ij], here we vectorize the forward process of state. 
      delta(j,t) = delta(j,t)+B(j,o(t,1));
      score(j,t) = sum(score(:,t-1) .* exp(A(:,j))) .* exp(B(j,o(t,1)));
   end
end
I = zeros(1,size(o,1));
[prob, I(end)] = max(delta(:,end));
for t = size(o):-1:2
   I(t-1) = psi(I(t),t);
end
score1 = sum(score(:,end));
end