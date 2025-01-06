X1 = feature1_train; % spatial representation obtained by the SPDNetCL
X2 = feature2_train; % temproal representation obtained by the TSCNNCL
X = double([X1 X2]');
Y = full(ind2vec(label))';
group_idx = [9, 144];
r1 = 1;
r2 = 1;
[W ,pi, obj] = Norm(X,Y,group_idx,r1,r2);

X1_test = feature1_test;
X2_test = feature2_test;
X_test = double([X1_test X2_test]');
label_test = double(label_test+1);
Y_test = full(ind2vec(label_test))';


[dumb idx] = sort(sum(W.*W,2),'descend'); % idx represents the ranking of features based on their importance


