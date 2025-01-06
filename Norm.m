function [W, pi, obj] = Norm(X,Y,group_idx,r1,r2)

group_num = 2; 

X = double(X);


[dim,n] = size(X);
class_num = size(Y,2);
y = sum(Y,2)';
YY = ones(dim,1)*y;

W = ones(dim,class_num); 

dimall = dim*class_num;
XY = X*Y;
XX = X*X';

t = cputime;
for iter = 1:10

    sid = 1;
    for c = 1:group_num
        idx = [sid:group_idx(c)];
        sid = group_idx(c)+1; 
        Vc = W(idx,:);   
        di = sqrt(sum(Vc.*Vc,1)+eps);   
        Vig(idx,:) = ones(length(idx),1)*di;
        ob(c) = sum(di);
    end
    D1 = 0.5./Vig;    
    Vi = sqrt(sum(W.*W,2)+eps); 
    D2 = diag(0.5./(Vi));   
    
    
    WX = W'*X;
    t1 = exp(WX);
    t2 = sum(t1);
    
    tt = ones(class_num,1)*t2;
    pi = t1./tt;
    
    te1 = WX'.*Y;
    te1 = sum(te1(:)); 
    te2 = log(tt').*Y;
    te2 = sum(te2(:));  
    obj(iter) = te2 - te1 + r1*sum(ob) + r2*sum(Vi); 
      

    d = zeros(dim,class_num);
    for k = 1:class_num
        temp = ones(dim,1)*pi(k,:);
        G = YY.*X.*temp;
        d(:,k) = sum(G,2); 
    end
    
    
    d = d - XY;
    Wd1 = D1(:,1:class_num).*W;
    Wd2 = D2*W;
    d = d(:) + 2*r1*Wd1(:) + 2*r2*Wd2(:);
    
    H = zeros(dimall);
    for k = 1:class_num
        kdix = (k-1)*dim+1:k*dim;
        for m = 1:class_num
            mdix = ((m-1)*dim+1):(m*dim);
            D = sparse(diag(y.*pi(k,:).*pi(m,:)));
            if k == m
                D11 = sparse(diag(y.*pi(k,:)));
                H(kdix, mdix) = X*D11*X' - X*D*X' + 2*r1*diag(D1(:,k)) + 2*r2*D2;
            else
                H(kdix, mdix) = -X*D*X';
            end
        end
    end
    
    b = H\d;
    W = W(:) - b;
    W = reshape(W,dim,class_num); 
	
end

