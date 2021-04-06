function y = pmean(x,p)

%calculates generalized mean
if p==0
    y = (prod(x,1)).^(1/size(x,1));
else
    y = (sum(x.^p,1)/size(x,1)).^(1/p);
end
