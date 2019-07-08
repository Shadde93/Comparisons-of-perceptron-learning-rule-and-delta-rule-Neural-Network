function [patterns, targets, x, y, ndata] = funApprox()

x=[-5:0.5:5]';
y=x;
z = exp(-x.*x*0.1) * exp(-y.*y*0.1)' - 0.5;
%mesh (x, y, z);

ndata = size(x,1)*size(y,1);
targets = reshape (z, 1, ndata);
[xx, yy] = meshgrid (x, y);
patterns = [reshape(xx, 1, ndata); reshape(yy, 1, ndata)];

permute = randperm(size(patterns,2));
patterns = patterns(:,permute);
targets = targets(:,permute);


end
