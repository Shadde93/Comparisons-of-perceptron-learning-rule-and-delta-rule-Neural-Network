close all;
verbose = 1; % verbose > 0 generates plot
batch = 1; % 0 for sequential, 1 for batch
separable = 0; % 0 for non-separable, otherwise 1

% patterns and training data
if separable == 1
    [patterns, targets] = sepdata(verbose);
else 
    [patterns, targets] = nsepdata(verbose);
end
% input
X = [patterns;ones(1,size(patterns,2))];
szX = size(X,1); % number of rows of input data X
szT = size(targets,1); % number of rows of training data T

% params
eta = 0.001;
T = 200; % epochsw

% initial weights 
w = randn(szT,szX);


errorb = zeros(1,T); % number of false classifications in the batch update
errors = zeros(1,T); % number of false classifications in the sequential update

s = w;
%-------Batch---------

if batch == 1;
    for i=1:T
        w = w - eta*(w*X-targets)*X';
        
        errorb(i) = sum(abs(sign(w*X) - targets)./2);
        
        p = w(1,1:2);
        k = -w(1, size(patterns,1)+1) / (p*p');
        l = sqrt(p*p');
        plot (patterns(1, find(targets>0)), ...
        patterns(2, find(targets>0)), '*', ...
        patterns(1, find(targets<0)), ...
        patterns(2, find(targets<0)), '+', ...
        [p(1), p(1)]*k + [-p(2), p(2)]/l, ...
        [p(2), p(2)]*k + [p(1), -p(1)]/l, '-');
        drawnow;
        
        axis ([-2, 2, -2, 2], 'square');
        title('batch')
    end

end

%------Sequential-------


figure(2)
batch = 0;


if batch == 0
    for j = 1:T
        for i = 1:size(X,2)
            w = w - eta*(w*(X(:,i)-targets(i)))*X(:,i)';
           
        end
        
        errors(j) = sum(abs(sign(w*X) - targets)./2);
        p = w(1,1:2);
        k = -w(1, size(patterns,1)+1) / (p*p');
        l = sqrt(p*p');
        plot (patterns(1, find(targets>0)), ...
        patterns(2, find(targets>0)), '*', ...
        patterns(1, find(targets<0)), ...
        patterns(2, find(targets<0)), '+', ...
        [p(1), p(1)]*k + [-p(2), p(2)]/l, ...
        [p(2), p(2)]*k + [p(1), -p(1)]/l, '-');
        drawnow;
        
        axis ([-2, 2, -2, 2], 'square');
        
        title('seq')
    end
       
end

figure(3)
plot(0:T,[sum(abs(sign(s*X) - targets)./2) errorb]) % first term is the number of
% classifications before the weigths are updated
title('Error batch')
xlabel('Epochs')
ylabel('Errors')
figure(4)
plot(0:T,[sum(abs(sign(s*X) - targets)./2) errors])
title('Error seq')
xlabel('Epochs')
ylabel('Errors')