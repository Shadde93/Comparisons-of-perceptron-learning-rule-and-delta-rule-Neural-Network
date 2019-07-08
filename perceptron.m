close all;
% patterns and training data
figure(5)
verbose = 1;
separable = 0;
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
T = 200; % epochs

% initial weights 
w = randn(szT,szX);
s = w;
    

error = zeros(1,T);

for j = 1:T
    for i=1:size(X,2)
        if sign(w*X(:,1))-targets(i) ==  2 % should be -1, but is 1
            w = w-eta*X(:,i)';
        elseif sign(w*X(:,1))-targets(i) ==  -2 % should be 1, but is -1
            w = w+eta*X(:,i)';
        end
    end
    %if (mod(T,10) == 0 && j > 90)
        error(j) = sum(abs(sign(w*X) - targets)./2);
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
        
        title('perceptron')
    %end
    %end
end

figure(6)
plot(0:T,[sum(abs(sign(s*X) - targets)./2) error])
title('Error perceptron')
xlabel('Epochs')
ylabel('Errors')