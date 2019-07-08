function [patterns, targets] = sepdata(verbose)

rng('default')
classA(1,:) = randn(1,100).*0.5 + 1.0;
classA(2,:) = randn(1,100).*0.5 + 1.0;
classB(1,:) = randn(1,100).*0.5 - 1.0;
classB(2,:) = randn(1,100).*0.5 - 1.0;

szA = size(classA,2);
szB = size(classB,2);

patterns = [classA,classB];
targets = [ones(1,szA), -ones(1,szB)];

permute = randperm(szA+szB);
patterns = patterns(:,permute);
targets = targets(:,permute);

if verbose > 0 
    plot (patterns(1, find(targets>0)), ...
    patterns(2, find(targets>0)), '*', ...
    patterns(1, find(targets<0)), ...
    patterns(2, find(targets<0)), '+');
end
end
