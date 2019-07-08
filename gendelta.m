close all;

data = load('binMNIST.mat');
Xtrain = data.bindata_trn;
Xtest = data.bindata_tst;
ytrain = data.digtargets_trn;
ytest = data.digtargets_tst;

%Params
alpha = 0.9;
epochs = 200;
hidden = 10; %3 for encoder
eta = 0.5;
 

problem = 3; %1 for classification, 2 for encoder, 3 for function approx
separable = 2;
enc = 0; %1 for encoder details (sign(w) etc.) 

%rng(15);


switch problem
    case 1
        if separable == 1
            [patterns, targets] = sepdata(verbose);
        elseif separable == 0
            [patterns, targets] = nsepdata(verbose);
        end
    case 2
        [patterns, targets] = encoder();
    case 3
        [patterns, targets,x,y,ndataVal] = funApprox();     
end

%validation

rng(15)
classAtest(1,:) = [randn(1,25) .* 0.2 - 1.0, ...
randn(1,25) .* 0.2 + 1.0];
classAtest(2,:) = randn(1,50) .* 0.2 + 0.3;
classBtest(1,:) = randn(1,50) .* 0.3 + 0.0;
classBtest(2,:) = randn(1,50) .* 0.3 - 0.1;

szA = size(classAtest,2);
szB = size(classBtest,2);

patternstest = [classAtest,classBtest];
targetstest = [ones(1,szA), -ones(1,szB)];

[insizetest, ndatatest] = size(patternstest);
[outsizetest, ndatatest] = size(targetstest);

Xtest = [patternstest;ones(1,ndatatest)];

% %For validation
% 
 patternsVal = patterns;
 targetsVal = targets;

%For training

trainingLength = size(patterns,2); %for full length
patterns = patterns(:,1:trainingLength);
targets = targets(:,1:trainingLength);


[insize, ndata] = size(patterns);
[outsize, ndata] = size(targets);

X = [patterns;ones(1,ndata)];
szX = size(X,1);

%Initialize W and V
%rng(10); 

w = rand(hidden,szX)-0.5; %??
v = rand(outsize,hidden+1)-0.5; %??
dw = rand(hidden,szX)-0.5; %??
dv = rand(outsize,hidden+1)-0.5; %??

error = zeros(1,epochs); %Error vector
error1 = zeros(1,epochs);
%Update rule

for epoch = 1:epochs

    hin = w * X;
    hout = [2 ./ (1+exp(-hin)) - 1 ; ones(1,ndata)];
    oin = v * hout;
    out = 2 ./ (1+exp(-oin)) - 1;


    delta_o = (out - targets) .* ((1 + out) .* (1 - out)) * 0.5;
    delta_h = (v' * delta_o) .* ((1 + hout) .* (1 - hout)) * 0.5;
    delta_h = delta_h(1:hidden, :);

    dw = (dw .* alpha) - (delta_h * X') .* (1-alpha);
    dv = (dv .* alpha) - (delta_o * hout') .* (1-alpha);
    w = w + dw .* eta;
    v = v + dv .* eta;
    
    if (problem == 3 && trainingLength == size(patternsVal,2));
        gridsize = sqrt(ndata);
        zz = reshape(out, gridsize, gridsize);
        mesh(x,y,zz);
        axis([-5 5 -5 5 -0.7 0.7]);
        drawnow;
    end
    
%     if mod(epoch,10) == 0
%         disp(epoch)
%     end
    error(epoch) = sum(sum(abs(sign(out) - targets)./2));
    %test error
    
    hintest = w * Xtest;
    houttest = [2 ./ (1+exp(-hintest)) - 1 ; ones(1,ndatatest)];
    ointest = v * houttest;
    outtest = 2 ./ (1+exp(-ointest)) - 1;
    error1(epoch) = sum(sum(abs(sign(outtest) - targetstest)./2));
    
end
figure(10)
plot(error)
figure(11)
plot(error1)

%For encoder problem 

if enc == 1
    disp('Sign of w =')
    disp(sign(w))
    plot(1:size(error,2),error)
    if verbose > 0
        hin = w * X;
        hout = [2 ./ (1+exp(-hin)) - 1 ; ones(1,ndata)];
        oin = v * hout;
        out = 2 ./ (1+exp(-oin)) - 1;
        %disp('Output =')
        %disp(sign(out))
        eq(patterns,sign(out))
    end
end

%For generalization

if problem == 3
    X = [patternsVal;ones(1,ndataVal)];
    szX = size(X,1);
    hin = w * X;
    hout = [2 ./ (1+exp(-hin)) - 1 ; ones(1,ndataVal)];
    oin = v * hout;
    out = 2 ./ (1+exp(-oin)) - 1;
    
    figure;
    gridsize = sqrt(ndataVal);
    zz = reshape(out, gridsize, gridsize);
    mesh(x,y,zz);
    axis([-5 5 -5 5 -0.7 0.7]);
    drawnow;
    
    z = exp(-x.*x*0.1) * exp(-y.*y*0.1)' - 0.5;
     figure;
     mesh (x, y, z);

    error = abs(zz-z);
    %figure;
    %mesh(x,y,error)
end