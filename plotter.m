function plotter(problemNum, patterns, targets, out, X, w, v, ndata, verbose)

switch problemNum
%     case 1
%         if verbose > 0
%             plot (patterns(1, find(targets>0)), ...
%             patterns(2, find(targets>0)), '*', ...
%             patterns(1, find(targets<0)), ...
%             patterns(2, find(targets<0)), '+');
%         end
    case 2
        hin = w * X;
        hout = [2 ./ (1+exp(-hin)) - 1 ; ones(1,ndata)];
        oin = v * hout;
        out = 2 ./ (1+exp(-oin)) - 1;
    case 3
        gridsize = sqrt(ndata);
        zz = reshape(out, gridsize, gridsize);
        mesh(x,y,zz);
        axis([-5 5 -5 5 -0.7 0.7]);
        drawnow;
end