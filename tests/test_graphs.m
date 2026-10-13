function test_graphs
% Run from the repository root: addpath('tests'); test_graphs
c = enas_config('full');
% Exercise both connection flags, stride 1..5, even/odd kernels and projections.
for stride=1:5
    for block=1:2
        for stack=1:2
            x = [2 1 1 2 2 3 4 stride block stack];
            a = enas_decode(x,c);
            graph = enas_graph(a,c,2);
            net = dlnetwork(removeLayers(graph,'classification'));
            p = extractdata(predict(net,dlarray(zeros(100,100,1,2,'single'),'SSCB')));
            assert(isequal(size(p),[2 2]) && all(isfinite(p(:))));
        end
    end
end
a = enas_decode(c.ub,c);
assert(a.blockCount<=160);
disp('Graph shape and forward-pass checks passed.');
end
