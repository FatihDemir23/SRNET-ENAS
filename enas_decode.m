function a = enas_decode(x,c)
assert(numel(x)==10 && all(x==round(x)) && ...
    all(x>=c.lb) && all(x<=c.ub),'Invalid integer chromosome.');
n = x(1); a.stacks = n;
a.blocks = c.book.blocks(x(2),1:n);
a.middle = c.book.middle(x(3),1:n);
a.output = c.book.output(x(4),1:n);
a.k1 = c.book.kernel(x(5),1:n);
a.k2 = c.book.kernel(x(6),1:n);
a.k3 = c.book.kernel(x(7),1:n);
a.stride = c.book.stride(x(8),1:n);
a.blockResidual = x(9)==2;
a.stackResidual = x(10)==2;
a.blockCount = sum(a.blocks);
end
