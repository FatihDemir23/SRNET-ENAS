function c = enas_config(mode)
% New reference implementation; not the historical experimental code.
if nargin < 1, mode = 'full'; end
c.seed = 42;
c.imageSize = [100 100 1];
c.population = 20;
c.generations = 50;
c.candidateEpochs = 50;
c.finalEpochs = 500;
c.batchSize = 128;
c.learningRate = 0.001;
c.executionEnvironment = 'auto';
c.errorWeight = 0.85;
c.sizeWeight = 0.15;
c.blockNormalizer = 160;
c.mutationIndividual = 0.1;
c.mutationGene = 0.01;
c.crossoverFraction = 0.8;
c.stallGenerations = 50;
c.functionTolerance = 1e-6;
c.testFraction = 0.2;
c.validationFraction = 0.1; % Fraction of the remaining development data.
% Explicit finite catalogues, newly defined for this implementation.
% A row is a sequence; only its first x(1) entries are used.
c.book.blocks = [(1:20)' * ones(1,8); 4 3 2 2 2 2 2 2];
c.book.middle = [(16:16:128)' * ones(1,8); 16 32 64 64 64 64 64 64];
c.book.output = [(16:16:256)' * ones(1,8); 32 64 128 128 128 128 128 128];
c.book.kernel = (1:11)' * ones(1,8); % Square k-by-k filters.
c.book.stride = [(1:5)' * ones(1,8); 1 2 2 1 1 1 1 1];
c.lb = ones(1,10);
c.ub = [8 size(c.book.blocks,1) size(c.book.middle,1) ...
    size(c.book.output,1) 11 11 11 size(c.book.stride,1) 2 2];
c.mode = char(mode);
switch c.mode
    case 'full'
    case 'smoke'
        c.population = 4; c.generations = 1;
        c.candidateEpochs = 1; c.finalEpochs = 1; c.batchSize = 4;
        c.ub = [2 2 2 2 3 3 3 2 2 2];
    otherwise
        error('ENAS:Mode','Use full or smoke.');
end
end
