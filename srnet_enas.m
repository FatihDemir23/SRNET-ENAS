function result = srnet_enas(dataDir,outDir,mode)
% SRNET_ENAS Search, retrain from scratch, and evaluate on held-out images.
% result = srnet_enas('data','results_run01','smoke');
if nargin<3, mode='full'; end
c = enas_config(mode);
assert(exist('ga','file')~=0,'Global Optimization Toolbox is required.');
assert(exist('trainNetwork','file')~=0,'Deep Learning Toolbox is required.');
assert(isfolder(dataDir),'Data directory does not exist.');
assert(~isfolder(outDir),'Use a new output directory for each run.');
rng(c.seed,'twister');
imds = imageDatastore(dataDir,'IncludeSubfolders',true,'LabelSource','foldernames');
assert(numel(categories(imds.Labels))==2,'Exactly two class folders required.');
counts = countEachLabel(imds);
assert(all(counts.Count>=20),'At least 20 images per class required.');
[dev,test] = splitEachLabel(imds,1-c.testFraction,'randomized');
[tr,va] = splitEachLabel(dev,1-c.validationFraction,'randomized');
mkdir(outDir);
split = table([tr.Files;va.Files;test.Files], ...
    [tr.Labels;va.Labels;test.Labels], ...
    [repmat("train",numel(tr.Files),1);repmat("validation",numel(va.Files),1); ...
    repmat("test",numel(test.Files),1)],'VariableNames',{'file','label','split'});
writetable(split,fullfile(outDir,'split.csv'));
environment = ver;
save(fullfile(outDir,'configuration.mat'),'c','environment');
cache = containers.Map('KeyType','char','ValueType','double');
logRows = cell(0,6);
bestSeen = inf;
opts = optimoptions('ga','PopulationSize',c.population, ...
    'MaxGenerations',c.generations,'EliteCount',1, ...
    'SelectionFcn',{@selectiontournament,2}, ...
    'CrossoverFraction',c.crossoverFraction,'CrossoverFcn',@crossOver, ...
    'MutationFcn',@mutateChildren,'FunctionTolerance',c.functionTolerance, ...
    'MaxStallGenerations',c.stallGenerations,'UseParallel',false, ...
    'UseVectorized',false,'Display','iter','OutputFcn',@checkpoint);
[x,fval,exitflag,output,population,scores] = ga(@fitness,10,[],[],[],[], ...
    c.lb,c.ub,[],1:10,opts);
a = enas_decode(x,c);
save(fullfile(outDir,'search_result.mat'),'x','a','fval','exitflag', ...
    'output','population','scores','c');
% Fixed epoch budget: no test-based or validation-based final selection.
rng(c.seed+1,'twister');
g = enas_graph(a,c,2);
finalOptions = trainingOptions('adam','InitialLearnRate',c.learningRate, ...
    'MaxEpochs',c.finalEpochs,'MiniBatchSize',c.batchSize, ...
    'Shuffle','every-epoch','ExecutionEnvironment',c.executionEnvironment, ...
    'Verbose',false,'Plots','none');
[net,info] = trainNetwork(prepared(dev,c),g,finalOptions);
% Test is opened only after architecture search and final fitting finish.
[pred,probabilities] = classify(net,prepared(test,c),'MiniBatchSize',c.batchSize);
truth = test.Labels;
accuracy = mean(pred==truth);
classOrder = net.Layers(end).Classes;
cm = confusionmat(truth,pred,'Order',classOrder);
result = struct('accuracy',accuracy,'confusionMatrix',cm, ...
    'classOrder',classOrder,'chromosome',x,'architecture',a,'mode',c.mode);
save(fullfile(outDir,'final_model.mat'),'net','info','result','c','-v7.3');
save(fullfile(outDir,'test_predictions.mat'),'truth','pred','probabilities','classOrder');
fprintf('Held-out accuracy: %.4f; mode: %s\n',accuracy,c.mode);

    function value = fitness(z)
        z = round(z); key = sprintf('%d,',z);
        if isKey(cache,key), value=cache(key); return; end
        % Candidate training must not consume the GA's random stream.
        state = rng; restore = onCleanup(@() rng(state)); %#ok<NASGU>
        rng(c.seed,'twister');
        arch = enas_decode(z,c); timer = tic;
        graph = enas_graph(arch,c,2);
        training = trainingOptions('adam','InitialLearnRate',c.learningRate, ...
            'MaxEpochs',c.candidateEpochs,'MiniBatchSize',c.batchSize, ...
            'Shuffle','every-epoch','ExecutionEnvironment',c.executionEnvironment, ...
            'Verbose',false,'Plots','none');
        % No catch-all penalty: data, memory and graph errors stop the run.
        model = trainNetwork(prepared(tr,c),graph,training);
        prediction = classify(model,prepared(va,c),'MiniBatchSize',c.batchSize);
        acc = mean(prediction==va.Labels);
        value = c.errorWeight*(1-acc)+c.sizeWeight*arch.blockCount/c.blockNormalizer;
        assert(isfinite(value),'Non-finite candidate fitness.');
        cache(key) = value;
        logRows(end+1,:) = {key,acc,arch.blockCount,value,toc(timer),size(logRows,1)+1};
        writetable(cell2table(logRows,'VariableNames', ...
            {'chromosome','validation_accuracy','blocks','fitness','seconds','evaluation'}), ...
            fullfile(outDir,'candidate_log.csv'));
        if value<bestSeen
            bestSeen=value;
            save(fullfile(outDir,'best_candidate.mat'),'z','arch','value','acc');
        end
    end

    function children = crossOver(parents,~,nvars,~,~,population)
        children = zeros(numel(parents)/2,nvars);
        for k=1:size(children,1)
            cut = randi(nvars-1);
            children(k,:) = [population(parents(2*k-1),1:cut), ...
                population(parents(2*k),cut+1:end)];
        end
        children = mutateRows(children);
    end

    function children = mutateChildren(parents,~,~,~,~,~,population)
        children = mutateRows(population(parents,:));
    end

    function rows = mutateRows(rows)
        for i=1:size(rows,1)
            if rand<c.mutationIndividual
                for j=1:10
                    if rand<c.mutationGene
                        rows(i,j)=randi([c.lb(j),c.ub(j)]);
                    end
                end
            end
        end
    end

    function [state,options,optchanged] = checkpoint(options,state,flag)
        optchanged = false;
        if strcmp(flag,'iter') || strcmp(flag,'done')
            gaState=state; randomState=rng; %#ok<NASGU>
            save(fullfile(outDir,'ga_checkpoint.mat'),'gaState','randomState','c');
        end
    end
end

function ds = prepared(imds,c)
% Resizing and grayscale conversion only; no random data augmentation.
ds = augmentedImageDatastore(c.imageSize(1:2),imds,'ColorPreprocessing','rgb2gray');
end
