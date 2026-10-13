function g = enas_graph(a,c,numClasses)
% Three-convolution bottlenecks with optional block and stack residuals.
g = layerGraph([imageInputLayer(c.imageSize,'Normalization','none','Name','input')
    convolution2dLayer(3,16,'Padding','same','Name','stem_conv')
    batchNormalizationLayer('Name','stem_bn')
    reluLayer('Name','stem_relu')]);
last = 'stem_relu'; channels = 16;
for s = 1:a.stacks
    stackInput = last; stackChannels = channels;
    for b = 1:a.blocks(s)
        p = sprintf('s%d_b%d',s,b);
        stride = 1; if b==1, stride=a.stride(s); end
        blockInput = last;
        layers = [convolution2dLayer(a.k1(s),a.middle(s),'Padding','same', ...
            'Stride',stride,'Name',[p '_c1'])
            batchNormalizationLayer('Name',[p '_bn1'])
            reluLayer('Name',[p '_r1'])
            convolution2dLayer(a.k2(s),a.middle(s),'Padding','same','Name',[p '_c2'])
            batchNormalizationLayer('Name',[p '_bn2'])
            reluLayer('Name',[p '_r2'])
            convolution2dLayer(a.k3(s),a.output(s),'Padding','same','Name',[p '_c3'])
            batchNormalizationLayer('Name',[p '_bn3'])];
        g = addLayers(g,layers);
        g = connectLayers(g,blockInput,[p '_c1']);
        last = [p '_bn3'];
        if a.blockResidual
            [g,last] = residual(g,blockInput,last,channels,a.output(s),stride,[p '_skip']);
        end
        g = addLayers(g,reluLayer('Name',[p '_out']));
        g = connectLayers(g,last,[p '_out']);
        last = [p '_out']; channels = a.output(s);
    end
    if a.stackResidual
        p = sprintf('s%d_long',s);
        [g,last] = residual(g,stackInput,last,stackChannels,channels,a.stride(s),p);
        g = addLayers(g,reluLayer('Name',[p '_out']));
        g = connectLayers(g,last,[p '_out']); last = [p '_out'];
    end
end
g = addLayers(g,[globalAveragePooling2dLayer('Name','gap')
    fullyConnectedLayer(numClasses,'Name','fc')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classification')]);
g = connectLayers(g,last,'gap');
end

function [g,last] = residual(g,source,target,inChannels,outChannels,stride,p)
skip = source;
if inChannels~=outChannels || stride~=1
    g = addLayers(g,[convolution2dLayer(1,outChannels,'Stride',stride, ...
        'Padding','same','Name',[p '_proj'])
        batchNormalizationLayer('Name',[p '_bn'])]);
    g = connectLayers(g,source,[p '_proj']); skip = [p '_bn'];
end
last = [p '_add'];
g = addLayers(g,additionLayer(2,'Name',last));
g = connectLayers(g,target,[last '/in1']);
g = connectLayers(g,skip,[last '/in2']);
end
