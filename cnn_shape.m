function net = cnn_shape(dataName, varargin)
%CNN_SHAPE Train an MVCNN on a provided dataset 
%
%   dataName:: 
%     must be name of a folder under data/
%   `baseModel`:: 'imagenet-matconvnet-vgg-m'
%     learning starting point
%   `fromScratch`:: false
%     if false, only the last layer is initialized randomly
%     if true, all the weight layers are initialized randomly
%   `numFetchThreads`::
%     #threads for vl_imreadjpeg
%   `aug`:: 'none'
%     specifies the operations (fliping, perturbation, etc.) used 
%     to get sub-regions
%   `viewpoolPos` :: 'relu5'
%     location of the viewpool layer, only used when multiview is true
%   `includeVal`:: false
%     if true, validation set is also used for training 
%   `useUprightAssumption`:: true
%     if true, 12 views will be used to render meshes, 
%     otherwise 80 views based on a dodecahedron
% 
%   `train` 
%     training parameters: 
%       `learningRate`:: [0.001*ones(1, 10) 0.0001*ones(1, 10) 0.00001*ones(1,10)]
%         learning rate
%       `batchSize`: 128
%         set to a smaller number on limited memory
%       `momentum`:: 0.9
%         learning momentum
%       `gpus` :: []
%         a list of available gpus
% 
% Hang Su

opts.networkType = 'simplenn'; % only simplenn is supported currently 
opts.baseModel = 'imagenet-matconvnet-vgg-m';
opts.fromScratch = false; 
opts.dataRoot = 'data' ;
opts.imageExt = '.jpg';
opts.numFetchThreads = 0 ;
opts.netvlad = false;
opts.netID = 'vgg-m';
opts.netvladPos = 'conv5';
opts.netvladMethod = 'vlad_preL2_intra';
opts.netvladUseAllData = false;
opts.netvladNumViews = 12;
opts.netvladXY = true;
opts.netvladTheta = true;
opts.multiview = true; 
opts.viewpoolPos = 'relu5';
opts.useUprightAssumption = true;
opts.aug = 'stretch';
opts.pad = 0; 
opts.border = 0; 
opts.numEpochs = [5 10 20]; 
opts.includeVal = false;
opts.test = false;
[opts, varargin] = vl_argparse(opts, varargin) ;

if opts.multiview, 
  opts.expDir = sprintf('%s-ft-%s-%s-%s', ...
    opts.baseModel, ...
    dataName, ...
    opts.viewpoolPos, ...
    opts.networkType); 
else
  opts.expDir = sprintf('%s-ft-%s-%s', ...
    opts.baseModel, ...
    dataName, ...
    opts.networkType); 
end
opts.expDir = fullfile(opts.dataRoot, opts.expDir);
[opts, varargin] = vl_argparse(opts,varargin) ;

opts.train.learningRate = [0.005*ones(1, 5) 0.001*ones(1, 5) 0.0001*ones(1,10) 0.00001*ones(1,10)];
opts.train.momentum = 0.9; 
opts.train.batchSize = 256; 
opts.train.maxIterPerEpoch = [Inf, Inf]; 
opts.train.balancingFunction = {[], []}; 
opts.train.gpus = []; 
opts.train = vl_argparse(opts.train, varargin) ;

opts.netvladOpts = struct('netID', opts.netID, ...
                          'layerName', opts.netvladPos, ...
                          'method', opts.netvladMethod, ...
                          'useGPU', numel(opts.train.gpus) > 0, ...
                          'useAllData', opts.netvladUseAllData, ...
                          'numViews', opts.netvladNumViews, ...
                          'xy', opts.netvladXY, ...
                          'theta', opts.netvladTheta, ...
                          'pad', opts.pad, ...
                          'border', opts.border);

if ~exist(opts.expDir, 'dir'), vl_xmkdir(opts.expDir) ; end

assert(strcmp(opts.networkType,'simplenn'), 'Only simplenn is supported currently'); 

% -------------------------------------------------------------------------
%                                                             Prepare data
% -------------------------------------------------------------------------
imdb = get_imdb(dataName); 
if ~opts.multiview, 
  nViews = 1;
else
  nShapes = length(unique(imdb.images.sid));
  nViews = length(imdb.images.id)/nShapes;
end
imdb.meta.nViews = nViews; 

opts.train.train = find(imdb.images.set==1);
opts.train.val = find(imdb.images.set==2);
if opts.includeVal, 
  opts.train.train = [opts.train.train opts.train.val];
  opts.train.val = [];
end
if opts.test
    opts.train.train = [];
    opts.train.val = find(imdb.images.set==3);
end
opts.train.train = opts.train.train(1:nViews:end);
opts.train.val = opts.train.val(1:nViews:end); 

% dbTrain used for NetVLAD
dbTrain = imdb;
train_val = find(dbTrain.images.set <= 2);
dbTrain.images = struct('name', {dbTrain.images.name(train_val)}, ...
                        'class', dbTrain.images.class(train_val), ...
                        'set', dbTrain.images.set(train_val), ...
                        'sid', dbTrain.images.sid(train_val), ...
                        'id', dbTrain.images.id(train_val));
dbTrain.name = dataName;
dbTrain.numImages = numel(dbTrain.images.name);

dbTrain.dbPath = imdb.imageDir;
dbTrain.dbImageFns = imdb.images.name(imdb.images.set <= 2);
dbTrain.name = dataName;
dbTrain.numImages = numel(dbTrain.dbImageFns);

% -------------------------------------------------------------------------
%                                                            Prepare model
% -------------------------------------------------------------------------
net = cnn_shape_init(imdb.meta.classes, ...
  'base', opts.baseModel, ...
  'restart', opts.fromScratch, ...
  'netvlad', opts.netvlad, ...
  'netvladOpts', opts.netvladOpts, ...
  'nViews', nViews, ...
  'viewpoolPos', opts.viewpoolPos, ...
  'networkType', opts.networkType, ...
  'dbTrain', dbTrain, ...
  'getBatchOpts', struct('pad', opts.pad, ...
                         'border', opts.border, ...
                         'aug', opts.aug, ...
                         'numFetchThreads', opts.numFetchThreads, ...
                         'networkType', opts.networkType) ...
  );  

% -------------------------------------------------------------------------
%                                                                    Learn 
% -------------------------------------------------------------------------
switch opts.networkType
  case 'simplenn', trainFn = @cnn_shape_train ;
  case 'dagnn', trainFn = @cnn_train_dag ;
end

trainable_layers = find(cellfun(@(l) isfield(l, 'weights') || ...
                                     isprop(l, 'weights'), net.layers));
disp('Trainable layers:')
for i = 1:numel(trainable_layers)
    l = trainable_layers(i);
    if isfield(net.layers{l}, 'learningRate') || isprop(net.layers{l}, 'learningRate')
        disp(net.layers{l}.name)
    else
        fprintf('Unexpected: layer %d (%s) has no learning rate\n', l, net.layers{l}.name)
    end
end
fc_layers = find(cellfun(@(s) numel(s.name)>=2 && strcmp(s.name(1:2),'fc'),net.layers));
fc_layers = intersect(fc_layers, trainable_layers); 
lr = cellfun(@(l) l.learningRate, net.layers(trainable_layers),'UniformOutput',false); 
layers_for_update = {trainable_layers(end), trainable_layers(end-1:end), trainable_layers}; 

for s=1:numel(opts.numEpochs), 
  if opts.numEpochs(s)<1, continue; end
  for i=1:numel(trainable_layers), 
    l = trainable_layers(i);
    if ismember(l,layers_for_update{s}), 
      net.layers{l}.learningRate = lr{i};
    else
      net.layers{l}.learningRate = lr{i}*0;
    end
  end

  net = trainFn(net, imdb, getBatchFn(opts, net.meta), ...
    'expDir', opts.expDir, ...
    net.meta.trainOpts, ...
    opts.train, ...
    'numEpochs', sum(opts.numEpochs(1:s)), ...
    'test', opts.test) ;
end

% -------------------------------------------------------------------------
%                                                                   Deploy
% -------------------------------------------------------------------------
net = cnn_imagenet_deploy(net) ;
modelPath = fullfile(opts.expDir, 'net-deployed.mat');

switch opts.networkType
  case 'simplenn'
    save(modelPath, '-struct', 'net') ;
  case 'dagnn'
    net_ = net.saveobj() ;
    save(modelPath, '-struct', 'net_') ;
    clear net_ ;
end
