function net = cnn_shape_init(classNames, varargin)
opts.base = 'imagenet-matconvnet-vgg-m'; 
opts.restart = false;
opts.netvlad = false;
opts.netvladOpts = struct('netID', 'vgg-m', ...
                          'layerName', 'conv5', ...
                          'method', 'vladv2_intra', ...
                          'useGPU', false);
opts.dbTrain = 'not provided';
opts.viewpoint = false;
opts.nViews = 12; 
opts.viewpoolPos = 'relu5'; 
opts.viewpoolType = 'max';
opts.weightInitMethod = 'xavierimproved';
opts.scale = 1;
opts.networkType = 'simplenn'; % only simplenn is supported currently
opts = vl_argparse(opts, varargin); 

assert(strcmp(opts.networkType,'simplenn'), 'Only simplenn is supported currently'); 

init_bias = 0.1;
nClass = length(classNames);

% Add .mat file extension if not already there and get path to file
if ~strcmpi(opts.base(end - 3:end),'.mat')
    opts.base = strcat(opts.base, '.mat');
end
if strcmp(opts.base(1:length('data')), 'data')
    netFilePath = opts.base;
else
    netFilePath = fullfile('data','models', opts.base);
end

% Download network if doesn't already exist
if ~exist(netFilePath,'file')
    fprintf('Downloading model (%s) ...', opts.base) ;
    vl_xmkdir(fullfile('data','models')) ;
    urlwrite(fullfile('http://www.vlfeat.org/matconvnet/models/', ...
      opts.base), netFilePath) ;
    fprintf(' done!\n');
end

% Load network and optionally add viewpoint and NetVLAD layers
if opts.netvlad
    % Load the network and split into two different groups of layers
    % (splitting is to allow for adding of NetVLAD and PCA layers)
    [frontNet, backNet] = loadNet(netFilePath, opts.netvladOpts.netID, ...
                                  opts.netvladOpts.layerName);
                              
    % Add viewpoint information: theta
    if opts.viewpoint
        % NOTE: Check how numViews is initialized when multiview is false!
        % Consider just printing it here.
        viewpointLayer = struct('name', 'viewpoint', ...
                                'type', 'custom', ...
                                'numViews', opts.numViews, ...
                                'forward', @viewpoint_fw, ...
                                'backward', @viewpoint_bw);
        frontNet = modify_net(frontNet, viewpointLayer, ...
                              'mode', 'add_layer', ...
                              'loc', opts.netvladOpts.layerName);
    end
                              
    % Add NetVLAD layers
    frontNet = addLayers(frontNet, opts.netvladOpts, opts.dbTrain);
    % Add PCA and whitening layers
%     frontNet = addPCA(frontNet, opts.dbTrain, 'doWhite', true, 'pcaDim', 4096, ...
%        'batchSize', 10, 'useGPU', opts.netvladOpts.useGPU);
    
    % Cut off all but last fully connected layer
    backNet.layers = backNet.layers(end-1:end);
    % Combine all layers
    net = frontNet;
    net.layers = [frontNet.layers backNet.layers];
    
    % Fix weight matrix dimensions where the two networks were put together
    iVlad = relja_whichLayer(net, 'vlad:core');
%     iPCA = relja_whichLayer(net, 'WPCA');
    iCutLayer = numel(frontNet.layers) + 1;
    dataType = class(net.layers{end-1}.weights{1});
    sz = [1 1 numel(net.layers{iVlad}.weights{end}) numel(net.layers{iCutLayer}.weights{2})]; 
    net.layers{iCutLayer}.weights{1} = init_weight(...
        struct('weightInitMethod', opts.weightInitMethod), ...
        sz(1), sz(2), sz(3), sz(4), dataType);
    net.layers{iCutLayer}.weights{2} = zeros(sz(4), 1, dataType);
else
    [net, ~] = loadNet(netFilePath, opts.netvladOpts.netID);
    dataType = class(net.layers{end-1}.weights{1});
end

assert(strcmp(net.layers{end}.type, 'softmax'), 'Wrong network format'); 

% Initiate the last but one layer w/ random weights
widthPrev = size(net.layers{end-1}.weights{1}, 3);
nClass0 = size(net.layers{end-1}.weights{1},4);
if nClass0 ~= nClass || opts.restart, 
  net.layers{end-1}.weights{1} = init_weight(opts, 1, 1, widthPrev, nClass, dataType);
  net.layers{end-1}.weights{2} = zeros(nClass, 1, dataType); 
end

% Initiate other layers w/ random weights if training from scratch is desired
if opts.restart, 
  w_layers = find(cellfun(@(c) isfield(c,'weights'),net.layers));
  for i=w_layers(1:end-1), 
    sz = size(net.layers{i}.weights{1}); 
    net.layers{i}.weights{1} = init_weight(opts, sz(1), sz(2), sz(3), sz(4), dataType);
    net.layers{i}.weights{2} = zeros(sz(4), 1, dataType); 
  end	
end

% Swap softmax w/ softmaxloss
net.layers{end} = struct('type', 'softmaxloss', 'name', 'loss') ;

% Insert viewpooling
if opts.nViews>1, 
  viewpoolLayer = struct('name', 'viewpool', ...
    'type', 'custom', ...
    'vstride', opts.nViews, ...
    'method', opts.viewpoolType, ...
    'forward', @viewpool_fw, ...
    'backward', @viewpool_bw);
  net = modify_net(net, viewpoolLayer, ...
        'mode','add_layer', ...
        'loc',opts.viewpoolPos);

  if strcmp(opts.viewpoolType, 'cat'),
    loc = find(cellfun(@(c) strcmp(c.name,'viewpool'), net.layers));
    assert(numel(loc)==1);
    w_layers = find(cellfun(@(c) isfield(c,'weights'), net.layers));
    loc = w_layers(find((w_layers-loc)>0,1)); % location of the adjacent weight layer
    if ~isempty(loc),
      sz = size(net.layers{loc}.weights{1});
      if length(sz)<4, sz = [sz ones(1,4-length(sz))]; end
      net.layers{loc}.weights{1} = init_weight(opts, sz(1), sz(2), sz(3)*opts.nViews, sz(4), dataType);
      net.layers{loc}.weights{2} = zeros(sz(4), 1, dataType);
      % random initialize layers after
      w_layers = w_layers(w_layers>loc);
      for i=w_layers(1:end-1),
        sz = size(net.layers{i}.weights{1});
        if length(sz)<4, sz = [sz ones(1,4-length(sz))]; end
        net.layers{i}.weights{1} = init_weight(opts, sz(1), sz(2), sz(3), sz(4), dataType);
        net.layers{i}.weights{2} = zeros(sz(4), 1, dataType);
      end
    end
  end

end

% update meta data
net.meta.classes.name = classNames;
net.meta.classes.description = classNames;

% special case: when no class names specified, remove fc8/prob layers
if nClass==0, 
    net.layers = net.layers(1:end-2);
end
    
end


% -------------------------------------------------------------------------
function weights = init_weight(opts, h, w, in, out, type)
% -------------------------------------------------------------------------
% See K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into
% rectifiers: Surpassing human-level performance on imagenet
% classification. CoRR, (arXiv:1502.01852v1), 2015.

switch lower(opts.weightInitMethod)
  case 'gaussian'
    sc = 0.01/opts.scale ;
    weights = randn(h, w, in, out, type)*sc;
  case 'xavier'
    sc = sqrt(3/(h*w*in)) ;
    weights = (rand(h, w, in, out, type)*2 - 1)*sc ;
  case 'xavierimproved'
    sc = sqrt(2/(h*w*out)) ;
    weights = randn(h, w, in, out, type)*sc ;
  otherwise
    error('Unknown weight initialization method''%s''', opts.weightInitMethod) ;
end

end


% -------------------------------------------------------------------------
function res_ip1 = viewpoint_fw(layer, res_i, res_ip1)
% -------------------------------------------------------------------------
[sz1, sz2, sz3, sz4] = size(res_i.x);
if mod(sz4,layer.vstride)~=0, 
    error('all shapes should have same number of views');
end
if strcmp(layer.method, 'avg'), 
    res_ip1.x = permute(...
        mean(reshape(res_i.x,[sz1 sz2 sz3 layer.vstride sz4/layer.vstride]), 4), ...
        [1,2,3,5,4]);
elseif strcmp(layer.method, 'max'), 
    res_ip1.x = permute(...
        max(reshape(res_i.x,[sz1 sz2 sz3 layer.vstride sz4/layer.vstride]), [], 4), ...
        [1,2,3,5,4]);
elseif strcmp(layer.method, 'cat'),
    res_ip1.x = reshape(res_i.x,[sz1 sz2 sz3*layer.vstride sz4/layer.vstride]);
else
    error('Unknown viewpool method: %s', layer.method);
end

end


% -------------------------------------------------------------------------
function res_i = viewpoint_bw(layer, res_i, res_ip1)
% -------------------------------------------------------------------------
[sz1, sz2, sz3, sz4] = size(res_ip1.dzdx);
if strcmp(layer.method, 'avg'), 
    res_i.dzdx = ...
        reshape(repmat(reshape(res_ip1.dzdx / layer.vstride, ...
                       [sz1 sz2 sz3 1 sz4]), ...
                [1 1 1 layer.vstride 1]),...
        [sz1 sz2 sz3 layer.vstride*sz4]);
elseif strcmp(layer.method, 'max'), 
    [~,I] = max(reshape(permute(res_i.x,[4 1 2 3]), ...
                [layer.vstride, sz4*sz1*sz2*sz3]),[],1);
    Ind = zeros(layer.vstride,sz4*sz1*sz2*sz3, 'single');
    Ind(sub2ind(size(Ind),I,1:length(I))) = 1;
    Ind = permute(reshape(Ind,[layer.vstride*sz4,sz1,sz2,sz3]),[2 3 4 1]);
    res_i.dzdx = ...
        reshape(repmat(reshape(res_ip1.dzdx, ...
                       [sz1 sz2 sz3 1 sz4]), ...
                [1 1 1 layer.vstride 1]),...
        [sz1 sz2 sz3 layer.vstride*sz4]) .* Ind;
elseif strcmp(layer.method, 'cat'),
    res_i.dzdx = reshape(res_ip1.dzdx, [sz1 sz2 sz3/layer.vstride sz4*layer.vstride]);
else
    error('Unknown viewpool method: %s', layer.method);
end

end


% -------------------------------------------------------------------------
function res_ip1 = viewpool_fw(layer, res_i, res_ip1)
% -------------------------------------------------------------------------
[sz1, sz2, sz3, sz4] = size(res_i.x);
if mod(sz4,layer.vstride)~=0, 
    error('all shapes should have same number of views');
end
if strcmp(layer.method, 'avg'), 
    res_ip1.x = permute(...
        mean(reshape(res_i.x,[sz1 sz2 sz3 layer.vstride sz4/layer.vstride]), 4), ...
        [1,2,3,5,4]);
elseif strcmp(layer.method, 'max'), 
    res_ip1.x = permute(...
        max(reshape(res_i.x,[sz1 sz2 sz3 layer.vstride sz4/layer.vstride]), [], 4), ...
        [1,2,3,5,4]);
elseif strcmp(layer.method, 'cat'),
    res_ip1.x = reshape(res_i.x,[sz1 sz2 sz3*layer.vstride sz4/layer.vstride]);
else
    error('Unknown viewpool method: %s', layer.method);
end

end


% -------------------------------------------------------------------------
function res_i = viewpool_bw(layer, res_i, res_ip1)
% -------------------------------------------------------------------------
[sz1, sz2, sz3, sz4] = size(res_ip1.dzdx);
if strcmp(layer.method, 'avg'), 
    res_i.dzdx = ...
        reshape(repmat(reshape(res_ip1.dzdx / layer.vstride, ...
                       [sz1 sz2 sz3 1 sz4]), ...
                [1 1 1 layer.vstride 1]),...
        [sz1 sz2 sz3 layer.vstride*sz4]);
elseif strcmp(layer.method, 'max'), 
    [~,I] = max(reshape(permute(res_i.x,[4 1 2 3]), ...
                [layer.vstride, sz4*sz1*sz2*sz3]),[],1);
    Ind = zeros(layer.vstride,sz4*sz1*sz2*sz3, 'single');
    Ind(sub2ind(size(Ind),I,1:length(I))) = 1;
    Ind = permute(reshape(Ind,[layer.vstride*sz4,sz1,sz2,sz3]),[2 3 4 1]);
    res_i.dzdx = ...
        reshape(repmat(reshape(res_ip1.dzdx, ...
                       [sz1 sz2 sz3 1 sz4]), ...
                [1 1 1 layer.vstride 1]),...
        [sz1 sz2 sz3 layer.vstride*sz4]) .* Ind;
elseif strcmp(layer.method, 'cat'),
    res_i.dzdx = reshape(res_ip1.dzdx, [sz1 sz2 sz3/layer.vstride sz4*layer.vstride]);
else
    error('Unknown viewpool method: %s', layer.method);
end

end
