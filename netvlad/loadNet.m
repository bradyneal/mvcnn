function [frontNet, backNet] = loadNet(netID, layerName)
    if nargin<2, layerName= '_relja_none_'; end
    
    switch netID
        case 'vd16'
            netname= 'imagenet-vgg-verydeep-16.mat';
        case 'vd19'
            netname= 'imagenet-vgg-verydeep-19.mat';
        case 'caffe'
            netname= 'imagenet-caffe-ref.mat';
        case 'places'
            netname= 'places-caffe.mat';
        case 'vgg-m'
            netname = 'imagenet-matconvnet-vgg-m.mat';
        otherwise
            error( 'Unknown network ID', netID );
    end
    
    paths= localPaths();
    net= load( fullfile(paths.pretrainedCNNs, netname));
    
    net= vl_simplenn_tidy(net); % matconvnet beta17 or newer is needed
    
    if isfield(net.meta, 'classes')
        net.meta= rmfield(net.meta, 'classes');
    end
    
    frontNet = net;
    backNet = {};
    if ~strcmp(layerName, '_relja_none_')
        frontNet = relja_cropToLayer(net, layerName);
        % default assumes there are ReLU and pooling layers following conv
        backNet = cropFront(net, layerName, 2);
        backNet= relja_swapLayersForEfficiency(backNet);
        layerNameStr= ['_', layerName];
    else
        layerNameStr= '';
    end
    
    frontNet= relja_swapLayersForEfficiency(frontNet);
    
    frontNet.meta.netID= netID;
    
    frontNet.meta.sessionID= sprintf('%s_offtheshelf%s', netID, layerNameStr);
    frontNet.meta.epoch= 0;
    
end


function net = cropFront(net, layerName, offset)
    if nargin < 3
        offset = 0;
    end
    net.layers= net.layers(relja_whichLayer(net, layerName) + 1 + offset:end);
end