function fn = getBatchFn(opts, meta)
bopts.numThreads = opts.numFetchThreads ;
bopts.pad = opts.pad; 
bopts.border = opts.border ;
bopts.transformation = opts.aug ;
bopts.imageSize = meta.normalization.imageSize ;
bopts.averageImage = meta.normalization.averageImage ;
bopts.rgbVariance = meta.augmentation.rgbVariance ;
% bopts.transformation = meta.augmentation.transformation ;

switch lower(opts.networkType)
  case 'simplenn'
    fn = @(x,y) getSimpleNNBatch(bopts,x,y) ;
  case 'dagnn'
    error('dagnn version not yet implemented');
end

% -------------------------------------------------------------------------
function [im,labels] = getSimpleNNBatch(opts, imdb, batch)
% -------------------------------------------------------------------------
if nargout > 1, labels = imdb.images.class(batch); end
isVal = ~isempty(batch) && imdb.images.set(batch(1)) ~= 1 ;
nViews = imdb.meta.nViews; 

batch = bsxfun(@plus,repmat(batch(:)',[nViews 1]),(0:nViews-1)');
batch = batch(:)'; 

images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;

if ~isVal, % training
  im = cnn_shape_get_batch(images, opts, ...
    'prefetch', nargout == 0, ...
    'nViews', nViews); 
else
  im = cnn_shape_get_batch(images, opts, ...
    'prefetch', nargout == 0, ...
    'nViews', nViews, ...
    'transformation', 'none'); 
end

nAugs = size(im,4)/numel(images); 
if nargout > 1, labels = repmat(labels(:)',[1 nAugs]); end
