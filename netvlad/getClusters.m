function clsts= getClusters(net, opts, clstFn, k, dbTrain, trainDescFn)
    
    if ~exist(clstFn, 'file')
        
        if ~exist(trainDescFn, 'file')
            
            simpleNnOpts= {'conserveMemory', true, 'mode', 'test'};
            
            if opts.useGPU
                net= relja_simplenn_move(net, 'gpu');
            end
            
            % ---------- extract training descriptors
            
            relja_display('Computing training descriptors');
            
            nTrain= 50000;
            nPerImage= 100;
            nIm= ceil(nTrain/nPerImage);
            
            rng(43);
            trainIDs= randsample(dbTrain.numImages, nIm);
            
            if opts.theta
                % Get thetas for training examples
                viewIds = repmat(1:opts.numViews, 1, dbTrain.numImages / opts.numViews);
                thetas = normalizeAngles(viewIds(trainIDs), opts.numViews, 'unit');

                % Remove append_theta layer (manually added later in this file)
                frontNet = net;
                frontNet.layers = net.layers(1:relja_whichLayer(net, 'append_theta') - 1);
                backNet = net;
                backNet.layers = net.layers(relja_whichLayer(net, 'append_theta') + 1:end);
            end
            
            if opts.useAllData
                nIm = dbTrain.numImages;
                trainIDs = 1:dbTrain.numImages;
            end
            
            nTotal= 0;
            
            prog= tic;
            
            for iIm = 1:nIm
                relja_progress(iIm, nIm, 'extract train descs', prog);
                
                % --- extract descriptors
                
                % didn't want to complicate with batches here as it's only done once (per network and training set)
                
                im = cnn_shape_get_batch(...
                    {fullfile(dbTrain.dbPath, dbTrain.dbImageFns{trainIDs(iIm)})}, ...
                    'pad', opts.pad, ...
                    'border', opts.border);
                
                % fix non-colour images
                if size(im,3)==1
                    im= cat(3,im,im,im);
                end
                
                if opts.useGPU
                    im= gpuArray(im);
                end
                
                % Get CNN descriptors
                if opts.theta
                    frontRes = vl_simplenn(frontNet, im, [], [], simpleNnOpts{:});
                    
                    % Manually append theta
                    [sz1, sz2, sz3] = size(frontRes(end).x);
                    frontRes(end).x(:, :, sz3 + 1) = repmat(thetas(iIm), sz1, sz2);
                    
                    res = vl_simplenn(backNet, frontRes(end).x, [], [], simpleNnOpts{:}); 
                else
                    res = vl_simplenn(net, im, [], [], simpleNnOpts{:});
                end
                
                descs= gather(res(end).x);
                assignin('base', 'trainId', trainIDs(iIm))
                assignin('base', 'descs', descs)
                assignin('base', 'clustsIm', im)
                fprintf('imName: %s\n', dbTrain.dbImageFns{trainIDs(iIm)})
                disp('paused in getClusters.m')
                pause
                descs= reshape( descs, [], size(descs,3) )';
                
                % --- sample descriptors
                
                nThis= min( min(nPerImage, size(descs,2)), nTrain - nTotal );
                descs= descs(:, randsample( size(descs,2), nThis ) );
                
                if iIm==1
                    trainDescs= zeros( size(descs,1), nTrain, 'single' );
                end
                
                trainDescs(:, nTotal+[1:nThis])= descs;
                nTotal= nTotal+nThis;
            end
            
            trainDescs= trainDescs(:, 1:nTotal);
            
            % move back to CPU addLayers() assumes it
            if opts.useGPU
                net= relja_simplenn_move(net, 'cpu');
            end
            
            save(trainDescFn, 'trainDescs');
        else
            relja_display('Loading training descriptors');
            load(trainDescFn, 'trainDescs');
        end
        
        % ---------- Cluster descriptors
        
        relja_display('Computing clusters');
        clsts= yael_kmeans(trainDescs, k, 'niter', 500, 'verbose', 0, 'seed', 43);
        clear trainDescs;
        
        save(clstFn, 'clsts');
    else
        relja_display('Loading clusters');
        load(clstFn, 'clsts');
        assert(size(clsts, 2)==k);
    end
    
end
