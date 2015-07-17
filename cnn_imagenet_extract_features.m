function cnn_imagenet_extract_features()
% CNN_IMAGENET_EXTRACT_FEATURES   
% Use pre-trained imagenet for extracting features 
% setup toolbox
run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', 'matlab', 'vl_setupnn.m')) ;

% download a pre-trained CNN from the web
%if ~exist('imagenet-vgg-f.mat', 'file')
%  urlwrite('http://www.vlfeat.org/matconvnet/models/imagenet-vgg-f.mat', ...
%    'imagenet-vgg-f.mat') ;
%end
net = load('pretrained_models/imagenet-vgg-m-2048.mat') ;
image_root='/home/wangxin/te/';
feature_root = '/home/wangxin/te_fea/';
% obtain and preprocess an image
dirs = dir([image_root])
N = length(dirs);
%Extract features in different ratio
for ratio = 30:100 
    %for class_index = 5:N
%        class_index
        %ims = dir([image_root dirs(class_index).name]);
        %im_N = length(ims);
        %if ~exist([feature_root   dirs(class_index).name '/'])
        %    mkdir([feature_root   dirs(class_index).name '/'])
        %end
        for im_index = 3:im_N
            %ims(im_index).name    
            im_ = imread([image_root dirs(class_index).name '/' ims(im_index).name]);
                
        for d1 =0:(100-ratio)/10
            for d2 = 0:(100-ratio)/10
                im_ = im(1+floor(d1*end*0.1):floor(d1*end*0.1 +end*ratio/100),1+floor(d2*end*0.1):floor(d2*end*0.1 +end*ratio/100),:);
                im_ = single(im_) ; % note: 255 range
                im_ = imresize(im_, net.normalization.imageSize(1:2)) ;
                try
                im_ = im_ - net.normalization.averageImage ;
                catch
                    continue
                end
                % run the CNN
                res = vl_simplenn(net, im_);
                features = res(end-2).x;
                dlmwrite([feature_root   dirs(class_index).name '/' ims(im_index).name(1:end-4) '.txt'],features,'delimiter','\n','precision',6)
            end
        end
    end
end
% show the classification result
%scores = squeeze(gather(res(end).x)) ;
%[bestScore, best] = max(scores) ;
%figure(1) ; clf ; imagesc(im) ;
%title(sprintf('%s (%d), score %.3f',...
%   net.classes.description{best}, best, bestScore)) ;
