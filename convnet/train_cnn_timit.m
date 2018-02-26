
if ~exist('imdb_m','var')
close all
gpud=gpuDevice(3);
reset(gpud)

addpath('../utils/')

%run('/home/bruna/matlab/matconvnet/matlab/vl_setupnn.m') ;
run('../matconvnet/matlab/vl_setupnn.m') ;

C = 1;
use_single = 1;

%representation = '/misc/vlgscratch3/LecunGroup/pablo/TIMIT/spect_fs16_NFFT1024_hop512/TRAIN/';
representation = '/tmp/';

load([representation 'female.mat']);
name = 'female';
imdb_f = prepareData_matconvnet(data,C,name,use_single);
clear data

load([representation 'male.mat']);
name = 'male';
imdb_m = prepareData_matconvnet(data,C,name,use_single);
clear data
fprintf('data ready \n')
end
%%

NFFT = size(imdb_f.images.data,3);
net.layers = {};
filter_num = 1024;
temp_context = 1;

%f1 = 1/sqrt(1*temp_context*NFFT);
f1 = 1;
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f1*randn(1, temp_context, NFFT,filter_num, 'single'), ...
                           'biases', zeros(1, filter_num, 'single'), ...
                           'stride', 1, ...
                           'pad',[0 0 floor(temp_context/2) floor(temp_context/2)]) ;

net.layers{end+1} = struct('type', 'relu') ;

%f2 = 1/sqrt(1*filter_num);
f2 = 1;
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f2*randn(1,1,filter_num,2*NFFT, 'single'), ...
                           'biases', zeros(1, 2*NFFT, 'single'), ...
                           'stride', 1, ...
                           'pad',0) ;
                       
%net.layers{end+1} = struct('type', 'relu') ;

% net.layers{end+1} = struct('type', 'normalize', ...
%                            'param', [2 1e-5 1 0.5]) ;

net.layers{end+1} = struct('type', 'normalize_audio', ...
                           'param', [2 1e-5 1 0.5]) ;
                       
net.layers{end+1} = struct('type', 'fitting', ...
                           'loss', 'L2') ;

opts.expDir = '/misc/vlgscratch3/LecunGroup/pablo/models/cnn/timit-cnn-relu/';
%opts.expDir = '/tmp/pablo/timit-cnn-test-lr/';
opts.train.batchSize = 500 ;
opts.train.numEpochs = 200;
opts.train.continue = false ;
opts.train.useGpu = true ;
opts.train.learningRate = [0.1*ones(1,10) 0.01*ones(1,50) 0.001];
opts.train.expDir = opts.expDir ;

% set validation set
epsilon = 0.001;
V = 2;
imdb_m.images.set(end-V*opts.train.batchSize+1:end) = 2;
imdb_f.images.set(end-V*opts.train.batchSize+1:end) = 2;

gB    = @(imdb1, imdb2, batch,batch2) getBatch_nmf(imdb1, imdb2, batch, batch2,epsilon);

%
[net,info_sc_init] = nmf_train(net, imdb_f, imdb_m, gB,opts.train) ;

