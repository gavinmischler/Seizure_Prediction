% Preprocess data

clear
clc

patient = 'Pat1';

data_dir = ['..\contest_data_downloader\test_download\' patient 'Train\'];
output_dir = 'preprocessed\';

files = dir(data_dir);
labels = zeros(size(3:length(files)));
preprocessed_data = [];
bad_patients = cell(0);

for filenum = 3:length(files)
    
    
    load([data_dir files(filenum).name]);
    filename = files(filenum).name;
    if min(std(data)) < .0001 % dropout data
        bad_patients{end+1} = filename;
        continue 
    end
    
    
    label = str2num(filename(end-4));
    labels(filenum-2) = label;
    
    %% DWT
    dwt_feats = [];
    for i = 1:size(data,2)
        [c,l] = wavedec(data(:,i), 6, 'db4');
        [cd1,cd2,cd3,cd4,cd5,cd6] = detcoef(c,l,[1 2 3 4 5 6]);
        details = {cd1, cd2, cd3, cd4, cd5, cd6};
        
        means = zeros(1,6);
        powers = zeros(1,6);
        stdevs = zeros(1,6);
        ratios = zeros(1,6);
        skews = zeros(1,6);
        kurts = zeros(1,6);
        
        for i = 1:6
            detail = details{i};
            means(i) = mean(detail);
            powers(i) = rms(detail) / length(detail);
            stdevs(i) = std(detail);
            if i < 6
                ratios(i) = abs(mean(detail) / mean(details{i+1}));
            end
            skews(i) = skewness(detail);
            kurts(i) = kurtosis(detail);
        end
        dwt_feats = [dwt_feats means powers stdevs ratios skews kurts];
        
    end
    
    %% EMD
    emd_feats = [];
    for i = 1:size(data,2)
        
        [imf,residual,info] = emd(data(:,i), 'MaxNumIMF', 7, 'Display', 0);
        if info.NumIMF < 7
            disp('Oh NOOOOOOO')
            disp(i)
            break
        end
        
        
        means = mean(imf);
        powers = rms(imf) ./ size(imf,1);
        stdevs = std(imf);
        ratios = abs(means(1:end-1) ./ means(2:end));
        skews = skewness(imf);
        kurts = kurtosis(imf);
        if (length([means powers stdevs ratios skews kurts]) ~= 7*5 + 6)
            disp('') 
        end
        
        emd_feats = [emd_feats means powers stdevs ratios skews kurts];
    end
    
    %% Wavelet Packet Decomposition
    wpd_feats = [];
    for i = 1:size(data,2)
        T = wpdec(data(:,i), 4, 'db4');
        
        means = zeros(1,16);
        powers = zeros(1,16);
        stdevs = zeros(1,16);
        ratios = zeros(1,15);
        skews = zeros(1,16);
        kurts = zeros(1,16);
        
        for i = 1:16
            X = wpcoef(T, i);
            means(i) = mean(X);
            powers(i) = rms(X) / length(X);
            stdevs(i) = std(X);
            if i < 16
                ratios(i) = abs(means(i) / mean(wpcoef(T, i+1)));
            end
            skews(i) = skewness(X);
            kurts(i) = kurtosis(X);
        end
        
        wpd_feats = [wpd_feats means powers stdevs ratios skews kurts];
    end
    
    full_feats = [dwt_feats emd_feats wpd_feats]';
    
    preprocessed_data = [preprocessed_data full_feats];
    
    fprintf('finished %d / %d\n', filenum-2, length(files)-2);
    
end

save([output_dir 'preprocessed_data.mat'], 'preprocessed_data');

% %% Split into training and testing
% data = processed_data';
% rng('default');
% cv = cvpartition(size(data,1), 'Holdout', 0.3);
% idx = cv.test;
% dataTrain = data(~idx, :);
% dataTest = data(idx, :);
% 
% %% PCA
% for k = [50, 100, 250, 500]
%     [coeffs, transformedTrain] = pca(data, 'NumComponents', k);
% 
%     
% end


