%% Post Process the data to reduce dimensionality

%% Split into training and testing
data = preprocessed_data';
rng('default');
cv = cvpartition(size(data,1), 'Holdout', 0.3);
idx = cv.test;
dataTrain = data(~idx, :);
dataTest = data(idx, :);

%% PCA
coeff = pca(dataTrain);
for k = [50, 100, 250, 500]
    transformed = dataTrain * coeff(:,1:k);
    
end
