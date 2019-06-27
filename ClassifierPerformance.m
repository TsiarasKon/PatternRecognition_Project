[X, Y] = loadCleanData();
indices = crossvalind('Kfold', Y, 10);

%{
having already split our data to k equal splits,
for each classifier:
    1) Create a classperf class (cp) using the true categories values
    for each split:
        2a) train the classifier using the k-1 other splits
        2b) use the remaining 1 as test set to make predictions
        2c) update cp with the current prediction
    3) Print Accuracy, Sensitivity and Specificity for the classifier
%}

% KNN
cp = classperf(Y);
disp("KNN (3 neighbors):");
for i = 1:10
    test = (indices == i); 
    train = ~test;
    model = fitcknn(X(train, :), Y(train), 'NumNeighbors', 3);
    predictions = predict(model, X(test, :));
    classperf(cp, predictions, test);
end
fprintf("  Accuracy: %f\n", cp.CorrectRate);
fprintf("  Sensitivity: %f\n", cp.Sensitivity);
fprintf("  Specificity: %f\n\n", cp.Specificity);

cp = classperf(Y);
disp("KNN (5 neighbors):");
for i = 1:10
    test = (indices == i); 
    train = ~test;
    model = fitcknn(X(train, :), Y(train), 'NumNeighbors', 5);
    predictions = predict(model, X(test, :));
    classperf(cp, predictions, test);
end
fprintf("  Accuracy: %f\n", cp.CorrectRate);
fprintf("  Sensitivity: %f\n", cp.Sensitivity);
fprintf("  Specificity: %f\n\n", cp.Specificity);

cp = classperf(Y);
disp("KNN (7 neighbors):");
for i = 1:10
    test = (indices == i); 
    train = ~test;
    model = fitcknn(X(train, :), Y(train), 'NumNeighbors', 7);
    predictions = predict(model, X(test, :));
    classperf(cp, predictions, test);
end
fprintf("  Accuracy: %f\n", cp.CorrectRate);
fprintf("  Sensitivity: %f\n", cp.Sensitivity);
fprintf("  Specificity: %f\n\n", cp.Specificity);

% Naive Bayes
cp = classperf(Y);
disp("Naive Bayes (normal):");
for i = 1:10
    test = (indices == i); 
    train = ~test;
    model = fitcnb(X(train, :), Y(train), 'DistributionNames', 'normal');
    predictions = predict(model, X(test, :));
    classperf(cp, predictions, test);
end
fprintf("  Accuracy: %f\n", cp.CorrectRate);
fprintf("  Sensitivity: %f\n", cp.Sensitivity);
fprintf("  Specificity: %f\n\n", cp.Specificity);

cp = classperf(Y);
disp("Naive Bayes (kernel):");
for i = 1:10
    test = (indices == i); 
    train = ~test;
    model = fitcnb(X(train, :), Y(train), 'DistributionNames', 'kernel');
    predictions = predict(model, X(test, :));
    classperf(cp, predictions, test);
end
fprintf("  Accuracy: %f\n", cp.CorrectRate);
fprintf("  Sensitivity: %f\n", cp.Sensitivity);
fprintf("  Specificity: %f\n\n", cp.Specificity);

% SVM
cp = classperf(Y);
disp("SVM (linear):");
for i = 1:10
    test = (indices == i); 
    train = ~test;
    model = fitcsvm(X(train, :), Y(train));
    predictions = predict(model, X(test, :));
    classperf(cp, predictions, test);
end
fprintf("  Accuracy: %f\n", cp.CorrectRate);
fprintf("  Sensitivity: %f\n", cp.Sensitivity);
fprintf("  Specificity: %f\n\n", cp.Specificity);

cp = classperf(Y);
disp("SVM (rbf):");
for i = 1:10
    test = (indices == i); 
    train = ~test;
    model = fitcsvm(X(train, :), Y(train), 'KernelFunction', 'rbf');
    predictions = predict(model, X(test, :));
    classperf(cp, predictions, test);
end
fprintf("  Accuracy: %f\n", cp.CorrectRate);
fprintf("  Sensitivity: %f\n", cp.Sensitivity);
fprintf("  Specificity: %f\n\n", cp.Specificity);

% Decision Tree
cp = classperf(Y);
disp("Decision Tree (allsplits):");
for i = 1:10
    test = (indices == i); 
    train = ~test;
    model = fitctree(X(train, :), Y(train));
    predictions = predict(model, X(test, :));
    classperf(cp, predictions, test);
end
fprintf("  Accuracy: %f\n", cp.CorrectRate);
fprintf("  Sensitivity: %f\n", cp.Sensitivity);
fprintf("  Specificity: %f\n\n", cp.Specificity);

cp = classperf(Y);
disp("Decision Tree (curvature):");
for i = 1:10
    test = (indices == i); 
    train = ~test;
    model = fitctree(X(train, :), Y(train), 'PredictorSelection', 'curvature');
    predictions = predict(model, X(test, :));
    classperf(cp, predictions, test);
end
fprintf("  Accuracy: %f\n", cp.CorrectRate);
fprintf("  Sensitivity: %f\n", cp.Sensitivity);
fprintf("  Specificity: %f\n\n", cp.Specificity);
