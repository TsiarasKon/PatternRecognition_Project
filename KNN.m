[X, Y] = loadCleanData();
indices = crossvalind('Kfold', Y, 10);

cp = classperf(Y);
disp("KNN results:");
for i = 1:10
    test = (indices == i); 
    train = ~test;
    mdl = fitcknn(X(train,:), Y(train), 'NumNeighbors', 3);
    predictions = predict(mdl, X(test,:));
    classperf(cp, predictions, test);
end
fprintf("  Accuracy: %f\n", cp.CorrectRate);
fprintf("  Sensitivity: %f\n", cp.Sensitivity);
fprintf("  Specificity: %f\n", cp.Specificity);
