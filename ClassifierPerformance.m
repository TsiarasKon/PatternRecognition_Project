[X, Y] = loadCleanData();
indices = crossvalind('Kfold', Y, 10);

cp = classperf(Y);
disp("KNN results:");
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
disp("Naive Bayes results:");
for i = 1:10
    test = (indices == i); 
    train = ~test;
    model = fitcnb(X(train, :), Y(train));
    predictions = predict(model, X(test, :));
    classperf(cp, predictions, test);
end
fprintf("  Accuracy: %f\n", cp.CorrectRate);
fprintf("  Sensitivity: %f\n", cp.Sensitivity);
fprintf("  Specificity: %f\n\n", cp.Specificity);

cp = classperf(Y);
disp("SVM results:");
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
disp("Decision Tree results:");
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

