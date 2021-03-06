function [X, Y] = loadCleanData()
% loads the data, removing the "id" column and returns:
% X: a matrix with the relevant clean data
% y: the Categories for each data row
    X = xlsread('breast-cancer-winsconsin-data');
    CategoriesNum = X(:, end);
    X = X(:, [2 : end-1]);      % remove "id" and "CategoriesNum" column
    Y = string(zeros(size(X,1), 1));
    % Replace NaN ('?' in the dataset) with the most frequent value
    ColModeVals = mode(X);
    for i = 1 : size(X, 1)
        for j = 1 : size(X, 2)
            if isnan(X(i, j))
                X(i, j) = ColModeVals(j);
            end
        end
        % also generate "Categories" matrix
        if CategoriesNum(i) == 2
            Y(i) = "benign";
        else
            Y(i) = "malignant";
        end
    end
    X = normalize(X, 'range');      % normalizes the data to [0, 1]
end
