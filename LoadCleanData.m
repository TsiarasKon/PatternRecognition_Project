A = xlsread('breast-cancer-winsconsin-data');
CategoriesNum = A(:, end);
A = A(:, [2 : end-1]);      % remove "id" and "CategoriesNum" column
Categories = string(zeros(size(A,1), 1));
% Replace NaN ('?' in the dataset) with the most frequent value
ColModeVals = mode(A);
for i = 1 : size(A, 1)
    for j = 1 : size(A, 2)
        if isnan(A(i, j))
            A(i, j) = A(j);
        end
    end
    % also generate "Categories" matrix
    if CategoriesNum(i) == 2
        Categories(i) = "benign";
    else
        Categories(i) = "malignant";
    end
end