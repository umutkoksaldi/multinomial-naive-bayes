% Author: Umut Mucahit Koksaldi
% ID: 21402234
% CS464 Assignment 1

function main(path)
    % QUESTION 3.4

    % import and pre-process the data
    train_file = fullfile(path, 'traindata.txt');
    test_file = fullfile(path, 'testdata.txt');
    train_data = importdata(train_file);
    test_data = importdata(test_file);
    X_train = train_data.data;
    y_train = train_data.textdata;
    X_test = test_data.data;
    y_test = test_data.textdata;
    y_train_categorical = zeros([1000 1]);
    y_test_categorical = zeros([400 1]);

    % replace categorical data in training set
    for i = 1:1000
        if strcmp(y_train(i), 'student')
            y_train_categorical(i) = 1;
        else 
            y_train_categorical(i) = 0;
        end
    end

    % replace categorical data in test set
    for i = 1:400
        if strcmp(y_test(i), 'student')
            y_test_categorical(i) = 1;
        else 
            y_test_categorical(i) = 0;
        end
    end

    % split the final categorical data into student and faculty subsbets
    X_train_student = X_train(1:500, :);
    X_train_faculty = X_train(501:1000, :);
    y_train_student = y_train(1:500, :);
    y_train_faculty = y_train(501:1000, :);

    student_thetas = zeros([1 1309]);
    faculty_thetas = zeros([1 1309]);
    % determine student thetas
    for i = 1:1309
        T_s = sum(X_train_student(:, i));
        totalSum = sum(sum(X_train_student));
        student_thetas(1, i) = T_s / totalSum;
    end
    % determine faculty thetas
    for i = 1:1309
        T_s = sum(X_train_faculty(:, i));
        totalSum = sum(sum(X_train_faculty));
        faculty_thetas(1, i) = T_s / totalSum;
    end

    pred = zeros([400 1]);

    % make predictions on the test set
    for i = 1:400
        sum1 = 0;
        sum2 = 0;
        for j = 1:1309
            % check if the expression is not a number for the case of 0 * log(0)
            if ~isnan(X_test(i, j) * log(student_thetas(1, j)))
                sum1 = sum1 + X_test(i, j) * log(student_thetas(1, j));

            end
            % check if the expression is not a number for the case of 0 * log(0)
            if ~isnan(X_test(i, j) * log(faculty_thetas(1, j)))
                sum2 = sum2 + X_test(i, j) * log(faculty_thetas(1, j));
            end
        end
        % calculate the probabilities of student and faculty class
        argmaxY1 = log(1/2) + sum1;
        argmaxY0 = log(1/2) + sum2;
        % assign the class 
        if argmaxY1 > argmaxY0
            pred(i, 1) = 1;
        else
            pred(i, 1) = 0;
        end
    end

    % construct the confusion matrices
    confusion_student = confusionmat(y_test_categorical(1:200, 1), pred(1:200, 1));
    confusion_faculty = confusionmat(y_test_categorical(201:400, 1), pred(201:400, 1));

    % report accuracy from the confusion matrices
    accuracy_student = confusion_student(2, 2) / sum(sum(confusion_student));
    accuracy_faculty = confusion_faculty(1, 1) / sum(sum(confusion_faculty));

    display(accuracy_student);
    display(accuracy_faculty);      

    % QUESTION 3.5
    % find the mutual information relationships
    mutual_information_matrix = zeros([1 1309]);

    accuracy_matrix = zeros([1 1309]);

    for i = 1:1309
        n = 1000;
        % calculate parameters
        n11 = sum(X_train_student(:,i)~=0);
        n10 = sum(X_train_student(:,i)==0);
        n01 = sum(X_train_faculty(:,i)~=0);
        n00 = sum(X_train_faculty(:,i)==0);

        % calculate the mutual information
        if ~isnan(((n11/n) * log2(n*n11 / ((n10 + n11) * (n10 + n11)))) + ((n01 / n) * log2(n*n01 / ((n01 + n00) * (n01 + n11)))) + ((n10 / n) * log2(n*n10 / ((n10 + n11) * (n10 + n00)))) + ((n00 / n) * log2(n*n00 / ((n00 + n01) * (n10 + n00)))))
            mutual_inf = ((n11/n) * log2(n*n11 / ((n10 + n11) * (n10 + n11)))) + ((n01 / n) * log2(n*n01 / ((n01 + n00) * (n01 + n11)))) + ((n10 / n) * log2(n*n10 / ((n10 + n11) * (n10 + n00)))) + ((n00 / n) * log2(n*n00 / ((n00 + n01) * (n10 + n00))));
        else
            mutual_inf = 0;
        end
        mutual_information_matrix(1, i) = mutual_inf;
    end

    % 10 most important features' indices
    sorted_mi = sort(mutual_information_matrix, 'descend');
    sorted_mi_t = sorted_mi.';
    sorted_mi_mod = reshape(sorted_mi_t(~isnan(sorted_mi_t)),[],size(sorted_mi,1)).';
    sorted_mi = sorted_mi_mod(1, 1:10);

    % indices of the 10 most important features stored in the array
    mi_top10_indices = arrayfun(@(x)find(mutual_information_matrix==x,1),sorted_mi);
    disp('Indices of 10 most important features:');
    disp(mi_top10_indices);

    % QUESTION 3.6
    % find indices of the least important features
    ascending_sorted = sort(mutual_information_matrix, 'ascend');
    ascending_indices = arrayfun(@(x)find(mutual_information_matrix==x,1),ascending_sorted);

    accuracy_matrix = zeros([1 1309]);

    % remove features at indices specified at ascending_indices, starting from
    % the last imporant one
    for i = 1:1308
        % reprocess data, removing features at specified indices
        removed_index = ascending_indices(1, i);
        X_train_student(:, removed_index) = 0;
        X_train_faculty(:, removed_index) = 0;
        X_test(:, removed_index) = 0;
        % recalculate thetas with the removed feature
        % determine student thetas
        for j = 1:(1309)
            T_s = sum(X_train_student(:, j));
            totalSum = sum(sum(X_train_student));
            student_thetas(1, j) = T_s / totalSum;
        end
        % determine faculty thetas
        for j = 1:(1309)
            T_s = sum(X_train_faculty(:, j));
            totalSum = sum(sum(X_train_faculty));
            faculty_thetas(1, j) = T_s / totalSum;
        end

        % make predictions on the test data again
        for k = 1:400
            sum1 = 0;
            sum2 = 0;
            for j = 1:(1309)
                % check if the expression is not a number for the case of 0 * log(0)
                if ~isnan(X_test(k, j) * log(student_thetas(1, j)))
                    sum1 = sum1 + X_test(k, j) * log(student_thetas(1, j));

                end
                % check if the expression is not a number for the case of 0 * log(0)
                if ~isnan(X_test(k, j) * log(faculty_thetas(1, j)))
                    sum2 = sum2 + X_test(k, j) * log(faculty_thetas(1, j));
                end
            end
            % calculate the probabilities of student and faculty class
            argmaxY1 = log(1/2) + sum1;
            argmaxY0 = log(1/2) + sum2;
            % assign the class 
            if argmaxY1 > argmaxY0
                pred(k, 1) = 1;
            else
                pred(k, 1) = 0;
            end
        end

        % construct the confusion matrices
        confusion_student = confusionmat(y_test_categorical(1:200, 1), pred(1:200, 1));
        confusion_faculty = confusionmat(y_test_categorical(201:400, 1), pred(201:400, 1));

        accuracy_matrix(1, i) = ((confusion_faculty(1,1) + confusion_student(2,2)) / (sum(sum(confusion_student)) + sum(sum(confusion_faculty))));

    end
    % plot the accuracies
    plot(accuracy_matrix) 
end
