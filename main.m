%Read test data
tempX = dataset('File','KDDTrain+.txt','Delimiter',',');
tempY = dataset('File','KDDTrain+.txt','Delimiter',',');
tempX(:,2) = [];
tempX(:,2) = [];
tempX(:,2) = [];

% Read labels into y
for i = 1:41
    tempY(:,1) = [];
end
tempY(:,2) = [];

X = double(tempX);
y = double(tempY);

%remove label columns from X
X(:,40) = [];
X(:,39) = [];

%network architecture parameters
input_size = size(X, 2);
hidden_size = 150;
num_labels = 6;
num_samples = 30000;

%randomly initialize weights 
Init_Theta1 = randInitializeWeights(input_size, hidden_size);
Init_Theta2 = randInitializeWeights(hidden_size, num_labels);
%set regularization parameter lambda
lambda = 4;


%----Training-------
%unroll theta's for the optimization function
initial_thetaVec = [Init_Theta1(:); Init_Theta2(:)];
%[J, grad] = nnCostFunction(initial_thetaVec, input_size, hidden_size, num_labels,X, y, lambda);
options = optimset('MaxIter', 1000);
[thetaVec, cost] = fmincg (@(t)(nnCostFunction(t, input_size, hidden_size, num_labels,X, y, lambda, num_samples)), initial_thetaVec, options);


%reshape theta vector into the weight matrices
Theta1 = reshape(thetaVec(1:hidden_size * (input_size + 1)), hidden_size, (input_size + 1));
Theta2 = reshape(thetaVec((1 + (hidden_size * (input_size + 1))):end), num_labels, (hidden_size + 1));
            

%-----TESTING---
% Read
tempX = dataset('File','KDDTest+.txt','Delimiter',',');
tempY = dataset('File','KDDTest+.txt','Delimiter',',');
tempX(:,2) = [];
tempX(:,2) = [];
tempX(:,2) = [];

% Read labels into y
for i = 1:41
    tempY(:,1) = [];
end
tempY(:,2) = [];

%remove label columns from X
tempX(:,40) = [];
tempX(:,39) = [];

X2 = double(tempX);
y2 = double(tempY);


%---TESTING------
p = predict(Theta1, Theta2, X);

confusion = zeros([num_labels num_labels]);
for i = 1:size(X, 1)
    confusion(y(i), p(i)) = confusion(y(i), p(i)) + 1;
end

sum = 0;
for i = 1:num_labels
    sum = sum + confusion(i, i);
end
training_accuracy = sum / size(X, 1) * 100;

p2 = predict(Theta1, Theta2, X2);
correct = zeros([1 10000]);

confusion2 = zeros([num_labels num_labels]);
for i = 1:size(X2, 1)
    confusion2(y2(i), p2(i)) = confusion2(y2(i), p2(i)) + 1;
    if y2(i) == p2(i)
        correct(i) = 1;
    end
end

sum = 0;
for i = 1:num_labels
    sum = sum + confusion2(i, i);
end
testing_accuracy_1 = sum /size(X2, 1) * 100;

sum = 0;
for i = 1:4
    sum = sum + confusion2(i, 1);
    sum = sum + confusion2(i, 2);
    sum = sum + confusion2(i, 3);
    sum = sum + confusion2(i, 4);
end
sum = sum + confusion2(5, 5);
sum = sum + confusion2(6, 6);
testing_accuracy_2 = sum /size(X2, 1) * 100;
