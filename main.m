%Read test data
X = dataset('File','KDDTrain+.txt','Delimiter',',');
y = dataset('File','KDDTrain+.txt','Delimiter',',');

% Read labels into y
for i = 1:41
    y(:,1) = [];
end
y(:,2) = [];

%remove label columns from X
X(:,42) = [];
X(:,42) = [];

%network architecture parameters
input_size = size(X, 1);
hidden_size = 200;
num_labels = 6;


%randomly initialize weights 
Init_Theta1 = randInitializeWeights(input_size, hidden_size);
Init_Theta2 = randInitializeWeights(hidden_size, num_labels);
%set regularization parameter lambda
lambda = 1.5;


%----Training-------
%unroll theta's for the optimization function
initial_thetaVec = [Init_Theta1(:); Init_Theta2(:)];
%[J, grad] = nnCostFunction(initial_thetaVec, input_size, hidden_size, num_labels,X, y, lambda);
options = optimset('MaxIter', 50);
[thetaVec, cost] = fmincg (@(t)(nnCostFunction(t, input_size, hidden_size, num_labels,X, y, lambda)), initial_thetaVec, options);


%reshape theta vector into the weight matrices
Theta1 = reshape(thetaVec(1:hidden_size * (input_size + 1)), hidden_size, (input_size + 1));
Theta2 = reshape(thetaVec((1 + (hidden_size * (input_size + 1))):end), num_labels, (hidden_size + 1));
            

%-----TESTING---
% Read digits
f3 = fopen('t10k-images-idx3-ubyte', 'r', 'b');
header = fread(f3, 4, 'int32');

train = fread(f3, 10000*784);
X2 = reshape(train, [784 10000]);
X2 = X2';

% Read digit labels
f4 = fopen('t10k-labels-idx1-ubyte', 'r', 'b');
header = fread(f4, 2, 'int32');
y2 = fread(f4, 10000);

for i = 1:size(y2, 1)
    if(y2(i) == 0)
        y2(i) = 10;
    end
end

fclose(f3);
fclose(f4);


%---TESTING------
p = predict(Theta1, Theta2, X);

confusion = zeros([num_labels num_labels]);
for i = 1:60000
    confusion(y(i), p(i)) = confusion(y(i), p(i)) + 1;
end

sum = 0;
for i = 1:10
    sum = sum + confusion(i, i);
end
training_accuracy = sum / 60000 * 100;

p2 = predict(Theta1, Theta2, X2);
correct = zeros([1 10000]);

confusion2 = zeros([num_labels num_labels]);
for i = 1:10000
    confusion2(y2(i), p2(i)) = confusion2(y2(i), p2(i)) + 1;
    if y2(i) == p2(i)
        correct(i) = 1;
    end
end

sum = 0;
for i = 1:10
    sum = sum + confusion2(i, i);
end
testing_accuracy = sum /10000 * 100;