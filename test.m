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
