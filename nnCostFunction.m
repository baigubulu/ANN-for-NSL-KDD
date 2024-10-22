function [J ,grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda, num_samples)
                               
% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


m = size(X, 1);
         
% need to return the following variables  
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


J = 0;
S = 0;
s = size(X);


    
    a = 1;
    b = 125973;
    r = (b-a).*rand(num_samples,1) + a;
    r = floor(r);
        

for jIterate = 1:num_samples
    
    i = r(jIterate, 1);
    
    y_new = zeros([num_labels 1]);
    label = y(i);
    y_new(label) = 1;
        
    a1 = ones([401 1]);
    a2 = ones([25 1]);
    a3 = ones([10 1]);

    a1 = X(i, 1:s(2));
    a1 = [1; a1'];
    z1 = Theta1 * a1;
    a2 = sigmoid(z1);
    a2 = [1; a2];
    z2 = Theta2 * a2;
    a3 = sigmoid(z2);
    h = a3;
     
    cost = 0;
    for k = 1:num_labels
        cost = cost + (-y_new(k)* log(h(k)) - ((1 - y_new(k)).* log(1 - h(k))));
    end
    
    S = S + cost;
end

J = S/num_samples;

square = 0;
for j = 1:hidden_layer_size
    for k = 2:input_layer_size+1
        square = square + (Theta1(j, k)^2);
    end
end

for j = 1:num_labels
    for k = 2:hidden_layer_size+1
        square = square + (Theta2(j, k)^2);
    end
end

J = J + (square * lambda / 2/ num_samples);


Delta_1 = zeros([hidden_layer_size input_layer_size + 1]); 
Delta_2 = zeros([num_labels hidden_layer_size+1]);

for jIter = 1:num_samples
    
    i = r(jIter, 1);
    
    y_new = zeros([num_labels 1]);
    y_new(y(i)) = 1;
        
    a1 = ones([401 1]);
    a2 = ones([25 1]);
    a3 = ones([10 1]);

    a1 = X(i, 1:s(2))';
    a1 = [1; a1];
    z2 = Theta1 * a1;
    a2 = sigmoid(z2);
    a2 = [1; a2];
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);
    h = a3;
    
    delta_3 = h - y_new;
    delta_2 = (Theta2(1:num_labels, 2:hidden_layer_size+1)'*delta_3) .* sigmoidGradient(z2);
    
    Delta_1 = Delta_1 + delta_2*a1';
    Delta_2 = Delta_2 + delta_3*a2';
end

Theta1_grad = Delta_1/num_samples;
Theta2_grad = Delta_2/num_samples;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
