function W = randInitializeWeights(L_in, L_out)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
     W = zeros(L_out, 1 + L_in);
    epsilon_init = 0.12;
    W = rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;
end

