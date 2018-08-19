function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
J_k = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


hidden_layer_gradient = zeros(size(Theta2));
input_layer_gradient = zeros(size(Theta1));

Y_out=zeros(m,num_labels);
% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

J_k = 0;

a1 = [ones(size(X, 1), 1) X];%[5000x401]%

z2 = a1*Theta1';%[5000*401]x[401*25] = [5000x25]%

a2_1 = sigmoid(z2);%[5000x25]%

a2 = [ones(size(a2_1, 1), 1) a2_1];%[5000x26]%

z3 = a2*Theta2';%[5000x26]x[26x10] = [5000x10]%

h = sigmoid(z3);%[5000x10]%

for k=1:num_labels
  J_k = J_k + (-1/m)*sum(((y == (k)).*log(h(:,k)) + (1 - (y == (k))).*(log(1 - h(:,k)))));%[1x1]
endfor

Theta1_reg = [zeros(size(Theta1, 1), 1) Theta1(:,2:end)];%[50001x26]%
Theta1_err = sum(sum(Theta1_reg.^2));%[1x1]%

Theta2_reg = [zeros(size(Theta2, 1), 1) Theta2(:,2:end)];%[50001x26]%
Theta2_err = sum(sum(Theta2_reg.^2));%[1x1]%

J = J_k + (lambda/(2*m))*(Theta1_err + Theta2_err);%[1x1]%


for j = 1:m;
  value = y(j,:);%y[5000x1]%
  Y_out(j,value)=1;%Y_out[5000x10]%
endfor

d3 = h - Y_out;

d2 = ((Theta2(:,2:end)'*d3').*(sigmoidGradient(z2))')';

delta1 = d2'*a1;

delta2 = d3'*a2;

regular_1 = (Theta1)*(lambda/m);

regular_1(:,1) = 0;

regular_2 = (Theta2)*(lambda/m);

regular_2(:,1) = 0;

Theta1_grad = delta1/m + regular_1;

Theta2_grad = delta2/m + regular_2;

% -------------------------------------------------------------



% =========================================================================

% Unroll gradients

grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
