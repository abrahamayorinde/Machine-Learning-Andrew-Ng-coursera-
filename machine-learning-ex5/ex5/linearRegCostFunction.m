function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%               You should set J to the cost and grad to the gradient.
h = X*theta;

err = (h-y).^2;

err_grad = (h-y);

J_sum = sum(err)/(2*m);

gamma = theta;

gamma(1,:) = 0;

%regularized_term = (lambda/(2*m))*sum(theta(2:end,:).^2);

%J = J_sum + regularized_term;
J = sum(err)/(2*m) + (lambda/(2*m))*sum(gamma.^2);

grad_reg = (1/m)*X'*err_grad + (lambda/m)*gamma;

grad = grad_reg;

% =========================================================================


end
