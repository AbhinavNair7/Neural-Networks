function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================

Y_matrix = eye(num_labels)(y,:);
X_matrix = [ones(m,1) X];


z2 = X_matrix * Theta1';
a2 = [ones(size(z2,1),1) sigmoid(z2)];
H = sigmoid(a2 * Theta2');
J = (1/m)*sum(sum(-Y_matrix.*log(H)-(1-Y_matrix).*log(1-H)));


temp1 = [zeros(size(Theta1,1),1) Theta1(:,2:end)];
temp2 = [zeros(size(Theta2,1),1) Theta2(:,2:end)];
J = J + (lambda/(2*m))*(sum(sum(temp1.^2))+sum(sum(temp2.^2)));


delta3 = H - Y_matrix;
Triangle_2 = delta3' * a2;
Theta2_grad = (1/m) * Triangle_2;

delta2 = (delta3 * temp2(:,2:end)) .* sigmoidGradient(z2);
Triange_1 = delta2' * X_matrix;
Theta1_grad = (1/m) * Triange_1;


Theta2_grad = Theta2_grad + (lambda/m)*temp2;
Theta1_grad = Theta1_grad + (lambda/m)*temp1;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
