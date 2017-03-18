function SVMDT(trainData,testData)

% Decision Tree using Support Vector Machine for classification

% Hybrid model of SVM and DT is used to achieve the good accuracy of SVM 
% with reduction of time using DT.

%--------------Training Phase----------------------------------------------

% Data to be used for training

[m1,n1] = size(trainData);
data1   = trainData(:,2:n1);
target1 = trainData(:,1);

% SVM is soft margin SVM classification problem with cost = 50
svm_struct=svmtrain(target1,data1,'-s 0 -t 2 -c 50');

sv = svm_struct.SVs;                    % Support Vectors
alph = svm_struct.sv_coef;              % Alpha values
weight = sv'*alph;                      % Weight vector
bias = -svm_struct.rho;                 % Bias value

%-----------------Testing Phase--------------------------------------------

% Data to be used for testing
[m2,n2] = size(testData);
data2   = testData(:,2:n1);
target2 = testData(:,1);

% To speed up SVMs in testing phase, conventional approach was reducing the 
% no. of support vectors, but SVMDT aims at "reducing no. of test datapoints" 
% that requires SVM's decision.

% In order to achieve this goal, we consider only those points which are close 
% to decision hyperplane for SVM classification.


decision_f = inline(' w*x + b ');            % Decision function
closeness_f = inline('(1/(1+exp(-y)))-0.5'); % Measure of closeness

new_target2 = target2;
% 'S' lies in the interval [0,0.5)
t = input('Enter threshold value for measure of closeness between 0 and 0.5: ');

for i=1:m2
    
    y=decision_f(bias, weight', data2(i,:)');   % Value of decision function
    S=closeness_f(y);                           % Measure of Closeness of data 
    S=abs(S);                                   % from decision hyperplane
                                            
    if S<=t                 % t is threshold value for measure of closeness
        new_target2(i)=3;   % If data is close to decision boundary,  
    end                     % then its new class is updated to 3.
    
end

%                     Structure of Decision Tree
%
%                           (root node)
%   (node1)                  (node2)                     (node3)
%  (class 1)                (class 2)              (class 1)  (class 2)


node1 = [];     % Univariate node for class 1
node2 = [];     % Univariate node for class 2

node3 = [];     % SVM will be used for classification in this case
                % when data is close to the decision boundary


% Testing in decision trees involves series of simple logical operations

for i = 1 : m2
    if new_target2(i)==1
        node1=[node1 i];
        
    elseif new_target2(i)==2
        node2=[node2 i];
        
    elseif new_target2(i)==3
        node3=[node3 i];
    end
end

fprintf('\n Out of 400 testing data samples, the no. of samples classified with SVM during testing is : %d \n\n',size(node3,2))

% Only data belonging to class 3 (i.e. closer to decision hyperplane) is
% considered for classification with SVM and remaining data is classified
% with DT. Thus, accuracy of SVM is achieved along with saving time in testing.

[predicted_class, acc, prob]=svmpredict( target2(node3), data2(node3,:), svm_struct);

end










