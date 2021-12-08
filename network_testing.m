%% Transpose the inputs, convert targets to doubles (only do once after loading in the inputs)
features_train = transpose(features_train);
features_test = transpose(features_test);
targets_train = double(transpose(targets_train));
targets_test = double(transpose(targets_test));

%% Test the network
testoutput = sim(network2`, features_test);

testoutput_binary = testoutput > 0.5;

accuracy = sum(sum(testoutput_binary==targets_test))/(2*length(testoutput))
good_accuracy = sum(testoutput_binary(1,:)==targets_test(1,:))/length(testoutput(1,:))
bad_accuracy = sum(testoutput_binary(2,:)==targets_test(2,:))/length(testoutput(2,:))

% Generate ROC curve
scores = testoutput(2, :);
labels = targets_test(2, :);
posclass = 'diaphragmatic_breathing';
[X,Y,T,AUC] = perfcurve(labels,scores,1);
plot(X, Y)
xlabel('False positive rate')
ylabel('True positive rate')
title('ROC Curve for Classification Neural Network')
dim= [.5 .3 .3 .3];
dim2 = [.5 .2 .3 .3];
str = strcat('AUC: ', string(AUC));
str2 = strcat('Accuracy: ', string(accuracy));
annotation('textbox',dim,'String',str,'FitBoxToText','on');
annotation('textbox',dim2,'String',str2,'FitBoxToText','on');
