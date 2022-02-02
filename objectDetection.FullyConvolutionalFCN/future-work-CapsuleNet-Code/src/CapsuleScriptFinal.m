% Train Capsule net
[CapsuleNet, CapsuleNetInfo] = trainNetwork(pximds,lgraph,options);

save('CapsuleNet.mat','CapsuleNet');
save('CapsuleNet.mat','CapsuleNetInfo');
save('CapsuleNet.mat','options');

% modelDateTime = datestr(now,'dd-mmm-yyyy-HH-MM-SS');
%     [net,info] = trainNetwork(dsTrain,lgraph,options);
%     save(['multispectralUnet-' modelDateTime '-Epoch-' num2str(maxEpochs) '.mat'],'net','options');