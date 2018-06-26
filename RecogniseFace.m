function ppl = RecogniseFace(I, featureType, classiferName)



%All classifier conditionals follow the same format so there's a lot of
%repeat comments... but the basic format is 
%1. Extract features specific to LBP/HOG > 
%2. Extract HOG specifically for feelings-classifier > 
%3. Feed both features into respective classifier and append to array.

faceDetector = vision.CascadeObjectDetector('MergeThreshold',5); %specify minimum to avoid artifacts
load('FeelingsClassifier.mat');
bbox = step(faceDetector, I);
ppl = [];

if isempty(bbox)
     bbox
     disp('No faces detected')
end


%% CONVOLUTIONAL NEURAL NET (ALEXNET)
if classiferName == 'CNN'
    
    load('CNNclassifier.mat');    
    [K, ~] = size(bbox);
    for x = 1:K
        face = imcrop(I,bbox(x,:)); %crop out area of interest
        face = imresize(face, [227 227]); %apply resizing if necessary
        face_f = imresize(face, [80 80]); %this resizing is specifically for the feelings-classifier as it uses HOG-SVM
        lvl = graythresh(face_f);
        face_f = im2bw(face_f, lvl);
        faceHog = extractHOGFeatures(face_f,'CellSize',[4 4]); %this is specifically for the feelings-classifier
        [feeling_lab, ~] = predict(feelingsClassifier, faceHog); %predict feels
        [label, ~] = classify(CNNclassifier, face); %classify face (666 = 'miscelaneous')
        co_x = round(bbox(x,1) + ((bbox(x,3)/2))); %this calculates the exact center of the bounding box coordinates
        co_y = round(bbox(x,2) + ((bbox(x,4)/2)));
        person = [str2num(char(label(1))) co_x co_y str2num(char(feeling_lab(1)))]; %create output array
        ppl = [ppl;person];
    end
    disp('Results:')
    ppl
end
 
%% SUPPORT VECTOR MACHINE       
if classiferName == 'SVM'
    
    %HOG
             if featureType == 'HOG'
                 load('SVMhogclassifier.mat'); %load up appropriate classifier
                 [K, ~] = size(bbox); %how many iterations? K = number of faces detected
                 for x = 1:K
                    face = imcrop(I,bbox(x,:));
                    face = imresize(face, [80 80]);
                    lvl = graythresh(face);
                    face = im2bw(face, lvl);
                    faceHog = extractHOGFeatures(face,'CellSize',[4 4]); 
                    [feeling_lab, ~] = predict(feelingsClassifier, faceHog);
                    [label, ~] = predict(SVMhog, faceHog);
                    co_x = round(bbox(x,1) + ((bbox(x,3)/2)));
                    co_y = round(bbox(x,2) + ((bbox(x,4)/2)));
                    person = [str2num(char(label(1))) co_x co_y str2num(char(feeling_lab(1)))];
                    ppl = [ppl;person];
                 end
             disp('Results:') 
             ppl
             
            
    %LBP
            elseif featureType == 'LBP'
                 load('SVMlbpclassifier.mat');
                 [K, ~] = size(bbox);
                 for x = 1:K
                    face = imcrop(I,bbox(x,:));
                    face = imresize(face, [125 125]); %This resize is done specifically for LBP feature extraction
                    face_f = imresize(face, [80 80]);
                    lvl = graythresh(face_f);
                    face_f = im2bw(face_f, lvl);
                    faceHog = extractHOGFeatures(face_f,'CellSize',[4 4]);
                    [feeling_lab, ~] = predict(feelingsClassifier, faceHog);
                    lvl = graythresh(face); %Get gray threshold levels
                    face = im2bw(face, lvl); %Convert to black/white according to above levels
                    faceLBP = extractLBPFeatures(face, 'CellSize', [32 32]); %extract features
                    [label, ~] = predict(SVMlbp, faceLBP); %predict
                    co_x = round(bbox(x,1) + ((bbox(x,3)/2)));
                    co_y = round(bbox(x,2) + ((bbox(x,4)/2)));
                    person = [str2num(char(label(1))) co_x co_y str2num(char(feeling_lab(1)))];
                    ppl = [ppl;person];
                 end
            disp('Results:')
            ppl
            
    %BAG OF FEATURES
    
            elseif featureType == 'Bag'
                load('SVMbagclassifier.mat');
            	[K,~] = size(bbox);
                 for x = 1:K
                    face = imcrop(I,bbox(x,:));
                    face_f = imresize(face, [80 80]); %no resizing needed specicialy for BagOfFeatures. This is for feelings HOGs
                    lvl = graythresh(face_f);
                    face_f = im2bw(face_f, lvl);
                    faceHog = extractHOGFeatures(face_f,'CellSize',[4 4]);
                    [feeling_lab, ~] = predict(feelingsClassifier, faceHog);
                    [label_1, ~] = predict(SVMbag, face); 
                    a = SVMbag.Labels(label_1);
                    label = a{1}; %label was stored in cell oddly so... this is here.
                    co_x = round(bbox(x,1) + ((bbox(x,3)/2)));
                    co_y = round(bbox(x,2) + ((bbox(x,4)/2)));
                    person = [str2num(char(label)) co_x co_y str2num(char(feeling_lab(1)))];
                    ppl = [ppl;person];
                 end
             disp('Results:')
             ppl
             end
end

                         
%% RANDOM FOREST           

if classiferName == 'TRE'
             
        %HOG
             if featureType == 'HOG'
             [K, ~] = size(bbox);
             load('treeHOGclassifier.mat');
                 for x = 1:K
                    face = imcrop(I,bbox(x,:));
                    face = imresize(face, [80 80]);
                    lvl = graythresh(face);
                    face = im2bw(face, lvl);
                    faceHog = extractHOGFeatures(face,'CellSize',[4 4]);
                    [feeling_lab, ~] = predict(feelingsClassifier, faceHog);
                    label = predict(treeHog,faceHog);
                    co_x = round(bbox(x,1) + ((bbox(x,3)/2)));
                    co_y = round(bbox(x,2) + ((bbox(x,4)/2)));
                    person = [str2num(char(label{1})) co_x co_y str2num(char(feeling_lab(1)))];
                    ppl = [ppl;person];
                 end
             disp('Results:')
             ppl
            
        %LBP
             elseif featureType == 'LBP'
             load('treeLBPclassifier.mat');
             [K, ~] = size(bbox);
                 for x = 1:K
                    face = imcrop(I,bbox(x,:));
                    face = imresize(face, [125 125]);
                    face_f = imresize(face, [80 80]);
                    lvl = graythresh(face_f);
                    face_f = im2bw(face_f, lvl);
                    faceHog = extractHOGFeatures(face_f,'CellSize',[4 4]); 
                    [feeling_lab, ~] = predict(feelingsClassifier, faceHog);
                    lvl = graythresh(face);
                    face = im2bw(face, lvl);
                    faceLBP = extractLBPFeatures(face, 'CellSize', [32 32]);
                    [label, ~] = predict(treeLBP, faceLBP);
                    co_x = round(bbox(x,1) + ((bbox(x,3)/2)));
                    co_y = round(bbox(x,2) + ((bbox(x,4)/2)));
                    person = [str2num(char(label(1))) co_x co_y str2num(char(feeling_lab(1)))];
                    ppl = [ppl;person];
                 end
             disp('Results:')
             ppl

          end
end
end

      