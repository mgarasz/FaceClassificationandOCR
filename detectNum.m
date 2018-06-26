function number = detectNum(filename)

faceDetector = vision.CascadeObjectDetector('MinSize', [130, 130]); %specify minimum to avoid artifacts 
ext = filename(end-2:end); %get extention to determine filetype
nums = {}; %Where detected numbers will be stored


%% VIDEO
    %check file type
    if strcmp(ext,'mov') || strcmp(ext,'avi') || strcmp(ext, 'mp4') || strcmp(ext,'AVI') || strcmp(ext, 'MP4') || strcmp(ext,'MOV')
        videoFileReader = vision.VideoFileReader(filename);
        frameCount = 0;
        videoPlayer = vision.VideoPlayer();

        while ~isDone(videoFileReader) % get the next frame
            videoFrame = step(videoFileReader);
            frameCount = frameCount + 1;
            s = size(videoFrame);
            
        if s(2) > s(1)   %Rotates if necessary
            videoFrame = imrotate(videoFrame,-90);
        end

        if mod(frameCount,10) == 0 %check only every 10 frames (better performance)
            
            %Assumption: The size of the card is almost x2 the size of the
            %persons face. Use this assumption to create a dynamic
            %threshold.
            
            bboxf = step(faceDetector, videoFrame);
            min1 = round(bboxf(1,3)*0.5); %these are size thresholds.
            min2 = round(bboxf(1,4)*0.5);
            max1 = round(bboxf(1,3)*1.5);
            max2 = round(bboxf(1,4)*1.5);
            
            
            %Detect any cards using the head-card size proportions as size
            %thresholds
            cardDetector = vision.CascadeObjectDetector('cardDetector.xml', 'MinSize', [min1, min2], 'MaxSize', [max1, max2]);
            bbox = step(cardDetector,videoFrame);
            [N, ~] = size(bbox);
            
            %Likely some false positives, iterate through them. 
            for i = 1:N
            cardroi = imcrop(videoFrame,bbox(i,:));
            cardroi = imresize(cardroi, [227 227]);

            ocrResults = ocr(cardroi,'CharacterSet', '0123456789'); 
            ocrResults.Text;
            numOut = ocrResults.Text(find(~isspace(ocrResults.Text)));

            if length(numOut) == 2
                nums{end+1} = numOut; %add detected numbers to array
            end
            end 
        end 
            % Display the annotated video frame using the video player object
            step(videoPlayer, videoFrame);
        end
        
    number = unique(nums); %Show all numbers detected that have 2 characters. 
    disp('Numbers seen in video are:')
    disp(unique(nums))

    end

%% IMAGE

    if strcmp(ext,'jpg') || strcmp(ext,'png') || strcmp(ext,'JPG') || strcmp(ext,'PNG')
        I = imread(filename);
        %I = rgb2gray(I);
        
        s = size(I);
        if s(2) > s(1)  
            I = imrotate(I,-90);
        end

        %Use face known proportions to filter card region proposals
        bboxf = step(faceDetector, I);
        if ~isempty(bboxf)
            %if a face IS detected.. continue on with creating a dynamic
            %size threshold as did with video files.
            
            min1 = round(bboxf(1,3)*0.5);
            min2 = round(bboxf(1,4)*0.5);
            max1 = round(bboxf(1,3)*1.5);
            max2 = round(bboxf(1,4)*1.5);
            
            
            %Secondary process that uses filtering to spot contiguous white
            %areas. In the event that cardDetector fails, this is a back
            %up.
            
            cardprops = (bboxf(1,3) * bboxf(1,4));
            Ibw = im2bw(I,0.73); %converts to binary
            BW2 = bwareafilt(Ibw,[(cardprops*1.1) (cardprops*2.25)]);         
            stats = regionprops(BW2);
            %imshow(BW2);
            bb = stats.BoundingBox;
            rectangle('position',bb,'edgecolor','r','linewidth',2);

            s1 = stats(1).BoundingBox(1).*(1.10);
            s2 = stats(1).BoundingBox(2).*(1.02);
            s3 = stats(1).BoundingBox(3).*(0.6); %
            s4 = stats(1).BoundingBox(4).*(0.7);

            roi = [s1 s2 s3 s4];
            I2 = imcrop(I,roi);
            imshow(I2);

            ocrResult1 = ocr(roi,'CharacterSet', '0123456789'); 
            ocrResult1.Text;
            numOut1 = ocrResult1.Text(find(~isspace(ocrResult1.Text)));

            if length(numOut1) == 2
                nums{end+1} = numOut1;
            end

        else
            min1 = 100;
            min2 = 100;
            max1 = 800;
            max2 = 800; %if no face detected, set bbox to some random values (wide). Doesnt really matter as the algo will
                        %be iterating through all bounding boxes detected
                        %by card Detector.
        end


        %Card detector
        cardDetector = vision.CascadeObjectDetector('cardDetector.xml', 'MinSize', [min1, min2], 'MaxSize', [max1, max2]);
        bbox = step(cardDetector,I);
        [N, ~] = size(bbox);

        %Likely lots of false positives, iterate through them. 
        for i = 1:N
        cardroi = imcrop(I,bbox(i,:));
        cardroi = imresize(cardroi, [227 227]);

        ocrResults = ocr(cardroi,'CharacterSet', '0123456789'); 
        ocrResults.Text;
        numOut = ocrResults.Text(find(~isspace(ocrResults.Text)));

        if length(numOut) == 2
            nums{end+1} = numOut;
        end
       
        end

    end
    number = unique(nums);    
    disp('Numbers seen are:')
    disp(unique(nums))

end