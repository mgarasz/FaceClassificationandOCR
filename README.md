# FaceClassificationandOCR
Applies various combinations of classifiers (SVM, CNN, Decision Tree) to extracted features (LBP, HOG) of individual faces.

PLEASE HAVE TESTING IMAGES/VIDS IN THIS SAME FOLDER 
AS THE CLASSIFIERS ARE, INCLUDING THE cardDetector.xml

(loads in classifier within the function (bug hotfix))


####################################################################################################

RecogniseFace(I, featureType, classiferName)

(ie., RecogniseFace(I, 'HOG', 'SVM')

 Acceptable inputs:

featureType: 'HOG', 'LBP', 'Bag'
classiferName: 'CNN', 'SVM, 'TRE' (important not to input 'Tree', but full caps 'TRE')

I: must be an image, not path. 

Combinations:
SVM+HOG
SVM+LBP
SVM+Bag

TRE+HOG
TRE+LBP

CNN+'nil'



###################################################################################################################
detectNum(filename)

(ie: detectNum('IMG_0654.jpg'); )

notice how in the conditional statements I've covered all case-sensitive extentions. But if there is an error, please look at your filetype extension.

Acceptable files: 
VIDEO: .mov, .avi, .mp4   (including capitalized versions)
IMAGE: .jpg, .png (including capitalized versions)

